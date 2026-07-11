"""Compose the raw datagen outputs into the actual training corpora.

Two products:
  * the Stage-1 multi-capability SFT mixture (kernel repair/opt + kernel QA +
    agentic tool-use trajectories + ~45% general replay), via mixing.py; and
  * the Stage-2 DPO set with >=8% labeled reward-hack hard negatives folded in.

Everything degrades gracefully: missing datagen dirs simply contribute nothing,
and general replay always falls back to bundled samples, so this runs offline
(and in tests) with a StubTeacher.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

from kore.data import build_datasets as bd
from kore.data.general_replay import load_general_replay
from kore.data.gen_qa import generate_kernel_qa
from kore.data.hard_negatives import (
    build_hard_negative_group,
    meets_hard_negative_target,
)
from kore.data.mixing import build_multicap_sft, mixture_report
from kore.data.schemas import read_jsonl
from kore.obs import get_logger

log = get_logger("data.assemble")


def _read_dir(data_root: Path, sub: str, typed: bool = True) -> list:
    d = data_root / sub
    recs: list = []
    if d.exists():
        for p in sorted(d.glob("*.jsonl")):
            recs += read_jsonl(p, typed=typed)
    return recs


def _agentic_rows(data_root: Path) -> list[dict]:
    """Agentic trajectory records -> SFT chat rows (their ``messages``)."""
    rows: list[dict] = []
    d = data_root / "agentic"
    if not d.exists():
        return rows
    for p in sorted(d.glob("*.jsonl")):
        for rec in read_jsonl(p, typed=False):
            msgs = rec.get("messages") if isinstance(rec, dict) else None
            if msgs:
                rows.append({"messages": msgs})
    return rows


def assemble_multicap_sources(
    data_root, tasks, teacher, config, total: int, *, seed: int = 0,
    use_hf: bool = False, kernel_records: Optional[list] = None,
    extra_records: Optional[list] = None,
) -> dict[str, list]:
    """Build the ``{source_key: [chat rows]}`` dict for build_multicap_sft.

    Kernel repair/opt comes from generated repair+wins records; agentic from
    generated trajectories; kernel QA is synthesized from task seeds via the
    teacher; the ~45% general half comes from general_replay.

    ``kernel_records`` overrides the on-disk repair+wins scan with an explicit
    record list — used by the campaign to build SFT from a leakage-split TRAIN
    partition only (so held-out op families never leak into training).

    ``extra_records`` folds in additional kernel-bucket records produced *after*
    the base scan — the on-policy DAgger repairs, evolutionary ``WinRecord``s and
    on-policy relabeled wins. They are appended to (not substituted for) the
    kernel repair/opt bucket so the multi-capability SFT mix always INCLUDES the
    DAgger repairs the on-policy loop mined on the current policy's own failures.
    ``build_sft`` dispatches by record type, so ``RankedGroupRecord``s mixed in
    here are simply ignored (they belong to the DPO product).
    """
    data_root = Path(data_root)
    tasks = list(tasks)

    if kernel_records is None:
        krecs = _read_dir(data_root, "repair") + _read_dir(data_root, "wins")
    else:
        krecs = list(kernel_records)
    if extra_records:
        krecs = krecs + list(extra_records)
    kernel_repair_opt = bd.build_sft(krecs) if krecs else []

    # Curation (Pillar 6), applied to the kernel SFT pool BEFORE mixing so the
    # mixer's fractions are computed on a balanced, high-quality set: drop trivial
    # (<1.1x) win demos and cap per-op-family over-representation (gemm has many
    # tasks and would otherwise drown rmsnorm/quant/...). Repairs are never dropped
    # (they carry correctness lessons). Toggle with KORE_CURATE=0.
    if kernel_repair_opt and os.environ.get("KORE_CURATE", "1") != "0":
        from kore.data.curate import balance_by_family, filter_trivial_wins
        kernel_repair_opt, _ct = filter_trivial_wins(kernel_repair_opt, min_speedup=1.1)
        kernel_repair_opt, _cb = balance_by_family(kernel_repair_opt, cap_frac=0.30)
        log.metric("curate_kernel_sft",
                   dropped_trivial=_ct["n_dropped_trivial_wins"],
                   family_capped=_cb.get("capped", 0), kept=len(kernel_repair_opt))

    # Agentic slice = KORE-native tool trajectories (real build/test/bench/pmc on our
    # tools) + a "generic layer" of real public function-calling data (ToolACE via the
    # tool_use replay) so the slice teaches the general tool-use skill AND our schema,
    # and is never starved when native trajectories are sparse. ~40% generic / rest native.
    agentic_rows = _agentic_rows(data_root)
    agentic_n = max(1, int(round(config.frac_agentic_tooluse * total)))
    n_generic = max(1, int(round(0.4 * agentic_n)))
    generic_agentic = load_general_replay("tool_use", n_generic, seed + 40, use_hf)
    agentic_rows = agentic_rows + generic_agentic

    qa_n = max(1, int(round(config.frac_kernel_qa * total)))
    qa_rows = generate_kernel_qa(tasks, teacher, n=qa_n, seed=seed) if teacher and tasks else []

    gc = load_general_replay("code", max(1, int(round(config.frac_general_code * total))), seed + 10, use_hf)
    gm = load_general_replay("math", max(1, int(round(config.frac_math_reasoning * total))), seed + 20, use_hf)
    gch = load_general_replay("chat", max(1, int(round(config.frac_general_chat * total))), seed + 30, use_hf)

    out = {
        "kernel_repair_opt": kernel_repair_opt,
        "kernel_qa": qa_rows,
        "agentic_tooluse": agentic_rows,
        "general_code": gc,
        "math_reasoning": gm,
        "general_chat": gch,
    }
    log.metric(
        "assemble_sources", total=total,
        kernel_records=len(krecs),
        source_counts={k: len(v) for k, v in out.items()},
    )
    return out


def build_multicap_dataset(
    data_root, tasks, teacher, config, total: int, *, seed: int = 0,
    use_hf: bool = False, verbose: bool = True, kernel_records: Optional[list] = None,
    extra_records: Optional[list] = None,
) -> list[dict]:
    """Assemble + mix the Stage-1 multi-capability SFT dataset.

    ``kernel_records``, when given, supplies the kernel repair/opt records
    directly (a leakage-split TRAIN partition) instead of scanning ``data_root``.
    ``extra_records`` folds in the on-policy DAgger repairs + evolutionary/
    on-policy wins so the mix includes them (see ``assemble_multicap_sources``).
    """
    with log.stage("build_multicap_dataset", total=total, seed=seed):
        sources = assemble_multicap_sources(data_root, tasks, teacher, config, total,
                                            seed=seed, use_hf=use_hf,
                                            kernel_records=kernel_records,
                                            extra_records=extra_records)
        rows = build_multicap_sft(sources, config, total, seed=seed, verbose=verbose)
        log.metric("multicap_dataset_built", total=total, rows=len(rows))
        return rows


def build_dpo_with_hard_negatives(data_root, tasks, *, correct_source_fn=None,
                                  group_records: Optional[list] = None,
                                  extra_group_records: Optional[list] = None,
                                  hard_target: Optional[float] = None,
                                  prompt_fn=None,
                                  seed: int = 0) -> dict:
    """Stage-2 DPO rows = ranked-group prefs + labeled reward-hack hard negatives.

    ``correct_source_fn(task)->str`` supplies the trusted 'chosen' kernel for the
    hard-negative group (defaults to the task's verified seed). ``group_records``,
    when given, supplies the ranked-group records directly (a leakage-split TRAIN
    partition) instead of scanning ``data_root``. ``extra_group_records`` folds in
    additional ranked groups produced on-policy (iterative-DPO relabeling) or by
    the evolutionary loop, so the preference set covers states the policy actually
    visits.

    ``hard_target`` (e.g. 0.12): when set, the abundant ranked-group ("base")
    pairs are deterministically SUBSAMPLED (seeded, order-preserving) so the
    reward-hack hard negatives reach at least this fraction of the total. Every
    hard-negative pair is kept — only the redundant base pairs are thinned (DPO
    converges well below the tens-of-thousands of group pairs available), so the
    crucial anti-reward-hacking signal is not diluted below the spec floor.

    Returns ``{rows, n_hard, n_total, meets_target}`` where meets_target checks
    the >=8% hard-negative floor.
    """
    data_root = Path(data_root)
    tasks = list(tasks)
    with log.stage("build_dpo_with_hard_negatives", n_tasks=len(tasks)):
        correct_source_fn = correct_source_fn or (lambda t: t.seed_source)

        if group_records is None:
            group_records = _read_dir(data_root, "groups")
        else:
            group_records = list(group_records)
        if extra_group_records:
            group_records = group_records + list(extra_group_records)
        base_rows = bd.build_dpo(group_records, prompt_fn=prompt_fn) if group_records else []

        hard_groups = [build_hard_negative_group(correct_source_fn(t), t) for t in tasks]
        hard_rows = bd.build_dpo(hard_groups, prompt_fn=prompt_fn)

        # Boost the hard-negative fraction to >= hard_target by thinning the
        # over-abundant base pairs (keep ALL hard negatives). Seeded + order-
        # preserving so the subsample stays diverse across tasks/families and is
        # reproducible.
        n_base_full = len(base_rows)
        if hard_target and 0.0 < hard_target < 1.0 and hard_rows and base_rows:
            max_base = int(len(hard_rows) * (1.0 - hard_target) / hard_target)
            if 0 < max_base < len(base_rows):
                rng = random.Random(seed)
                keep = sorted(rng.sample(range(len(base_rows)), max_base))
                base_rows = [base_rows[i] for i in keep]

        rows = base_rows + hard_rows
        meets = meets_hard_negative_target(len(hard_rows), len(rows))
        hard_frac = (len(hard_rows) / len(rows)) if rows else 0.0
        log.metric(
            "dpo_hard_negatives", n_group_records=len(group_records),
            n_base_pairs=len(base_rows), n_base_full=n_base_full,
            n_hard=len(hard_rows), n_total=len(rows),
            hard_fraction=round(hard_frac, 4), hard_target=hard_target,
            meets_target=meets,
        )
        return {
            "rows": rows,
            "n_hard": len(hard_rows),
            "n_total": len(rows),
            "meets_target": meets,
        }


def summarize_multicap(rows: list[dict]) -> dict:
    return mixture_report(rows)
