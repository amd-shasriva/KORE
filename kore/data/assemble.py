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


def _agentic_row_is_heldout(rec: dict) -> bool:
    """True iff an agentic record belongs to the held-out generalization split --
    by task_id, by operator FAMILY (mla/paged), OR by foreign arch. Mirrors
    run_campaign._rec_is_heldout so the agentic slice enforces the SAME authority as
    every other SFT source (audit R2 sft I3: the old task_id-only check missed the
    MLA/paged family holdout + foreign-arch trajectories)."""
    from types import SimpleNamespace

    from kore.tasks.registry import (HELDOUT_FAMILIES, HELDOUT_TASKS, TRAIN_ARCHS,
                                     operator_family)
    prov = rec.get("_provenance") or {}
    tid = rec.get("task_id") or prov.get("task_id")
    if tid and tid in HELDOUT_TASKS:
        return True
    arch = rec.get("arch") or rec.get("gpu") or prov.get("arch")
    if arch is not None and arch not in TRAIN_ARCHS:
        return True
    op = rec.get("operation") or prov.get("op") or ((tid or "").split("_")[0] if tid else "")
    if op and operator_family(SimpleNamespace(operation=op, task_id=tid or "")) in HELDOUT_FAMILIES:
        return True
    return False


def _agentic_rows(data_root: Path) -> list[dict]:
    """Agentic trajectory records -> SFT chat rows (their ``messages``)."""
    rows: list[dict] = []
    d = data_root / "agentic"
    if not d.exists():
        return rows
    for p in sorted(d.glob("*.jsonl")):
        for rec in read_jsonl(p, typed=False):
            if not isinstance(rec, dict):
                continue
            # Defense-in-depth: never route a held-out generalization trajectory into
            # SFT (task / FAMILY / arch), even from a legacy on-disk shard (audit C2/R2).
            if _agentic_row_is_heldout(rec):
                continue
            msgs = rec.get("messages")
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
        # Headroom rebalance (WS-C3): cap the low-headroom (trivial/memory-bound)
        # kernel rows so compute-bound (gemm/attention/moe) reaches the target share
        # -- the audited pool was ~82% low-headroom, starving the demos that matter
        # on MI300X. Opt-in (frontier default ON); degrades gracefully on thin pools.
        if os.environ.get("KORE_REBALANCE_HEADROOM", "1") != "0":
            from kore.data.curate import rebalance_by_headroom
            _tc = float(os.environ.get("KORE_COMPUTE_FRAC", "0.5"))
            kernel_repair_opt, _rh = rebalance_by_headroom(kernel_repair_opt, target_compute_frac=_tc)
            log.metric("curate_headroom", **_rh)

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

    # Decontaminate the general-replay + generic-agentic slices against the held-out
    # eval sources (Pillar 5): a mined code/math row could carry a held-out attention
    # kernel. ALSO decontaminate against the RETENTION eval benchmarks (MMLU/HumanEval/
    # LCB/IFEval/BFCL/MT) -- the general slices are ~45% of the SFT mix, so a mined row
    # carrying an eval question is train-on-test that inflates the retention gate
    # (audit R2 sft, mirroring the midtrain corpus fix). One shared n-gram set.
    if os.environ.get("KORE_DECONTAM", "1") != "0":
        from kore.data.decontam import (build_heldout_ngrams, decontaminate_chat_rows,
                                        eval_benchmark_texts)
        _extra = eval_benchmark_texts() if os.environ.get(
            "KORE_DECONTAM_EVAL_BENCH", "1") != "0" else None
        _hn = build_heldout_ngrams(8, extra_sources=_extra)
        _dropped = 0
        _slices = {"general_code": gc, "math_reasoning": gm, "general_chat": gch,
                   "agentic_tooluse": agentic_rows}
        for _name, _rows in list(_slices.items()):
            _clean, _st = decontaminate_chat_rows(_rows, heldout_ngrams=_hn)
            _slices[_name] = _clean
            _dropped += _st["n_dropped_contaminated"]
        gc, gm, gch, agentic_rows = (_slices["general_code"], _slices["math_reasoning"],
                                     _slices["general_chat"], _slices["agentic_tooluse"])
        if _dropped:
            log.metric("general_replay_decontam", dropped=_dropped)

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
                                  pref_policy=None,
                                  anchor_baseline: Optional[bool] = None,
                                  anchor_min: Optional[float] = None,
                                  margin_min: Optional[float] = None,
                                  subbaseline_mode: Optional[str] = None,
                                  weighting: Optional[bool] = None,
                                  baseline_speedup_fn=None,
                                  seed: int = 0) -> dict:
    """Stage-2 DPO rows = ranked-group prefs + labeled reward-hack hard negatives.

    ``correct_source_fn(task)->str`` supplies the trusted 'chosen' kernel for the
    hard-negative group (defaults to the task's verified seed). ``group_records``,
    when given, supplies the ranked-group records directly (a leakage-split TRAIN
    partition) instead of scanning ``data_root``. ``extra_group_records`` folds in
    additional ranked groups produced on-policy (iterative-DPO relabeling) or by
    the evolutionary loop, so the preference set covers states the policy actually
    visits.

    Preference quality (audit fix): the ranked-group ("base") pairs are built through
    the baseline-anchored, margin-weighted :func:`build_datasets.build_dpo` policy —
    sub-baseline "wins" (chosen slower than production) are relabelled/dropped,
    noise-band near-ties are dropped, and high-margin compute-bound pairs are
    up-weighted. Pass ``pref_policy`` (a :class:`build_datasets.DPOPrefPolicy`) or the
    individual ``anchor_baseline`` / ``anchor_min`` / ``margin_min`` /
    ``subbaseline_mode`` / ``weighting`` / ``baseline_speedup_fn`` overrides (all
    default to the ``KORE_PREF_*`` env, frontier defaults ON). The hard negatives are
    built with the SAME policy but are correctness pairs, so anchoring is a safe no-op
    on them — every hard-negative pair is preserved at neutral weight.

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
    _pref = dict(policy=pref_policy, anchor_baseline=anchor_baseline, anchor_min=anchor_min,
                 margin_min=margin_min, subbaseline_mode=subbaseline_mode,
                 weighting=weighting, baseline_speedup_fn=baseline_speedup_fn)
    with log.stage("build_dpo_with_hard_negatives", n_tasks=len(tasks)):
        correct_source_fn = correct_source_fn or (lambda t: t.seed_source)

        if group_records is None:
            group_records = _read_dir(data_root, "groups")
        else:
            group_records = list(group_records)
        if extra_group_records:
            group_records = group_records + list(extra_group_records)
        base_rows = bd.build_dpo(group_records, prompt_fn=prompt_fn, **_pref) if group_records else []

        hard_groups = [build_hard_negative_group(correct_source_fn(t), t) for t in tasks]
        hard_rows = bd.build_dpo(hard_groups, prompt_fn=prompt_fn, **_pref)

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
