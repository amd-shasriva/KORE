"""Turn raw KORE records into training-ready, HF-style chat datasets.

  - ``build_sft``: repair turns + winning trajectories -> {"messages": [...]}.
  - ``build_dpo``: ranked groups -> {"prompt", "chosen", "rejected"} preference
    pairs (chosen/rejected are assistant completions wrapping each candidate).
  - ``build_rft``: rejection-sampled SFT on the best candidate of each group and
    on winning trajectories -> {"messages": [...]}.

Plus corpus hygiene:
  - ``dedup_by_source_hash``: drop records with a duplicate representative source.
  - ``leakage_split``: split by a grouping key (default operation-family+arch) so
    the same op family never appears in more than one of train/val/test.

Everything here is PURE (no GPU / teacher) and unit-testable.
"""

from __future__ import annotations

from typing import Any, Iterable

from kore.data.prompts import SYSTEM_PROMPT, extract_kernel, wrap_full_kernel
from kore.data.schemas import (
    RepairRecord,
    RankedGroupRecord,
    WinRecord,
    record_from_dict,
)
from kore.env.replay import kernel_hash
from kore.obs import get_logger

log = get_logger("data.build_datasets")


# --- coercion helpers ---
def _as_record(rec: Any):
    if isinstance(rec, (RepairRecord, RankedGroupRecord, WinRecord)):
        return rec
    if isinstance(rec, dict) and rec.get("type"):
        return record_from_dict(rec)
    return rec


# Canonical FULL_KERNEL completion wrapper (single source of truth: policy.format).
_wrap_full_kernel = wrap_full_kernel


def _generic_prompt(task_id: str, gpu: str = "gfx942") -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Optimize the Triton kernel for task '{task_id}' on {gpu}. "
                "Output the complete kernel under the FULL_KERNEL contract."
            ),
        },
    ]


# --- provenance (Pillar 5): auditable per-row metadata for curation ---
def _prov_common(rec: Any) -> dict:
    d = rec.to_dict() if hasattr(rec, "to_dict") else {}
    return {
        "task_id": getattr(rec, "task_id", None) or d.get("task_id"),
        "operation": getattr(rec, "operation", None) or d.get("operation"),
        "arch": (getattr(rec, "arch", None) or d.get("arch")
                 or getattr(rec, "gpu", None) or d.get("gpu")),
        "shape": getattr(rec, "shape", None) or d.get("shape"),
    }


def _prov_win(rec: Any) -> dict:
    p = _prov_common(rec)
    p.update({"kind": "win", "verified": True, "baseline": "measured",
              "speedup": getattr(rec, "speedup", None),
              "snr_db": getattr(rec, "snr_db", None)})
    return p


def _prov_repair(rec: Any) -> dict:
    p = _prov_common(rec)
    p.update({"kind": "repair", "verified": True,
              "failure_class": getattr(rec, "failure_class", None),
              "snr_db": getattr(rec, "child_snr_db", None)})
    return p


# --- SFT ---
def build_sft(records: Iterable[Any]) -> list[dict]:
    """Chat-SFT rows from repair turns and winning trajectories.

    Each row carries a ``_provenance`` block (kind / task / op / arch / measured
    speedup + snr / verified) — ignored by the trainer but consumed by the curation
    stage and available for audit. ``mixing._tag`` shallow-copies rows, so it
    survives into the final multicap shard.
    """
    out: list[dict] = []
    n_repair = 0
    n_win = 0
    for raw in records:
        rec = _as_record(raw)
        if isinstance(rec, RepairRecord):
            if rec.messages:
                out.append({"messages": list(rec.messages), "_provenance": _prov_repair(rec)})
                n_repair += 1
        elif isinstance(rec, WinRecord):
            if rec.trajectory:
                out.append({"messages": list(rec.trajectory), "_provenance": _prov_win(rec)})
                n_win += 1
    log.metric("build_sft", rows=len(out), from_repair=n_repair, from_wins=n_win)
    return out


# --- DPO ---
def build_dpo(records: Iterable[Any], prompt_fn=None) -> list[dict]:
    """Preference rows from ranked groups, in trl's *conversational* DPO shape.

    ``prompt_fn(task_id) -> messages`` supplies the DPO prompt. When given (the
    campaign passes an in-context builder = the GRPO turn-1 transcript with the seed
    kernel + contract), preferences are learned in the SAME context the policy sees
    at inference. Falls back to the generic one-shot prompt when ``prompt_fn`` is
    None or returns falsy (keeps CPU tests + legacy callers working).

    Each ``[chosen_idx, rejected_idx]`` preference becomes a DPO row whose
    ``prompt`` is a chat-message list and whose ``chosen``/``rejected`` are each a
    single-message assistant completion list wrapping the candidate source under
    the FULL_KERNEL contract — i.e. ``trl.DPOTrainer``'s conversational schema:

        {"prompt": [ ...chat... ],
         "chosen":   [{"role": "assistant", "content": "FULL_KERNEL:..."}],
         "rejected": [{"role": "assistant", "content": "FULL_KERNEL:..."}]}

    Degenerate pairs where the chosen and rejected sources are identical are
    skipped (no learnable preference signal)."""
    out: list[dict] = []
    n_groups = 0
    n_prefs = 0
    n_degenerate = 0
    for raw in records:
        rec = _as_record(raw)
        if not isinstance(rec, RankedGroupRecord):
            continue
        n_groups += 1
        cands = rec.candidates
        prompt = (prompt_fn(rec.task_id) if prompt_fn else None) or _generic_prompt(rec.task_id, rec.gpu)
        for pair in rec.preferences:
            if len(pair) != 2:
                continue
            ci, ri = pair
            if not (0 <= ci < len(cands) and 0 <= ri < len(cands)):
                continue
            n_prefs += 1
            chosen_c, rejected_c = cands[ci], cands[ri]
            chosen_src = chosen_c.get("source", "")
            rejected_src = rejected_c.get("source", "")
            if chosen_src == rejected_src:
                n_degenerate += 1
                continue  # degenerate: identical sources carry no preference
            cw, rw = chosen_c.get("wall_us"), rejected_c.get("wall_us")
            speedup = (rw / cw) if (isinstance(cw, (int, float)) and cw
                                    and isinstance(rw, (int, float)) and rw) else None
            out.append(
                {
                    "prompt": prompt,
                    "chosen": [
                        {"role": "assistant", "content": _wrap_full_kernel(chosen_src)}
                    ],
                    "rejected": [
                        {"role": "assistant", "content": _wrap_full_kernel(rejected_src)}
                    ],
                    # Speed-grounding metadata (Pillar 5/3): the preference is a
                    # verified faster-correct > slower-correct ranking. Ignored by the
                    # trainer, consumed by curation.
                    "_provenance": {
                        "kind": "dpo_group", "task_id": rec.task_id,
                        "operation": getattr(rec, "operation", None),
                        "arch": getattr(rec, "gpu", None), "verified": True,
                        "chosen_wall_us": cw, "rejected_wall_us": rw,
                        "chosen_snr_db": chosen_c.get("snr_db"),
                        "rejected_snr_db": rejected_c.get("snr_db"),
                        "speedup": round(speedup, 4) if speedup else None,
                    },
                }
            )
    log.metric("build_dpo", groups=n_groups, pairs_considered=n_prefs,
               degenerate_dropped=n_degenerate, pairs=len(out))
    return out


# --- RFT (rejection-sampled SFT) ---
def build_rft(records: Iterable[Any]) -> list[dict]:
    """Chat-SFT rows on the single best candidate per group + win trajectories."""
    out: list[dict] = []
    n_group = 0
    n_win = 0
    for raw in records:
        rec = _as_record(raw)
        if isinstance(rec, RankedGroupRecord):
            n_group += 1
            best = None
            for c in rec.candidates:
                if c.get("rank") == 0:
                    best = c
                    break
            if best is None and rec.candidates:
                best = min(rec.candidates, key=lambda c: c.get("rank", 1 << 30))
            if best is not None:
                out.append(
                    {
                        "messages": _generic_prompt(rec.task_id, rec.gpu)
                        + [
                            {
                                "role": "assistant",
                                "content": _wrap_full_kernel(best.get("source", "")),
                            }
                        ]
                    }
                )
        elif isinstance(rec, WinRecord):
            if rec.trajectory:
                out.append({"messages": list(rec.trajectory)})
                n_win += 1
    log.metric("build_rft", rows=len(out), from_groups=n_group, from_wins=n_win)
    return out


# --- hygiene: dedup ---
def _record_source(rec: Any) -> str:
    """A representative source string for a record, for dedup hashing."""
    rec = _as_record(rec)
    if isinstance(rec, RepairRecord):
        for m in reversed(rec.messages):
            if m.get("role") == "assistant":
                k = extract_kernel(m.get("content", ""))
                if k:
                    return k
        return rec.parent_hash
    if isinstance(rec, WinRecord):
        return rec.final_source or ""
    if isinstance(rec, RankedGroupRecord):
        return "||".join(c.get("source", "") for c in rec.candidates)
    return repr(rec)


def dedup_by_source_hash(records: Iterable[Any]) -> list:
    """Keep the first record for each distinct representative-source hash."""
    seen: set[str] = set()
    out: list = []
    n_in = 0
    for rec in records:
        n_in += 1
        h = kernel_hash(_record_source(rec))
        if h in seen:
            continue
        seen.add(h)
        out.append(rec)
    log.metric("dedup_by_source_hash", n_in=n_in, kept=len(out),
               dropped=n_in - len(out))
    return out


def _record_score(rec: Any) -> float:
    """Preference score for near-dup dedup: higher = keep. Fastest win wins."""
    rec = _as_record(rec)
    if isinstance(rec, WinRecord):
        try:
            return float(getattr(rec, "speedup", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def dedup_near_source(records: Iterable[Any], per_fingerprint_cap: int = 1,
                      fuzzy_threshold: float = 0.0) -> list:
    """Near-duplicate dedup on the representative kernel source (STRUCTURAL).

    Complements :func:`dedup_by_source_hash` (exact) by collapsing kernels that
    differ only by variable renaming, whitespace, or comments (see
    ``kore.data.dedup``). Keeps the highest-scoring record per structural cluster
    (fastest :class:`WinRecord`), up to ``per_fingerprint_cap``.

    Apply this to WIN/gold records — NOT to repair records: each broken->fixed
    transition is a distinct lesson even when the fixed kernels converge, so
    collapsing repairs by fixed-kernel structure would delete real signal.
    """
    from kore.data.dedup import dedup_near

    recs = list(records)
    items = [{"_rec": r, "source": _record_source(r), "_score": _record_score(r)}
             for r in recs]
    kept, stats = dedup_near(items, source_key="source",
                             scorer=lambda d: d["_score"],
                             per_fingerprint_cap=per_fingerprint_cap,
                             fuzzy_threshold=fuzzy_threshold)
    log.metric("dedup_near_source", **stats)
    return [it["_rec"] for it in kept]


# --- hygiene: leakage-aware split ---
def _group_key(rec: Any, by: tuple = ("operation", "arch")) -> str:
    """Build a grouping key from ``by`` fields, tolerating missing fields.

    Fields are looked up on the record's dict. The ``operation`` field is
    normalized to its op *family* via ``mutate.infer_family`` (so gemm_bf16 and
    gemm_fp8_a8w8 group together as 'gemm'), falling back to ``task_id`` when the
    provenance field is absent. This replaces the brittle leading-``_`` split so
    the same op family never leaks across train/val/test."""
    from kore.data.mutate import infer_family

    rec = _as_record(rec)
    d = rec.to_dict() if hasattr(rec, "to_dict") else dict(rec)
    parts: list[str] = []
    for field in by:
        val = d.get(field)
        if field == "operation":
            val = infer_family(val or d.get("task_id", ""))
        parts.append(str(val) if val is not None else "")
    key = "|".join(parts)
    return key or str(d.get("task_id", ""))


def leakage_split(
    records: Iterable[Any],
    by: tuple = ("operation", "arch"),
    ratios: tuple = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> tuple[list, list, list]:
    """Split records into (train, val, test) so no ``by``-group crosses splits.

    Whole groups are assigned to a single split; deterministic given ``seed``."""
    records = list(records)
    # bucket records by group key
    groups: dict[str, list] = {}
    for rec in records:
        groups.setdefault(_group_key(rec, by), []).append(rec)

    keys = sorted(groups.keys())
    # deterministic shuffle by seed
    import random

    random.Random(seed).shuffle(keys)

    n = len(keys)
    tr, va, _te = ratios
    n_train = int(round(n * tr))
    n_val = int(round(n * va))
    # guard rounding so all keys are assigned
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train : n_train + n_val])
    test_keys = set(keys[n_train + n_val :])

    def collect(kset):
        out: list = []
        for k in kset:
            out.extend(groups[k])
        return out

    train, val, test = collect(train_keys), collect(val_keys), collect(test_keys)
    log.metric(
        "leakage_split", by=list(by), n_records=len(records), n_groups=n,
        train_groups=len(train_keys), val_groups=len(val_keys),
        test_groups=len(test_keys),
        train=len(train), val=len(val), test=len(test),
    )
    return train, val, test
