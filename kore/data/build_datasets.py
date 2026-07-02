"""Turn raw KORE records into training-ready, HF-style chat datasets.

  - ``build_sft``: repair turns + winning trajectories -> {"messages": [...]}.
  - ``build_dpo``: ranked groups -> {"prompt", "chosen", "rejected"} preference
    pairs (chosen/rejected are assistant completions wrapping each candidate).
  - ``build_rft``: rejection-sampled SFT on the best candidate of each group and
    on winning trajectories -> {"messages": [...]}.

Plus corpus hygiene:
  - ``dedup_by_source_hash``: drop records with a duplicate representative source.
  - ``leakage_split``: split by a grouping key (default operation+shape) so the
    same op/shape never appears in more than one of train/val/test.

Everything here is PURE (no GPU / teacher) and unit-testable.
"""

from __future__ import annotations

from typing import Any, Iterable

from kore.data.prompts import SYSTEM_PROMPT, extract_kernel
from kore.data.schemas import (
    RepairRecord,
    RankedGroupRecord,
    WinRecord,
    record_from_dict,
)
from kore.env.replay import kernel_hash


# --- coercion helpers ---
def _as_record(rec: Any):
    if isinstance(rec, (RepairRecord, RankedGroupRecord, WinRecord)):
        return rec
    if isinstance(rec, dict) and rec.get("type"):
        return record_from_dict(rec)
    return rec


def _wrap_full_kernel(source: str) -> str:
    """Wrap a kernel source in the FULL_KERNEL assistant-response contract."""
    return "FULL_KERNEL:\n```python\n" + source.strip() + "\n```\n"


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


# --- SFT ---
def build_sft(records: Iterable[Any]) -> list[dict]:
    """Chat-SFT rows from repair turns and winning trajectories."""
    out: list[dict] = []
    for raw in records:
        rec = _as_record(raw)
        if isinstance(rec, RepairRecord):
            if rec.messages:
                out.append({"messages": list(rec.messages)})
        elif isinstance(rec, WinRecord):
            if rec.trajectory:
                out.append({"messages": list(rec.trajectory)})
    return out


# --- DPO ---
def build_dpo(records: Iterable[Any]) -> list[dict]:
    """Preference rows from ranked groups.

    Each ``[chosen_idx, rejected_idx]`` preference becomes a DPO row whose
    ``chosen``/``rejected`` are the candidate sources wrapped as assistant
    completions, sharing a single ``prompt`` chat context."""
    out: list[dict] = []
    for raw in records:
        rec = _as_record(raw)
        if not isinstance(rec, RankedGroupRecord):
            continue
        cands = rec.candidates
        prompt = _generic_prompt(rec.task_id, rec.gpu)
        for pair in rec.preferences:
            if len(pair) != 2:
                continue
            ci, ri = pair
            if not (0 <= ci < len(cands) and 0 <= ri < len(cands)):
                continue
            out.append(
                {
                    "prompt": prompt,
                    "chosen": _wrap_full_kernel(cands[ci].get("source", "")),
                    "rejected": _wrap_full_kernel(cands[ri].get("source", "")),
                }
            )
    return out


# --- RFT (rejection-sampled SFT) ---
def build_rft(records: Iterable[Any]) -> list[dict]:
    """Chat-SFT rows on the single best candidate per group + win trajectories."""
    out: list[dict] = []
    for raw in records:
        rec = _as_record(raw)
        if isinstance(rec, RankedGroupRecord):
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
    for rec in records:
        h = kernel_hash(_record_source(rec))
        if h in seen:
            continue
        seen.add(h)
        out.append(rec)
    return out


# --- hygiene: leakage-aware split ---
def _op_from_task_id(task_id: str) -> str:
    return (task_id or "").split("_")[0] if task_id else ""


def _group_key(rec: Any, by: tuple) -> str:
    """Build a grouping key from ``by`` fields, tolerating missing fields.

    Fields are looked up on the record's dict; ``operation`` falls back to the
    leading token of ``task_id`` so gemm_bf16 -> 'gemm'."""
    rec = _as_record(rec)
    d = rec.to_dict() if hasattr(rec, "to_dict") else dict(rec)
    parts: list[str] = []
    for field in by:
        val = d.get(field)
        if val is None and field == "operation":
            val = _op_from_task_id(d.get("task_id", ""))
        parts.append(str(val) if val is not None else "")
    key = "|".join(parts)
    return key or str(d.get("task_id", ""))


def leakage_split(
    records: Iterable[Any],
    by: tuple = ("operation", "shape"),
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

    return collect(train_keys), collect(val_keys), collect(test_keys)
