"""Mint DPO preference pairs from verified repair records (fixed > broken).

Each ``repair`` record is a proven correctness contrast: a broken kernel + the
exact verifier error + a teacher fix that PASSED the oracle. `build_dpo` only
consumes ``RankedGroupRecord``s, so this module packages each repair as a
two-candidate ranked group - ``candidates=[fixed, broken]`` with the single
preference ``[0, 1]`` (fixed preferred over broken) - and writes them to
``<data_root>/groups/_repair_pairs.jsonl``. The campaign build stage's raw
gather then folds them through the SAME leakage split + `build_dpo` path as the
real ranked groups, adding a clean "correct kernel > broken kernel" signal on
top of the speed-ranked group prefs and the reward-hack hard negatives.

CPU-only, no GPU/teacher: the fix already passed the oracle at datagen time.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any, Optional

from kore.data.schemas import (
    RankedGroupRecord,
    read_jsonl,
    stamp_production_record,
    write_jsonl,
)
from kore.obs import get_logger

log = get_logger("data.repair_dpo")

DEFAULT_ARCH = "gfx950"  # KORE target = MI350X/CDNA4 (matches registry.TRAIN_ARCH)
_FENCE_RE = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.DOTALL)


def _fences(text: str) -> list[str]:
    return [m.group(1).strip() for m in _FENCE_RE.finditer(text or "")]


def _largest_fence(text: str) -> Optional[str]:
    f = _fences(text)
    return max(f, key=len) if f else None


def _last_fence(text: str) -> Optional[str]:
    f = _fences(text)
    return f[-1] if f else None


def mint_repair_pair(rec: dict, arch: Optional[str] = None) -> Optional[RankedGroupRecord]:
    """One repair record -> a 2-candidate ranked group (fixed > broken)."""
    msgs = rec.get("messages") or []
    if len(msgs) < 3:
        return None
    broken = _largest_fence(msgs[1].get("content", ""))   # broken kernel in the user turn
    fixed = _last_fence(msgs[-1].get("content", ""))       # teacher fix in the assistant turn
    if not broken or not fixed or broken.strip() == fixed.strip():
        return None
    a = str(arch or rec.get("arch") or rec.get("gpu") or DEFAULT_ARCH)
    op = str(rec.get("operation") or rec.get("operator") or rec.get("task_id") or "kernel")
    return RankedGroupRecord(
        task_id=str(rec.get("task_id", "repair")),
        parent_id=str(rec.get("parent_hash", "") or "repair"),
        candidates=[
            {"source": fixed, "rank": 0, "snr_db": rec.get("child_snr_db")},
            {"source": broken, "rank": 1, "snr_db": None,
             "failure_class": rec.get("failure_class")},
        ],
        preferences=[[0, 1]],
        gpu=a,
        operation=op,
        arch=a,
        shape=rec.get("shape"),
    )


def mint_repair_dpo(
    data_root: Any, *, cap: int = 8000, per_task_cap: int = 60,
    seed: int = 0, arch: Optional[str] = None, write: bool = True,
    out_name: str = "_repair_pairs.jsonl",
) -> dict:
    """Scan ``<data_root>/repair`` and mint up to ``cap`` fixed>broken DPO groups.

    Deterministic given ``seed``; ``per_task_cap`` keeps any one task from
    dominating. Writes ``<data_root>/groups/<out_name>`` (picked up by the build
    raw gather). Never touches existing per-task shards.
    """
    data_root = Path(data_root)
    rng = random.Random(seed)
    recs: list[dict] = []
    d = data_root / "repair"
    if d.exists():
        for p in sorted(d.glob("*.jsonl")):
            try:
                if p.stat().st_size == 0 or p.name.startswith("_"):
                    continue
            except OSError:
                continue
            for r in read_jsonl(
                p, typed=False, mode="generic_training_row"):
                if isinstance(r, dict):
                    recs.append(r)
    rng.shuffle(recs)

    per_task: dict[str, int] = {}
    out: list[RankedGroupRecord] = []
    for r in recs:
        if len(out) >= cap:
            break
        tid = str(r.get("task_id", "?"))
        if per_task.get(tid, 0) >= per_task_cap:
            continue
        try:
            g = mint_repair_pair(r, arch)
        except Exception as e:  # noqa: BLE001 - one bad record must not abort
            log.debug("repair_dpo_skip", task=tid, err=str(e)[:120])
            g = None
        if g is not None:
            out.append(g)
            per_task[tid] = per_task.get(tid, 0) + 1

    if write and out:
        (data_root / "groups").mkdir(parents=True, exist_ok=True)
        rows = [
            stamp_production_record(
                group,
                provenance_id="repair_dpo_v1",
                evaluation_id=f"repair_dpo:{group.task_id}:{group.parent_id}",
            )
            for group in out
        ]
        write_jsonl(data_root / "groups" / out_name, rows)

    summary = {
        "repair_pairs": len(out),
        "tasks_covered": len(per_task),
        "repair_scanned": len(recs),
    }
    log.event("repair_dpo_minted", cap=cap, **summary)
    return summary


__all__ = ["mint_repair_pair", "mint_repair_dpo"]
