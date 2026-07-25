#!/usr/bin/env python3
"""Quarantine held-out (eval) task shards that leaked into the datagen corpus.

Held-out/near-generalization tasks must NEVER carry training wins. This moves any
wins/repair/groups shard whose task is is_heldout into <root>/_quarantine/<lane>/,
printing the exact task_ids + reason so the leak is auditable and reversible.

Usage:
  python scripts/quarantine_heldout_wins.py <data_root> [--apply]
Dry-run by default; pass --apply to actually move shards.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_root")
    ap.add_argument("--apply", action="store_true", help="actually move shards (default: dry-run)")
    a = ap.parse_args(argv)

    from kore.tasks.registry import is_heldout, get_task, task_ids

    known = set(task_ids())
    root = Path(a.data_root)
    lanes = ("wins", "repair", "groups")
    qroot = root / "_quarantine"

    def _heldout(tid: str) -> tuple[bool, str]:
        if tid not in known:
            return False, "unknown-task-id"
        try:
            dec = get_task(tid)
        except Exception:
            return False, "get_task-failed"
        try:
            if is_heldout(dec):
                # surface the split reason if available
                try:
                    from kore.tasks import taxonomy
                    sd = taxonomy.split_decision(dec, strict=True)
                    return True, getattr(sd, "reason", "heldout")
                except Exception:
                    return True, "heldout"
        except Exception:
            return False, "is_heldout-failed"
        return False, "train"

    leaked: dict[str, str] = {}
    per_lane_moved: dict[str, int] = {L: 0 for L in lanes}
    for lane in lanes:
        d = root / lane
        if not d.is_dir():
            continue
        for shard in sorted(d.glob("*.jsonl")):
            tid = shard.stem
            held, reason = _heldout(tid)
            if not held:
                continue
            leaked[tid] = reason
            if a.apply:
                dest = qroot / lane
                dest.mkdir(parents=True, exist_ok=True)
                os.replace(shard, dest / shard.name)
                per_lane_moved[lane] += 1

    print(f"data_root={root}  heldout-leaked distinct task_ids: {len(leaked)}")
    for tid in sorted(leaked):
        print(f"  LEAK {tid}  reason={leaked[tid]}")
    if a.apply:
        print(f"MOVED to {qroot}: " + ", ".join(f"{L}={per_lane_moved[L]}" for L in lanes))
    else:
        print("DRY-RUN (no files moved). Re-run with --apply to quarantine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
