#!/usr/bin/env python
"""Shard an arbitrary task-id list for the datagen array, registry or pool.

``spur_partition.py`` validates every id against ``registry.all_tasks()`` and
rejects anything it does not recognise, which is correct for registry work and
fatal for pool work: the external pool is deliberately outside the registry, so
all 13,570 of its tasks are "unknown task ids" to it. That is the whole reason
none of them has ever been mined.

This writes the same on-disk contract the array job expects -- ``deep_NNN.txt``,
``base_NNN.txt``, ``manifest.json`` -- over any id list, and validates ids by
actually resolving them the way datagen will, rather than by membership in a
registry that is the wrong authority for pool tasks.

    python scripts/partition_any_tasks.py --task-file runs/unmined.txt \
        --out-dir runs/shards_pool --data-root data/v5pool --shards 6
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def _resolvable(task_ids, sample: int = 40):
    """Confirm a sample resolves before committing a node-hours-long array.

    Resolving all 13,570 costs minutes of import per id; a sample is enough to
    catch the failure mode that matters, which is 'none of these resolve at all'.
    """
    from run_campaign import _resolve_task_anywhere

    step = max(1, len(task_ids) // sample)
    checked = bad = 0
    for tid in task_ids[::step][:sample]:
        checked += 1
        try:
            _resolve_task_anywhere(tid)
        except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
            bad += 1
            if bad <= 3:
                print(f"  UNRESOLVABLE {tid}: {type(exc).__name__}: {exc}"[:160])
    return checked, bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-file", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--target", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap the task count (0 = all)")
    ap.add_argument("--skip-check", action="store_true")
    args = ap.parse_args()

    ids = [ln.strip() for ln in Path(args.task_file).read_text().splitlines()
           if ln.strip()]
    if args.limit:
        ids = ids[: args.limit]
    if not ids:
        print("no task ids", file=sys.stderr)
        return 2

    if not args.skip_check:
        checked, bad = _resolvable(ids)
        print(f"resolution check: {checked - bad}/{checked} sampled ids resolve")
        if bad == checked:
            print("FATAL: nothing resolves; refusing to queue an array that can "
                  "only fail", file=sys.stderr)
            return 3

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Round-robin rather than contiguous blocks: ids arrive grouped by source
    # (kbk_*, syn_*, genb_*) and those groups differ in cost, so contiguous
    # slices would leave some nodes idle while others carry the expensive family.
    for i in range(args.shards):
        shard = ids[i::args.shards]
        body = ",".join(shard)
        (out / f"deep_{i:03d}.txt").write_text(body)
        # The array job requires both lists to exist. Both stages should visit
        # every task in the shard: an unmined task needs a first win (deep) and
        # has no repair/ranked shard yet (base).
        (out / f"base_{i:03d}.txt").write_text(body)
        print(f"shard={i:03d} tasks={len(shard)}")

    head = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                   cwd=REPO, text=True).strip()
    (out / "manifest.json").write_text(json.dumps({
        "repo_commit": head,
        "data_root": str(Path(args.data_root).resolve()),
        "target_wins": args.target,
        "n_shards": args.shards,
        "n_tasks": len(ids),
        "source_task_file": str(Path(args.task_file).resolve()),
    }, indent=2) + "\n")
    print(f"PARTITION tasks={len(ids)} shards={args.shards} out={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
