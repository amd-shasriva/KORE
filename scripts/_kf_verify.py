"""Verify dataset completeness and emit the remaining-undone list for cleanup.

Prints a one-line summary (fully-complete count, wins histogram, missing repair/
groups) and writes /tmp/cleanup.txt with every task still short of
repair+groups+wins>=target, so the supervisor can mop up stragglers on b05-2.
"""
from __future__ import annotations

import collections
import os
import sys


def main() -> int:
    root = sys.argv[1]
    target = int(sys.argv[2])
    from kore.tasks.registry import train_tasks

    def nwins(t: str) -> int:
        f = f"{root}/wins/{t}.jsonl"
        if not os.path.exists(f) or os.path.getsize(f) == 0:
            return 0
        return sum(1 for _ in open(f))

    def has(kind: str, t: str) -> bool:
        f = f"{root}/{kind}/{t}.jsonl"
        return os.path.exists(f) and os.path.getsize(f) > 0

    tasks = [t.task_id for t in train_tasks() if t.task_id.startswith("genb_")]
    wh = collections.Counter()
    miss_r = miss_g = full = 0
    undone = []
    for t in tasks:
        w = nwins(t)
        wh[min(w, target)] += 1
        r = has("repair", t)
        g = has("groups", t)
        if not r:
            miss_r += 1
        if not g:
            miss_g += 1
        if w >= target and r and g:
            full += 1
        else:
            undone.append(t)

    with open("/tmp/cleanup.txt", "w") as fh:
        fh.write(",".join(undone))

    print(f"VERIFY tasks={len(tasks)} fully_complete={full} "
          f"wins_hist={dict(sorted(wh.items()))} "
          f"missing_repair={miss_r} missing_groups={miss_g} remaining_undone={len(undone)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
