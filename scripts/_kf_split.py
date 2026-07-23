"""Partition undone train tasks into two DISJOINT, cost-balanced halves.

Reads the current dataset state and, for every genb train task, computes what work
remains: wins short of target, and/or a missing/empty repair or groups shard. Tasks
needing nothing are dropped. The rest are cost-weighted (a zero-win task is the most
expensive to deepen) and dealt snake-draft into halves A (local/b05-2) and B
(peer/b05-1) so each node gets ~equal *work*, not just an equal count.

Writes four comma-lists to /tmp:
  half_A.txt / half_B.txt  - tasks each node must DEEPEN (wins < target)
  base_A.txt / base_B.txt  - tasks each node must fill REPAIR/GROUPS
"""
from __future__ import annotations

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
    work = []  # (cost, task, needs_deepen, needs_base)
    for t in tasks:
        w = nwins(t)
        nw = max(0, target - w)
        nr = 0 if has("repair", t) else 1
        ng = 0 if has("groups", t) else 1
        if nw or nr or ng:
            # deepen dominates; a zero-win task is the worst (up to 9 trajectories).
            cost = nw * (3 if w == 0 else 1) + nr * 2 + ng
            work.append((cost, t, nw > 0, (nr or ng) > 0))
    work.sort(reverse=True)  # most expensive first -> snake draft balances tails

    A, B = [], []
    ca = cb = 0
    for cost, t, nd, nb in work:
        if ca <= cb:
            A.append((t, nd, nb)); ca += cost
        else:
            B.append((t, nd, nb)); cb += cost

    def wr(fn, items):
        with open(fn, "w") as fh:
            fh.write(",".join(items))

    wr("/tmp/half_A.txt", [t for t, nd, nb in A if nd])
    wr("/tmp/half_B.txt", [t for t, nd, nb in B if nd])
    wr("/tmp/base_A.txt", [t for t, nd, nb in A if nb])
    wr("/tmp/base_B.txt", [t for t, nd, nb in B if nb])

    print(f"SPLIT undone={len(work)} "
          f"A(deepen={sum(nd for _, nd, _ in A)},base={sum(nb for _, _, nb in A)},cost={ca}) "
          f"B(deepen={sum(nd for _, nd, _ in B)},base={sum(nb for _, _, nb in B)},cost={cb})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
