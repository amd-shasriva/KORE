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

import argparse
import json
import os
from pathlib import Path
import tempfile

from kore.ops.runtime import deprecated_entrypoint, task_set_identity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?")
    parser.add_argument("target", nargs="?", type=int)
    parser.add_argument("--out-dir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not deprecated_entrypoint(
        "scripts/_kf_split.py",
        "use scripts/spur_partition.py for immutable cost-balanced production shards",
        dry_run=args.dry_run,
    ):
        print(
            json.dumps(
                {"root": args.root, "target": args.target, "out_dir": args.out_dir},
                sort_keys=True,
            )
        )
        return 0
    if args.root is None or args.target is None or args.out_dir is None:
        parser.error("root, target, and --out-dir are required")
    if args.target < 1:
        parser.error("target must be positive")
    root = args.root
    target = args.target
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if out_dir.is_symlink() or out_dir.stat().st_uid != os.getuid():
        raise SystemExit(f"unsafe output directory: {out_dir}")
    os.chmod(out_dir, 0o700)
    from kore.tasks.registry import train_tasks

    def nwins(t: str) -> int:
        f = f"{root}/wins/{t}.jsonl"
        if not os.path.exists(f) or os.path.getsize(f) == 0:
            return 0
        return sum(1 for _ in open(f))

    def has(kind: str, t: str) -> bool:
        f = f"{root}/{kind}/{t}.jsonl"
        return os.path.exists(f) and os.path.getsize(f) > 0

    tasks = sorted(t.task_id for t in train_tasks() if t.task_id.startswith("genb_"))
    identity = task_set_identity(tasks)
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

    def wr(name: str, items: list[str]) -> None:
        fd, raw_tmp = tempfile.mkstemp(prefix=f".{name}.", dir=out_dir)
        tmp = Path(raw_tmp)
        target_path = out_dir / name
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w") as fh:
                fh.write(",".join(items))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, target_path)
        finally:
            tmp.unlink(missing_ok=True)

    wr("half_A.txt", [t for t, nd, nb in A if nd])
    wr("half_B.txt", [t for t, nd, nb in B if nd])
    wr("base_A.txt", [t for t, nd, nb in A if nb])
    wr("base_B.txt", [t for t, nd, nb in B if nb])
    wr(
        "task-set.json",
        [
            json.dumps(
                {
                    "schema": 1,
                    "count": identity.count,
                    "sha256": identity.sha256,
                    "task_ids": list(identity.task_ids),
                },
                sort_keys=True,
            )
            + "\n"
        ],
    )

    print(f"SPLIT undone={len(work)} "
          f"A(deepen={sum(nd for _, nd, _ in A)},base={sum(nb for _, _, nb in A)},cost={ca}) "
          f"B(deepen={sum(nd for _, nd, _ in B)},base={sum(nb for _, _, nb in B)},cost={cb}) "
          f"task_count={identity.count} task_sha256={identity.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
