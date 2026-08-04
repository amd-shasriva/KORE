#!/usr/bin/env python3
"""Hand b501 the tail of each cluster shard, so the two boxes do not overlap.

The cluster and b501 do not share a filesystem, so they cannot coordinate through
the resume ledger the way six shards on one volume do -- each only skips work it
can see in its OWN output directory. Splitting by position is what keeps them off
each other: the cluster works each shard front to back, so the tail is the part
it reaches last and the part b501 can take without redoing anything.

`--done-frac` is where the cluster has got to. Everything before that point is
already generated and is skipped by both; b501 takes from the far end and works
backwards toward the cluster, so the two converge on the middle rather than
colliding at a boundary.
"""
from __future__ import annotations

import argparse
import json
import pathlib


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster-shards", default="data/b05factory/shards_v2")
    ap.add_argument("--out-dir", default="data/b501remain/shards")
    ap.add_argument("--shards", type=int, default=5, help="b501 GPUs available")
    ap.add_argument("--done-frac", type=float, default=0.40,
                    help="fraction of each cluster shard already generated")
    ap.add_argument("--take-frac", type=float, default=0.35,
                    help="fraction of each shard, from the END, to give b501")
    args = ap.parse_args()

    src = pathlib.Path(args.cluster_shards)
    files = sorted(src.glob("shard_*.txt"))
    if not files:
        print(f"no shard files under {src}")
        return 1

    taken: list[str] = []
    for f in files:
        ids = [ln.strip() for ln in f.read_text().splitlines() if ln.strip()]
        n = len(ids)
        # Take from the end, and never cross into what the cluster has finished.
        take = min(int(n * args.take_frac), n - int(n * args.done_frac))
        if take > 0:
            taken.extend(ids[-take:])
        print(f"{f.name}: {n} tasks, giving b501 the last {take}")

    if not taken:
        print("nothing to take")
        return 1

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.shards):
        part = taken[i::args.shards]
        (out / f"shard_{i:03d}.txt").write_text("\n".join(part) + "\n")
        print(f"  shard_{i:03d}.txt: {len(part)} tasks")

    (out / "manifest.json").write_text(json.dumps({
        "purpose": "b501's half of the remaining cluster wave, taken from the "
                   "END of each cluster shard so the two boxes do not overlap",
        "source_shards": [f.name for f in files],
        "n_tasks": len(taken),
        "shards": args.shards,
        "done_frac_assumed": args.done_frac,
        "take_frac": args.take_frac,
    }, indent=2))
    print(f"\ntotal handed to b501: {len(taken)} tasks across {args.shards} shards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
