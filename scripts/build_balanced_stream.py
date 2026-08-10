#!/usr/bin/env python3
"""Build a shard set whose family mix stays balanced however far it gets.

Mining a task list in whatever order it came in produces whatever mix the
source happens to have, and the sources here are wildly lopsided: the hard pool
is 1,691 GEMM tasks against a single MoE, and the registry frontier is 108
attention against 26 GEMM. Concatenating them and sharding gives a corpus that
is 94% GEMM in one stream and 45% attention in the other, which is what we
actually measured after a night of mining.

Equalising by count is the obvious fix and the wrong one. Across both HIP
sources there are 1,717 GEMM tasks and 43 MoE, so an equal-count list is 43
per family and throws away 1,600 usable GEMM tasks.

Round-robin instead: take one task from each family in turn, so that *any
prefix* of the list is as balanced as the supply allows. A miner that gets a
third of the way through a shard has mined a third of each family rather than
all of the first one. When a scarce family runs out the rotation carries on
without it, which spends the abundant families rather than discarding them.

The rotation starts with whichever family is furthest behind in what has
already been mined, so the deficit closes rather than persisting.

    python scripts/build_balanced_stream.py \\
        --task-file runs/hard_pool_hip.txt --task-file runs/frontier_hip.txt \\
        --out-dir runs/shards_frontierhip --data-root data/v5frontierhip \\
        --task-pool data/frontier_hip_all --shards 2
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

FAMILIES = ("attention", "moe", "quant", "gemm", "norm_fusion")


def strip_twin(task_id: str) -> str:
    return re.sub(r"__(hip|hipf|flydsl|triton)$", "", task_id)


def mined_by_family(roots: list[str], fam_of) -> collections.Counter:
    """Win-groups already on disk per family, to seed the rotation order."""
    counts: collections.Counter = collections.Counter()
    for root in roots:
        for path in glob.glob(f"data/{root}/**/*.jsonl", recursive=True):
            with open(path, errors="ignore") as fh:
                for line in fh:
                    if len(line) < 2 or '"ranked_group"' not in line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if row.get("type") != "ranked_group":
                        continue
                    f = fam_of(row.get("task_id") or "")
                    if f:
                        counts[f] += 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-file", action="append", default=[],
                    help="file of task ids; repeatable")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--task-pool", default="",
                    help="root the ids resolve against, recorded in the manifest")
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--target", type=int, default=3)
    ap.add_argument("--coverage-roots", nargs="*", default=[],
                    help="mined roots to read the current family deficit from")
    ap.add_argument("--families", nargs="*", default=list(FAMILIES))
    args = ap.parse_args()

    from select_frontier_tasks import collect
    ranked = {n: (s, w) for s, n, w in collect()}

    def fam_of(task_id: str):
        e = ranked.get(task_id) or ranked.get(strip_twin(task_id))
        return e[1]["family"] if e else None

    def score_of(task_id: str) -> float:
        e = ranked.get(task_id) or ranked.get(strip_twin(task_id))
        return e[0] if e else 0.0

    seen: set[str] = set()
    buckets: dict[str, list[str]] = collections.defaultdict(list)
    for path in args.task_file:
        for line in open(path):
            t = line.split("#", 1)[0].strip()
            if not t or t in seen:
                continue
            seen.add(t)
            f = fam_of(t)
            if f in args.families:
                buckets[f].append(t)

    if not buckets:
        print("no tasks matched the requested families", file=sys.stderr)
        return 2

    # Hardest first inside a family, so a shard that is only part-mined has
    # spent its time on the tasks worth the most.
    for f in buckets:
        buckets[f].sort(key=lambda t: -score_of(t))

    deficit = mined_by_family(args.coverage_roots, fam_of) if args.coverage_roots else {}
    order = sorted(buckets, key=lambda f: (deficit.get(f, 0), -len(buckets[f])))
    print("family supply (rotation starts with the least-mined):")
    for f in order:
        print(f"  {f:<14}{len(buckets[f]):>6} tasks   already mined: {deficit.get(f, 0)}")

    interleaved: list[str] = []
    idx = {f: 0 for f in order}
    while True:
        progressed = False
        for f in order:
            i = idx[f]
            if i < len(buckets[f]):
                interleaved.append(buckets[f][i])
                idx[f] = i + 1
                progressed = True
        if not progressed:
            break

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    listing = out_dir / "balanced_tasks.txt"
    listing.write_text("\n".join(interleaved) + "\n")
    print(f"\nwrote {len(interleaved)} tasks -> {listing}")
    head = collections.Counter(fam_of(t) for t in interleaved[:40])
    print(f"  first 40 are {dict(head)}")

    env = dict(os.environ)
    if args.task_pool:
        env["KORE_TASK_POOL"] = str((REPO / args.task_pool).resolve())
    env["PYTHONPATH"] = str(REPO)
    cmd = [sys.executable, str(REPO / "scripts" / "partition_any_tasks.py"),
           "--task-file", str(listing), "--out-dir", str(out_dir),
           "--data-root", args.data_root, "--shards", str(args.shards),
           "--target", str(args.target), "--skip-check"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(r.stdout.strip()[-600:] or r.stderr.strip()[-600:])
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
