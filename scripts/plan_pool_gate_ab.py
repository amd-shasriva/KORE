#!/usr/bin/env python
"""Two matched task lists for the pool-admission A/B.

``before`` is sampled from the pre-gate index, ``after`` from the post-gate one.
Both are drawn with the same RNG from the same materialized pool, so the arms
differ only in which tasks the plan contains -- which is what the gate changes.

The ``before`` sample deliberately keeps the pre-gate FAMILY MIX rather than
sampling uniformly at random over ids, because a random draw from a corpus that
is 64% convolution would otherwise be compared against a post-gate corpus with a
different composition, and the comparison would be confounded by family rather
than by admissibility.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--pool-root", default="")
    p.add_argument("--tasks-per-arm", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _ids(path):
    out = []
    with pathlib.Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                out.append((row["task_id"], row.get("family", "?")))
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from kore.tasks import external

    root = external.pool_root(args.pool_root or None)
    post = root / external.POOL_INDEX_NAME
    pre = post.with_suffix(".jsonl.prefilter")
    if not pre.is_file():
        print(f"no pre-gate index at {pre}; run --rescreen-index first",
              file=sys.stderr)
        return 2

    tasks_dir = external.pool_tasks_dir(root)

    def materialized(pairs):
        return [(t, f) for t, f in pairs
                if (tasks_dir / t / "task.yaml").is_file()]

    before_all = materialized(_ids(pre))
    after_all = materialized(_ids(post))
    after_ids = {t for t, _ in after_all}
    print(f"pre-gate  index: {len(before_all)} materialized")
    print(f"post-gate index: {len(after_all)} materialized")
    if not after_all:
        print("the gate admitted nothing; nothing to compare", file=sys.stderr)
        return 3

    rng = random.Random(args.seed)
    n = min(args.tasks_per_arm, len(after_all))
    before = rng.sample(before_all, min(args.tasks_per_arm, len(before_all)))
    after = rng.sample(after_all, n)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for arm, chosen in (("before", before), ("after", after)):
        path = out_dir / f"shard_{arm}.txt"
        path.write_text("".join(f"{t}\n" for t, _ in chosen), encoding="utf-8")
        fams = {}
        for _, f in chosen:
            fams[f] = fams.get(f, 0) + 1
        admissible = sum(1 for t, _ in chosen if t in after_ids)
        print(f"{arm:7s}: {len(chosen)} tasks  admissible={admissible} "
              f"families={dict(sorted(fams.items()))}")
        print(f"         -> {path}")

    (out_dir / "plan.json").write_text(json.dumps({
        "pre_gate_materialized": len(before_all),
        "post_gate_materialized": len(after_all),
        "tasks_per_arm": args.tasks_per_arm,
        "seed": args.seed,
        "before_admissible_fraction": round(
            sum(1 for t, _ in before if t in after_ids) / max(1, len(before)), 4),
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
