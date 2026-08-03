#!/usr/bin/env python
"""Partition trainable tasks into disjoint per-node shards for agentic datagen.

Each Slurm array element owns exactly one shard file and one output JSONL, so
nodes never write to the same file and a requeued element resumes its own work
instead of racing a sibling. That is the whole reason the assignment is pinned in
a manifest rather than recomputed on the node: another agent is expanding the
task registry toward 10-20K, and a partition derived live from the registry would
silently reshuffle under a requeued node, stranding half-finished work under a
shard id that no longer covers it.

Tasks are dealt round-robin across shards after sorting by operator family, so
every node gets the same mix of cheap elementwise kernels and expensive
attention/GEMM/MoE ones. Balancing by task count alone would leave the nodes that
drew the attention kernels running hours after the rest finished.

Re-running is the resume operation: work already durable in the output directory
is removed from the plan, so a second wave only covers what is left.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _family(task) -> str:
    try:
        from kore.tasks.registry import operator_family

        return operator_family(task) or "unclassified"
    except Exception:  # noqa: BLE001 - taxonomy is advisory for balancing only
        return "unclassified"


def completed_counts(out_dir: Path) -> Counter:
    """Episodes already durable per task, across every shard in ``out_dir``."""
    from kore.data.saturated_agentic import completed_keys

    done: Counter = Counter()
    for shard in sorted(out_dir.glob("shard_*.jsonl")):
        for key in completed_keys(shard):
            task_id, _, _episode = key.rpartition("#")
            if task_id:
                done[task_id] += 1
    return done


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True,
                        help="directory that will hold shard_NNN.jsonl outputs")
    parser.add_argument("--shard-dir", required=True,
                        help="directory for the immutable shard plan of this wave")
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=6)
    parser.add_argument("--max-tasks", type=int, default=0,
                        help="cap the number of tasks planned (0 = all trainable)")
    args = parser.parse_args()

    from kore.tasks.registry import train_tasks

    out_dir = Path(args.out_dir)
    shard_dir = Path(args.shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = list(train_tasks())
    done = completed_counts(out_dir)
    # A task whose episode quota is already durable is finished; anything short is
    # replanned in full, because run_node_shard skips the individual work items it
    # already holds and only pays for the gap.
    remaining = [
        task for task in tasks
        if done.get(task.task_id, 0) < args.episodes_per_task
    ]
    remaining.sort(key=lambda task: (_family(task), task.task_id))
    if args.max_tasks > 0:
        remaining = remaining[:args.max_tasks]

    n_shards = max(1, int(args.shards))
    buckets: list[list[str]] = [[] for _ in range(n_shards)]
    for index, task in enumerate(remaining):
        buckets[index % n_shards].append(task.task_id)

    for index, bucket in enumerate(buckets):
        (shard_dir / f"shard_{index:03d}.txt").write_text("\n".join(bucket) + "\n")

    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=str(Path(__file__).resolve().parents[1]), text=True).strip()
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_commit": head,
        "out_dir": str(out_dir.resolve()),
        "shard_dir": str(shard_dir.resolve()),
        "n_shards": n_shards,
        "episodes_per_task": args.episodes_per_task,
        "n_trainable_tasks": len(tasks),
        "n_tasks_planned": len(remaining),
        "n_tasks_already_complete": len(tasks) - len(remaining),
        "planned_episodes": len(remaining) * args.episodes_per_task,
        "family_mix": dict(Counter(_family(task) for task in remaining).most_common()),
        "shard_sizes": [len(bucket) for bucket in buckets],
    }
    (shard_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps({k: v for k, v in manifest.items() if k != "family_mix"}, indent=2))
    print(f"planned {len(remaining)} tasks x {args.episodes_per_task} episodes "
          f"across {n_shards} shards -> {shard_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
