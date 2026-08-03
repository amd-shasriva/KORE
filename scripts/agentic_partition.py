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

Tasks are also screened against the held-out eval BEFORE any GPU time is spent on
them. 398 of the 1,289 trainable tasks have a seed kernel that the repo's own
decontamination rules flag against a held-out task - almost all of them
``genb_*`` breadth tasks whose held-out sibling differs only in a shape or dtype
constant, so the two seeds are structurally the same program. Those trajectories
are rejected at filter time no matter how good they are, and generating them
first would burn roughly a third of the campaign to produce data that is thrown
away. Screening on the seed is the cheap, deterministic proxy: it is the text the
trajectory shares with the sibling, and it is known before the episode runs.

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


def screen_heldout_seeds(tasks) -> tuple[list, list[dict]]:
    """Split tasks into (clean, flagged) by their seed's held-out overlap.

    Uses the same ``analyze_text_contamination`` the filter applies to finished
    trajectories, so a task cleared here is a task whose trajectories will not be
    thrown away for a reason that was knowable up front. A task whose seed cannot
    be read is kept: an unreadable seed is a task-registry problem, and silently
    dropping tasks on a read error would shrink the campaign invisibly.
    """
    from kore.data.decontam import analyze_text_contamination, build_heldout_ngrams

    index = build_heldout_ngrams()
    clean: list = []
    flagged: list[dict] = []
    for task in tasks:
        try:
            source = task.seed_source
        except Exception:  # noqa: BLE001 - missing seed is not a contamination verdict
            clean.append(task)
            continue
        if not source:
            clean.append(task)
            continue
        match = analyze_text_contamination(source, index)
        if match is None:
            clean.append(task)
        else:
            flagged.append({
                "task_id": task.task_id,
                "reason": match.reason,
                "score": round(float(match.score), 4),
                "heldout_reference": match.reference_id,
            })
    return clean, flagged


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
    parser.add_argument("--no-decontam-screen", action="store_true",
                        help="plan tasks whose seed overlaps a held-out task; their "
                             "trajectories will be rejected at filter time")
    # On by default: the registry alone is ~1,289 trainable tasks, of which the
    # seed screen rejects ~398, leaving under 900. At 6 episodes each that is a
    # campaign spent re-sampling the same few hundred programs, which is
    # redundancy rather than data. The external pool adds ~13.5k screened,
    # deduplicated tasks and is the whole reason it was built.
    parser.add_argument("--no-pool", dest="include_pool", action="store_false",
                        default=True,
                        help="plan only registry tasks, ignoring data/task_pool")
    args = parser.parse_args()

    from kore.tasks.registry import train_tasks

    out_dir = Path(args.out_dir)
    shard_dir = Path(args.shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = list(train_tasks())
    n_registry = len(tasks)
    if args.include_pool:
        try:
            from kore.tasks.external import load_pool

            # Registry ids win on collision: those tasks carry the authoritative
            # train/held-out split, and letting a pool entry shadow one could
            # quietly move a held-out task into training.
            seen = {t.task_id for t in tasks}
            added = [t for t in load_pool() if t.task_id not in seen]
            tasks.extend(added)
            print(f"task pool: {n_registry} registry + {len(added)} external "
                  f"= {len(tasks)} planned")
        except Exception as exc:  # noqa: BLE001 - pool is additive, never required
            print(f"task pool: unavailable ({type(exc).__name__}: {exc}); "
                  f"planning {n_registry} registry tasks only")
    flagged: list[dict] = []
    if not args.no_decontam_screen:
        tasks, flagged = screen_heldout_seeds(tasks)
        (shard_dir / "heldout_seed_screen.json").write_text(
            json.dumps(flagged, indent=2) + "\n")
        print(f"held-out seed screen: {len(flagged)} tasks excluded, "
              f"{len(tasks)} eligible")
        print(f"  reasons: {dict(Counter(f['reason'] for f in flagged).most_common())}")

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
        "n_eligible_tasks": len(tasks),
        "n_excluded_heldout_seed_overlap": len(flagged),
        "heldout_screen_reasons": dict(
            Counter(f["reason"] for f in flagged).most_common()),
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
