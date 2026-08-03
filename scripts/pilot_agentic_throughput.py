#!/usr/bin/env python
"""Measure real per-node agentic-datagen throughput, and find the worker count.

Everything downstream of datagen is sized from one number - episodes per hour per
node - and there is no way to derive it. An episode interleaves a teacher round
trip (network; the AMD gateway allows 4000 rpm, so it is not the constraint) with
verification against production vendor baselines (GPU; the env takes an exclusive
per-physical-device lock around timing, so eight devices cap concurrent
measurement no matter how many workers exist). Those two legs overlap, the
overlap is what makes oversubscription pay, and how far it pays before the timing
lock and CPU contention eat the gain is a property of the node.

So: run real episodes with the real teacher against the real verified env at
several worker counts and report the curve.

Two rates are reported per point and they answer different questions:

  measured        completed episodes over the point's wall clock. Includes the
                  pool ramping up and draining at the tail, so a short point
                  understates a long run.
  steady_state    workers / mean-episode-seconds. What a long run converges to
                  once the tail stops mattering. This is the sizing number.

Two controls keep the comparison between points honest, because both defaults
would make whichever point ran last look fastest:

* Each point gets a DISJOINT slice of tasks, so no point reads another's replay
  cache instead of running the evaluation.
* Each point gets its own Triton/inductor compile cache, so no point inherits
  warm compiles paid for by the point before it.

Both make every point pay cold-compile cost, so the absolute numbers are a floor:
a long production run amortizes those compiles across tasks and does better. The
comparison between points - which is what picks the worker count - is what these
controls protect.

Writes a JSON report and nothing else; it never touches the dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def _summarize_point(workers: int, result, wall: float) -> dict:
    outcomes = [o for o in result.outcomes if o.error is None]
    durations = [o.wall_seconds for o in outcomes]
    turns = [o.turns for o in outcomes if o.turns]
    teacher_seconds = sum(o.meter.get("teacher_seconds", 0.0) for o in result.outcomes)
    env_seconds = sum(o.meter.get("env_seconds", 0.0) for o in result.outcomes)
    teacher_calls = sum(o.meter.get("teacher_calls", 0) for o in result.outcomes)
    env_calls = sum(o.meter.get("env_calls", 0) for o in result.outcomes)
    busy_seconds = sum(o.wall_seconds for o in result.outcomes)
    mean_episode = statistics.fmean(durations) if durations else 0.0
    total_turns = sum(turns)

    return {
        "workers": workers,
        "wall_seconds": round(wall, 1),
        "attempted": result.attempted,
        "kept": result.kept,
        "errors": result.errors,
        "by_category": dict(result.by_category),
        "measured_episodes_per_hour": round(result.attempted / wall * 3600.0, 1) if wall else 0.0,
        "measured_kept_per_hour": round(result.kept / wall * 3600.0, 1) if wall else 0.0,
        "steady_state_episodes_per_hour": (
            round(workers * 3600.0 / mean_episode, 1) if mean_episode else 0.0
        ),
        "episode_seconds_mean": round(mean_episode, 1),
        "episode_seconds_p50": round(_percentile(durations, 0.50), 1),
        "episode_seconds_p90": round(_percentile(durations, 0.90), 1),
        "turns_mean": round(statistics.fmean(turns), 2) if turns else 0.0,
        "total_turns": total_turns,
        "seconds_per_turn": round(busy_seconds / total_turns, 1) if total_turns else 0.0,
        "teacher_seconds": round(teacher_seconds, 1),
        "env_seconds": round(env_seconds, 1),
        "teacher_calls": teacher_calls,
        "env_calls": env_calls,
        "teacher_seconds_per_call": (
            round(teacher_seconds / teacher_calls, 2) if teacher_calls else 0.0
        ),
        "env_seconds_per_call": round(env_seconds / env_calls, 2) if env_calls else 0.0,
        # Of the wall clock an episode occupies, how much is accounted for by the
        # two instrumented legs. A large remainder means time is going somewhere
        # neither the gateway nor the GPU explains (harness overhead, contention).
        "teacher_share": round(teacher_seconds / busy_seconds, 3) if busy_seconds else 0.0,
        "env_share": round(env_seconds / busy_seconds, 3) if busy_seconds else 0.0,
        "unattributed_share": (
            round(1.0 - (teacher_seconds + env_seconds) / busy_seconds, 3)
            if busy_seconds else 0.0
        ),
        "bytes_written": result.bytes_written,
        "bytes_per_kept": (
            round(result.bytes_written / result.kept) if result.kept else 0
        ),
        "stopped_reason": result.stopped_reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", default="8,16,32,64",
                        help="comma-separated worker counts to sweep")
    parser.add_argument("--rounds", type=float, default=2.0,
                        help="episodes per point = rounds * workers, so every point "
                             "runs the same number of sequential episode-lengths")
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--teacher", default="claude")
    parser.add_argument("--model-teacher", default="")
    parser.add_argument("--out-dir", default="runs/pilot_agentic")
    parser.add_argument("--report", default="runs/pilot_agentic/report.json")
    parser.add_argument("--task-seed", type=int, default=1337)
    parser.add_argument("--min-free-gb", type=float, default=60.0)
    parser.add_argument("--point-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--keep-only-useful", action="store_true")
    parser.add_argument("--cold-compile-cache", default="",
                        help="root for per-point compile caches; empty keeps the "
                             "inherited (shared, warm) cache")
    args = parser.parse_args()

    from kore.data.saturated_agentic import run_node_shard
    from kore.data.teacher import load_env_local, make_teacher
    from kore.tasks.registry import train_tasks

    worker_points = [int(w) for w in args.workers.split(",") if w.strip()]
    gpu_ids = [int(g) for g in args.gpu_ids.split(",") if g.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic task sample, then a disjoint contiguous slice per sweep point.
    import random

    all_ids = sorted(task.task_id for task in train_tasks())
    rng = random.Random(args.task_seed)
    rng.shuffle(all_ids)
    needed = sum(
        max(1, int(round(args.rounds * w)) // max(1, args.episodes_per_task))
        for w in worker_points
    )
    if needed > len(all_ids):
        print(f"FATAL: sweep needs {needed} distinct tasks, registry has {len(all_ids)}")
        return 2

    load_env_local()
    teacher_kwargs = {"model": args.model_teacher} if args.model_teacher else {}
    teacher = make_teacher(args.teacher, resilient=True, **teacher_kwargs)

    report = {
        "host": os.uname().nodename,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu_ids": gpu_ids,
        "teacher": args.teacher,
        "model_teacher": args.model_teacher or None,
        "max_turns": args.max_turns,
        "episodes_per_task": args.episodes_per_task,
        "rounds": args.rounds,
        "keep_only_useful": bool(args.keep_only_useful),
        "runs_dir": os.environ.get("KORE_RUNS_DIR"),
        "cold_compile_cache": bool(args.cold_compile_cache),
        "points": [],
    }

    cursor = 0
    for workers in worker_points:
        episodes = max(1, int(round(args.rounds * workers)))
        n_tasks = max(1, episodes // max(1, args.episodes_per_task))
        slice_ids = all_ids[cursor:cursor + n_tasks]
        cursor += n_tasks
        shard = out_dir / f"w{workers:03d}.jsonl"
        telemetry = out_dir / f"w{workers:03d}.telemetry.jsonl"
        print(f"\n=== worker sweep point: workers={workers} tasks={len(slice_ids)} "
              f"episodes={len(slice_ids) * args.episodes_per_task} ===", flush=True)

        # KoreEnv reads this per evaluation subprocess, so pointing it somewhere
        # new gives this point a cold compile cache. Without it the last point in
        # the sweep would win on inherited compiles rather than on worker count.
        if args.cold_compile_cache:
            cache_root = Path(args.cold_compile_cache) / f"w{workers:03d}"
            (cache_root / "triton").mkdir(parents=True, exist_ok=True)
            (cache_root / "inductor").mkdir(parents=True, exist_ok=True)
            os.environ["KORE_COMPILE_CACHE_DIR"] = str(cache_root)
            os.environ.pop("TRITON_CACHE_DIR", None)
            os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)

        start = time.monotonic()
        result = run_node_shard(
            task_ids=slice_ids,
            episodes_per_task=args.episodes_per_task,
            workers=workers,
            gpu_ids=gpu_ids,
            out_path=shard,
            telemetry_path=telemetry,
            max_turns=args.max_turns,
            teacher=teacher,
            keep_only_useful=bool(args.keep_only_useful),
            min_free_bytes=int(args.min_free_gb * 1e9),
            deadline=time.monotonic() + args.point_timeout_seconds,
            resume=True,
            shard_meta={"pilot": True, "workers": workers},
            progress_every=5,
        )
        wall = time.monotonic() - start
        point = _summarize_point(workers, result, wall)
        report["points"].append(point)
        print(json.dumps(point, indent=2), flush=True)

        # Write incrementally: a preempted pilot still leaves the points it finished.
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")

        if result.stopped_reason == "disk_floor":
            print("FATAL: stopping sweep, disk floor reached")
            break

    best = max(report["points"], key=lambda p: p["steady_state_episodes_per_hour"],
               default=None)
    if best:
        report["recommended_workers"] = best["workers"]
        report["recommended_steady_state_episodes_per_hour"] = (
            best["steady_state_episodes_per_hour"])
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.report}")
    if best:
        print(f"best point: workers={best['workers']} "
              f"steady_state={best['steady_state_episodes_per_hour']} ep/h/node")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
