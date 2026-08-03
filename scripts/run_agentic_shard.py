#!/usr/bin/env python
"""Run one node's shard of the agentic datagen campaign.

Thin driver: read the shard's task list, build the teacher once, and hand the
whole thing to ``run_node_shard``, which owns the pool, the per-episode
checkpointing and the disk floor. Everything interesting is there; this exists so
the sbatch has something to call and so the shard's identity, deadline and
disk budget are decided in Python rather than in shell arithmetic.

Exits non-zero when the run stopped for a reason the campaign must not paper
over - the disk floor especially, because a shard that quietly stops writing
looks exactly like a shard that finished.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=6)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--teacher", default="claude")
    parser.add_argument("--model-teacher", default="")
    parser.add_argument("--min-free-gb", type=float, default=150.0)
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="stop cleanly after this many seconds (0 = no limit)")
    parser.add_argument("--keep-only-useful", action="store_true",
                        help="drop episodes that never reached a correct kernel")
    args = parser.parse_args()

    from kore.data.saturated_agentic import run_node_shard
    from kore.data.teacher import load_env_local, make_teacher

    task_ids = [
        line.strip()
        for line in Path(args.shard_file).read_text().splitlines()
        if line.strip()
    ]
    if not task_ids:
        print(f"[shard {args.shard_index}] empty task list; nothing to do")
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = out_dir / f"shard_{args.shard_index:03d}.jsonl"
    telemetry_path = out_dir / f"shard_{args.shard_index:03d}.telemetry.jsonl"

    load_env_local()
    teacher_kwargs = {"model": args.model_teacher} if args.model_teacher else {}
    teacher = make_teacher(args.teacher, resilient=True, **teacher_kwargs)

    print(f"[shard {args.shard_index}] host={os.uname().nodename} "
          f"tasks={len(task_ids)} episodes_per_task={args.episodes_per_task} "
          f"workers={args.workers} out={shard_path}", flush=True)

    result = run_node_shard(
        task_ids=task_ids,
        episodes_per_task=args.episodes_per_task,
        workers=args.workers,
        gpu_ids=[int(g) for g in args.gpu_ids.split(",") if g.strip()],
        out_path=shard_path,
        telemetry_path=telemetry_path,
        max_turns=args.max_turns,
        teacher=teacher,
        keep_only_useful=bool(args.keep_only_useful),
        min_free_bytes=int(args.min_free_gb * 1e9),
        deadline=(time.monotonic() + args.seconds) if args.seconds > 0 else None,
        resume=True,
        shard_meta={"shard": args.shard_index, "host": os.uname().nodename},
        progress_every=25,
    )

    summary = result.summary()
    summary["shard"] = args.shard_index
    print("[shard] " + json.dumps(summary), flush=True)
    (out_dir / f"shard_{args.shard_index:03d}.summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    # A deadline stop is the normal end of a time-boxed element: the work is
    # durable and the requeue picks it up. A disk stop is not, and must not be
    # reported as a finished shard.
    if result.stopped_reason == "disk_floor":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
