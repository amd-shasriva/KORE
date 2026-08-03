#!/usr/bin/env python
"""Run the evolve-agent loop over a set of tasks and write the results.

Thin driver around :func:`kore.search.evolve_agent.evolve`: resolve the task
list, build the generator once, and give each task its own ``KoreEnv``. Every
interesting decision lives in the module; this exists so an sbatch has something
to call and so the shard's identity, budget and output paths are decided in
Python rather than in shell arithmetic.

Parallelism is across JOBS, not threads: ``--shard i --shards n`` takes the
i-th stripe of the task list, so a job array fans out without this script owning
a worker pool it cannot test on a login node. One GPU per process
(``--gpu``), because ``KoreEnv`` benches in a subprocess pinned by
``HIP_VISIBLE_DEVICES`` and two evolve loops sharing a device would contend and
report each other's contention as kernel slowness.

Exits non-zero when the run produced no verified kernel at all, because a run
that silently optimised nothing looks exactly like a run that finished.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _resolve_tasks(args) -> list[str]:
    from kore.tasks import registry

    if args.task_file:
        ids = [line.strip() for line in Path(args.task_file).read_text().splitlines()
               if line.strip() and not line.startswith("#")]
    elif args.tasks:
        ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        ids = registry.task_ids()
    if args.shards > 1:
        ids = ids[args.shard::args.shards]
    if args.limit > 0:
        ids = ids[:args.limit]
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="", help="comma-separated task ids")
    parser.add_argument("--task-file", default="", help="file with one task id per line")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--gpu", default=None, help="HIP device for the bench subprocess")
    parser.add_argument("--teacher", default="vllm",
                        help="generator kind: vllm | hf | claude | stub")
    parser.add_argument("--model", default="", help="generator model id")
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--turns", type=int, default=4,
                        help="harness turns per generation (the mutation operator)")
    parser.add_argument("--exemplars", type=int, default=4,
                        help="archive exemplars in the prompt (Dr. Kernel's w=4)")
    parser.add_argument("--budget", type=int, default=400,
                        help="verifier-call cap per task")
    parser.add_argument("--capacity", type=int, default=32, help="archive capacity")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="stop cleanly after this many seconds (0 = no limit)")
    args = parser.parse_args()

    from kore.data.teacher import load_env_local, make_teacher
    from kore.env.kore_env import KoreEnv
    from kore.search.evolve_agent import EvolveAgentConfig, HarnessProposer, evolve
    from kore.tasks import registry

    task_ids = _resolve_tasks(args)
    if not task_ids:
        print("[evolve] empty task list; nothing to do", flush=True)
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"evolve_shard_{args.shard:03d}.jsonl"

    load_env_local()
    teacher_kwargs = {"model": args.model} if args.model else {}
    generator = make_teacher(args.teacher, resilient=True, **teacher_kwargs)
    proposer = HarnessProposer(generator, max_turns=args.turns)
    cfg = EvolveAgentConfig(
        generations=args.generations,
        max_env_calls=args.budget,
        turns_per_generation=args.turns,
        exemplars=args.exemplars,
        archive_capacity=args.capacity,
        seed=args.seed,
    )

    print(f"[evolve] host={os.uname().nodename} tasks={len(task_ids)} "
          f"shard={args.shard}/{args.shards} generations={cfg.generations} "
          f"budget={cfg.max_env_calls} out={results_path}", flush=True)

    deadline = (time.monotonic() + args.seconds) if args.seconds > 0 else None
    wins = 0
    done = 0
    with results_path.open("a", encoding="utf-8") as sink:
        for task_id in task_ids:
            if deadline is not None and time.monotonic() > deadline:
                print(f"[evolve] deadline reached after {done} tasks", flush=True)
                break
            try:
                task = registry.get_task(task_id)
            except KeyError as exc:
                print(f"[evolve] {task_id}: {exc}", flush=True)
                continue
            env = KoreEnv(task, gpu=args.gpu)
            result = evolve(task, proposer, env, cfg)
            payload = result.to_dict()
            champion = result.best
            # The winning SOURCE travels with the record: a summary that names a
            # speedup but not the kernel that achieved it cannot be re-verified.
            payload["best_source"] = champion.source if champion is not None else None
            sink.write(json.dumps(payload) + "\n")
            sink.flush()
            done += 1
            speedup = result.scaling.best_speedup
            if speedup is not None and speedup > 1.0:
                wins += 1
            print(f"[evolve] {task_id}: best={speedup} "
                  f"coverage={result.archive.coverage()} "
                  f"env_calls={result.env_calls} "
                  f"hacks={result.stats['rejected_hacks']} "
                  f"implausible={result.stats['rejected_implausible']}", flush=True)

    summary = {"shard": args.shard, "tasks": done, "wins": wins,
               "results": str(results_path)}
    (out_dir / f"evolve_shard_{args.shard:03d}.summary.json").write_text(
        json.dumps(summary, indent=2))
    print("[evolve] " + json.dumps(summary), flush=True)
    # No verified kernel anywhere is a failure, not a quiet success.
    return 0 if done and wins else 1


if __name__ == "__main__":
    raise SystemExit(main())
