#!/usr/bin/env python
"""End-to-end proof that a HIP task is runnable through the real KoreEnv path.

Feeds each HIP task its OWN declared seed as the candidate and reports what the
environment concluded: did it compile, did the oracle verify it on every declared
shape, was the timing publication-eligible, and what speedup was measured against
the production baseline.

This is deliberately the whole environment, not a shortcut: the same staging, the
same read-only oracle, the same reward-hack scan, the same paired cold-cache
timing protocol, and the same post-timing re-verification a model's candidate
gets.  A task that passes here is runnable; a task that does not is a liability
and must not be claimed as coverage.

Usage
-----
    PYTHONPATH=. python scripts/verify_hip_tasks_e2e.py --gpu 7 \
        [--tasks hip_gemm_bf16,...] [--json out.json] [--no-bench]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Optional


def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="", help="comma-separated task ids")
    parser.add_argument("--backend", default="hip", help="filter the registry by backend")
    parser.add_argument("--gpu", default=None, help="physical GPU index")
    parser.add_argument("--json", default="", help="write the report here")
    parser.add_argument("--no-bench", action="store_true",
                        help="correctness only (skip the timed protocol)")
    parser.add_argument("--verified-correctness", action="store_true",
                        help="also run the enumerated adversarial regimes")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args(argv)

    if args.gpu is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.verified_correctness:
        os.environ["KORE_VERIFIED_CORRECTNESS"] = "1"

    from kore.env.kore_env import KoreEnv
    from kore.reward.reward import _worst_speedup
    from kore.tasks.registry import all_tasks, get_task

    if args.tasks:
        tasks = [get_task(t) for t in args.tasks.split(",") if t]
    else:
        tasks = [t for t in all_tasks() if t.backend == args.backend]
    tasks.sort(key=lambda t: t.task_id)
    if not tasks:
        print(f"no tasks with backend={args.backend!r}")
        return 1

    print(f"evaluating {len(tasks)} task(s) through KoreEnv "
          f"(bench={'off' if args.no_bench else 'on'}, "
          f"adversarial={'on' if args.verified_correctness else 'off'})")

    rows: list[dict[str, Any]] = []
    for task in tasks:
        started = time.time()
        env = KoreEnv(task, use_replay=False, gpu=args.gpu,
                      correctness_timeout=args.timeout, bench_timeout=args.timeout)
        obs = env.step(task.seed_source, full_validation=not args.no_bench)
        row = {
            "task_id": task.task_id,
            "backend": task.backend,
            "dtype": task.dtype,
            "operation": task.operation,
            "compiled": bool(obs.compiled),
            "correct": bool(obs.validation_passed),
            "flagged_hack": bool(obs.flagged_hack),
            "hack_reason": obs.hack_reason,
            "infra_error": bool(obs.infra_error),
            "snr_db": obs.snr_db,
            "snr_by_shape": dict(obs.snr_by_shape or {}),
            "gate_db": task.snr_threshold,
            # The worst per-shape speedup, which is what the reward uses -- not a
            # best-shape number that would flatter a kernel tuned for one size.
            "speedup": _worst_speedup(obs),
            "wall_by_shape": dict(getattr(obs, "wall_by_shape", None) or {}),
            "cv_pct": getattr(obs, "cv_pct", None),
            "performance_eligible": getattr(obs, "performance_eligible", None),
            "timing_protocol": getattr(obs, "timing_protocol", None),
            "timing_grade": getattr(obs, "timing_grade", None),
            "error_text": (obs.error_text or "")[:400],
            "seconds": round(time.time() - started, 1),
        }
        row["runnable"] = bool(
            row["compiled"] and row["correct"] and not row["infra_error"]
            and not row["flagged_hack"]
            and (args.no_bench or row["performance_eligible"] is True))
        rows.append(row)
        _print(row, args.no_bench)

    runnable = [r for r in rows if r["runnable"]]
    print(f"\n{len(runnable)}/{len(rows)} tasks PROVEN runnable end-to-end")
    for row in rows:
        if not row["runnable"]:
            print(f"  NOT runnable: {row['task_id']}: {row['error_text'] or 'see above'}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"rows": rows}, fh, indent=2)
        print(f"wrote {args.json}")
    return 0 if runnable and len(runnable) == len(rows) else 1


def _print(row: dict, no_bench: bool) -> None:
    verdict = "RUNNABLE" if row["runnable"] else "FAILED  "
    snr = row["snr_db"]
    snr_text = f"{snr:.1f}dB" if isinstance(snr, (int, float)) else str(snr)
    speed = row["speedup"]
    speed_text = f"{speed:.3f}x" if isinstance(speed, (int, float)) else "-"
    print(f"  {row['task_id']:26s} {verdict} snr={snr_text:>9s} "
          f"(gate {row['gate_db']}) "
          + ("" if no_bench else f"speedup={speed_text:>8s} "
             f"eligible={row['performance_eligible']} "
             f"protocol={row['timing_protocol']} ")
          + f"{row['seconds']}s")
    if row["error_text"] and not row["runnable"]:
        print(f"      {row['error_text'][:300]}")


if __name__ == "__main__":
    raise SystemExit(_main())
