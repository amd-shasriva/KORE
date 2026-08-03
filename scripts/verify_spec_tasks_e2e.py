#!/usr/bin/env python
"""End-to-end proof that a SPEC-SYNTHESIS task is runnable through real KoreEnv.

A spec task cannot be proven the way an optimize task is.  ``verify_hip_tasks_e2e``
feeds each task its own declared seed, because for an optimize task the seed IS a
working kernel and "the seed verifies" is exactly the claim that matters.  A spec
task's seed is a signature stub with no implementation, so feeding it back would
prove nothing except that the stub is a stub.

So this script proves the two facts that actually make a spec task sound, and it
fails unless BOTH hold:

1. **Solvable.**  A real from-scratch Triton kernel for the declared entry point
   compiles, clears the SNR gate on EVERY declared shape through the same
   read-only oracle, survives the reward-hack scan and the post-timing
   re-verification, and produces a publication-eligible timing against the
   production baseline.  Without this the task is a trap: it would burn datagen
   GPU time and report as a model error, which is how a campaign reaches a 97%
   error rate while its throughput counter reads 13,277 episodes/hour, because
   failures are fast.

2. **Genuinely unsolved.**  The declared stub does NOT pass.  This is what
   distinguishes a synthesis task from one that ships its own answer.  If a stub
   ever verified, the task would be scoring the model for returning what it was
   handed.

The solution used for (1) is ``_genops.seed_source(op, family, dtype)`` -- the
same generator whose output the ``gen_*`` tasks ship as their seeds, all of which
are recorded PASS on gfx950 in ``data/gfx950_task_verification.json``.  Using it
here is deliberate: the point is to prove the TASK is well-posed, and a
known-good solution isolates that from the question of whether some newly
hand-written kernel happens to be correct.  It also means a failure here
implicates the task, not the prover.

Usage
-----
    PYTHONPATH=. python scripts/verify_spec_tasks_e2e.py --gpu 0 \
        [--tasks spec_row_rms_bf16,...] [--json out.json] [--no-bench]
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
    parser.add_argument("--gpu", default=None, help="physical GPU index")
    parser.add_argument("--json", default="", help="write the report here")
    parser.add_argument("--no-bench", action="store_true",
                        help="correctness only (skip the timed protocol)")
    parser.add_argument("--skip-stub-check", action="store_true",
                        help="do not verify that the stub fails (NOT recommended: "
                             "the stub check is what proves the task is unsolved)")
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
    from kore.tasks._genops import seed_source as reference_solution
    from kore.tasks.registry import all_tasks, get_task

    if args.tasks:
        tasks = [get_task(t) for t in args.tasks.split(",") if t]
    else:
        tasks = [t for t in all_tasks() if getattr(t, "is_spec_synthesis", False)]
    tasks.sort(key=lambda t: t.task_id)
    if not tasks:
        print("no spec-synthesis tasks found")
        return 1

    print(f"evaluating {len(tasks)} spec task(s) through KoreEnv "
          f"(bench={'off' if args.no_bench else 'on'}, "
          f"stub-check={'off' if args.skip_stub_check else 'on'}, "
          f"adversarial={'on' if args.verified_correctness else 'off'})")

    rows: list[dict[str, Any]] = []
    for task in tasks:
        started = time.time()
        family = task.source_family or ""
        try:
            solution = reference_solution(task.operation, family, task.dtype)
        except Exception as exc:  # noqa: BLE001 - a missing solution is a task defect
            rows.append({
                "task_id": task.task_id, "runnable": False,
                "error_text": f"no reference solution: {type(exc).__name__}: {exc}",
                "seconds": round(time.time() - started, 1),
            })
            print(f"  {task.task_id:28s} FAILED   no reference solution: {exc}")
            continue

        env = KoreEnv(task, use_replay=False, gpu=args.gpu,
                      correctness_timeout=args.timeout, bench_timeout=args.timeout)
        obs = env.step(solution, full_validation=not args.no_bench)

        row = {
            "task_id": task.task_id,
            "task_kind": task.task_kind,
            "backend": task.backend,
            "dtype": task.dtype,
            "operation": task.operation,
            "op_family": family,
            "spec_chars": len(task.spec_source),
            "compiled": bool(obs.compiled),
            "correct": bool(obs.validation_passed),
            "flagged_hack": bool(obs.flagged_hack),
            "hack_reason": obs.hack_reason,
            "infra_error": bool(obs.infra_error),
            "snr_db": obs.snr_db,
            "snr_by_shape": dict(obs.snr_by_shape or {}),
            "gate_db": task.snr_threshold,
            "speedup": _worst_speedup(obs),
            "wall_by_shape": dict(getattr(obs, "wall_by_shape", None) or {}),
            "cv_pct": getattr(obs, "cv_pct", None),
            "performance_eligible": getattr(obs, "performance_eligible", None),
            "timing_protocol": getattr(obs, "timing_protocol", None),
            "timing_grade": getattr(obs, "timing_grade", None),
            "error_text": (obs.error_text or "")[:400],
        }

        # Fact 2: the stub must NOT pass. Run it through the same oracle.
        row["stub_checked"] = not args.skip_stub_check
        row["stub_passes"] = None
        if not args.skip_stub_check:
            stub_env = KoreEnv(task, use_replay=False, gpu=args.gpu,
                               correctness_timeout=args.timeout,
                               bench_timeout=args.timeout)
            stub_obs = stub_env.step(task.seed_source, full_validation=False)
            row["stub_passes"] = bool(stub_obs.compiled and stub_obs.validation_passed)
            row["stub_error"] = (stub_obs.error_text or "")[:200]

        row["solvable"] = bool(
            row["compiled"] and row["correct"] and not row["infra_error"]
            and not row["flagged_hack"]
            and (args.no_bench or row["performance_eligible"] is True))
        row["unsolved_by_stub"] = (
            True if args.skip_stub_check else row["stub_passes"] is False)
        row["runnable"] = bool(row["solvable"] and row["unsolved_by_stub"])
        row["seconds"] = round(time.time() - started, 1)
        rows.append(row)
        _print(row, args.no_bench)

    runnable = [r for r in rows if r.get("runnable")]
    print(f"\n{len(runnable)}/{len(rows)} spec tasks PROVEN "
          f"(solvable AND unsolved by their stub)")
    for row in rows:
        if not row.get("runnable"):
            why = []
            if not row.get("solvable"):
                why.append(f"not solvable: {row.get('error_text') or 'see above'}")
            if row.get("stub_passes"):
                why.append("STUB PASSES -- the task ships its own answer")
            print(f"  NOT proven: {row['task_id']}: {'; '.join(why) or 'unknown'}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"rows": rows}, fh, indent=2)
        print(f"wrote {args.json}")
    return 0 if runnable and len(runnable) == len(rows) else 1


def _print(row: dict, no_bench: bool) -> None:
    verdict = "PROVEN " if row["runnable"] else "FAILED "
    snr = row["snr_db"]
    snr_text = f"{snr:.1f}dB" if isinstance(snr, (int, float)) else str(snr)
    speed = row["speedup"]
    speed_text = f"{speed:.3f}x" if isinstance(speed, (int, float)) else "-"
    stub = row.get("stub_passes")
    stub_text = "stub_fails" if stub is False else (
        "STUB_PASSES!" if stub else "stub_unchecked")
    print(f"  {row['task_id']:28s} {verdict} snr={snr_text:>9s} "
          f"(gate {row['gate_db']}) "
          + ("" if no_bench else f"speedup={speed_text:>8s} "
             f"eligible={row['performance_eligible']} "
             f"protocol={row['timing_protocol']} ")
          + f"{stub_text} {row['seconds']}s")
    if row["error_text"] and not row["runnable"]:
        print(f"      {row['error_text'][:300]}")


if __name__ == "__main__":
    raise SystemExit(_main())
