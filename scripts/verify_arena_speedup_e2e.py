"""Prove on real GPUs that the arena can measure a speedup, end to end.

Why this exists: the 2026-08-10 sweep scored 302 correct kernels and produced 49
speedups. Every unit test passed the whole time, because each one exercises a
parser or a matcher in isolation and none of them runs a task on a GPU and asks
"did a number come out?". That gap cost 253 measurements, unrecoverably, because
the workspaces are deleted after scoring.

The check is an IDENTITY transformation: stage a real task, time it once as the
baseline, then submit the byte-identical source as the "optimized" answer and time
it again. The expected speedup is 1.0x, which is what makes it a test -- the
result is known in advance, so anything else is a defect in the measurement path
rather than an opinion about a kernel. Specifically it proves:

  * the baseline pass produces per-case denominators at all
  * the optimized pass parses its own timings
  * cases pair up by shape/params across two independent runs
  * a ratio comes out, and lands within noise of 1.0

Run it on the GEAK-style suites, not the gpumode ones. gpumode harnesses print
their own ratio in-process and were the only family that ever worked; the vLLM and
rocmbench families are the 216 tasks that came back dark, and they are the ones
whose denominator has to survive a round trip through a ledger.

    python scripts/verify_arena_speedup_e2e.py \
        --arena-root ~/third_party/AgentKernelArena \
        --task triton2triton/vllm/triton_rms_norm

Nothing here touches a cluster or a scheduler; it runs locally on whatever GPU
HIP_VISIBLE_DEVICES points at.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kore.eval.agent_kernel_arena import (  # noqa: E402
    _case_time,
    evaluate_task,
    load_task,
    speedup_from_cases,
)

# Reused so the staging is identical to a real run rather than a second
# implementation that could drift from it.
from scripts.run_agent_kernel_arena import _workspace  # noqa: E402


def _fmt(cases: list) -> str:
    if not cases:
        return "none"
    bits = []
    for c in cases[:4]:
        key = c.get("params") or c.get("shape") or c.get("test_case_id") or "?"
        # Via _case_time, not a raw key lookup: the rocmbench dialect carries its
        # time under timing_ms.mean and has no execution_time_ms at all, so reading
        # the raw key printed "Nonems" for cases that were in fact timed fine.
        t = _case_time(c)
        m = c.get("benchmark_method") or "no-method-reported"
        bits.append(f"{key}={t}ms[{m}]")
    more = "" if len(cases) <= 4 else f" (+{len(cases) - 4} more)"
    return ", ".join(str(b) for b in bits) + more


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arena-root", required=True, type=Path)
    ap.add_argument("--task", required=True,
                    help="task id relative to tasks/, e.g. "
                         "triton2triton/vllm/triton_rms_norm")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--tolerance", type=float, default=0.35,
                    help="how far from 1.0x an identity transform may land. GPU "
                         "timing is noisy and a re-measured baseline moves; this "
                         "is a sanity bound, not a precision claim")
    args = ap.parse_args()

    cfg = args.arena_root / "tasks" / args.task / "config.yaml"
    if not cfg.is_file():
        print(f"no such task: {cfg}")
        return 2
    task = load_task(cfg)
    print(f"task      : {task.task_id}  ({task.task_type})")
    print(f"perf cmd  : {task.performance_command}")

    root = Path(tempfile.mkdtemp(prefix="aka_e2e_"))
    rc = 1
    try:
        # ---- pass 1: the baseline, i.e. the task exactly as shipped ----------
        ws1 = _workspace(task, root / "baseline", args.arena_root)
        base = evaluate_task(task, ws1, timeout=args.timeout)
        print("\n-- baseline pass --")
        print(f"  compiled={base.compiled} correct={base.correct}")
        print(f"  optimized_seconds={base.optimized_seconds}")
        print(f"  cases: {_fmt(base.perf_cases)}")
        print(f"  note : {base.speedup_note or '-'}")
        if not base.perf_cases and base.optimized_seconds is None:
            print("\nFAIL: the baseline pass produced no timing at all. Every "
                  "speedup for this suite depends on it.")
            return 1

        # Round-trip the cases through JSON exactly as the ledger does, so this
        # also covers the serialization the real two-job flow depends on.
        carried = json.loads(json.dumps(base.perf_cases))

        # ---- pass 2: identity "optimization", scored against pass 1 ----------
        ws2 = _workspace(task, root / "optimized", args.arena_root)
        opt = evaluate_task(task, ws2, timeout=args.timeout,
                            reference_latency=base.optimized_seconds,
                            reference_cases=carried)
        print("\n-- optimized pass (identical source) --")
        print(f"  compiled={opt.compiled} correct={opt.correct}")
        print(f"  optimized_seconds={opt.optimized_seconds}")
        print(f"  cases: {_fmt(opt.perf_cases)}")
        print(f"  speedup={opt.speedup}")
        print(f"  note : {opt.speedup_note or '-'}")

        # ---- verdict ---------------------------------------------------------
        print("\n-- verdict --")
        if opt.speedup is None:
            print("FAIL: no speedup was produced even with a baseline in hand.")
            print("      This is the exact failure that made 253 correct kernels "
                  "score 120 in the last sweep.")
            return 1
        lo, hi = 1.0 - args.tolerance, 1.0 + args.tolerance
        direct, why = speedup_from_cases(carried, opt.perf_cases)
        print(f"  speedup via evaluate_task : {opt.speedup:.4f}x")
        if direct:
            print(f"  speedup via per-case match: {direct:.4f}x  ({why or 'clean'})")
        if lo <= opt.speedup <= hi:
            print(f"  PASS: identity transform measured {opt.speedup:.4f}x, "
                  f"within [{lo:.2f}, {hi:.2f}] of the expected 1.0x")
            rc = 0
        else:
            print(f"  FAIL: identity transform measured {opt.speedup:.4f}x, "
                  f"outside [{lo:.2f}, {hi:.2f}]. The path produces a number, "
                  f"but not a trustworthy one.")
            rc = 1
        return rc
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
