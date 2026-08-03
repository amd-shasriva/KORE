#!/usr/bin/env python3
"""Produce the receipt that arms ``profiling_reward_weight``.

``scripts/validate_rocprofv3_coverage.py`` proves the PARSER reads this ROCm
build's kernel-trace export. That is necessary but not what the gate asks for:
:func:`kore.policy.capabilities` requires evidence that
``KoreEnv.collect_kernel_trace`` -- which stages its own isolated workdir,
resolves a shape, and invokes the profiler through the sandbox -- produced a
usable trace on THIS hardware. Every one of those steps can fail independently
of the parser, and each failure is silent by design: the collector returns
``None`` so a rollout never dies for want of a profile.

So this runs the real env method on real registry tasks with their own seed
kernels, and writes a receipt only for what it actually observed. A task whose
trace comes back ``None`` is recorded as a failure, not skipped, because the
useful number here is the fraction of tasks the coverage term will be live on --
an armed weight that measures a tenth of the task pool is a different (and much
worse) thing than one that measures nearly all of it.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def rocm_version() -> str:
    try:
        out = subprocess.run(["rocprofv3", "--version"], capture_output=True,
                             text=True, timeout=120).stdout
        for line in out.splitlines():
            if "version" in line.lower():
                return line.strip()
        return out.strip().splitlines()[0] if out.strip() else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def gpu_arch() -> str:
    try:
        import torch
        return torch.cuda.get_device_properties(0).gcnArchName
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=8,
                    help="how many registry tasks to probe")
    ap.add_argument("--gpu", default=os.environ.get("KORE_PROBE_GPU", "0"))
    ap.add_argument("--out", default="data/ktrace_receipt.json")
    ap.add_argument("--task-id", action="append", default=[],
                    help="probe specific task ids instead of a sample")
    args = ap.parse_args()

    from kore.env.kore_env import KoreEnv
    from kore.reward.coverage import candidate_kernel_names, kernel_coverage
    from kore.tasks.registry import get_task, task_ids

    if args.task_id:
        chosen = args.task_id
    else:
        # Prefer cheap elementwise/normalisation tasks: this receipt is about
        # whether the profiler path works, and a flash-attention backward pass
        # would spend all its time proving something we are not asking.
        prefer = ("add", "mul", "relu", "gelu", "silu", "softmax", "layernorm",
                  "rmsnorm", "elementwise", "scale", "bias", "dropout")
        ids = task_ids()
        ranked = sorted(
            ids,
            key=lambda t: (0 if any(p in t.lower() for p in prefer) else 1, len(t)))
        chosen = ranked[:args.tasks]

    print(f"# gpu   : {args.gpu}   arch: {gpu_arch()}")
    print(f"# rocm  : {rocm_version()}")
    print(f"# tasks : {len(chosen)}")

    observations: list[dict] = []
    for task_id in chosen:
        entry: dict = {"task_id": task_id}
        t0 = time.time()
        try:
            task = get_task(task_id)
            source = task.seed_source
            env = KoreEnv(task, use_replay=False, gpu=args.gpu)
            dispatches = env.collect_kernel_trace(source)
        except Exception as exc:  # noqa: BLE001
            entry.update(ok=False, error=f"{type(exc).__name__}: {exc}")
            observations.append(entry)
            print(f"  {task_id[:44]:<44} ERROR {entry['error'][:60]}")
            continue

        entry["seconds"] = round(time.time() - t0, 1)
        if not dispatches:
            entry.update(ok=False, error="collect_kernel_trace returned no dispatches")
            observations.append(entry)
            print(f"  {task_id[:44]:<44} NONE  ({entry['seconds']}s)")
            continue

        names = sorted(candidate_kernel_names(source))
        report = kernel_coverage(dispatches, names)
        entry.update(
            ok=True,
            n_dispatches=len(dispatches),
            candidate_kernels=names,
            total_ns=sum(int(d.duration_ns) for d in dispatches),
        )
        if report is not None:
            entry.update(
                coverage=report.coverage,
                n_candidate_dispatches=report.n_candidate_dispatches,
                never_ran=report.never_ran,
            )
            cov = f"cov={report.coverage:.4f}"
        else:
            entry["coverage"] = None
            cov = "cov=None"
        observations.append(entry)
        print(f"  {task_id[:44]:<44} OK    {len(dispatches):>4} disp  {cov}  "
              f"({entry['seconds']}s)")

    ok = [o for o in observations if o.get("ok")]
    measured = [o for o in ok if o.get("coverage") is not None]
    rate = len(ok) / len(observations) if observations else 0.0

    receipt = {
        "what": "KoreEnv.collect_kernel_trace produced usable rocprofv3 "
                "kernel traces on this hardware",
        "generated_by": "scripts/make_ktrace_receipt.py",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "gpu_arch": gpu_arch(),
        "rocprofv3": rocm_version(),
        "tasks_probed": len(observations),
        "tasks_traced": len(ok),
        "tasks_with_coverage": len(measured),
        "trace_success_rate": round(rate, 4),
        "parser_validation": "scripts/validate_rocprofv3_coverage.py "
                             "(dominant/minor/decoy on gfx950)",
        "observations": observations,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2))
    print(f"\n# wrote {out}")
    print(f"# traced {len(ok)}/{len(observations)} tasks "
          f"({rate:.0%}), coverage computed on {len(measured)}")

    # A receipt is only worth writing if the path it attests to actually works
    # on the majority of tasks. Below that the honest move is to leave the
    # weight at 0.0 rather than arm a term that is inert most of the time.
    if rate < 0.5:
        print("\nFAILED -- trace success rate too low to arm the reward")
        return 1
    print("\nPASSED -- receipt is valid evidence for profiling_reward_weight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
