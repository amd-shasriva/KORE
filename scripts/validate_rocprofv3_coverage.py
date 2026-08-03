#!/usr/bin/env python3
"""Validate the coverage reward's rocprofv3 path against real gfx950 hardware.

``kore.env.kore_env.KoreEnv.collect_kernel_trace`` and
``kore.verifier.parsers.rocprofv3.parse_kernel_dispatches`` were written against
AMD's *documented* kernel-trace schema but never run on silicon, so
``profiling_reward_weight`` ships disarmed. This script is the evidence that
gate is waiting on.

It runs three workloads under a real ``rocprofv3 --kernel-trace`` and checks the
parse against a ground truth known independently of the profiler:

  dominant  a Triton kernel doing ~all of the GPU work -> coverage near 1.0
  minor     the same kernel next to a much larger torch matmul -> coverage well
            under 0.5, which is the case that makes a big local speedup a small
            end-to-end one
  decoy     a Triton kernel that is defined and compiled but never launched ->
            a parse that works, zero matching dispatches, coverage exactly 0.0

The decoy is the one that matters. Coverage exists to catch a candidate that
reports a huge speedup because its kernel never ran, and a profiler integration
that cannot separate "did not run" from "was not measured" would let exactly
that through. Passing dominant and minor only proves we can read a CSV.

Exit code is 0 only if every check passes; anything else leaves the reward
disarmed.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Where the probe workloads write their own timing, so we have a ground truth
# that does not come from the profiler we are trying to validate.
WORKLOADS: dict[str, str] = {}

WORKLOADS["dominant"] = '''
import torch, triton, triton.language as tl

@triton.jit
def kore_probe_dominant(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    # Deliberately arithmetic-heavy so this kernel dominates wall time.
    acc = x
    for _ in range(64):
        acc = acc * 1.000001 + 0.000001
    tl.store(y_ptr + offs, acc, mask=mask)

n = 1 << 24
x = torch.randn(n, device="cuda", dtype=torch.float32)
y = torch.empty_like(x)
BLOCK = 1024
grid = (triton.cdiv(n, BLOCK),)
for _ in range(20):
    kore_probe_dominant[grid](x, y, n, BLOCK=BLOCK)
torch.cuda.synchronize()
'''

WORKLOADS["minor"] = '''
import torch, triton, triton.language as tl

@triton.jit
def kore_probe_minor(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=mask) * 2.0, mask=mask)

# Small elementwise kernel ...
n = 1 << 16
x = torch.randn(n, device="cuda", dtype=torch.float32)
y = torch.empty_like(x)
BLOCK = 256
grid = (triton.cdiv(n, BLOCK),)
kore_probe_minor[grid](x, y, n, BLOCK=BLOCK)

# ... next to a matmul that costs far more, so the candidate owns a small share.
a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(20):
    c = a @ b
kore_probe_minor[grid](x, y, n, BLOCK=BLOCK)
torch.cuda.synchronize()
'''

WORKLOADS["decoy"] = '''
import torch, triton, triton.language as tl

@triton.jit
def kore_probe_decoy(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=mask) * 2.0, mask=mask)

# The kernel above is NEVER launched. Real GPU work happens, so the trace is
# non-empty and healthy -- but none of it belongs to the candidate. This is the
# reward-hacking signature coverage has to catch.
a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
b = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
for _ in range(10):
    c = a @ b
torch.cuda.synchronize()
'''


def run_profile(name: str, source: str, workdir: Path, gpu: str) -> tuple[Path | None, str]:
    """Run one workload under rocprofv3; return (kernel-trace csv, log tail)."""
    d = workdir / name
    d.mkdir(parents=True, exist_ok=True)
    script = d / "workload.py"
    script.write_text(source)

    env = dict(os.environ)
    env["HIP_VISIBLE_DEVICES"] = gpu
    env["ROCR_VISIBLE_DEVICES"] = gpu

    cmd = [
        "rocprofv3", "--kernel-trace",
        "--output-format", "csv",
        "-d", str(d), "-o", "trace",
        "--", sys.executable, str(script),
    ]
    proc = subprocess.run(
        cmd, cwd=str(d), env=env, capture_output=True, text=True, timeout=1800)
    log = (proc.stdout[-1500:] + proc.stderr[-2500:]).strip()

    # rocprofv3 prefixes and/or nests its output; take any kernel-trace csv.
    hits = sorted(d.rglob("*kernel_trace.csv")) or sorted(d.rglob("*.csv"))
    return (hits[0] if hits else None), log


def main() -> int:
    from kore.reward.coverage import (
        candidate_kernel_names, kernel_coverage, coverage_ceiling)
    from kore.verifier.parsers.rocprofv3 import parse_kernel_dispatches

    gpu = os.environ.get("KORE_PROBE_GPU", "0")
    workdir = Path(os.environ.get(
        "KORE_PROBE_DIR", tempfile.mkdtemp(prefix="rocprof_probe_")))
    print(f"# workdir : {workdir}")
    print(f"# gpu     : {gpu}")

    results: dict[str, dict] = {}
    failures: list[str] = []

    for name, source in WORKLOADS.items():
        print(f"\n=== {name} ===", flush=True)
        csv_path, log = run_profile(name, source, workdir, gpu)
        if csv_path is None:
            print(f"FAIL: no kernel-trace csv produced\n{log}")
            failures.append(f"{name}: no csv")
            continue
        print(f"csv: {csv_path.name}")

        # Report the real header once: this is the schema question the whole
        # gate hinges on, and it should be in the record, not inferred.
        with csv_path.open() as fh:
            header = next(csv.reader(fh), [])
        print(f"columns ({len(header)}): {','.join(header)}")

        dispatches = parse_kernel_dispatches(csv_path)
        print(f"parsed dispatches: {len(dispatches)}")
        if not dispatches:
            print(f"FAIL: parser returned nothing from a non-empty trace\n{log}")
            failures.append(f"{name}: parsed 0 dispatches")
            continue

        top = sorted(dispatches, key=lambda d: -d.duration_ns)[:3]
        for d in top:
            print(f"  {d.duration_ns:>12} ns  {d.kernel_name[:70]}")

        names = candidate_kernel_names(source)
        print(f"candidate kernels: {sorted(names)}")
        report = kernel_coverage(dispatches, names)
        if report is None:
            print("FAIL: coverage unknowable from a good trace")
            failures.append(f"{name}: coverage None")
            continue

        ceiling = coverage_ceiling(report.coverage)
        print(f"coverage: {report.coverage:.6f}  "
              f"({report.n_candidate_dispatches}/{report.n_dispatches} dispatches, "
              f"{report.candidate_ns}/{report.total_ns} ns)")
        print(f"amdahl ceiling: {ceiling:.4f}x  never_ran={report.never_ran}")

        results[name] = {
            "coverage": report.coverage,
            "candidate_ns": report.candidate_ns,
            "total_ns": report.total_ns,
            "n_candidate": report.n_candidate_dispatches,
            "n_total": report.n_dispatches,
            "never_ran": report.never_ran,
            "ceiling": ceiling,
            "columns": header,
        }

        # Ground-truth checks. These are properties of the workload, decided
        # before the profiler ran, not thresholds fitted to what it reported.
        if name == "dominant":
            if report.coverage < 0.80:
                failures.append(
                    f"dominant: coverage {report.coverage:.4f} < 0.80 -- the "
                    "candidate does nearly all the GPU work here")
            if report.never_ran:
                failures.append("dominant: never_ran on a kernel that ran 20x")
        elif name == "minor":
            if not (0.0 < report.coverage < 0.50):
                failures.append(
                    f"minor: coverage {report.coverage:.4f} outside (0, 0.5) -- "
                    "a small kernel beside 20 4096^3 matmuls")
        elif name == "decoy":
            if report.coverage != 0.0 or not report.never_ran:
                failures.append(
                    f"decoy: coverage {report.coverage:.6f} never_ran="
                    f"{report.never_ran} -- a kernel that never launched must "
                    "read as exactly zero, or reward hacking goes undetected")
            if report.n_dispatches == 0:
                failures.append(
                    "decoy: trace was empty, so zero coverage proves nothing")

    out = workdir / "coverage_validation.json"
    out.write_text(json.dumps(
        {"results": results, "failures": failures}, indent=2))
    print(f"\n# wrote {out}")

    print("\n" + "=" * 60)
    if failures:
        print("FAILED -- coverage reward stays disarmed")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASSED -- rocprofv3 kernel-trace path validated on gfx950")
    print("  dominant/minor/decoy all behaved as predicted from the workload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
