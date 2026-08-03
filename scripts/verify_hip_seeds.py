#!/usr/bin/env python
"""Compile and verify every HIP seed on real gfx950 silicon, then report.

Run this BEFORE generating HIP task directories.  A task whose seed does not
compile, does not clear its own declared SNR gate, or cannot be timed is worse
than no task at all: it burns GPU time at datagen and reports as a model error,
which is exactly how a task-resolution bug once produced a 97% error rate while
claiming 13,277 episodes/hour.  So the GPU decides which (op, dtype) pairs become
tasks -- not this file's author.

Usage
-----
    PYTHONPATH=. python scripts/verify_hip_seeds.py [--ops gemm,silu] [--gpu 5]
        [--json out.json] [--adversarial]

Reports, per (op, dtype): compile seconds, worst SNR over the declared shapes,
whether the declared gate is cleared, and the seed's speedup against the
production baseline (expected to be BELOW 1.0 -- the seed is deliberately naive,
and a seed that already beat the vendor kernel would mean the task has no
headroom to learn).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Optional


def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops", default="", help="comma-separated op ids (default: all)")
    parser.add_argument("--dtypes", default="", help="comma-separated dtype ids (default: all)")
    parser.add_argument("--gpu", default=None, help="physical GPU index to use")
    parser.add_argument("--json", default="", help="write the full report here")
    parser.add_argument("--adversarial", action="store_true",
                        help="also run the enumerated adversarial regimes")
    parser.add_argument("--shapes", default="minimal,primary",
                        help="which declared shape lanes to check")
    parser.add_argument("--bench-iters", type=int, default=20)
    args = parser.parse_args(argv)

    if args.gpu is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from kore.env import hip_toolchain
    from kore.tasks import hip_ops

    status = hip_toolchain.probe_toolchain()
    if not status.available:
        print(f"FATAL: HIP toolchain unusable: missing {status.missing}; {status.detail}")
        return 2

    # Pin the arch before the first compile: 15.4s vs 114.6s per kernel.
    hip_toolchain.compile_environment(os.environ, hip_ops.GPU_TARGET)
    os.environ.update({
        k: v for k, v in hip_toolchain.compile_environment(
            dict(os.environ), hip_ops.GPU_TARGET).items()
        if k in ("PATH", "PYTORCH_ROCM_ARCH", "TORCH_EXTENSIONS_DIR", "MAX_JOBS")
    })

    import torch

    from kore.reward.reward import scan_for_hacks

    print(f"toolchain: hipcc={status.hipcc} ninja={status.ninja} "
          f"torch_hip={status.torch_hip_version} arch={os.environ.get('PYTORCH_ROCM_ARCH')}")
    print(f"device: {torch.cuda.get_device_properties(0).gcnArchName}")

    want_ops = [o for o in args.ops.split(",") if o] or sorted(hip_ops.HIP_OPS)
    want_dtypes = [d for d in args.dtypes.split(",") if d]
    lanes = [s for s in args.shapes.split(",") if s]

    rows: list[dict[str, Any]] = []
    for op_id in want_ops:
        spec = hip_ops.HIP_OPS[op_id]
        source = hip_ops.seed_source(op_id)
        hack = scan_for_hacks(source, "cpp")
        for dtype_id in (want_dtypes or list(spec.dtypes)):
            row: dict[str, Any] = {
                "op": op_id, "dtype": dtype_id, "gate_db": spec.snr_db,
                "hack_reason": hack, "compiled": False, "gate_cleared": False,
            }
            if hack:
                row["error"] = f"seed rejected by the hack scanner: {hack}"
                rows.append(row)
                print(f"  {op_id:14s} {dtype_id:6s} HACK-REJECTED {hack}")
                continue
            try:
                _check_one(spec, op_id, dtype_id, source, lanes, args, row)
            except Exception as exc:  # noqa: BLE001 - a failure is the datum
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            _print_row(row)

    ok = [r for r in rows if r.get("gate_cleared")]
    print(f"\n{len(ok)}/{len(rows)} (op, dtype) pairs compiled AND cleared their gate")
    failed = [r for r in rows if not r.get("gate_cleared")]
    if failed:
        print("NOT usable as tasks:")
        for r in failed:
            print(f"  {r['op']}/{r['dtype']}: {r.get('error') or 'gate not cleared'}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"rows": rows, "toolchain": status.as_dict()}, fh, indent=2)
        print(f"wrote {args.json}")
    return 0 if ok and not failed else 1


def _leaf_pairs(name: str, got, expected):
    """Flatten a single-tensor or multi-output result, checking the ABI exactly.

    A HIP entry point that returns ``std::vector<torch::Tensor>`` arrives as a
    Python list while the oracle returns a tuple, so arity and per-leaf
    dtype/shape are what must agree -- not the container type.
    """
    import torch

    got_seq = got if isinstance(got, (tuple, list)) else (got,)
    want_seq = expected if isinstance(expected, (tuple, list)) else (expected,)
    if len(got_seq) != len(want_seq):
        raise AssertionError(
            f"{name}: returned {len(got_seq)} output(s), oracle has {len(want_seq)}")
    for index, (out, want) in enumerate(zip(got_seq, want_seq)):
        if not torch.is_tensor(out):
            raise AssertionError(f"{name}: output {index} is not a tensor")
        if out.dtype != want.dtype or tuple(out.shape) != tuple(want.shape):
            raise AssertionError(
                f"{name}: output {index} ABI mismatch got "
                f"{out.dtype}/{tuple(out.shape)} want {want.dtype}/{tuple(want.shape)}")
        yield out, want


def _snr_db(out, ref) -> float:
    o, r = out.float(), ref.float()
    noise = (o - r).norm().item()
    signal = r.norm().item()
    if noise == 0.0:
        return 999.0
    return 20.0 * math.log10(signal / noise) if signal > 0 else -999.0


def _time_median(fn, warmup: int, iters: int) -> float:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    scratch = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
    for i in range(iters):
        scratch.zero_()          # cold-cache, enqueued before the start event
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times[len(times) // 2]


def _check_one(spec, op_id: str, dtype_id: str, source: str, lanes, args, row) -> None:
    import tempfile

    import torch

    from kore.env import hip_toolchain
    from kore.tasks import hip_ops

    workdir = tempfile.mkdtemp(prefix=f"hipseed_{op_id}_{dtype_id}_")
    path = os.path.join(workdir, "kernel.hip")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(source)

    t0 = time.time()
    fn = hip_toolchain.load_hip_candidate(workdir, "forward", gpu_target=hip_ops.GPU_TARGET)
    row["compile_s"] = round(time.time() - t0, 1)
    row["compiled"] = True

    ns = hip_ops.make_reference(op_id, dtype_id)
    get_inputs, ref_fn, baseline_fn = ns["get_inputs"], ns["ref_fn"], ns["baseline_fn"]

    shape_specs: list[tuple[str, dict]] = []
    for lane in lanes:
        value = spec.shapes.get(lane)
        if isinstance(value, dict):
            shape_specs.append((lane, value))
        elif isinstance(value, list):
            shape_specs.extend((f"{lane}_{i}", v) for i, v in enumerate(value))

    worst = 999.0
    per_shape: dict[str, float] = {}
    for name, dims in shape_specs:
        for seed in range(3):
            inputs = get_inputs(dims, device="cuda", seed=seed)
            expected = ref_fn(*inputs)
            got = fn(*inputs)
            torch.cuda.synchronize()
            for out, want in _leaf_pairs(name, got, expected):
                snr = _snr_db(out, want)
                worst = min(worst, snr)
                per_shape[name] = min(per_shape.get(name, 999.0), snr)
    row["snr_by_shape"] = {k: round(v, 2) for k, v in per_shape.items()}
    row["worst_snr_db"] = round(worst, 2)

    if args.adversarial:
        from kore.tasks._genops import _adversarial_fills

        adv_worst = 999.0
        inputs = get_inputs(spec.shapes["minimal"], device="cuda", seed=0)
        for name, adv in _adversarial_fills(inputs):
            expected = ref_fn(*adv)
            got = fn(*adv)
            torch.cuda.synchronize()
            finite = torch.isfinite(expected.float())
            snr = _snr_db(got.float()[finite], expected.float()[finite]) \
                if finite.any() else 999.0
            structure_ok = bool(
                torch.equal(torch.isnan(got.float()), torch.isnan(expected.float()))
                and torch.equal(torch.isinf(got.float()), torch.isinf(expected.float())))
            if not structure_ok:
                raise AssertionError(f"adversarial[{name}]: non-finite structure differs")
            adv_worst = min(adv_worst, snr)
        row["adversarial_worst_snr_db"] = round(adv_worst, 2)
        worst = min(worst, adv_worst)

    row["gate_cleared"] = worst >= spec.snr_db

    dims = spec.shapes["primary"]
    inputs = get_inputs(dims, device="cuda", seed=0)
    cand_ms = _time_median(lambda: fn(*inputs), 5, args.bench_iters)
    base_ms = _time_median(lambda: baseline_fn(*inputs), 5, args.bench_iters)
    row["candidate_ms"] = round(cand_ms, 4)
    row["baseline_ms"] = round(base_ms, 4)
    row["seed_speedup"] = round(base_ms / cand_ms, 3) if cand_ms > 0 else None


def _print_row(row: dict) -> None:
    if row.get("error"):
        print(f"  {row['op']:14s} {row['dtype']:6s} FAIL  {row['error'][:150]}")
        return
    gate = "PASS" if row["gate_cleared"] else "GATE-MISS"
    adv = row.get("adversarial_worst_snr_db")
    print(f"  {row['op']:14s} {row['dtype']:6s} {gate:9s} "
          f"snr={row['worst_snr_db']:>8}dB (gate {row['gate_db']}) "
          + (f"adv={adv}dB " if adv is not None else "")
          + f"compile={row['compile_s']}s "
          f"seed_speedup={row.get('seed_speedup')}x")


if __name__ == "__main__":
    raise SystemExit(_main())
