#!/usr/bin/env python
"""Why the HIP GEMM task is not timing-admissible, measured rather than argued.

``hip_gemm`` compiles and verifies at 85-130 dB, but it is not shipped: the
hipBLASLt baseline's OWN coefficient of variation lands above the 3% publication
gate.  The recorded explanation is a CANDIDATE/BASELINE ASYMMETRY -- the seed is
10-16x slower than hipBLASLt, so every ~30 us baseline median is measured
immediately after milliseconds of sustained candidate work.

That is a hypothesis, and it makes a prediction this script tests directly:

    if the asymmetry causes it, hipBLASLt timed ALONE at the same shape must be
    stable, and only the INTERLEAVED measurement should be noisy.

So each shape is measured two ways, in one process, back to back:

  solo        the baseline alone, ``repeat`` medians, nothing else on the GPU
  interleaved the real paired protocol -- fresh inputs per pair, balanced AB/BA
              order, L2 flush per iteration -- exactly as kore.tasks._genops
              runs it for a graded episode

Both are then judged by the SAME admission function the environment uses
(:func:`kore.reward.stats.publication_admission_error`) at the SAME publication
thresholds.  Nothing here weakens a gate: the point is to locate the noise, and
a shape only "passes" if it clears the unmodified 3% bar.

    PYTHONPATH=. python scripts/probe_hip_gemm_timing.py --gpu 7 --trials 3
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
from typing import Any, Optional

#: The publication gates, verbatim from kore.config.KoreConfig.
PUBLICATION_CV_PCT = 3.0
MIN_PAIRS = 2

#: Shape lanes to sweep.  ``asymmetry`` is what the hypothesis is about, so the
#: sweep deliberately spans it: shallow-K lanes keep the output size while
#: shrinking the candidate's work, and the large-K lanes make it worse.
SHAPES: tuple[dict, ...] = (
    {"M": 2048, "N": 8192, "K": 512},     # the current `minimal` lane
    {"M": 4096, "N": 4096, "K": 1024},    # the current `primary` lane
    {"M": 4096, "N": 4096, "K": 512},
    {"M": 1024, "N": 5120, "K": 1792},
    {"M": 1023, "N": 4097, "K": 511},
    {"M": 2048, "N": 2048, "K": 2048},    # square: the known-bad case
    {"M": 8192, "N": 8192, "K": 128},     # very shallow K, large output
    {"M": 512, "N": 512, "K": 512},       # small enough to be launch-bound
    {"M": 16384, "N": 16384, "K": 64},    # baseline pushed into the ms range
)


def _shape_id(shape: dict) -> str:
    return f"M{shape['M']}xN{shape['N']}xK{shape['K']}"


def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default=None, help="physical GPU index")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--trials", type=int, default=3,
                        help="independent repetitions of the whole sweep")
    parser.add_argument("--repeat", type=int, default=5,
                        help="paired repeats per trial (KoreConfig.max_variance_runs)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--json", default="", help="write the report here")
    args = parser.parse_args(argv)

    if args.gpu is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch

    from kore.env.hip_toolchain import compile_hip_source
    from kore.reward.stats import (
        cv_pct,
        paired_timing_stats,
        publication_admission_error,
    )
    from kore.tasks._genops import _clone_inputs, _time_median
    from kore.tasks.hip_ops import GPU_TARGET, HIP_OPS

    spec = HIP_OPS["gemm"]
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                   "fp32": torch.float32}[args.dtype]

    # Build the seed exactly as the environment would stage it.
    path = "/tmp/kore_gemm_probe.hip"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(spec.seed)
    module = compile_hip_source(path, gpu_target=GPU_TARGET)
    candidate = module.forward
    baseline = spec.baseline

    print(f"probing hip_gemm ({args.dtype}) on {GPU_TARGET}: {len(SHAPES)} shapes "
          f"x {args.trials} trials, {args.repeat} paired repeats, "
          f"publication gate {PUBLICATION_CV_PCT}%")

    rows: list[dict[str, Any]] = []
    for shape in SHAPES:
        for trial in range(args.trials):
            inputs = spec.make_inputs(shape, torch_dtype, "cuda", 0)

            # --- solo: the vendor baseline with nothing else running --------- #
            solo = [
                _time_median(lambda: baseline(*inputs), args.warmup, args.iters)
                for _ in range(args.repeat)
            ]

            # --- interleaved: the graded paired protocol --------------------- #
            cand_ms: list[float] = []
            base_ms: list[float] = []
            candidate_first = bool(random.getrandbits(1))
            for pair in range(args.repeat):
                fresh = spec.make_inputs(shape, torch_dtype, "cuda", 0)
                ci, bi = _clone_inputs(fresh), _clone_inputs(fresh)
                w = random.randint(max(4, args.warmup - 3), args.warmup + 4)
                it = random.randint(max(8, args.iters - 5), args.iters + 6)
                first = candidate_first if pair % 2 == 0 else not candidate_first
                if first:
                    cm = _time_median(lambda: candidate(*ci), w, it)
                    bm = _time_median(lambda: baseline(*bi), w, it)
                else:
                    bm = _time_median(lambda: baseline(*bi), w, it)
                    cm = _time_median(lambda: candidate(*ci), w, it)
                cand_ms.append(cm)
                base_ms.append(bm)

            stats = paired_timing_stats(cand_ms, base_ms)
            error = publication_admission_error(
                stats,
                min_pairs=MIN_PAIRS,
                candidate_cv_threshold_pct=PUBLICATION_CV_PCT,
                baseline_cv_threshold_pct=PUBLICATION_CV_PCT,
                paired_ratio_cv_threshold_pct=PUBLICATION_CV_PCT,
                paired_ci_threshold_pct=PUBLICATION_CV_PCT,
            )
            row = {
                "shape": _shape_id(shape),
                "trial": trial,
                "asymmetry": statistics.fmean(cand_ms) / statistics.fmean(base_ms),
                "baseline_ms": statistics.fmean(base_ms),
                "candidate_ms": statistics.fmean(cand_ms),
                "solo_baseline_cv_pct": cv_pct(solo),
                "baseline_cv_pct": stats["baseline_cv_pct"],
                "candidate_cv_pct": stats["candidate_cv_pct"],
                "paired_ratio_cv_pct": stats["paired_ratio_cv_pct"],
                "ci_half_width_pct": stats["ci_half_width_pct"],
                "admitted": error is None,
                "admission_error": error or "",
            }
            rows.append(row)
            print(f"  {row['shape']:22s} t{trial} "
                  f"asym={row['asymmetry']:6.1f}x "
                  f"base={row['baseline_ms']*1000:8.1f}us "
                  f"solo_cv={row['solo_baseline_cv_pct']:5.2f}% "
                  f"base_cv={row['baseline_cv_pct']:5.2f}% "
                  f"cand_cv={row['candidate_cv_pct']:5.2f}% "
                  f"ratio_cv={row['paired_ratio_cv_pct']:5.2f}% "
                  + ("ADMITTED" if row["admitted"] else f"rejected: {error}"))

    print("\nper-shape summary (a shape is only usable if it is admitted EVERY trial):")
    usable: list[str] = []
    for shape in SHAPES:
        sid = _shape_id(shape)
        mine = [r for r in rows if r["shape"] == sid]
        admitted = sum(r["admitted"] for r in mine)
        solo = statistics.fmean(r["solo_baseline_cv_pct"] for r in mine)
        inter = statistics.fmean(r["baseline_cv_pct"] for r in mine)
        print(f"  {sid:22s} admitted {admitted}/{len(mine)}  "
              f"baseline CV solo={solo:5.2f}% interleaved={inter:5.2f}%")
        if admitted == len(mine):
            usable.append(sid)
    print(f"\nshapes admitted in every trial: {usable or 'NONE'}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({
                "dtype": args.dtype,
                "gate_pct": PUBLICATION_CV_PCT,
                "repeat": args.repeat,
                "trials": args.trials,
                "rows": rows,
                "usable_shapes": usable,
            }, fh, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
