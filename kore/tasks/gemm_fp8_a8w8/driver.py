"""Driver for the fp8 (a8w8) GEMM task (KernelForge verifier contract).

Correctness (no --bench-mode): candidate ``gemm_fp8(xq, wq, x_scale, w_scale)``
vs exact torch-fp32 matmul of the dequantized fp8 inputs; prints SNR/allclose/
max_diff. Gate is 25 dB (fp8).
Bench (--bench-mode):
    --impl reference  -> AITER ``gemm_a8w8`` (the real fp8 serving bar)
    --impl candidate  -> kernel.py
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference as ref  # noqa: E402
from kore.tasks.aiter_ref import aiter_gemm_a8w8  # noqa: E402


def _load_candidate():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel.py")
    spec = importlib.util.spec_from_file_location("candidate_kernel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.gemm_fp8


def snr_db(out, ref_out) -> float:
    o, r = out.float(), ref_out.float()
    noise = (o - r).norm().item()
    signal = r.norm().item()
    if noise == 0:
        return 999.0
    return 20.0 * math.log10(signal / noise) if signal > 0 else -999.0


def run_correctness(shape, mode) -> int:
    dev = "cuda"
    seeds = [0] if mode not in ("stability", "determinism") else [0, 1, 2, 3, 4]
    fn = _load_candidate()
    worst, maxd, ok = 999.0, 0.0, True
    for s in seeds:
        xq, wq, xs, ws = ref.get_inputs(shape, device=dev, seed=s)
        r = ref.matmul_ref(xq, wq, xs, ws)
        try:
            o = fn(xq, wq, xs, ws)
        except Exception as e:
            print("SNR: -999.00 dB"); print("allclose: False"); print("max_diff: inf")
            print(f"CANDIDATE_ERROR: {type(e).__name__}: {e}")
            return 0
        torch.cuda.synchronize()
        worst = min(worst, snr_db(o, r))
        maxd = max(maxd, (o.float() - r.float()).abs().max().item())
        # fp8 GEMM: absolute tolerance scaled by magnitude; SNR is the real gate.
        ok = ok and torch.allclose(o.float(), r.float(), atol=5e-1, rtol=5e-2)
    print(f"SNR: {worst:.2f} dB"); print(f"allclose: {ok}"); print(f"max_diff: {maxd:.6f}")
    return 0


def run_bench(shape, impl, warmup, iters) -> int:
    dev = "cuda"
    xq, wq, xs, ws = ref.get_inputs(shape, device=dev, seed=0)
    if impl == "reference":
        fn = lambda: aiter_gemm_a8w8(xq, wq, xs, ws)
    else:
        cand = _load_candidate()
        fn = lambda: cand(xq, wq, xs, ws)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        st[i].record(); fn(); en[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(st, en))
    for t in times:
        print(f"wall_ms: {t:.4f}")
    print(f"median_ms: {times[len(times) // 2]:.4f}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shape", default="default")
    p.add_argument("--mode", default=None)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--bench-mode", action="store_true")
    p.add_argument("--impl", default="candidate", choices=["candidate", "reference"])
    a = p.parse_args()
    shape = ref.parse_shape(a.shape)
    return run_bench(shape, a.impl, a.warmup, a.iters) if a.bench_mode else run_correctness(shape, a.mode)


if __name__ == "__main__":
    raise SystemExit(main())
