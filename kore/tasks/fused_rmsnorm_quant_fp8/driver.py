"""Driver for FUSED RMSNorm -> per-token fp8 quant (KernelForge verifier contract).

Correctness (no --bench-mode): candidate ``quant(x, w) -> (xq, scale)``.
  * SNR  = SNR of the *dequantized* output ``xq*scale`` vs the TRUE fp32 RMSNorm
           activation (the real fidelity gate; fp8 e4m3 ~ low-20s dB).
  * allclose = candidate scales match the oracle scales (LOOSE rtol: the fused
           kernel recomputes the normed value, so its rowamax legitimately differs
           from the fp32 oracle by a hair) AND the candidate dequant matches the
           ORACLE dequant to fp8 rounding (>=35 dB) - so a kernel cannot pass by
           emitting a bogus scale or a wrong RMSNorm.
Bench (--bench-mode):
    --impl reference  -> AITER rms_norm + dynamic_per_token quant (the 2-kernel bar)
    --impl candidate  -> the kernel written to kernel.py
Prints ``wall_ms: X`` per iter + ``median_ms: X``.
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
from kore.tasks.aiter_ref import aiter_dynamic_per_token_quant, aiter_rms_norm  # noqa: E402


def _load_candidate():
    if getattr(_load_candidate, "_mod", None) is not None:
        mod = _load_candidate._mod
    else:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel.py")
        spec = importlib.util.spec_from_file_location("candidate_kernel", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _load_candidate._mod = mod
    return mod.quant


def snr_db(out, ref_out) -> float:
    o, r = out.float(), ref_out.float()
    noise = (o - r).norm().item()
    signal = r.norm().item()
    if noise == 0:
        return 999.0
    return 20.0 * math.log10(signal / noise) if signal > 0 else -999.0


def _num_correct_trials() -> int:
    try:
        n = int(os.environ.get("KORE_CORRECTNESS_TRIALS", "5"))
    except ValueError:
        n = 5
    return max(5, n)


def _bench_cold() -> bool:
    return os.environ.get("KORE_BENCH_COLD", "1") != "0"


_L2_SCRATCH = None


def _flush_l2(device: str = "cuda") -> None:
    global _L2_SCRATCH
    if _L2_SCRATCH is None:
        _L2_SCRATCH = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)
    _L2_SCRATCH.zero_()


def _time_fn(fn, warmup: int, iters: int) -> int:
    cold = _bench_cold()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if cold:
            _flush_l2()
        st[i].record(); fn(); en[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(st, en))
    for t in times:
        print(f"wall_ms: {t:.4f}")
    print(f"median_ms: {times[len(times) // 2]:.4f}")
    return 0


def run_correctness(shape, mode) -> int:
    dev = "cuda"
    fn = _load_candidate()
    worst, maxd, ok = 999.0, 0.0, True
    for s in range(_num_correct_trials()):
        (x, w) = ref.get_inputs(shape, device=dev, seed=s)
        y_true = ref.rmsnorm_ref(x, w)
        ref_xq, ref_scale = ref.fused_ref(x, w)
        ref_deq = ref.dequant(ref_xq, ref_scale)
        try:
            xq, scale = fn(x, w)
        except Exception as e:  # noqa: BLE001
            print("SNR: -999.00 dB"); print("allclose: False"); print("max_diff: inf")
            print(f"CANDIDATE_ERROR: {type(e).__name__}: {e}")
            return 0
        torch.cuda.synchronize()
        cand_deq = ref.dequant(xq, scale)
        worst = min(worst, snr_db(cand_deq, y_true))
        maxd = max(maxd, (cand_deq - y_true).abs().max().item())
        scale_ok = torch.allclose(scale.float(), ref_scale.float(), atol=1e-6, rtol=5e-2)
        code_ok = snr_db(cand_deq, ref_deq) >= 35.0
        ok = ok and scale_ok and code_ok
    print(f"SNR: {worst:.2f} dB"); print(f"allclose: {ok}"); print(f"max_diff: {maxd:.6f}")
    return 0


def run_bench(shape, impl, warmup, iters) -> int:
    dev = "cuda"
    (x, w) = ref.get_inputs(shape, device=dev, seed=0)
    if impl == "reference":
        def fn():
            y = aiter_rms_norm(x, w, ref.EPS)
            return aiter_dynamic_per_token_quant(y)
    elif impl == "torch":
        fn = lambda: ref.fused_ref(x, w)
    else:
        cand = _load_candidate()
        fn = lambda: cand(x, w)
    return _time_fn(fn, warmup, iters)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shape", default="default")
    p.add_argument("--mode", default=None)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--bench-mode", action="store_true")
    p.add_argument("--impl", default="candidate", choices=["candidate", "reference", "torch"])
    a = p.parse_args()
    shape = ref.parse_shape(a.shape)
    if a.bench_mode:
        rc = run_bench(shape, a.impl, a.warmup, a.iters)
        if a.impl == "candidate":
            run_correctness(shape, a.mode)
        return rc
    return run_correctness(shape, a.mode)


if __name__ == "__main__":
    raise SystemExit(main())
