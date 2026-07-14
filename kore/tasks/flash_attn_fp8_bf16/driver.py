"""Driver for fp8 causal (GQA) flash attention (fp8 QKV -> bf16 out).

Correctness (no --bench-mode): candidate ``flash_attn(q, k, v, sq, sk, sv, causal)``
vs exact fp32 SDPA on the dequantized fp8 q/k/v; prints SNR / allclose / max_diff.
Bench (--bench-mode):
    --impl reference  -> AITER bf16 ``flash_attn_func`` on the dequantized-to-bf16
                         q/k/v (the FMHA bar the fp8 kernel beats on QKV bandwidth)
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
from kore.tasks.aiter_ref_attn import aiter_flash_attn  # noqa: E402

CAUSAL = True


def _load_candidate():
    if getattr(_load_candidate, "_mod", None) is not None:
        mod = _load_candidate._mod
    else:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel.py")
        spec = importlib.util.spec_from_file_location("candidate_kernel", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _load_candidate._mod = mod
    return mod.flash_attn


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
        q, k, v, sq, sk, sv = ref.get_inputs(shape, device=dev, seed=s)
        r = ref.attn_ref(q, k, v, sq, sk, sv, causal=CAUSAL)
        try:
            o = fn(q, k, v, sq, sk, sv, causal=CAUSAL)
        except Exception as e:  # noqa: BLE001
            print("SNR: -999.00 dB"); print("allclose: False"); print("max_diff: inf")
            print(f"CANDIDATE_ERROR: {type(e).__name__}: {e}")
            return 0
        torch.cuda.synchronize()
        worst = min(worst, snr_db(o, r))
        maxd = max(maxd, (o.float() - r.float()).abs().max().item())
        ok = ok and torch.allclose(o.float(), r.float(), atol=3e-2, rtol=3e-2)
    print(f"SNR: {worst:.2f} dB"); print(f"allclose: {ok}"); print(f"max_diff: {maxd:.6f}")
    return 0


def run_bench(shape, impl, warmup, iters) -> int:
    dev = "cuda"
    q, k, v, sq, sk, sv = ref.get_inputs(shape, device=dev, seed=0)
    if impl == "reference":
        qb = (q.float() * float(sq)).to(torch.bfloat16)
        kb = (k.float() * float(sk)).to(torch.bfloat16)
        vb = (v.float() * float(sv)).to(torch.bfloat16)
        fn = lambda: aiter_flash_attn(qb, kb, vb, causal=CAUSAL)
    elif impl == "torch":
        fn = lambda: ref.attn_ref(q, k, v, sq, sk, sv, causal=CAUSAL)
    else:
        cand = _load_candidate()
        fn = lambda: cand(q, k, v, sq, sk, sv, causal=CAUSAL)
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
