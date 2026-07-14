"""Driver for bf16 variable-length causal (GQA) flash attention.

Correctness (no --bench-mode): candidate ``flash_attn(q, k, v, cu_seqlens,
max_seqlen, causal)`` vs per-sequence exact fp32 causal SDPA; SNR/allclose/max_diff.
Bench (--bench-mode):
    --impl reference  -> pad the ragged batch to [B,max_s,H,D] + AITER dense
                         ``flash_attn_func`` (the padded-FMHA bar varlen beats by
                         not wasting compute on padding)
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
        q, k, v, cu, ms = ref.get_inputs(shape, device=dev, seed=s)
        r = ref.attn_ref(q, k, v, cu, ms, causal=CAUSAL)
        try:
            o = fn(q, k, v, cu, ms, causal=CAUSAL)
        except Exception as e:  # noqa: BLE001
            print("SNR: -999.00 dB"); print("allclose: False"); print("max_diff: inf")
            print(f"CANDIDATE_ERROR: {type(e).__name__}: {e}")
            return 0
        torch.cuda.synchronize()
        worst = min(worst, snr_db(o, r))
        maxd = max(maxd, (o.float() - r.float()).abs().max().item())
        ok = ok and torch.allclose(o.float(), r.float(), atol=2e-2, rtol=2e-2)
    print(f"SNR: {worst:.2f} dB"); print(f"allclose: {ok}"); print(f"max_diff: {maxd:.6f}")
    return 0


def _pad_batch(q, k, v, cu, ms):
    H, D = q.shape[1], q.shape[2]
    KV = k.shape[1]
    cul = cu.tolist()
    B = len(cul) - 1
    qp = torch.zeros((B, ms, H, D), device=q.device, dtype=q.dtype)
    kp = torch.zeros((B, ms, KV, D), device=q.device, dtype=q.dtype)
    vp = torch.zeros((B, ms, KV, D), device=q.device, dtype=q.dtype)
    for b in range(B):
        s, e = cul[b], cul[b + 1]
        L = e - s
        qp[b, :L] = q[s:e]; kp[b, :L] = k[s:e]; vp[b, :L] = v[s:e]
    return qp, kp, vp


def run_bench(shape, impl, warmup, iters) -> int:
    dev = "cuda"
    q, k, v, cu, ms = ref.get_inputs(shape, device=dev, seed=0)
    if impl == "reference":
        qp, kp, vp = _pad_batch(q, k, v, cu, ms)
        fn = lambda: aiter_flash_attn(qp, kp, vp, causal=CAUSAL)   # padded dense FMHA bar
    elif impl == "torch":
        fn = lambda: ref.attn_ref(q, k, v, cu, ms, causal=CAUSAL)
    else:
        cand = _load_candidate()
        fn = lambda: cand(q, k, v, cu, ms, causal=CAUSAL)
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
