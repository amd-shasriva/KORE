"""Driver for the bf16 fused-MoE (top-k grouped GEMM + SiLU-mul) task.

Correctness (no --bench-mode): candidate
``fused_moe(hidden_states, w1, w2, topk_weight, topk_ids)`` vs exact fp32 top-k
fused-MoE oracle; prints SNR / allclose / max_diff. bf16 gate ~25 dB.
Bench (--bench-mode):
    --impl reference  -> AITER ``fused_moe`` (CK 2-stage SiLU). Weights are
                         pre-shuffled ONCE outside the timed region (load-time
                         cost in production), so only the GEMM+act+reduce is timed.
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
from kore.tasks.aiter_ref_attn import aiter_fused_moe, shuffle_moe_weights  # noqa: E402


def _load_candidate():
    # Cache the loaded candidate MODULE for the life of the process so a stateful
    # kernel's module globals (e.g. an invocation counter) PERSIST across the bench
    # timing loop and the post-timing correctness re-verification. Without this,
    # run_correctness would reload the module and reset the counter, letting a
    # "correct-for-first-N-calls, garbage-after" kernel evade the anti-hack check.
    if getattr(_load_candidate, "_mod", None) is not None:
        mod = _load_candidate._mod
    else:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel.py")
        spec = importlib.util.spec_from_file_location("candidate_kernel", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _load_candidate._mod = mod
    return mod.fused_moe


def snr_db(out, ref_out) -> float:
    o, r = out.float(), ref_out.float()
    noise = (o - r).norm().item()
    signal = r.norm().item()
    if noise == 0:
        return 999.0
    return 20.0 * math.log10(signal / noise) if signal > 0 else -999.0


def _num_correct_trials() -> int:
    """KernelBench-fidelity: >=5 reseeded correctness trials (env-overridable)."""
    try:
        n = int(os.environ.get("KORE_CORRECTNESS_TRIALS", "5"))
    except ValueError:
        n = 5
    return max(5, n)


def _bench_cold() -> bool:
    """Cold-cache (L2-flushed) timing by default; KORE_BENCH_COLD=0 -> warm."""
    return os.environ.get("KORE_BENCH_COLD", "1") != "0"


_L2_SCRATCH = None


def _flush_l2(device: str = "cuda") -> None:
    """Evict the GPU last-level cache (L2/Infinity) between timed iters by
    overwriting a scratch buffer larger than it, so each iter is cold-cache
    like KernelBench. Enqueued BEFORE the start event, so it is never timed."""
    global _L2_SCRATCH
    if _L2_SCRATCH is None:
        _L2_SCRATCH = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)
    _L2_SCRATCH.zero_()


def _time_fn(fn, warmup: int, iters: int) -> int:
    """Warmup + median-of-iters timing. Flushes the L2 between timed iters when
    cold-cache is enabled (default), matching KernelBench's cold measurement."""
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
    # KernelBench multi-trial: >=5 DIFFERENT input seeds; the candidate passes
    # only if EVERY trial passes both allclose and the SNR gate. We print the
    # worst-trial SNR + the AND of allclose so the env's last-match parse gets
    # the conservative worst case.
    seeds = list(range(_num_correct_trials()))
    fn = _load_candidate()
    worst, maxd, ok = 999.0, 0.0, True
    for s in seeds:
        hidden, w1, w2, tw, ti = ref.get_inputs(shape, device=dev, seed=s)
        r = ref.moe_ref(hidden, w1, w2, tw, ti)
        try:
            o = fn(hidden, w1, w2, tw, ti)
        except Exception as e:
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
    hidden, w1, w2, tw, ti = ref.get_inputs(shape, device=dev, seed=0)
    if impl == "reference":
        w1s, w2s = shuffle_moe_weights(w1, w2)   # load-time shuffle, outside timing
        fn = lambda: aiter_fused_moe(hidden, w1s, w2s, tw, ti)
    elif impl == "torch":
        fn = lambda: ref.moe_ref(hidden, w1, w2, tw, ti)
    else:
        cand = _load_candidate()
        fn = lambda: cand(hidden, w1, w2, tw, ti)
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
        # Anti-hack post-timing correctness: a stateful kernel that returns correct
        # output for the correctness calls but garbage during timing is caught here
        # (the verify runs on LATE invocations; the env invalidates the bench on a
        # False verdict).
        if a.impl == "candidate":
            run_correctness(shape, a.mode)
        return rc
    return run_correctness(shape, a.mode)


if __name__ == "__main__":
    raise SystemExit(main())
