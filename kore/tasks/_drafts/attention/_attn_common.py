"""Shared driver contract + fp32 attention oracle for the DRAFT attention tasks.

STAGING NOTE (safety): this module and every task under
``kore/tasks/_drafts/attention/`` are STAGED, not live. The registry discovers
tasks via ``kore/tasks/*/task.yaml`` (ONE directory level, see
``registry._discover``), so a task nested at
``kore/tasks/_drafts/attention/<id>/task.yaml`` is THREE levels deep and is NOT
auto-discovered -- no run/campaign can pick these up until a human promotes them.

Promotion (per task, after on-gfx950 verification): move
``kore/tasks/_drafts/attention/<id>/`` to ``kore/tasks/<id>/`` AND copy this file
to ``kore/tasks/_attn_common.py`` (each task's thin ``driver.py`` imports
``_attn_common`` from its parent directory, mirroring how the live attention tasks
share ``kore.tasks.aiter_ref_attn``). See VERIFICATION_CHECKLIST.md.

This centralizes the two things that must be correct in ONE place:
  * :func:`sdpa_fp32` -- the exact fp32 attention oracle (softmax over q.k^T with an
    optional additive mask and an optional gpt-oss-style per-head SINK logit). This
    IS the correctness ground truth for every drafted task; each ``reference.py``
    only builds the appropriate mask and calls it.
  * :func:`driver_main` -- the KernelForge verifier contract (multi-trial reseeded
    correctness printing ``SNR`` / ``allclose`` / ``max_diff``, plus cold-cache
    CUDA-event median timing with post-timing anti-hack re-verification), identical
    in spirit to ``kore.tasks._genops.driver_main`` so we do not duplicate the
    145-line per-task driver 11 times.

reference.py contract (each drafted task implements):
    parse_shape(s) -> dict
    get_inputs(shape, device="cuda", seed=0) -> tuple            # positional entry args
    reference_output(shape, inputs) -> torch.Tensor              # fp32 oracle -> out dtype
    candidate_output(fn, shape, inputs) -> torch.Tensor          # invoke candidate entry
    baseline_output(shape, inputs) -> torch.Tensor               # REAL vendor (AITER) op
    ENTRY: str                                                   # candidate attr name
    ATOL: float, RTOL: float                                     # allclose tolerances

torch/aiter are imported lazily inside the GPU paths so importing a reference (for
the CPU oracle sanity check) never needs a GPU or the aiter runtime.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os


# --------------------------------------------------------------------------- #
# fp32 attention oracle (the correctness ground truth)
# --------------------------------------------------------------------------- #
def sdpa_fp32(q, k, v, scale, attn_mask=None, sink=None):
    """Exact fp32 scaled-dot-product attention.

    Args (all already in the [B, H, S, D] "heads-second" layout, KV heads already
    expanded to H via repeat_interleave so this is plain multi-head attention):
        q:  fp32 [B, H, Sq, D]
        k:  fp32 [B, H, Sk, D]
        v:  fp32 [B, H, Sk, D]
        scale: python float (typically 1/sqrt(D))
        attn_mask: additive fp32 mask broadcastable to [B, H, Sq, Sk] (0.0 to keep,
                   -inf to block), or None for full (bidirectional) attention.
        sink: per-head additive SINK logit fp32 [H] (gpt-oss / StreamingLLM
              learned sink). It adds an extra term ``exp(sink - m)`` to the softmax
              DENOMINATOR with NO corresponding value, i.e. it leaks probability
              mass out of the real keys (a always-available no-op attention slot).
              None disables it.

    Returns fp32 [B, H, Sq, D]. Uses the numerically-stable max-subtraction form
    (the same online-softmax the flash kernels approximate), so it is the honest
    reference the SNR gate measures a candidate against.
    """
    import torch

    scores = torch.matmul(q, k.transpose(-1, -2)) * scale        # [B,H,Sq,Sk]
    if attn_mask is not None:
        scores = scores + attn_mask
    m = scores.amax(dim=-1, keepdim=True)                        # [B,H,Sq,1]
    if sink is not None:
        s = sink.to(scores.dtype).view(1, -1, 1, 1)             # [1,H,1,1]
        m = torch.maximum(m, s)
    # A fully-masked row has m = -inf; force it finite so exp(-inf - m) -> 0 rather
    # than exp(-inf + inf) = NaN (matches the flash kernels' m_safe guard).
    m = torch.nan_to_num(m, neginf=0.0)
    p = torch.exp(scores - m)                                    # [B,H,Sq,Sk]
    denom = p.sum(dim=-1, keepdim=True)                          # [B,H,Sq,1]
    if sink is not None:
        denom = denom + torch.exp(sink.to(scores.dtype).view(1, -1, 1, 1) - m)
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    return torch.matmul(p / denom, v)                            # [B,H,Sq,D]


def expand_kv(t, H):
    """Expand a [B, KV, S, D] tensor to [B, H, S, D] by repeat_interleave (GQA/MQA).

    H must be a multiple of KV. rep = H // KV consecutive query heads share one KV
    head, matching AITER ``flash_attn_func`` GQA (and torch ``enable_gqa``)."""
    KV = t.shape[1]
    if KV == H:
        return t
    return t.repeat_interleave(H // KV, dim=1)


def causal_mask(Sq, Sk, device, q_offset=0):
    """Additive [Sq, Sk] causal mask, bottom-right aligned via ``q_offset``.

    Query row i is at global key position ``q_offset + i`` and may attend to keys
    ``j <= q_offset + i``. With ``q_offset = Sk - Sq`` this is the standard
    flash-attention bottom-right causal alignment used when Sq < Sk (decode /
    chunked-prefill against a longer KV context). ``q_offset = 0`` with Sq == Sk is
    ordinary causal prefill."""
    import torch

    i = torch.arange(Sq, device=device)[:, None] + q_offset
    j = torch.arange(Sk, device=device)[None, :]
    allow = j <= i
    return torch.where(allow, 0.0, float("-inf")).to(torch.float32)


def sliding_window_mask(Sq, Sk, window, device, q_offset=0):
    """Additive [Sq, Sk] sliding-window causal mask.

    Query row i (global position ``q_offset + i``) attends to keys j with
    ``q_offset + i - window < j <= q_offset + i`` (a causal band of width
    ``window``)."""
    import torch

    i = torch.arange(Sq, device=device)[:, None] + q_offset
    j = torch.arange(Sk, device=device)[None, :]
    allow = (j <= i) & (j > i - window)
    return torch.where(allow, 0.0, float("-inf")).to(torch.float32)


# --------------------------------------------------------------------------- #
# Verifier driver (correctness + cold-cache bench + post-timing anti-hack)
# --------------------------------------------------------------------------- #
def _snr_db(out, ref_out) -> float:
    o, r = out.float(), ref_out.float()
    noise = (o - r).norm().item()
    signal = r.norm().item()
    if noise == 0:
        return 999.0
    return 20.0 * math.log10(signal / noise) if signal > 0 else -999.0


def _num_correct_trials() -> int:
    """KernelBench-fidelity: >=5 reseeded correctness trials (env-overridable)."""
    try:
        return max(5, int(os.environ.get("KORE_CORRECTNESS_TRIALS", "5")))
    except ValueError:
        return 5


def _bench_cold() -> bool:
    return os.environ.get("KORE_BENCH_COLD", "1") != "0"


_L2_SCRATCH = None


def _flush_l2(device: str = "cuda") -> None:
    """Evict the GPU last-level cache between timed iters (cold-cache, KernelBench)."""
    import torch

    global _L2_SCRATCH
    if _L2_SCRATCH is None:
        _L2_SCRATCH = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)
    _L2_SCRATCH.zero_()


def _time_fn(fn, warmup: int, iters: int) -> int:
    import torch

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


def _load_candidate(task_dir: str, entry: str):
    # Cache the module so a stateful kernel's globals persist from the bench timing
    # loop into the post-timing re-verification (anti invocation-count timing hack).
    if getattr(_load_candidate, "_mod", None) is None:
        path = os.path.join(task_dir, "kernel.py")
        spec = importlib.util.spec_from_file_location("candidate_kernel", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _load_candidate._mod = mod
    return getattr(_load_candidate._mod, entry)


def _run_correctness(ref, task_dir, shape) -> int:
    import torch

    fn = _load_candidate(task_dir, ref.ENTRY)
    atol = getattr(ref, "ATOL", 2e-2)
    rtol = getattr(ref, "RTOL", 2e-2)
    worst, maxd, ok = 999.0, 0.0, True
    for s in range(_num_correct_trials()):
        inputs = ref.get_inputs(shape, device="cuda", seed=s)
        r = ref.reference_output(shape, inputs)
        try:
            o = ref.candidate_output(fn, shape, inputs)
        except Exception as e:  # noqa: BLE001
            print("SNR: -999.00 dB"); print("allclose: False"); print("max_diff: inf")
            print(f"CANDIDATE_ERROR: {type(e).__name__}: {e}")
            return 0
        torch.cuda.synchronize()
        worst = min(worst, _snr_db(o, r))
        maxd = max(maxd, (o.float() - r.float()).abs().max().item())
        ok = ok and torch.allclose(o.float(), r.float(), atol=atol, rtol=rtol)
    print(f"SNR: {worst:.2f} dB"); print(f"allclose: {ok}"); print(f"max_diff: {maxd:.6f}")
    return 0


def _run_bench(ref, task_dir, shape, impl, warmup, iters) -> int:
    inputs = ref.get_inputs(shape, device="cuda", seed=0)
    if impl == "reference":
        fn = lambda: ref.baseline_output(shape, inputs)          # REAL vendor op
    elif impl == "torch":
        fn = lambda: ref.reference_output(shape, inputs)         # fp32 oracle
    else:
        cand = _load_candidate(task_dir, ref.ENTRY)
        fn = lambda: ref.candidate_output(cand, shape, inputs)
    return _time_fn(fn, warmup, iters)


def driver_main(ref, task_dir: str, argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shape", default="default")
    p.add_argument("--mode", default=None)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--bench-mode", action="store_true")
    p.add_argument("--impl", default="candidate", choices=["candidate", "reference", "torch"])
    a = p.parse_args(argv)
    shape = ref.parse_shape(a.shape)
    if a.bench_mode:
        rc = _run_bench(ref, task_dir, shape, a.impl, a.warmup, a.iters)
        # Post-timing anti-hack correctness re-verification on the cached candidate.
        if a.impl == "candidate":
            _run_correctness(ref, task_dir, shape)
        return rc
    return _run_correctness(ref, task_dir, shape)
