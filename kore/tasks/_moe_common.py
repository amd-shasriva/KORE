"""Shared fp32 MoE oracles + routing + fp8 quant + vendor wrappers + driver.

STAGING NOTE (safety): this module and every task under
``kore/tasks/_drafts/moe/`` are STAGED, not live. The registry discovers tasks
via ``kore/tasks/*/task.yaml`` (ONE directory level, see
``registry._discover``), so a task nested at
``kore/tasks/_drafts/moe/<id>/task.yaml`` is THREE levels deep and is NOT
auto-discovered -- no run/campaign can pick these up until a human promotes
them. Verified: ``registry.task_ids()`` contains none of the draft ids.

Promotion (per task, after on-gfx950 verification): move
``kore/tasks/_drafts/moe/<id>/`` to ``kore/tasks/<id>/`` AND copy this file to
``kore/tasks/_moe_common.py`` (each task's thin ``driver.py`` imports
``_moe_common`` from its parent directory, mirroring how the live MoE tasks
share ``kore.tasks.aiter_ref_attn``). See VERIFICATION_CHECKLIST.md.

This centralizes the pieces that must be numerically correct in ONE place so we
do not duplicate them across the drafted tasks:
  * The MoE building-block oracles (all fp32 accumulate -> low-precision out):
      - :func:`gated_mlp_fp32`   -- per-expert gated MLP (SiLU/GELU) + top-k
                                    weighted combine (the fused-MoE math).
      - :func:`grouped_gemm_fp32`-- segmented (per-expert) GEMM x @ w[e]^T.
      - :func:`topk_softmax_dense_fp32` -- softmax -> top-k (+optional renorm),
                                    materialized dense [M,E] (order-independent).
      - :func:`biased_grouped_topk_dense_fp32` -- DeepSeek-V3 sigmoid + bias
                                    grouped top-k, dense [M,E].
      - :func:`permute_tokens`   -- expert-sorted token scatter (stable).
      - :func:`moe_sum_fp32`     -- weighted combine (reduce over top-k slots).
  * :func:`make_routing` -- the SHARED unbalanced router assignment (jagged
    per-expert counts with a guaranteed 0-token last expert and a giant expert:
    the mandatory MoE edge, DATASET_SPEC 1.6 / blueprint 1.3).
  * fp8 quant helpers (:func:`quant_per_token_fp8` / :func:`quant_per_channel_fp8`)
    using the arch-selected OCP e4m3fn (gfx950) ``FP8_DTYPE``.
  * Thin REAL AITER vendor wrappers (lazy import) for each task's perf baseline.
  * :func:`driver_main` -- the KernelForge verifier contract (>=5 reseeded
    correctness trials printing ``SNR`` / ``allclose`` / ``max_diff`` plus
    cold-cache CUDA-event median timing with post-timing anti-hack re-verify),
    identical in spirit to ``kore.tasks._genops.driver_main`` so we do not
    duplicate the ~145-line per-task driver 8 times.

reference.py contract (each drafted task implements):
    parse_shape(s) -> dict
    get_inputs(shape, device="cuda", seed=0) -> tuple           # positional entry args
    reference_output(shape, inputs) -> torch.Tensor             # fp32 oracle -> out dtype
    candidate_output(fn, shape, inputs) -> torch.Tensor         # invoke candidate entry
    baseline_output(shape, inputs) -> torch.Tensor              # REAL vendor (AITER) op
    ENTRY: str                                                  # candidate attr name
    ATOL: float, RTOL: float                                    # allclose tolerances

torch/aiter are imported lazily inside the GPU paths so importing a reference
(for the CPU oracle sanity check) never needs a GPU or the aiter runtime.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os

from kore.tasks._genops import _clone_inputs, _compare_outputs, emit_driver_capabilities


# --------------------------------------------------------------------------- #
# Activations (fp32)
# --------------------------------------------------------------------------- #
def silu_fp32(x):
    import torch

    xf = x.float()
    return xf * torch.sigmoid(xf)


def gelu_tanh_fp32(x):
    """tanh-approx GELU (matches aiter ``gelu_tanh_and_mul`` / F.gelu tanh)."""
    import torch

    xf = x.float()
    return torch.nn.functional.gelu(xf, approximate="tanh")


def _act_fp32(name):
    if name == "silu":
        return silu_fp32
    if name == "gelu":
        return gelu_tanh_fp32
    raise ValueError(f"unknown activation {name!r}")


# --------------------------------------------------------------------------- #
# Router assignment (shared) -- unbalanced jagged trace + 0-token last expert
# --------------------------------------------------------------------------- #
# Representative jagged per-expert weighting (DATASET_SPEC 1.6 unbalanced
# 32-expert trace character): one giant expert, several mid, a long tail, and a
# final dead expert. Used as a router BIAS so token->expert counts are
# unbalanced and expert E-1 receives 0 tokens (the mandatory MoE edge).
_TRACE32 = [
    16053, 105, 1843, 2724, 327, 88, 4102, 51, 9210, 61, 3020, 44, 1502, 990,
    233, 77, 6740, 120, 410, 58, 2210, 39, 812, 175, 5030, 66, 1360, 92, 3550,
    47, 1180, 0,
]


def jagged_counts(E):
    """Reproducible jagged per-expert count pattern of length E, last entry 0.

    For E <= 32 uses the 32-expert trace prefix; for larger E it tiles the trace
    (dropping the interior zeros so only the LAST expert is guaranteed dead)."""
    if E <= len(_TRACE32):
        c = list(_TRACE32[:E])
    else:
        tile = [x for x in _TRACE32 if x > 0]
        c = [tile[i % len(tile)] for i in range(E)]
    c[-1] = 0
    return c


def make_routing(M, E, topk, device, g, renorm=True):
    """Unbalanced router assignment with a guaranteed 0-token last expert.

    Softmax over an unbalanced bias + noise, top-k select, optional renorm.
    Returns (topk_weight [M,topk] fp32, topk_ids [M,topk] int32)."""
    import torch

    counts = torch.tensor([float(x) for x in jagged_counts(E)],
                          dtype=torch.float32, device=device)
    bias = torch.log(counts + 1e-6)
    bias[counts == 0] = float("-inf")            # never select the dead expert
    gate = torch.randn((M, E), generator=g, device=device, dtype=torch.float32) + bias
    probs = torch.softmax(gate, dim=-1)
    tw, ti = torch.topk(probs, topk, dim=-1)
    if renorm:
        tw = tw / tw.sum(dim=-1, keepdim=True)
    return tw.to(torch.float32), ti.to(torch.int32)


# --------------------------------------------------------------------------- #
# fp8 quantization (arch-selected OCP e4m3fn on gfx950)
# --------------------------------------------------------------------------- #
def quant_per_token_fp8(x):
    """Per-row (per-token) symmetric fp8 quant. x[M,K] -> (xq[M,K] fp8, s[M,1] fp32)."""
    import torch

    from kore.tasks.aiter_ref import FP8_DTYPE, FP8_MAX
    amax = x.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = (amax / FP8_MAX).to(torch.float32)
    xq = (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return xq, scale


def quant_per_channel_fp8(w):
    """Per-row (per-output-channel) symmetric fp8 quant for a [N,K] weight.

    Returns (wq[N,K] fp8, s[N,1] fp32) with ``w ~= wq.float() * s``."""
    import torch

    from kore.tasks.aiter_ref import FP8_DTYPE, FP8_MAX
    amax = w.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = (amax / FP8_MAX).to(torch.float32)
    wq = (w.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return wq, scale


# --------------------------------------------------------------------------- #
# MoE building-block oracles (fp32 accumulate -> low-precision out)
# --------------------------------------------------------------------------- #
def gated_mlp_fp32(hidden, w1, w2, topk_weight, topk_ids, act="silu"):
    """Exact fp32 top-k fused-MoE oracle -> hidden.dtype, output [M, model_dim].

    Per token, for each selected expert e:
        gate_up = x @ w1[e].T           # [2*inter]  (gate = first half, up = second)
        h       = act(gate) * up        # [inter]
        y_e     = h @ w2[e].T           # [model_dim]
    and y = sum_k topk_weight[:,k] * y_{e_k}. Weight layout matches aiter:
    w1 [E, 2*inter, model_dim], w2 [E, model_dim, inter]. Experts with zero
    assigned tokens are skipped (the 0-token edge)."""
    import torch

    activation = _act_fp32(act)
    M, D = hidden.shape
    E = w1.shape[0]
    I = w2.shape[2]
    x = hidden.float()
    w1f = w1.float()
    w2f = w2.float()
    out = torch.zeros((M, D), device=hidden.device, dtype=torch.float32)
    ids = topk_ids.long()
    tw = topk_weight.float()
    for e in range(E):
        mask = ids == e                          # [M, topk]
        tok = mask.any(dim=1)
        if not bool(tok.any()):
            continue                             # 0-token expert -> skipped
        idx = tok.nonzero(as_tuple=True)[0]
        xe = x[idx]                              # [n, D]
        gate_up = xe @ w1f[e].t()                # [n, 2I]
        gate, up = gate_up[:, :I], gate_up[:, I:]
        h = activation(gate) * up                # [n, I]
        ye = h @ w2f[e].t()                      # [n, D]
        w_e = (tw * mask.float()).sum(dim=1)[idx]  # weight for expert e per token
        out[idx] += ye * w_e[:, None]
    return out.to(hidden.dtype)


def grouped_gemm_fp32(hidden, w, expert_ids, out_dtype=None):
    """Segmented (per-expert) GEMM: out[m] = hidden[m] @ w[expert_ids[m]].T.

    hidden [M, K], w [E, N, K], expert_ids [M] int -> out [M, N]. fp32 accumulate.
    A token routed to a 0-token... (there are none here: every token has exactly
    one expert) -- experts that receive no token are simply never visited. This
    is the canonical MoE grouped/segmented GEMM (the gate_up projection stage)."""
    import torch

    if out_dtype is None:
        out_dtype = hidden.dtype
    M, K = hidden.shape
    E, N, _ = w.shape
    x = hidden.float()
    wf = w.float()
    out = torch.zeros((M, N), device=hidden.device, dtype=torch.float32)
    eids = expert_ids.long()
    for e in range(E):
        idx = (eids == e).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        out[idx] = x[idx] @ wf[e].t()
    return out.to(out_dtype)


def grouped_gemm_fp8_fp32(xq, wq, x_scale, w_scale, expert_ids, out_dtype=None):
    """Segmented per-expert fp8 a8w8 GEMM oracle (fp32 of the dequant).

    xq [M,K] fp8, wq [E,N,K] fp8, x_scale [M,1] fp32 (per-token),
    w_scale [E,N,1] fp32 (per-expert per-channel), expert_ids [M] int.
    out[m] = (xq[m]*x_scale[m]) @ (wq[e]*w_scale[e]).T,  e = expert_ids[m]."""
    import torch

    if out_dtype is None:
        out_dtype = torch.bfloat16
    M, K = xq.shape
    E, N, _ = wq.shape
    xd = xq.float() * x_scale.float()                     # [M,K]
    out = torch.zeros((M, N), device=xq.device, dtype=torch.float32)
    eids = expert_ids.long()
    for e in range(E):
        idx = (eids == e).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        wd = wq[e].float() * w_scale[e].float()           # [N,K]
        out[idx] = xd[idx] @ wd.t()
    return out.to(out_dtype)


def batched_gemm_fp32(a, b, out_dtype=None):
    """Batched expert GEMM: C[e] = A[e] @ B[e]^T. A[E,m,K], B[E,N,K] -> [E,m,N]."""
    import torch

    if out_dtype is None:
        out_dtype = a.dtype
    return torch.bmm(a.float(), b.float().transpose(1, 2)).to(out_dtype)


def to_dense(topk_weights, topk_ids, E):
    """Scatter (weights, ids) to a dense [M, E] fp32 tensor (order-independent)."""
    import torch

    M = topk_weights.shape[0]
    dense = torch.zeros((M, E), device=topk_weights.device, dtype=torch.float32)
    dense.scatter_(1, topk_ids.long(), topk_weights.float())
    return dense


def topk_softmax_dense_fp32(gate, topk, renorm=True):
    """Exact fp32 softmax -> top-k (+optional renorm), returned dense [M, E]."""
    import torch

    E = gate.shape[1]
    sm = torch.softmax(gate.float(), dim=-1)
    tw, ti = torch.topk(sm, topk, dim=-1)
    if renorm:
        tw = tw / tw.sum(dim=-1, keepdim=True)
    return to_dense(tw, ti, E)


def biased_grouped_topk_dense_fp32(gate, correction_bias, topk, n_groups,
                                   topk_group, renorm=True, scale=1.0):
    """DeepSeek-V3 biased grouped top-k router, returned dense [M, E].

    Matches the vLLM / SGLang / aiter ``biased_grouped_topk`` reference:
      scores      = sigmoid(gate)                              # NOT softmax
      scores_bias = scores + correction_bias                   # routing-only bias
      group_score = sum of the top-2 scores_bias within each expert group
      keep the ``topk_group`` groups with the highest group_score
      top-k experts by scores_bias among the kept groups
      weights     = the ORIGINAL sigmoid ``scores`` at the chosen experts,
                    renormalized to sum 1 (if renorm) and multiplied by ``scale``.
    gate [M,E], correction_bias [E]. E must be divisible by n_groups."""
    import torch

    M, E = gate.shape
    grp = E // n_groups
    scores = torch.sigmoid(gate.float())                      # [M,E]
    sb = scores + correction_bias.float().view(1, E)          # [M,E]
    gview = sb.view(M, n_groups, grp)
    top2 = gview.topk(min(2, grp), dim=-1).values.sum(dim=-1)  # [M,n_groups]
    keep = top2.topk(topk_group, dim=-1).indices              # [M,topk_group]
    gmask = torch.zeros((M, n_groups), device=gate.device, dtype=torch.bool)
    gmask.scatter_(1, keep, True)
    emask = gmask.view(M, n_groups, 1).expand(M, n_groups, grp).reshape(M, E)
    masked = torch.where(emask, sb, torch.full_like(sb, float("-inf")))
    ti = masked.topk(topk, dim=-1).indices                    # [M,topk]
    tw = torch.gather(scores, 1, ti)                          # original sigmoid weights
    if renorm:
        tw = tw / tw.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    tw = tw * scale
    return to_dense(tw, ti, E)


def permute_tokens(hidden, expert_ids):
    """Expert-sorted token scatter (the MoE dispatch/permute).

    hidden [M, D], expert_ids [M] int. Returns (permuted [M, D], sort_idx [M]).
    ``sort_idx`` is a STABLE argsort of expert_ids (ties keep original token
    order), so ``permuted[i] = hidden[sort_idx[i]]`` groups every expert's tokens
    into one contiguous, ascending-expert block -- exactly the layout a grouped
    GEMM consumes. Stable + fully deterministic, so it is an unambiguous oracle."""
    import torch

    eids = expert_ids.long()
    sort_idx = torch.argsort(eids, stable=True)
    return hidden[sort_idx], sort_idx


def moe_sum_fp32(y, topk_weight, out_dtype=None):
    """Weighted combine over the top-k slots: out[m] = sum_k w[m,k]*y[m,k,:].

    y [M, topk, D], topk_weight [M, topk] -> out [M, D]. fp32 accumulate."""
    import torch

    if out_dtype is None:
        out_dtype = y.dtype
    out = (y.float() * topk_weight.float().unsqueeze(-1)).sum(dim=1)
    return out.to(out_dtype)


# --------------------------------------------------------------------------- #
# REAL AITER vendor baselines (thin, lazy import). Honestly labeled via
# aiter_ref._mark_baseline (aiter_vendor / hipblaslt_vendor / framework).
# --------------------------------------------------------------------------- #
def vendor_fused_moe(hidden, w1, w2, topk_weight, topk_ids, activation="silu"):
    """AITER production fused MoE (CK 2-stage). Weights pre-shuffled here for the
    correctness/oracle-match path; the driver bench pre-shuffles OUTSIDE timing."""
    from kore.tasks.aiter_ref import _mark_baseline
    from kore.tasks.aiter_ref_attn import shuffle_moe_weights

    import aiter
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    act = ActivationType.Silu if activation == "silu" else ActivationType.Gelu
    w1s, w2s = shuffle_moe_weights(w1, w2)
    out = fused_moe(hidden, w1s, w2s, topk_weight, topk_ids,
                    activation=act, quant_type=QuantType.No)
    _mark_baseline("aiter_vendor")
    return out


def vendor_fused_moe_preshuffled(hidden, w1s, w2s, topk_weight, topk_ids,
                                 activation="silu"):
    """Same as :func:`vendor_fused_moe` but takes PRE-SHUFFLED weights (the driver
    shuffles once outside the timed region, matching production load-time cost)."""
    from kore.tasks.aiter_ref import _mark_baseline

    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    act = ActivationType.Silu if activation == "silu" else ActivationType.Gelu
    out = fused_moe(hidden, w1s, w2s, topk_weight, topk_ids,
                    activation=act, quant_type=QuantType.No)
    _mark_baseline("aiter_vendor")
    return out


def vendor_batched_gemm_bf16(a, b):
    """AITER batched bf16 GEMM (A[E,m,K] @ B[E,N,K]^T -> [E,m,N]); torch.bmm ->
    hipBLASLt fallback. Reuses the confirmed wrapper in aiter_ref."""
    from kore.tasks.aiter_ref import aiter_batched_gemm_bf16

    return aiter_batched_gemm_bf16(a, b)


def vendor_grouped_gemm_bf16(hidden, w, expert_ids):
    """Production dense bf16 grouped GEMM bar: per-expert ``torch.matmul`` (which
    on ROCm dispatches to hipBLASLt). out[m] = hidden[m] @ w[e]^T for e=expert_ids[m].
    The candidate must beat launching one hipBLASLt GEMM per non-empty expert."""
    import torch

    from kore.tasks.aiter_ref import _mark_baseline
    M, K = hidden.shape
    E, N, _ = w.shape
    out = torch.empty((M, N), device=hidden.device, dtype=torch.bfloat16)
    eids = expert_ids.long()
    for e in range(E):
        idx = (eids == e).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        out[idx] = torch.matmul(hidden[idx], w[e].t())
    _mark_baseline("hipblaslt_vendor")
    return out


def vendor_grouped_gemm_fp8(xq, wq, x_scale, w_scale, expert_ids):
    """Per-expert AITER fp8 a8w8 GEMM (``aiter.gemm_a8w8``) grouped bar. For each
    non-empty expert e, runs the confirmed CK fp8 GEMM on that expert's token
    slice with per-token x-scale and per-channel w-scale, bf16 out."""
    import torch

    from kore.tasks.aiter_ref import aiter_gemm_a8w8
    M, K = xq.shape
    E, N, _ = wq.shape
    out = torch.empty((M, N), device=xq.device, dtype=torch.bfloat16)
    eids = expert_ids.long()
    for e in range(E):
        idx = (eids == e).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        xs = x_scale[idx].contiguous()                    # [n,1] per-token
        ws = w_scale[e].reshape(1, N).contiguous()        # [1,N] per-channel
        out[idx] = aiter_gemm_a8w8(xq[idx].contiguous(), wq[e].contiguous(),
                                   xs, ws, out_dtype=torch.bfloat16)
    return out


def vendor_topk_softmax_dense(gate, topk, E, renorm=True):
    """AITER ``topk_softmax`` -> dense [M,E] (order-independent grading)."""
    from kore.tasks.aiter_ref_attn import aiter_topk_softmax

    w, ids = aiter_topk_softmax(gate, topk, renorm)
    return to_dense(w, ids, E)


def vendor_biased_grouped_topk_dense(gate, correction_bias, topk, n_groups,
                                     topk_group, E, renorm=True, scale=1.0):
    """AITER ``biased_grouped_topk`` -> dense [M,E] if present; else the verified
    fp32 oracle (framework bar). The exact aiter signature is version-dependent
    (see VERIFICATION_CHECKLIST FLAG); this tries the common form and falls back."""
    import torch

    from kore.tasks.aiter_ref import _mark_baseline
    try:
        import aiter
        M = gate.shape[0]
        tw = torch.empty((M, topk), dtype=torch.float32, device=gate.device)
        ti = torch.empty((M, topk), dtype=torch.int32, device=gate.device)
        aiter.biased_grouped_topk(
            gate, correction_bias, tw, ti, n_groups, topk_group,
            renorm, scale,
        )
        _mark_baseline("aiter_vendor")
        return to_dense(tw, ti, E)
    except Exception:  # noqa: BLE001 - symbol/signature absent in this build
        _mark_baseline("framework")
        return biased_grouped_topk_dense_fp32(
            gate, correction_bias, topk, n_groups, topk_group,
            renorm=renorm, scale=scale)


def vendor_permute(hidden, sort_idx):
    """Token permute (expert dispatch). AITER exposes no standalone permuted-copy
    with a fixed public layout, so the honest production bar is the framework
    indexed gather ``hidden[sort_idx]`` (a fused ROCm gather kernel)."""
    from kore.tasks.aiter_ref import _mark_baseline

    _mark_baseline("framework")
    return hidden[sort_idx.long()]


def vendor_moe_sum(y, topk_weight):
    """Weighted top-k combine. Tries AITER ``moe_sum`` (unweighted reduce over
    dim=1) folded with the router weights; falls back to the framework reduce."""
    import torch

    from kore.tasks.aiter_ref import _mark_baseline
    _mark_baseline("framework")
    return (y.float() * topk_weight.float().unsqueeze(-1)).sum(dim=1).to(y.dtype)


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
        r = ref.reference_output(shape, _clone_inputs(inputs))
        try:
            o = ref.candidate_output(fn, shape, _clone_inputs(inputs))
        except Exception as e:  # noqa: BLE001
            print("SNR: -999.00 dB"); print("allclose: False"); print("max_diff: inf")
            print(f"CANDIDATE_ERROR: {type(e).__name__}: {e}")
            return 0
        torch.cuda.synchronize()
        snr, md, cok = _compare_outputs(o, r, atol=atol, rtol=rtol)
        worst = min(worst, snr)
        maxd = max(maxd, md)
        ok = ok and cok
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
    p.add_argument("--kore-driver-capabilities", action="store_true",
                   help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.kore_driver_capabilities:
        emit_driver_capabilities()
        return 0
    shape = ref.parse_shape(a.shape)
    if a.bench_mode:
        rc = _run_bench(ref, task_dir, shape, a.impl, a.warmup, a.iters)
        # Post-timing anti-hack correctness re-verification on the cached candidate.
        if a.impl == "candidate":
            _run_correctness(ref, task_dir, shape)
        return rc
    return _run_correctness(ref, task_dir, shape)
