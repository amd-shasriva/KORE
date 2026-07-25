"""Shared AITER baseline helpers for the KORE attention / MoE tasks.

The KORE convention (see ``aiter_ref.py``): the *performance baseline* for every
task is the kernel the production serving stack actually calls (AITER), not an
unfused torch path. This module centralizes the thin AITER wrappers for the
attention + MoE family so each task's ``driver.py --impl reference`` measures the
honest production bar.

This file is intentionally separate from ``aiter_ref.py`` (owned elsewhere) and
only adds attention / paged-KV / MoE / router wrappers.

Import-safe: ``aiter`` and ``torch`` heavy work happens lazily inside the
wrappers so registry discovery never needs a GPU or the aiter runtime.

gfx942 / CDNA3 notes
--------------------
* Attention fwd baseline is CK/ASM FMHA via ``aiter.flash_attn_func`` (bf16,
  fp32 accumulation), layout ``(batch, seqlen, nheads, head_dim)``, GQA by
  passing fewer KV heads than Q heads.
* Paged decode baseline is the ROCm custom paged attention
  (``aiter.paged_attention_rocm``) with the vLLM KV-cache layout
  (key ``[num_blocks, kv_heads, head_dim//x, block_size, x]`` with
  ``x = 16 // itemsize``, value ``[num_blocks, kv_heads, head_dim, block_size]``).
* Fused MoE baseline (``aiter.fused_moe.fused_moe``) requires the CK 2-stage
  *shuffled* weight layout; production shuffles weights **once at load time**, so
  a fair per-call benchmark must pre-shuffle outside the timed region (the driver
  does this) and only time the GEMM+activation+reduce.
* Router baseline is ``aiter.topk_softmax`` (softmax over experts -> top-k select
  -> optional renormalize), written in place.
"""

from __future__ import annotations

from typing import Optional

import torch

from kore.tasks.aiter_ref import _mark_baseline

# ROCm custom paged-attention partition size (see aiter/paged_attn.py).
PARTITION_SIZE_ROCM = 256


# --- Flash attention (prefill + decode), CK/ASM FMHA fwd -----------------
def aiter_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """AITER ``flash_attn_func``: q/k/v are ``(B, S, H, D)`` bf16.

    Supports GQA/MQA by passing K/V with fewer heads than Q (Q heads must be a
    multiple of KV heads). Returns ``(B, S, H, D)``.
    """
    import aiter

    out = aiter.flash_attn_func(q, k, v, causal=causal, softmax_scale=softmax_scale)
    _mark_baseline("aiter_vendor")
    return out


# --- fp8 flash attention (per-tensor descale), CK/ASM FMHA fwd -----------
def aiter_flash_attn_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
) -> torch.Tensor:
    """AITER fp8 flash attention fwd (per-tensor descale).

    Dense form ``aiter.flash_attn_fp8_pertensor_func(q, k, v, q_descale,
    k_descale, v_descale, causal=, softmax_scale=)`` with q/k/v in the
    arch-selected **OCP** fp8 e4m3 (``FP8_DTYPE`` -- e4m3fn on gfx950/CDNA4, NOT
    FNUZ) and scalar fp32 descales; layout ``(B, S, H, D)``. When the varlen
    packing args (``cu_seqlens_*`` + ``max_seqlen_*``) are supplied we call
    ``aiter.flash_attn_varlen_fp8_pertensor_func`` instead (q/k/v packed
    ``(total_tokens, H, D)``). Returns attention output in the fp8-fwd dtype.

    Falls back to the bf16 CK FMHA path (:func:`aiter_flash_attn`, dequantizing
    q/k/v via the descales) when the fp8 kernel is unavailable, and finally to
    torch SDPA (framework).
    """
    import aiter

    varlen = cu_seqlens_q is not None and cu_seqlens_k is not None
    try:
        if varlen:
            out = aiter.flash_attn_varlen_fp8_pertensor_func(
                q, k, v, q_descale, k_descale, v_descale,
                cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                causal=causal, softmax_scale=softmax_scale,
            )
        else:
            out = aiter.flash_attn_fp8_pertensor_func(
                q, k, v, q_descale, k_descale, v_descale,
                causal=causal, softmax_scale=softmax_scale,
            )
        _mark_baseline("aiter_vendor")
        return out
    except Exception:  # noqa: BLE001 - fp8 fmha kernel unavailable
        # Dequantize with the per-tensor descales and run the bf16 CK FMHA path.
        try:
            qd = (q.float() * q_descale.float()).to(torch.bfloat16)
            kd = (k.float() * k_descale.float()).to(torch.bfloat16)
            vd = (v.float() * v_descale.float()).to(torch.bfloat16)
            out = aiter.flash_attn_func(
                qd, kd, vd, causal=causal, softmax_scale=softmax_scale
            )
            _mark_baseline("aiter_vendor")
            return out
        except Exception:  # noqa: BLE001 - no CK fmha at all -> torch SDPA
            _mark_baseline("framework")
            import torch.nn.functional as F
            qd = (q.float() * q_descale.float()).to(torch.bfloat16)
            kd = (k.float() * k_descale.float()).to(torch.bfloat16)
            vd = (v.float() * v_descale.float()).to(torch.bfloat16)
            # (B, S, H, D) -> (B, H, S, D) for SDPA.
            o = F.scaled_dot_product_attention(
                qd.transpose(1, 2), kd.transpose(1, 2), vd.transpose(1, 2),
                is_causal=causal, scale=softmax_scale,
            )
            return o.transpose(1, 2)


# --- Flash attention BACKWARD (CK/ASM FMHA bwd) --------------------------
def aiter_flash_attn_backward_prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
):
    """Build the autograd graph for one FMHA fwd and return a **zero-arg closure**
    that runs ONLY the backward pass, so a caller can time the bwd in isolation.

    This uses ``aiter.ops.mha.FlashAttnFunc`` (via ``aiter.flash_attn_func``,
    which requires ``return_lse``/grad bookkeeping): the forward is executed here
    (OUTSIDE the timed region) to populate the saved tensors + a random ``dout``;
    the returned ``run_backward()`` invokes ``out.backward(dout, retain_graph=True)``
    which dispatches into ``FlashAttnFunc.backward`` -> ``_flash_attn_backward``
    (the CK/ASM FMHA bwd kernel producing dq/dk/dv). ``retain_graph=True`` lets
    the harness call it repeatedly for a stable median.

    q/k/v are ``(B, S, H, D)`` bf16 and MUST have ``requires_grad=True`` (the
    caller sets this before building inputs). Returns ``run_backward`` plus the
    leaf tensors so the caller can read/zero their ``.grad``.
    """
    import aiter

    out = aiter.flash_attn_func(
        q, k, v, causal=causal, softmax_scale=softmax_scale
    )
    dout = torch.randn_like(out)

    def run_backward():
        # ONLY this region is what a caller would time: the FMHA backward kernel.
        for t in (q, k, v):
            if t.grad is not None:
                t.grad = None
        out.backward(dout, retain_graph=True)
        _mark_baseline("aiter_vendor")
        return q.grad, k.grad, v.grad

    return run_backward, (q, k, v), out, dout


def aiter_flash_attn_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
):
    """Convenience one-shot FMHA backward baseline (fwd + a single bwd).

    Prefer :func:`aiter_flash_attn_backward_prepare` in a benchmark so the
    forward is excluded from timing; this helper runs the fwd then a single
    backward and returns ``(dq, dk, dv)``. Uses ``aiter.ops.mha.FlashAttnFunc``
    via ``aiter.flash_attn_func``.
    """
    run_backward, _leaves, _out, _dout = aiter_flash_attn_backward_prepare(
        q, k, v, causal=causal, softmax_scale=softmax_scale
    )
    return run_backward()


# --- Paged-KV decode attention (ROCm custom PA) --------------------------
def aiter_paged_attention_decode(
    query: torch.Tensor,          # [num_seqs, num_q_heads, head_dim]
    key_cache: torch.Tensor,      # [num_blocks, kv_heads, head_dim//x, block_size, x]
    value_cache: torch.Tensor,    # [num_blocks, kv_heads, head_dim, block_size]
    block_tables: torch.Tensor,   # [num_seqs, max_blocks_per_seq] int32
    seq_lens: torch.Tensor,       # [num_seqs] int32
    block_size: int,
    max_seq_len: int,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    """AITER ROCm custom paged attention decode (bf16 KV, ``kv_cache_dtype='auto'``)."""
    import aiter
    from aiter import dtypes

    num_seqs, num_heads, head_size = query.shape
    out = torch.empty_like(query)
    num_partitions = (max_seq_len + PARTITION_SIZE_ROCM - 1) // PARTITION_SIZE_ROCM
    tmp_out = torch.empty(
        (num_seqs, num_heads, num_partitions, head_size),
        dtype=query.dtype, device=query.device,
    )
    exp_sums = torch.empty(
        (num_seqs, num_heads, num_partitions), dtype=dtypes.fp32, device=query.device
    )
    max_logits = torch.empty_like(exp_sums)
    one = torch.tensor(1.0, dtype=torch.float32, device=query.device)
    aiter.paged_attention_rocm(
        out, exp_sums, max_logits, tmp_out,
        query, key_cache, value_cache,
        num_kv_heads, scale, block_tables, seq_lens,
        block_size, max_seq_len, None, "auto",
        one, one, None, PARTITION_SIZE_ROCM, 1,
    )
    _mark_baseline("aiter_vendor")
    return out


# --- Fused MoE (grouped GEMM + SiLU-mul + top-k reduce) ------------------
def shuffle_moe_weights(w1: torch.Tensor, w2: torch.Tensor):
    """Pre-shuffle MoE weights into the CK 2-stage layout (load-time cost).

    Returns ``(w1_shuffled, w2_shuffled)`` tagged with ``is_shuffled``.
    """
    from aiter.ops.shuffle import shuffle_weight

    return shuffle_weight(w1, layout=(16, 16)), shuffle_weight(w2, layout=(16, 16))


def aiter_fused_moe(
    hidden_states: torch.Tensor,  # [M, model_dim] bf16
    w1_shuffled: torch.Tensor,    # [E, 2*inter, model_dim] bf16, pre-shuffled
    w2_shuffled: torch.Tensor,    # [E, model_dim, inter] bf16, pre-shuffled
    topk_weight: torch.Tensor,    # [M, topk] fp32
    topk_ids: torch.Tensor,       # [M, topk] int32
) -> torch.Tensor:
    """AITER production fused MoE (bf16, SiLU gate, CK 2-stage). Weights must be
    pre-shuffled via :func:`shuffle_moe_weights` (done once, outside timing)."""
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    out = fused_moe(
        hidden_states, w1_shuffled, w2_shuffled, topk_weight, topk_ids,
        activation=ActivationType.Silu, quant_type=QuantType.No,
    )
    _mark_baseline("aiter_vendor")
    return out


# --- Router: softmax + top-k select (+ renormalize) ----------------------
def aiter_topk_softmax(gating_output: torch.Tensor, topk: int, renormalize: bool = True):
    """AITER ``topk_softmax``: softmax over experts -> top-k -> optional renorm.

    ``gating_output`` is ``[M, E]``. Returns ``(topk_weights[M,topk] fp32,
    topk_ids[M,topk] int32)``.
    """
    import aiter
    from aiter import dtypes

    M = gating_output.shape[0]
    topk_weights = torch.empty(M, topk, dtype=dtypes.fp32, device=gating_output.device)
    topk_ids = torch.empty(M, topk, dtype=dtypes.i32, device=gating_output.device)
    token_expert_idx = torch.empty(M, topk, dtype=dtypes.i32, device=gating_output.device)
    aiter.topk_softmax(topk_weights, topk_ids, token_expert_idx, gating_output, renormalize)
    _mark_baseline("aiter_vendor")
    return topk_weights, topk_ids
