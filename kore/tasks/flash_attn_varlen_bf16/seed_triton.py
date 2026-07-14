"""Seed Triton bf16 variable-length causal (GQA) flash attention for gfx942.

Exposes ``flash_attn(q, k, v, cu_seqlens, max_seqlen, causal=True)`` with packed
q ``[T,H,D]``, k/v ``[T,KV,D]`` and ``cu_seqlens[B+1]`` int32. Grid is
(m-blocks over max_seqlen, batch, head); each program reads its sequence's
``[cu[b], cu[b+1])`` range and runs online-softmax flash within it (no cross-
sequence attention, no padding). Boundary masking handles ragged seqlens. A correct
baseline the KORE policy optimizes against the AITER varlen FMHA bar.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fa_varlen_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, cu_ptr,
    sqt, sqh, skt, skh, svt, svh, sot, soh,
    scale,
    H: tl.constexpr, KV: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr,
):
    m_block = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    kv_h = h // (H // KV)

    seq_start = tl.load(cu_ptr + b)
    seq_end = tl.load(cu_ptr + b + 1)
    seq_len = seq_end - seq_start

    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)   # row within the sequence
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    row_ok = offs_m < seq_len

    q_base = q_ptr + seq_start * sqt + h * sqh
    q = tl.load(q_base + offs_m[:, None] * sqt + offs_d[None, :],
                mask=row_ok[:, None], other=0.0).to(tl.float32)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    k_base = k_ptr + seq_start * skt + kv_h * skh
    v_base = v_ptr + seq_start * svt + kv_h * svh

    n_end = seq_len
    if CAUSAL:
        n_end = tl.minimum(seq_len, (m_block + 1) * BLOCK_M)

    for n_start in range(0, n_end, BLOCK_N):
        cols = n_start + offs_n
        col_ok = cols < seq_len
        k = tl.load(k_base + cols[:, None] * skt + offs_d[None, :],
                    mask=col_ok[:, None], other=0.0).to(tl.float32)
        qk = tl.dot(q, tl.trans(k)) * scale
        qk = tl.where(col_ok[None, :], qk, -float("inf"))
        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= cols[None, :], qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        p = tl.exp(qk - m_safe[:, None])
        alpha = tl.exp(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        v = tl.load(v_base + cols[:, None] * svt + offs_d[None, :],
                    mask=col_ok[:, None], other=0.0).to(tl.float32)
        acc += tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16)).to(tl.float32)
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    o_base = o_ptr + seq_start * sot + h * soh
    tl.store(o_base + offs_m[:, None] * sot + offs_d[None, :],
             acc.to(tl.bfloat16), mask=row_ok[:, None])


def flash_attn(q, k, v, cu_seqlens, max_seqlen, causal: bool = True):
    T, H, D = q.shape
    KV = k.shape[1]
    scale = 1.0 / (D ** 0.5)
    B = cu_seqlens.numel() - 1
    o = torch.empty_like(q)
    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(int(max_seqlen), BLOCK_M), B, H)
    _fa_varlen_kernel[grid](
        q, k, v, o, cu_seqlens,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        o.stride(0), o.stride(1),
        scale,
        H=H, KV=KV, D=D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, CAUSAL=causal,
        num_warps=4, num_stages=2,
    )
    return o
