"""Seed Triton bf16 NON-causal VARLEN / packed (GQA) flash attention for gfx950.

Exposes ``flash_attn_varlen(q, k, v, cu_seqlens, max_seqlen, causal=False)`` with q
``[T,H,D]``, k/v ``[T,KV,D]`` packed (no padding), ``cu_seqlens`` int32 ``[num_seqs+1]``.
This correct-but-naive seed loops the sequences in Python and runs a per-sequence flash
kernel over each ``[cu[s]:cu[s+1]]`` slice (online softmax, fp32 accumulation, bf16 IO);
with causal=False each sequence attends bidirectionally within itself. The KORE policy's
job is to fuse this into a single persistent varlen kernel (one launch, cu_seqlens-
indexed) that beats the AITER ragged FMHA bar; the Python loop is the obviously-correct
starting point. The CAUSAL flag is retained so the same kernel also serves causal varlen.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fa_varlen_seq_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    tok_off,                 # first token index of this sequence (runtime int)
    sqt, sqh,                # q strides: token, head
    skt, skh,                # k strides
    svt, svh,                # v strides
    sot, soh,                # o strides
    L, scale,
    H: tl.constexpr, KV: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr,
):
    m_block = tl.program_id(0)
    h = tl.program_id(1)
    kv_h = h // (H // KV)

    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    q_base = q_ptr + tok_off * sqt + h * sqh
    q = tl.load(
        q_base + offs_m[:, None] * sqt + offs_d[None, :],
        mask=offs_m[:, None] < L, other=0.0,
    ).to(tl.float32)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    k_base = k_ptr + tok_off * skt + kv_h * skh
    v_base = v_ptr + tok_off * svt + kv_h * svh

    n_end = L
    if CAUSAL:
        n_end = tl.minimum(L, (m_block + 1) * BLOCK_M)

    for n_start in range(0, n_end, BLOCK_N):
        cols = n_start + offs_n
        col_mask = cols < L
        k = tl.load(
            k_base + cols[:, None] * skt + offs_d[None, :],
            mask=col_mask[:, None], other=0.0,
        ).to(tl.float32)
        qk = tl.dot(q, tl.trans(k)) * scale
        qk = tl.where(col_mask[None, :], qk, -float("inf"))
        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= cols[None, :], qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        p = tl.exp(qk - m_safe[:, None])
        alpha = tl.exp(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        v = tl.load(
            v_base + cols[:, None] * svt + offs_d[None, :],
            mask=col_mask[:, None], other=0.0,
        ).to(tl.float32)
        acc += tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16)).to(tl.float32)
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    o_base = o_ptr + tok_off * sot + h * soh
    tl.store(
        o_base + offs_m[:, None] * sot + offs_d[None, :],
        acc.to(tl.bfloat16), mask=offs_m[:, None] < L,
    )


def flash_attn_varlen(q, k, v, cu_seqlens, max_seqlen, causal: bool = False):
    total, H, D = q.shape
    KV = k.shape[1]
    scale = 1.0 / (D ** 0.5)
    o = torch.empty_like(q)
    BLOCK_M, BLOCK_N = 64, 64
    cu = cu_seqlens.tolist()
    for s in range(len(cu) - 1):
        a, b = cu[s], cu[s + 1]
        L = b - a
        if L <= 0:
            continue
        grid = (triton.cdiv(L, BLOCK_M), H)
        _fa_varlen_seq_kernel[grid](
            q, k, v, o,
            a,
            q.stride(0), q.stride(1),
            k.stride(0), k.stride(1),
            v.stride(0), v.stride(1),
            o.stride(0), o.stride(1),
            L, scale,
            H=H, KV=KV, D=D,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, CAUSAL=causal,
            num_warps=4, num_stages=2,
        )
    return o
