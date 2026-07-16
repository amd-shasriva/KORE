"""Seed Triton bf16 causal (MHA/GQA) flash-attention prefill for gfx950 (CDNA4).

Exposes ``flash_attn(q, k, v, causal=True)`` with q ``[B,S,H,D]``, k/v
``[B,S,KV,D]``. Online-softmax (max-subtraction + running rescale), fp32
accumulation, bf16 IO. Each Q head maps to its shared KV head (kv_h = h // (H//KV));
with KV == H this is plain multi-head attention. Boundary masking keeps it correct
for seqlens that are not a multiple of the tile. A correct, reasonably-tuned seed
the KORE policy optimizes against the AITER FMHA serving bar.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fa_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    sqb, sqs, sqh,
    skb, sks, skh,
    svb, svs, svh,
    sob, sos, soh,
    S, scale,
    H: tl.constexpr, KV: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr,
):
    m_block = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    kv_h = h // (H // KV)

    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    q_base = q_ptr + b * sqb + h * sqh
    q = tl.load(
        q_base + offs_m[:, None] * sqs + offs_d[None, :],
        mask=offs_m[:, None] < S, other=0.0,
    ).to(tl.float32)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    k_base = k_ptr + b * skb + kv_h * skh
    v_base = v_ptr + b * svb + kv_h * svh

    n_end = S
    if CAUSAL:
        n_end = tl.minimum(S, (m_block + 1) * BLOCK_M)

    for n_start in range(0, n_end, BLOCK_N):
        cols = n_start + offs_n
        col_mask = cols < S
        k = tl.load(
            k_base + cols[:, None] * sks + offs_d[None, :],
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
            v_base + cols[:, None] * svs + offs_d[None, :],
            mask=col_mask[:, None], other=0.0,
        ).to(tl.float32)
        acc += tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16)).to(tl.float32)
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    o_base = o_ptr + b * sob + h * soh
    tl.store(
        o_base + offs_m[:, None] * sos + offs_d[None, :],
        acc.to(tl.bfloat16), mask=offs_m[:, None] < S,
    )


def flash_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True):
    B, S, H, D = q.shape
    KV = k.shape[2]
    scale = 1.0 / (D ** 0.5)
    o = torch.empty_like(q)
    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    _fa_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        S, scale,
        H=H, KV=KV, D=D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, CAUSAL=causal,
        num_warps=4, num_stages=2,
    )
    return o
