"""Seed Triton bf16 sliding-window (GQA) flash-attention *decode* for gfx950 (CDNA4).

Exposes ``flash_attn_decode(q, k, v, window=1024)`` with q ``[B,1,H,D]``, k/v
``[B,Skv,KV,D]``. The single query (global position Skv-1) attends only to the most
recent ``window`` keys: the kernel starts the KV loop at the window's lower edge (aligned
down to BLOCK_N) so KV blocks fully before the band are SKIPPED (bounded work regardless
of context length). Online softmax (max-subtraction + running rescale), fp32
accumulation, bf16 IO, GQA/MQA head mapping. A correct seed the KORE policy optimizes.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fa_sliding_decode_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    sqb, sqh,
    skb, sks, skh,
    svb, svs, svh,
    sob, soh,
    Skv, scale,
    H: tl.constexpr, KV: tl.constexpr, D: tl.constexpr,
    WINDOW: tl.constexpr, BLOCK_N: tl.constexpr,
):
    bh = tl.program_id(0)
    b = bh // H
    h = bh % H
    kv_h = h // (H // KV)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)

    q = tl.load(q_ptr + b * sqb + h * sqh + offs_d).to(tl.float32)   # [D]

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([D], tl.float32)

    k_base = k_ptr + b * skb + kv_h * skh
    v_base = v_ptr + b * svb + kv_h * svh

    # Query global position is Skv-1; it attends keys j with j > Skv-1-WINDOW. Start
    # the loop at the window's lower edge aligned down to BLOCK_N so earlier KV blocks
    # are skipped entirely.
    lo = Skv - WINDOW
    lo = tl.maximum(lo, 0)
    lo = (lo // BLOCK_N) * BLOCK_N
    low_edge = Skv - 1 - WINDOW

    for n_start in range(lo, Skv, BLOCK_N):
        cols = n_start + offs_n
        mask = (cols < Skv) & (cols > low_edge)
        k = tl.load(
            k_base + cols[:, None] * sks + offs_d[None, :],
            mask=mask[:, None], other=0.0,
        ).to(tl.float32)                                  # [BLOCK_N, D]
        qk = tl.sum(q[None, :] * k, axis=1) * scale       # [BLOCK_N]
        qk = tl.where(mask, qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=0))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        p = tl.exp(qk - m_safe)
        alpha = tl.exp(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        v = tl.load(
            v_base + cols[:, None] * svs + offs_d[None, :],
            mask=mask[:, None], other=0.0,
        ).to(tl.float32)                                  # [BLOCK_N, D]
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe
    tl.store(o_ptr + b * sob + h * soh + offs_d, acc.to(tl.bfloat16))


def flash_attn_decode(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int = 1024):
    B, Sq, H, D = q.shape
    KV = k.shape[2]
    Skv = k.shape[1]
    scale = 1.0 / (D ** 0.5)
    o = torch.empty_like(q)
    BLOCK_N = 128
    grid = (B * H,)
    _fa_sliding_decode_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(2),
        Skv, scale,
        H=H, KV=KV, D=D, WINDOW=window, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )
    return o
