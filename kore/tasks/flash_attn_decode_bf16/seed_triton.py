"""Seed Triton bf16 (GQA) flash-attention *decode* kernel for gfx942.

Exposes ``flash_attn_decode(q, k, v)`` with q ``[B,1,H,D]``, k/v ``[B,Skv,KV,D]``.
One program per (batch, q-head): loads the single query vector, streams the KV
context in BLOCK_N chunks with online softmax (max-subtraction + running
rescale) and fp32 accumulation, bf16 IO. GQA maps each Q head to its shared KV
head. Boundary masking keeps it correct for KV lengths that are not a multiple of
the tile.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fa_decode_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    sqb, sqh,               # q strides: batch, head  (seq_q=1)
    skb, sks, skh,          # k strides: batch, seq, head
    svb, svs, svh,          # v strides
    sob, soh,               # o strides: batch, head
    Skv, scale,
    H: tl.constexpr, KV: tl.constexpr, D: tl.constexpr,
    BLOCK_N: tl.constexpr,
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

    for n_start in range(0, Skv, BLOCK_N):
        cols = n_start + offs_n
        mask = cols < Skv
        k = tl.load(
            k_base + cols[:, None] * sks + offs_d[None, :],
            mask=mask[:, None], other=0.0,
        ).to(tl.float32)                                  # [BLOCK_N, D]
        qk = tl.sum(q[None, :] * k, axis=1) * scale       # [BLOCK_N]
        qk = tl.where(mask, qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=0))
        p = tl.exp(qk - m_new)                            # [BLOCK_N]
        alpha = tl.exp(m_i - m_new)
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


def flash_attn_decode(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    B, Sq, H, D = q.shape
    KV = k.shape[2]
    Skv = k.shape[1]
    scale = 1.0 / (D ** 0.5)
    o = torch.empty_like(q)
    BLOCK_N = 128
    grid = (B * H,)
    _fa_decode_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(2),
        Skv, scale,
        H=H, KV=KV, D=D, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )
    return o
