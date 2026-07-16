"""Seed Triton fp8 (GQA) flash-attention *decode* for gfx950/CDNA4 (fp8 QKV -> bf16 out).

Exposes ``flash_attn_decode(q, k, v, sq, sk, sv)`` with q ``[B,1,H,D]`` fp8, k/v
``[B,Skv,KV,D]`` fp8, per-tensor fp32 scales sq/sk/sv. One program per (batch, q-head):
loads the single fp8 query, streams the fp8 KV context in BLOCK_N chunks with online
softmax and fp32 accumulation, folds the dequant scales (``qk_scale = sq*sk/sqrt(D)``
into the QK product, ``v_scale = sv`` onto the output), bf16 store. GQA/MQA maps each Q
head to its shared KV head. A correct seed the KORE policy optimizes against the AITER
FMHA decode bar (the fp8 kernel wins by moving ~half the KV bytes of bf16).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fa_decode_fp8_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    sqb, sqh,
    skb, sks, skh,
    svb, svs, svh,
    sob, soh,
    Skv, qk_scale, v_scale,
    H: tl.constexpr, KV: tl.constexpr, D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    bh = tl.program_id(0)
    b = bh // H
    h = bh % H
    kv_h = h // (H // KV)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)

    q = tl.load(q_ptr + b * sqb + h * sqh + offs_d).to(tl.float32)   # [D] fp8 value

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
        qk = tl.sum(q[None, :] * k, axis=1) * qk_scale    # fold sq*sk/sqrt(D)
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
    acc = acc / l_safe * v_scale                          # apply v dequant scale
    tl.store(o_ptr + b * sob + h * soh + offs_d, acc.to(tl.bfloat16))


def flash_attn_decode(q, k, v, sq, sk, sv):
    B, Sq, H, D = q.shape
    KV = k.shape[2]
    Skv = k.shape[1]
    qk_scale = float(sq) * float(sk) / (D ** 0.5)
    v_scale = float(sv)
    o = torch.empty((B, 1, H, D), device=q.device, dtype=torch.bfloat16)
    BLOCK_N = 128
    grid = (B * H,)
    _fa_decode_fp8_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(2),
        Skv, qk_scale, v_scale,
        H=H, KV=KV, D=D, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )
    return o
