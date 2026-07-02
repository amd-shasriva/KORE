"""Seed Triton bf16 paged-KV-cache decode attention kernel for gfx942.

Exposes ``paged_attn_decode(query, key_cache, value_cache, block_tables,
seq_lens, block_size, scale)``. One program per (sequence, q-head): walks the
sequence's block table one page at a time (page == tile == block_size=16),
gathers K/V straight out of the vLLM paged layout, and does online-softmax
attention with fp32 accumulation and bf16 IO. Per-sequence context length is read
at runtime so a partial last page is handled by masking.

KV-cache layout (x = 16 // itemsize = 8 for bf16):
    key_cache   : [num_blocks, KV, D // x, block_size, x]
    value_cache : [num_blocks, KV, D, block_size]
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode_kernel(
    q_ptr, kc_ptr, vc_ptr, o_ptr, bt_ptr, sl_ptr,
    sqs, sqh,                       # query strides: seq, head
    skb, skh, skd, sks, skx,        # key_cache strides
    svb, svh, svd, svs,             # value_cache strides
    sos, soh,                       # out strides: seq, head
    sbt_s, sbt_c,                   # block_tables strides
    scale,
    H: tl.constexpr, KV: tl.constexpr, D: tl.constexpr,
    X: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    s = pid // H
    h = pid % H
    kv_h = h // (H // KV)

    ctx = tl.load(sl_ptr + s)
    offs_d = tl.arange(0, D)
    offs_i = tl.arange(0, BLOCK_N)
    d_hi = offs_d // X
    d_lo = offs_d % X

    q = tl.load(q_ptr + s * sqs + h * sqh + offs_d).to(tl.float32)   # [D]

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([D], tl.float32)

    n_pages = (ctx + BLOCK_N - 1) // BLOCK_N
    for c in range(0, n_pages):
        page = tl.load(bt_ptr + s * sbt_s + c * sbt_c)
        t = c * BLOCK_N + offs_i
        mask = t < ctx                                    # [BLOCK_N]
        k_off = (page * skb + kv_h * skh
                 + d_hi[None, :] * skd + offs_i[:, None] * sks + d_lo[None, :] * skx)
        k = tl.load(kc_ptr + k_off, mask=mask[:, None], other=0.0).to(tl.float32)  # [BN,D]
        qk = tl.sum(q[None, :] * k, axis=1) * scale       # [BN]
        qk = tl.where(mask, qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=0))
        p = tl.exp(qk - m_new)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        v_off = (page * svb + kv_h * svh
                 + offs_d[None, :] * svd + offs_i[:, None] * svs)
        v = tl.load(vc_ptr + v_off, mask=mask[:, None], other=0.0).to(tl.float32)  # [BN,D]
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe
    tl.store(o_ptr + s * sos + h * soh + offs_d, acc.to(tl.bfloat16))


def paged_attn_decode(query, key_cache, value_cache, block_tables, seq_lens,
                      block_size, scale):
    B, H, D = query.shape
    KV = key_cache.shape[1]
    X = key_cache.shape[-1]
    out = torch.empty((B, H, D), device=query.device, dtype=query.dtype)
    grid = (B * H,)
    _paged_decode_kernel[grid](
        query, key_cache, value_cache, out, block_tables, seq_lens,
        query.stride(0), query.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        key_cache.stride(3), key_cache.stride(4),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        value_cache.stride(3),
        out.stride(0), out.stride(1),
        block_tables.stride(0), block_tables.stride(1),
        scale,
        H=H, KV=KV, D=D, X=X, BLOCK_N=block_size,
        num_warps=4, num_stages=2,
    )
    return out
