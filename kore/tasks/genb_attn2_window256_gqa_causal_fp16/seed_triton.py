"""GENERATED breadth attn2_window256_gqa_causal seed (fp16). Naive but correct fused flash-attention: one program per (query-block, batch*head) streams KV blocks with an online (max, sum) softmax (fp32 math); window serving/inference variant, GQA via kv_head = head // group. The policy fuses/tiles it. tl.float16 store."""
from __future__ import annotations
import torch
import triton
import triton.language as tl


@triton.jit
def _attn2_fwd(Q, K, V, O, Bias, sm_scale, H, HKV, SQ, SK,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
               IS_CAUSAL: tl.constexpr, WINDOW: tl.constexpr, DILATION: tl.constexpr,
               USE_BIAS: tl.constexpr):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H
    group = H // HKV
    off_hkv = off_h // group

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_row = (off_b * H + off_h) * SQ + offs_m
    q_mask = offs_m[:, None] < SQ
    q = tl.load(Q + q_row[:, None] * HEAD_DIM + offs_d[None, :], mask=q_mask, other=0.0).to(tl.float32)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q_pos = (SK - SQ) + offs_m
    kv_base = (off_b * HKV + off_hkv) * SK

    lo = 0
    if WINDOW > 0:
        lo = start_m * BLOCK_M + (SK - SQ) - WINDOW + 1
        lo = tl.maximum((lo // BLOCK_N) * BLOCK_N, 0)
    if IS_CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M + (SK - SQ), SK)
    else:
        hi = SK

    for start_n in range(lo, hi, BLOCK_N):
        n = start_n + offs_n
        n_mask = n < SK
        k = tl.load(K + (kv_base + n)[None, :] * HEAD_DIM + offs_d[:, None],
                    mask=n_mask[None, :], other=0.0).to(tl.float32)
        qk = tl.dot(q, k) * sm_scale
        if USE_BIAS:
            b = tl.load(Bias + (off_h * SQ + offs_m[:, None]) * SK + n[None, :],
                        mask=q_mask & n_mask[None, :], other=0.0).to(tl.float32)
            qk = qk + b
        keep = n_mask[None, :]
        if IS_CAUSAL:
            keep = keep & (n[None, :] <= q_pos[:, None])
        if WINDOW > 0:
            keep = keep & (q_pos[:, None] - n[None, :] < WINDOW)
        if DILATION > 1:
            keep = keep & (n[None, :] <= q_pos[:, None]) & (((q_pos[:, None] - n[None, :]) % DILATION) == 0)
        qk = tl.where(keep, qk, -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + (kv_base + n)[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=n_mask[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha[:, None] + tl.dot(p, v)
        m_i = m_ij

    acc = acc / l_i[:, None]
    tl.store(O + q_row[:, None] * HEAD_DIM + offs_d[None, :],
             acc.to(O.dtype.element_ty), mask=q_mask)


def attn2_window256_gqa_causal(q, k, v):
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    B, H, SQ, D = q.shape
    HKV = k.shape[1]
    SK = k.shape[2]
    o = torch.empty((B, H, SQ, D), dtype=q.dtype, device=q.device)
    sm_scale = 1.0 / (D ** 0.5)
    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(SQ, BLOCK_M), B * H)
    _attn2_fwd[grid](
        q, k, v, o, q, sm_scale, H, HKV, SQ, SK,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
        IS_CAUSAL=True, WINDOW=256, DILATION=1,
        USE_BIAS=False, num_warps=4)
    return o
