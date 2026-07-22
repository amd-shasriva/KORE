"""GENERATED breadth attn_bwd_mqa_causal seed (bf16). Fused flash-attention BACKWARD (dQ/dK/dV) for causal attention: recompute the online-softmax log-sum-exp, then the standard flash dQ and dK/dV passes (GQA/MQA reduce over the kv group). Naive but correct; the policy fuses/tiles it. tl.bfloat16 store."""
from __future__ import annotations
import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_lse(Q, K, V, O, LSE, sm_scale, H, HKV, SQ, SK,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
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
    kv_base = (off_b * HKV + off_hkv) * SK
    hi = tl.minimum((start_m + 1) * BLOCK_M, SK)
    for start_n in range(0, hi, BLOCK_N):
        n = start_n + offs_n
        n_mask = n < SK
        k = tl.load(K + (kv_base + n)[None, :] * HEAD_DIM + offs_d[:, None],
                    mask=n_mask[None, :], other=0.0).to(tl.float32)
        qk = tl.dot(q, k) * sm_scale
        keep = n_mask[None, :] & (n[None, :] <= offs_m[:, None])
        qk = tl.where(keep, qk, -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + (kv_base + n)[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=n_mask[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha[:, None] + tl.dot(p, v)
        m_i = m_ij
    tl.store(O + q_row[:, None] * HEAD_DIM + offs_d[None, :],
             (acc / l_i[:, None]).to(O.dtype.element_ty), mask=q_mask)
    tl.store(LSE + q_row, m_i + tl.log(l_i), mask=offs_m < SQ)


@triton.jit
def _attn_bwd_dq(Q, K, V, DO, LSE, Delta, DQ, sm_scale, H, HKV, SQ, SK,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
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
    m_mask = offs_m[:, None] < SQ
    q = tl.load(Q + q_row[:, None] * HEAD_DIM + offs_d[None, :], mask=m_mask, other=0.0).to(tl.float32)
    do = tl.load(DO + q_row[:, None] * HEAD_DIM + offs_d[None, :], mask=m_mask, other=0.0).to(tl.float32)
    lse = tl.load(LSE + q_row, mask=offs_m < SQ, other=0.0)
    delta = tl.load(Delta + q_row, mask=offs_m < SQ, other=0.0)
    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    kv_base = (off_b * HKV + off_hkv) * SK
    hi = tl.minimum((start_m + 1) * BLOCK_M, SK)
    for start_n in range(0, hi, BLOCK_N):
        n = start_n + offs_n
        n_mask = n < SK
        k = tl.load(K + (kv_base + n)[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=n_mask[:, None], other=0.0).to(tl.float32)
        v = tl.load(V + (kv_base + n)[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=n_mask[:, None], other=0.0).to(tl.float32)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        p = tl.exp(qk - lse[:, None])
        keep = n_mask[None, :] & (n[None, :] <= offs_m[:, None])
        p = tl.where(keep, p, 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds, k)
    tl.store(DQ + q_row[:, None] * HEAD_DIM + offs_d[None, :], dq * sm_scale, mask=m_mask)


@triton.jit
def _attn_bwd_dkdv(Q, K, V, DO, LSE, Delta, DK, DV, sm_scale, H, HKV, SQ, SK,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
    start_n = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H
    group = H // HKV
    off_hkv = off_h // group
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    n_mask = offs_n[:, None] < SK
    kv_base = (off_b * HKV + off_hkv) * SK
    k = tl.load(K + (kv_base + offs_n)[:, None] * HEAD_DIM + offs_d[None, :],
                mask=n_mask, other=0.0).to(tl.float32)
    v = tl.load(V + (kv_base + offs_n)[:, None] * HEAD_DIM + offs_d[None, :],
                mask=n_mask, other=0.0).to(tl.float32)
    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    q_head = (off_b * H + off_h) * SQ
    lo = (start_n * BLOCK_N // BLOCK_M) * BLOCK_M
    for start_m in range(lo, SQ, BLOCK_M):
        m = start_m + offs_m
        m_mask = m[:, None] < SQ
        q = tl.load(Q + (q_head + m)[:, None] * HEAD_DIM + offs_d[None, :], mask=m_mask, other=0.0).to(tl.float32)
        do = tl.load(DO + (q_head + m)[:, None] * HEAD_DIM + offs_d[None, :], mask=m_mask, other=0.0).to(tl.float32)
        lse = tl.load(LSE + q_head + m, mask=m < SQ, other=0.0)
        delta = tl.load(Delta + q_head + m, mask=m < SQ, other=0.0)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        p = tl.exp(qk - lse[:, None])
        keep = (offs_n[None, :] <= m[:, None]) & (offs_n[None, :] < SK) & (m[:, None] < SQ)
        p = tl.where(keep, p, 0.0)
        dv += tl.dot(tl.trans(p), do)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dk += tl.dot(tl.trans(ds), q)
    dk_row = (off_b * H + off_h) * SK + offs_n
    tl.store(DK + dk_row[:, None] * HEAD_DIM + offs_d[None, :], dk * sm_scale, mask=n_mask)
    tl.store(DV + dk_row[:, None] * HEAD_DIM + offs_d[None, :], dv, mask=n_mask)


def attn_bwd_mqa_causal(q, k, v, do):
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous(); do = do.contiguous()
    B, H, SQ, D = q.shape
    HKV = k.shape[1]
    SK = k.shape[2]
    group = H // HKV
    sm_scale = 1.0 / (D ** 0.5)
    o = torch.empty((B, H, SQ, D), dtype=q.dtype, device=q.device)
    lse = torch.empty((B, H, SQ), dtype=torch.float32, device=q.device)
    BLOCK_M = 64
    BLOCK_N = 64
    _attn_fwd_lse[(triton.cdiv(SQ, BLOCK_M), B * H)](
        q, k, v, o, lse, sm_scale, H, HKV, SQ, SK,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D, num_warps=4)
    delta = (o.float() * do.float()).sum(-1).contiguous()
    dq = torch.zeros((B, H, SQ, D), dtype=torch.float32, device=q.device)
    dkf = torch.zeros((B, H, SK, D), dtype=torch.float32, device=q.device)
    dvf = torch.zeros((B, H, SK, D), dtype=torch.float32, device=q.device)
    _attn_bwd_dq[(triton.cdiv(SQ, BLOCK_M), B * H)](
        q, k, v, do, lse, delta, dq, sm_scale, H, HKV, SQ, SK,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D, num_warps=4)
    _attn_bwd_dkdv[(triton.cdiv(SK, BLOCK_N), B * H)](
        q, k, v, do, lse, delta, dkf, dvf, sm_scale, H, HKV, SQ, SK,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D, num_warps=4)
    dk = dkf.view(B, HKV, group, SK, D).sum(2).to(k.dtype)
    dv = dvf.view(B, HKV, group, SK, D).sum(2).to(v.dtype)
    return dq.to(q.dtype), dk, dv
