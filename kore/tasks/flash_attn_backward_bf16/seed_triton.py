"""Seed causal MHA FLASH-ATTENTION BACKWARD (dQ/dK/dV) for gfx950 (CDNA4).

Exposes ``flash_attn_backward(q, k, v, o, do, lse, causal=True) -> (dq, dk, dv)``
with q,k,v,o,do,dq,dk,dv [B,S,H,D] bf16 and lse [B,H,S] fp32 (saved forward output
and row log-sum-exp). Implements the FlashAttention-2 backward without ever
materializing the S x S matrix:

    delta_i = sum_d o_id * do_id                              (preprocess)
    P_ij    = exp(scale * q_i.k_j - lse_i)   (causal: j <= i, else 0)
    dP_ij   = do_i . v_j        dS_ij = P_ij * (dP_ij - delta_i)
    dV_j = sum_i P_ij do_i    dK_j = scale * sum_i dS_ij q_i    dQ_i = scale * sum_j dS_ij k_j

Three kernels: (1) delta preprocess; (2) one program per KV block accumulates
dK/dV by streaming query blocks; (3) one program per Q block accumulates dQ by
streaming KV blocks. fp32 accumulation, matrix-core (bf16) dots for the reductions.

CORRECTNESS-FIRST SEED: it streams over ALL blocks and relies on the causal
mask (query >= key) to zero the upper triangle, so it is unconditionally correct
but does ~2x the minimal causal work. The KORE policy optimizes it (skip fully
masked blocks, fuse dK/dV/dQ, tune BLOCK / num_warps / num_stages). MHA only
(H == KV); GQA/MQA backward (grad accumulation across grouped query heads) is a
future extension -- see VERIFICATION_CHECKLIST.md.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_preprocess(
    o_ptr, do_ptr, delta_ptr, H, S,
    sb, ss, sh, D: tl.constexpr, BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, D)
    base = b * sb + h * sh
    m_mask = offs_m[:, None] < S
    o = tl.load(o_ptr + base + offs_m[:, None] * ss + offs_d[None, :], mask=m_mask, other=0.0).to(tl.float32)
    do = tl.load(do_ptr + base + offs_m[:, None] * ss + offs_d[None, :], mask=m_mask, other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)                       # [BLOCK]
    tl.store(delta_ptr + bh * S + offs_m, delta, mask=offs_m < S)


@triton.jit
def _bwd_dkdv(
    q_ptr, k_ptr, v_ptr, do_ptr, lse_ptr, delta_ptr, dk_ptr, dv_ptr,
    H, S, scale, sb, ss, sh,
    D: tl.constexpr, BLOCK: tl.constexpr, CAUSAL: tl.constexpr,
):
    pid_n = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_n = pid_n * BLOCK + tl.arange(0, BLOCK)         # keys
    offs_d = tl.arange(0, D)
    base = b * sb + h * sh
    n_mask = offs_n[:, None] < S
    kn = tl.load(k_ptr + base + offs_n[:, None] * ss + offs_d[None, :], mask=n_mask, other=0.0).to(tl.float32)
    vn = tl.load(v_ptr + base + offs_n[:, None] * ss + offs_d[None, :], mask=n_mask, other=0.0).to(tl.float32)
    dk = tl.zeros([BLOCK, D], tl.float32)
    dv = tl.zeros([BLOCK, D], tl.float32)
    lse_base = bh * S

    for start_m in range(0, S, BLOCK):
        offs_m = start_m + tl.arange(0, BLOCK)          # queries
        m_mask = offs_m[:, None] < S
        qm = tl.load(q_ptr + base + offs_m[:, None] * ss + offs_d[None, :], mask=m_mask, other=0.0).to(tl.float32)
        dom = tl.load(do_ptr + base + offs_m[:, None] * ss + offs_d[None, :], mask=m_mask, other=0.0).to(tl.float32)
        lse_m = tl.load(lse_ptr + lse_base + offs_m, mask=offs_m < S, other=0.0)      # [BLOCK]
        delta_m = tl.load(delta_ptr + lse_base + offs_m, mask=offs_m < S, other=0.0)  # [BLOCK]

        qk = tl.dot(qm, tl.trans(kn))                    # [BM,BN] fp32
        p = tl.exp(qk * scale - lse_m[:, None])          # [BM,BN]
        valid = (offs_m[:, None] < S) & (offs_n[None, :] < S)
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        p = tl.where(valid, p, 0.0)
        dp = tl.dot(dom, tl.trans(vn))                   # [BM,BN] fp32
        ds = p * (dp - delta_m[:, None])                 # [BM,BN]
        dv += tl.dot(tl.trans(p).to(tl.bfloat16), dom.to(tl.bfloat16))
        dk += tl.dot(tl.trans(ds).to(tl.bfloat16), qm.to(tl.bfloat16))

    dk = dk * scale
    tl.store(dk_ptr + base + offs_n[:, None] * ss + offs_d[None, :], dk.to(dk_ptr.dtype.element_ty), mask=n_mask)
    tl.store(dv_ptr + base + offs_n[:, None] * ss + offs_d[None, :], dv.to(dv_ptr.dtype.element_ty), mask=n_mask)


@triton.jit
def _bwd_dq(
    q_ptr, k_ptr, v_ptr, do_ptr, lse_ptr, delta_ptr, dq_ptr,
    H, S, scale, sb, ss, sh,
    D: tl.constexpr, BLOCK: tl.constexpr, CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)         # queries
    offs_d = tl.arange(0, D)
    base = b * sb + h * sh
    m_mask = offs_m[:, None] < S
    qm = tl.load(q_ptr + base + offs_m[:, None] * ss + offs_d[None, :], mask=m_mask, other=0.0).to(tl.float32)
    dom = tl.load(do_ptr + base + offs_m[:, None] * ss + offs_d[None, :], mask=m_mask, other=0.0).to(tl.float32)
    lse_base = bh * S
    lse_m = tl.load(lse_ptr + lse_base + offs_m, mask=offs_m < S, other=0.0)
    delta_m = tl.load(delta_ptr + lse_base + offs_m, mask=offs_m < S, other=0.0)
    dq = tl.zeros([BLOCK, D], tl.float32)

    for start_n in range(0, S, BLOCK):
        offs_n = start_n + tl.arange(0, BLOCK)          # keys
        n_mask = offs_n[:, None] < S
        kn = tl.load(k_ptr + base + offs_n[:, None] * ss + offs_d[None, :], mask=n_mask, other=0.0).to(tl.float32)
        vn = tl.load(v_ptr + base + offs_n[:, None] * ss + offs_d[None, :], mask=n_mask, other=0.0).to(tl.float32)

        qk = tl.dot(qm, tl.trans(kn))                    # [BM,BN] fp32
        p = tl.exp(qk * scale - lse_m[:, None])          # [BM,BN]
        valid = (offs_m[:, None] < S) & (offs_n[None, :] < S)
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        p = tl.where(valid, p, 0.0)
        dp = tl.dot(dom, tl.trans(vn))                   # [BM,BN] fp32
        ds = p * (dp - delta_m[:, None])                 # [BM,BN]
        dq += tl.dot(ds.to(tl.bfloat16), kn.to(tl.bfloat16))

    dq = dq * scale
    tl.store(dq_ptr + base + offs_m[:, None] * ss + offs_d[None, :], dq.to(dq_ptr.dtype.element_ty), mask=m_mask)


def flash_attn_backward(q, k, v, o, do, lse, causal: bool = True):
    B, S, H, D = q.shape
    scale = 1.0 / (D ** 0.5)
    lse = lse.contiguous()
    delta = torch.empty((B, H, S), device=q.device, dtype=torch.float32)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    sb, ss, sh = q.stride(0), q.stride(1), q.stride(2)
    BLOCK = 64
    grid = (triton.cdiv(S, BLOCK), B * H)
    _bwd_preprocess[grid](o, do, delta, H, S, sb, ss, sh, D=D, BLOCK=BLOCK)
    _bwd_dkdv[grid](q, k, v, do, lse, delta, dk, dv, H, S, scale, sb, ss, sh,
                    D=D, BLOCK=BLOCK, CAUSAL=causal, num_warps=8, num_stages=1)
    _bwd_dq[grid](q, k, v, do, lse, delta, dq, H, S, scale, sb, ss, sh,
                  D=D, BLOCK=BLOCK, CAUSAL=causal, num_warps=8, num_stages=1)
    return dq, dk, dv
