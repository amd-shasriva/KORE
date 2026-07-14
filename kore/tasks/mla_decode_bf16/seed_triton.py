"""Seed Triton bf16 MLA (Multi-head Latent Attention) decode for gfx942.

Exposes ``mla(q_nope, q_pe, c_kv, k_pe, w_uk, w_uv) -> out``. Absorbed-form MLA:
per (batch, head) it projects the query into the latent space
(``lat_q = q_nope @ W_UK``), runs online-softmax attention over the S-token latent
cache using ``lat_q . c_kv + q_pe . k_pe``, accumulates ``lat_o = softmax @ c_kv``,
then projects out ``out = lat_o @ W_UV^T``. One program per (batch, head); SQ query
rows are padded to BLOCK_SQ=16 so the MFMA ``tl.dot`` is legal for decode (SQ=1).
Correct baseline the KORE policy optimizes; this variant is HELD OUT (generalization).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _mla_kernel(
    qn_ptr, qp_ptr, ckv_ptr, kpe_ptr, wuk_ptr, wuv_ptr, o_ptr,
    qnb, qns, qnh, qpb, qps, qph,
    ckvb, ckvs, kpeb, kpes,
    wukh, wukr, wuvh, wuvr,
    ob, os_, oh,
    S, scale,
    DC: tl.constexpr, DNOPE: tl.constexpr, DROPE: tl.constexpr, DV: tl.constexpr,
    SQ: tl.constexpr, BLOCK_SQ: tl.constexpr, BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs_sq = tl.arange(0, BLOCK_SQ)
    sq_ok = offs_sq < SQ
    offs_dn = tl.arange(0, DNOPE)
    offs_dc = tl.arange(0, DC)
    offs_dr = tl.arange(0, DROPE)
    offs_dv = tl.arange(0, DV)

    qn = tl.load(qn_ptr + b * qnb + offs_sq[:, None] * qns + h * qnh + offs_dn[None, :],
                 mask=sq_ok[:, None], other=0.0).to(tl.float32)          # [BLOCK_SQ, DNOPE]
    qp = tl.load(qp_ptr + b * qpb + offs_sq[:, None] * qps + h * qph + offs_dr[None, :],
                 mask=sq_ok[:, None], other=0.0).to(tl.float32)          # [BLOCK_SQ, DROPE]
    wuk = tl.load(wuk_ptr + h * wukh + offs_dn[:, None] * wukr + offs_dc[None, :]).to(tl.float32)  # [DNOPE, DC]
    lat_q = tl.dot(qn.to(tl.bfloat16), wuk.to(tl.bfloat16)).to(tl.float32)  # [BLOCK_SQ, DC]

    m_i = tl.full([BLOCK_SQ], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_SQ], tl.float32)
    acc = tl.zeros([BLOCK_SQ, DC], tl.float32)

    for n_start in range(0, S, BLOCK_N):
        cols = n_start + tl.arange(0, BLOCK_N)
        col_ok = cols < S
        ckv = tl.load(ckv_ptr + b * ckvb + cols[:, None] * ckvs + offs_dc[None, :],
                      mask=col_ok[:, None], other=0.0).to(tl.float32)    # [BLOCK_N, DC]
        kpe = tl.load(kpe_ptr + b * kpeb + cols[:, None] * kpes + offs_dr[None, :],
                      mask=col_ok[:, None], other=0.0).to(tl.float32)    # [BLOCK_N, DROPE]
        s_nope = tl.dot(lat_q.to(tl.bfloat16), tl.trans(ckv).to(tl.bfloat16)).to(tl.float32)
        s_pe = tl.dot(qp.to(tl.bfloat16), tl.trans(kpe).to(tl.bfloat16)).to(tl.float32)
        scores = (s_nope + s_pe) * scale
        scores = tl.where(col_ok[None, :], scores, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        p = tl.exp(scores - m_safe[:, None])
        alpha = tl.exp(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(tl.bfloat16), ckv.to(tl.bfloat16)).to(tl.float32)  # [BLOCK_SQ, DC]
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    lat_o = acc / l_safe[:, None]                                        # [BLOCK_SQ, DC]
    wuv = tl.load(wuv_ptr + h * wuvh + offs_dv[:, None] * wuvr + offs_dc[None, :]).to(tl.float32)  # [DV, DC]
    out = tl.dot(lat_o.to(tl.bfloat16), tl.trans(wuv).to(tl.bfloat16)).to(tl.float32)  # [BLOCK_SQ, DV]

    tl.store(o_ptr + b * ob + offs_sq[:, None] * os_ + h * oh + offs_dv[None, :],
             out.to(tl.bfloat16), mask=sq_ok[:, None])


def mla(q_nope, q_pe, c_kv, k_pe, w_uk, w_uv):
    B, SQ, H, DNOPE = q_nope.shape
    DROPE = q_pe.shape[3]
    S, DC = c_kv.shape[1], c_kv.shape[2]
    DV = w_uv.shape[1]
    scale = 1.0 / ((DNOPE + DROPE) ** 0.5)
    o = torch.empty((B, SQ, H, DV), device=q_nope.device, dtype=torch.bfloat16)
    BLOCK_SQ = 16
    BLOCK_N = 64
    grid = (B, H)
    _mla_kernel[grid](
        q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, o,
        q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
        q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
        c_kv.stride(0), c_kv.stride(1),
        k_pe.stride(0), k_pe.stride(1),
        w_uk.stride(0), w_uk.stride(1),
        w_uv.stride(0), w_uv.stride(1),
        o.stride(0), o.stride(1), o.stride(2),
        S, scale,
        DC=DC, DNOPE=DNOPE, DROPE=DROPE, DV=DV,
        SQ=SQ, BLOCK_SQ=BLOCK_SQ, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )
    return o
