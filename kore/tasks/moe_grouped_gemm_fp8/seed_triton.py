"""Seed Triton fp8 grouped (segmented) expert GEMM (a8w8) for gfx950.

Exposes ``grouped_gemm_fp8(xq, wq, x_scale, w_scale, expert_ids)`` with xq
``[M,K]`` fp8, wq ``[E,N,K]`` fp8, x_scale ``[M,1]`` fp32 (per-token), w_scale
``[E,N,1]`` fp32 (per-channel), expert_ids ``[M]`` -> out ``[M,N]`` bf16. One
program per (token, n-tile): reads the token's expert id, up-converts the fp8
operands to fp32, fp32-accumulates over K, then folds the per-token and
per-channel scales onto the accumulator. A correct, simple seed the KORE policy
optimizes against the per-expert AITER gemm_a8w8 bar.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_fp8_kernel(
    xq_ptr, wq_ptr, xs_ptr, ws_ptr, e_ptr, o_ptr,
    sxq_m, sxq_k, swq_e, swq_n, swq_k, sxs_m, sws_e, sws_n, so_m, so_n,
    N, K,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    token = tl.program_id(0)
    nblk = tl.program_id(1)
    e = tl.load(e_ptr + token)
    xscale = tl.load(xs_ptr + token * sxs_m).to(tl.float32)

    offs_n = nblk * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = offs_n < N
    wscale = tl.load(ws_ptr + e * sws_e + offs_n * sws_n, mask=nmask, other=0.0).to(tl.float32)

    acc = tl.zeros([BLOCK_N], tl.float32)
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        kmask = offs_k < K
        xk = tl.load(xq_ptr + token * sxq_m + offs_k * sxq_k, mask=kmask, other=0.0).to(tl.float32)
        wk = tl.load(
            wq_ptr + e * swq_e + offs_n[:, None] * swq_n + offs_k[None, :] * swq_k,
            mask=nmask[:, None] & kmask[None, :], other=0.0,
        ).to(tl.float32)                                   # [BLOCK_N, BLOCK_K]
        acc += tl.sum(xk[None, :] * wk, axis=1)            # [BLOCK_N]

    acc = acc * xscale * wscale
    tl.store(o_ptr + token * so_m + offs_n * so_n, acc.to(tl.bfloat16), mask=nmask)


def grouped_gemm_fp8(xq, wq, x_scale, w_scale, expert_ids) -> torch.Tensor:
    M, K = xq.shape
    E, N, _ = wq.shape
    out = torch.empty((M, N), device=xq.device, dtype=torch.bfloat16)
    BLOCK_N, BLOCK_K = 64, 64
    grid = (M, triton.cdiv(N, BLOCK_N))
    _grouped_fp8_kernel[grid](
        xq, wq, x_scale, w_scale, expert_ids.contiguous(), out,
        xq.stride(0), xq.stride(1),
        wq.stride(0), wq.stride(1), wq.stride(2),
        x_scale.stride(0), w_scale.stride(0), w_scale.stride(1),
        out.stride(0), out.stride(1),
        N, K,
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=4, num_stages=2,
    )
    return out
