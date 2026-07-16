"""Seed Triton bf16 grouped (segmented) expert GEMM for gfx950.

Exposes ``grouped_gemm(hidden, w, expert_ids)`` with hidden ``[M,K]``, w
``[E,N,K]``, expert_ids ``[M]`` -> out ``[M,N]`` bf16, fp32 accumulate. One
program per (token, n-tile): reads the token's expert id, then contracts the
token's hidden vector with that expert's weight rows over K in BLOCK_K tiles.
Tokens whose expert has 0 assigned tokens never arise (each token has an expert);
a 0-token expert simply has no program reference it. A correct, simple seed the
KORE policy optimizes against the per-expert hipBLASLt grouped bar.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_kernel(
    x_ptr, w_ptr, e_ptr, o_ptr,
    sx_m, sx_k, sw_e, sw_n, sw_k, so_m, so_n,
    N, K,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    token = tl.program_id(0)
    nblk = tl.program_id(1)
    e = tl.load(e_ptr + token)

    offs_n = nblk * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = offs_n < N
    acc = tl.zeros([BLOCK_N], tl.float32)
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        kmask = offs_k < K
        x = tl.load(x_ptr + token * sx_m + offs_k * sx_k, mask=kmask, other=0.0).to(tl.float32)
        w = tl.load(
            w_ptr + e * sw_e + offs_n[:, None] * sw_n + offs_k[None, :] * sw_k,
            mask=nmask[:, None] & kmask[None, :], other=0.0,
        ).to(tl.float32)                                   # [BLOCK_N, BLOCK_K]
        acc += tl.sum(x[None, :] * w, axis=1)              # [BLOCK_N]

    tl.store(o_ptr + token * so_m + offs_n * so_n, acc.to(tl.bfloat16), mask=nmask)


def grouped_gemm(hidden: torch.Tensor, w: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
    M, K = hidden.shape
    E, N, _ = w.shape
    out = torch.empty((M, N), device=hidden.device, dtype=torch.bfloat16)
    BLOCK_N, BLOCK_K = 64, 64
    grid = (M, triton.cdiv(N, BLOCK_N))
    _grouped_kernel[grid](
        hidden, w, expert_ids.contiguous(), out,
        hidden.stride(0), hidden.stride(1),
        w.stride(0), w.stride(1), w.stride(2),
        out.stride(0), out.stride(1),
        N, K,
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=4, num_stages=2,
    )
    return out
