"""Seed W4A16 (int4 weight, bf16 activation) GEMM for gfx942 (CDNA3).

Exposes ``matmul(a, w_packed, scale) -> y`` where a is [M,K] bf16, w_packed is
[N,K//2] uint8 (2 int4 nibbles/byte along K), scale is [N,1] fp32, y is [M,N] bf16:
    W_deq[n,k] = (nibble(n,k) - 8) * scale[n]
    y = a @ W_deq^T

The packed layout stores even-K in the low nibble and odd-K in the high nibble of
each byte, so a tile is dequantized as TWO half-K operands (lo=even, hi=odd) and
accumulated with two ``tl.dot`` calls over the strided even/odd activation columns.
A correct baseline the KORE policy optimizes (fuse the two dots, widen tiles, load
the weight bytes vectorized, pipeline K, tune num_warps) to beat the bf16-weight bar.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _w4a16_kernel(
    a_ptr, wp_ptr, s_ptr, y_ptr,
    M, N, K,
    sa_m, sa_k, swp_n, swp_j, sy_m, sy_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    HALF: tl.constexpr = BLOCK_K // 2

    sc = tl.load(s_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)  # [BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ke = k0 + 2 * tl.arange(0, HALF)      # even absolute-K positions
        ko = ke + 1                            # odd absolute-K positions
        a_e = tl.load(a_ptr + offs_m[:, None] * sa_m + ke[None, :] * sa_k,
                      mask=(offs_m[:, None] < M) & (ke[None, :] < K), other=0.0).to(tl.float32)
        a_o = tl.load(a_ptr + offs_m[:, None] * sa_m + ko[None, :] * sa_k,
                      mask=(offs_m[:, None] < M) & (ko[None, :] < K), other=0.0).to(tl.float32)
        jcol = (k0 // 2) + tl.arange(0, HALF)  # packed byte columns for this tile
        b = tl.load(wp_ptr + offs_n[:, None] * swp_n + jcol[None, :] * swp_j,
                    mask=(offs_n[:, None] < N) & (jcol[None, :] < (K // 2)), other=0).to(tl.int32)
        lo = (b & 0xF) - 8                     # [BLOCK_N, HALF] even-K codes
        hi = ((b >> 4) & 0xF) - 8              # [BLOCK_N, HALF] odd-K codes
        w_e = lo.to(tl.float32) * sc[:, None]
        w_o = hi.to(tl.float32) * sc[:, None]
        acc += tl.dot(a_e, tl.trans(w_e))
        acc += tl.dot(a_o, tl.trans(w_o))
    y = acc.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs_m[:, None] * sy_m + offs_n[None, :] * sy_n, y,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul(a: torch.Tensor, w_packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    N = w_packed.shape[0]
    y = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    _w4a16_kernel[grid](
        a, w_packed, scale, y,
        M, N, K,
        a.stride(0), a.stride(1), w_packed.stride(0), w_packed.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, num_warps=4,
    )
    return y
