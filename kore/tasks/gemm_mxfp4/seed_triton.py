"""Seed MXFP4 weight-only GEMM for gfx950 / CDNA4 (MI350X / MI355X).

Exposes ``matmul(a, w_packed, scale_e8m0) -> y`` where a is [M,K] bf16, w_packed is
[N,K//2] uint8 (2 e2m1 nibbles/byte along K), scale_e8m0 is [N,K//32] uint8 (OCP
E8M0 shared exponents), y is [M,N] bf16:
    W_deq[n,k] = e2m1(code[n,k]) * 2^(scale_e8m0[n, k//32] - 127)
    y = a @ W_deq^T

The tile is one MXFP4 block wide (BLOCK_K=32), so one shared E8M0 scale per row
covers the tile. Each byte holds even-K in the low nibble and odd-K in the high
nibble, so the tile dequantizes as TWO half-K operands (lo=even, hi=odd), each
decoded from e2m1 arithmetic (no LUT) and accumulated with two ``tl.dot`` calls.
A correct baseline the KORE policy optimizes (widen BLOCK_N, batch multiple
32-blocks per K-tile, vectorize the byte loads, use the native CDNA4 MXFP4 matrix
path via tl.dot on the packed operands, tune num_warps) to beat the bf16-weight bar.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _e2m1_decode(code):
    """4-bit e2m1 code (int) -> fp32 value. Magnitudes {0,.5,1,1.5,2,3,4,6}."""
    idx = code & 0x7
    sign = tl.where((code & 0x8) != 0, -1.0, 1.0)
    exp = idx // 2                         # 0,0,1,1,2,2,3,3
    mant = (idx & 1).to(tl.float32)
    mag_sub = mant * 0.5                   # exp==0 (subnormal): 0 or 0.5
    mag_norm = (1.0 + 0.5 * mant) * tl.exp2((exp - 1).to(tl.float32))
    mag = tl.where(exp == 0, mag_sub, mag_norm)
    return sign * mag


@triton.jit
def _mxfp4_kernel(
    a_ptr, wp_ptr, s_ptr, y_ptr,
    M, N, K,
    sa_m, sa_k, swp_n, swp_j, ss_n, ss_b, sy_m, sy_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    HALF: tl.constexpr = BLOCK_K // 2

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        blk = k0 // BLOCK_K                       # MXFP4 block index (BLOCK_K==32)
        e8 = tl.load(s_ptr + offs_n * ss_n + blk * ss_b,
                     mask=offs_n < N, other=127).to(tl.float32)
        sc = tl.exp2(e8 - 127.0)                  # [BLOCK_N] block scale per row

        ke = k0 + 2 * tl.arange(0, HALF)          # even absolute-K positions
        ko = ke + 1                                # odd absolute-K positions
        a_e = tl.load(a_ptr + offs_m[:, None] * sa_m + ke[None, :] * sa_k,
                      mask=(offs_m[:, None] < M) & (ke[None, :] < K), other=0.0).to(tl.float32)
        a_o = tl.load(a_ptr + offs_m[:, None] * sa_m + ko[None, :] * sa_k,
                      mask=(offs_m[:, None] < M) & (ko[None, :] < K), other=0.0).to(tl.float32)

        jcol = (k0 // 2) + tl.arange(0, HALF)      # packed byte columns for this tile
        b = tl.load(wp_ptr + offs_n[:, None] * swp_n + jcol[None, :] * swp_j,
                    mask=(offs_n[:, None] < N) & (jcol[None, :] < (K // 2)), other=0).to(tl.int32)
        lo = b & 0xF                               # [BLOCK_N, HALF] even-K codes
        hi = (b >> 4) & 0xF                        # [BLOCK_N, HALF] odd-K codes
        w_e = _e2m1_decode(lo) * sc[:, None]
        w_o = _e2m1_decode(hi) * sc[:, None]
        acc += tl.dot(a_e, tl.trans(w_e))
        acc += tl.dot(a_o, tl.trans(w_o))

    y = acc.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs_m[:, None] * sy_m + offs_n[None, :] * sy_n, y,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul(a: torch.Tensor, w_packed: torch.Tensor, scale_e8m0: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    N = w_packed.shape[0]
    y = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    _mxfp4_kernel[grid](
        a, w_packed, scale_e8m0, y,
        M, N, K,
        a.stride(0), a.stride(1), w_packed.stride(0), w_packed.stride(1),
        scale_e8m0.stride(0), scale_e8m0.stride(1), y.stride(0), y.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4,
    )
    return y
