"""Seed MXFP4 a4w4 GEMM (both operands microscaling FP4) for gfx950 / CDNA4.

Exposes ``gemm(a_packed, a_e8m0, w_packed, w_e8m0) -> y`` where a_packed[M,K//2] and
w_packed[N,K//2] are uint8 (2 e2m1 nibbles/byte along K), a_e8m0[M,K//32] and
w_e8m0[N,K//32] are OCP E8M0 shared exponents, y is [M,N] bf16:
    A_deq[m,k] = e2m1(a_code) * 2^(a_e8m0[m,k//32] - 127)
    W_deq[n,k] = e2m1(w_code) * 2^(w_e8m0[n,k//32] - 127)
    y = A_deq @ W_deq^T

The tile is one MX block wide (BLOCK_K=32), so one shared E8M0 scale per row (A) and per
row (W) covers the tile. Each packed byte holds even-K in the low nibble and odd-K in the
high nibble, so a tile dequantizes as TWO half-K operands (lo=even, hi=odd), each decoded
from e2m1 arithmetic (no LUT) and accumulated with two ``tl.dot`` calls. A correct
starter the KORE policy optimizes (widen BLOCK_N, batch 32-blocks per K-tile, use the
native CDNA4 MXFP4 matrix path via scaled tl.dot, tune num_warps) to beat the bf16 bar.
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
    mag_sub = mant * 0.5                    # exp==0 (subnormal): 0 or 0.5
    mag_norm = (1.0 + 0.5 * mant) * tl.exp2((exp - 1).to(tl.float32))
    mag = tl.where(exp == 0, mag_sub, mag_norm)
    return sign * mag


@triton.jit
def _mxfp4_a4w4_kernel(
    ap_ptr, as_ptr, wp_ptr, ws_ptr, y_ptr, M, N, K,
    sap_m, sap_j, sas_m, sas_b, swp_n, swp_j, sws_n, sws_b, sy_m, sy_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    HALF: tl.constexpr = BLOCK_K // 2

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        blk = k0 // BLOCK_K                        # MX block index (BLOCK_K == 32)
        a_e8 = tl.load(as_ptr + offs_m * sas_m + blk * sas_b,
                       mask=offs_m < M, other=127).to(tl.float32)
        w_e8 = tl.load(ws_ptr + offs_n * sws_n + blk * sws_b,
                       mask=offs_n < N, other=127).to(tl.float32)
        a_sc = tl.exp2(a_e8 - 127.0)              # [BLOCK_M]
        w_sc = tl.exp2(w_e8 - 127.0)              # [BLOCK_N]

        jcol = (k0 // 2) + tl.arange(0, HALF)     # packed byte columns for this tile
        ab = tl.load(ap_ptr + offs_m[:, None] * sap_m + jcol[None, :] * sap_j,
                     mask=(offs_m[:, None] < M) & (jcol[None, :] < (K // 2)), other=0).to(tl.int32)
        wb = tl.load(wp_ptr + offs_n[:, None] * swp_n + jcol[None, :] * swp_j,
                     mask=(offs_n[:, None] < N) & (jcol[None, :] < (K // 2)), other=0).to(tl.int32)
        a_e = _e2m1_decode(ab & 0xF) * a_sc[:, None]           # [BLOCK_M, HALF] even-K
        a_o = _e2m1_decode((ab >> 4) & 0xF) * a_sc[:, None]    # [BLOCK_M, HALF] odd-K
        w_e = _e2m1_decode(wb & 0xF) * w_sc[:, None]           # [BLOCK_N, HALF] even-K
        w_o = _e2m1_decode((wb >> 4) & 0xF) * w_sc[:, None]    # [BLOCK_N, HALF] odd-K
        acc += tl.dot(a_e, tl.trans(w_e))
        acc += tl.dot(a_o, tl.trans(w_o))

    y = acc.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs_m[:, None] * sy_m + offs_n[None, :] * sy_n, y,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def gemm(a_packed: torch.Tensor, a_e8m0: torch.Tensor,
         w_packed: torch.Tensor, w_e8m0: torch.Tensor) -> torch.Tensor:
    M = a_packed.shape[0]
    N = w_packed.shape[0]
    K = a_packed.shape[1] * 2
    y = torch.empty((M, N), device=a_packed.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    _mxfp4_a4w4_kernel[grid](
        a_packed, a_e8m0, w_packed, w_e8m0, y, M, N, K,
        a_packed.stride(0), a_packed.stride(1), a_e8m0.stride(0), a_e8m0.stride(1),
        w_packed.stride(0), w_packed.stride(1), w_e8m0.stride(0), w_e8m0.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4,
    )
    return y
