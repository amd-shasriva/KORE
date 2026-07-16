"""Seed group-wise ASYMMETRIC int4 weight GEMM (W4A16) for gfx950 / CDNA4.

Exposes ``gemm(a, w_packed, scale, zero) -> y`` where a is [M,K] bf16, w_packed is
[N,K//2] uint8 (2 uint4 codes/byte along K), scale is [N,K//group] fp32, zero is
[N,K//group] uint8, y is [M,N] bf16:
    W_deq[n,k] = (code(n,k) - zero[n, k//group]) * scale[n, k//group]
    y = a @ W_deq^T

The K-tile equals one quant group (BLOCK_K = group inferred from ``K // scale.shape[1]``),
so each tile applies exactly one (scale, zero) pair per output row. The packed layout
stores even-K in the low nibble and odd-K in the high nibble, so a tile is dequantized as
TWO half-K operands (lo=even, hi=odd) and accumulated with two ``tl.dot`` calls. A correct
starter the KORE policy optimizes (fuse the two dots, widen tiles, vectorize byte loads,
support sub-group BLOCK_K) to beat the bf16-weight bar.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _w4a16_group_kernel(
    a_ptr, wp_ptr, s_ptr, z_ptr, y_ptr, M, N, K,
    sa_m, sa_k, swp_n, swp_j, ss_n, ss_g, sz_n, sz_g, sy_m, sy_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    HALF: tl.constexpr = BLOCK_K // 2

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        grp = k0 // BLOCK_K                        # one quant group per K-tile
        sc = tl.load(s_ptr + offs_n * ss_n + grp * ss_g,
                     mask=offs_n < N, other=0.0).to(tl.float32)              # [BLOCK_N]
        zr = tl.load(z_ptr + offs_n * sz_n + grp * sz_g,
                     mask=offs_n < N, other=0).to(tl.float32)               # [BLOCK_N]
        ke = k0 + 2 * tl.arange(0, HALF)          # even absolute-K positions
        ko = ke + 1                                # odd absolute-K positions
        a_e = tl.load(a_ptr + offs_m[:, None] * sa_m + ke[None, :] * sa_k,
                      mask=(offs_m[:, None] < M) & (ke[None, :] < K), other=0.0).to(tl.float32)
        a_o = tl.load(a_ptr + offs_m[:, None] * sa_m + ko[None, :] * sa_k,
                      mask=(offs_m[:, None] < M) & (ko[None, :] < K), other=0.0).to(tl.float32)
        jcol = (k0 // 2) + tl.arange(0, HALF)      # packed byte columns for this tile
        b = tl.load(wp_ptr + offs_n[:, None] * swp_n + jcol[None, :] * swp_j,
                    mask=(offs_n[:, None] < N) & (jcol[None, :] < (K // 2)), other=0).to(tl.int32)
        code_lo = (b & 0xF).to(tl.float32)         # [BLOCK_N, HALF] even-K codes
        code_hi = ((b >> 4) & 0xF).to(tl.float32)  # [BLOCK_N, HALF] odd-K codes
        w_e = (code_lo - zr[:, None]) * sc[:, None]
        w_o = (code_hi - zr[:, None]) * sc[:, None]
        acc += tl.dot(a_e, tl.trans(w_e))
        acc += tl.dot(a_o, tl.trans(w_o))
    y = acc.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs_m[:, None] * sy_m + offs_n[None, :] * sy_n, y,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def gemm(a: torch.Tensor, w_packed: torch.Tensor,
         scale: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    N = w_packed.shape[0]
    group = K // scale.shape[1]                    # BLOCK_K == quant group
    y = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    _w4a16_group_kernel[grid](
        a, w_packed, scale, zero, y, M, N, K,
        a.stride(0), a.stride(1), w_packed.stride(0), w_packed.stride(1),
        scale.stride(0), scale.stride(1), zero.stride(0), zero.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=group, num_warps=4,
    )
    return y
