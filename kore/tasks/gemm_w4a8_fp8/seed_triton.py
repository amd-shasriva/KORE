"""Seed W4A8 GEMM (int4 per-channel weight, fp8 per-token activation) for gfx950 / CDNA4.

Exposes ``gemm(xq, x_scale, w_packed, w_scale) -> y`` where xq is [M,K] fp8, x_scale is
[M,1] fp32 (per-token), w_packed is [N,K//2] uint8 (2 int4 codes/byte along K), w_scale is
[N,1] fp32 (per-output-channel), y is [M,N] bf16:
    y[m,n] = x_scale[m] * w_scale[n] * sum_k xq[m,k] * (code(n,k) - 8)

The fp8 activation and int4 weight codes are up-converted to fp32 and accumulated RAW; the
per-token and per-channel scales (scalars per row / per col) are applied to the fp32
accumulator at the end (so they compose on the correct axes). The packed weight stores
even-K in the low nibble and odd-K in the high nibble, so a tile is dequantized as TWO
half-K operands and accumulated with two ``tl.dot`` calls (with the fp8 activation loaded
on the matching even/odd K columns). A correct starter the KORE policy optimizes to beat
the bf16 bar.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _w4a8_kernel(
    a_ptr, wp_ptr, xs_ptr, ws_ptr, y_ptr, M, N, K,
    sa_m, sa_k, swp_n, swp_j, sy_m, sy_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    HALF: tl.constexpr = BLOCK_K // 2

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ke = k0 + 2 * tl.arange(0, HALF)          # even absolute-K positions
        ko = ke + 1                                # odd absolute-K positions
        a_e = tl.load(a_ptr + offs_m[:, None] * sa_m + ke[None, :] * sa_k,
                      mask=(offs_m[:, None] < M) & (ke[None, :] < K), other=0.0).to(tl.float32)
        a_o = tl.load(a_ptr + offs_m[:, None] * sa_m + ko[None, :] * sa_k,
                      mask=(offs_m[:, None] < M) & (ko[None, :] < K), other=0.0).to(tl.float32)
        jcol = (k0 // 2) + tl.arange(0, HALF)      # packed weight byte columns
        b = tl.load(wp_ptr + offs_n[:, None] * swp_n + jcol[None, :] * swp_j,
                    mask=(offs_n[:, None] < N) & (jcol[None, :] < (K // 2)), other=0).to(tl.int32)
        w_e = ((b & 0xF) - 8).to(tl.float32)       # [BLOCK_N, HALF] even-K values
        w_o = (((b >> 4) & 0xF) - 8).to(tl.float32)  # [BLOCK_N, HALF] odd-K values
        acc += tl.dot(a_e, tl.trans(w_e))
        acc += tl.dot(a_o, tl.trans(w_o))

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    xs = tl.load(xs_ptr + offs_cm, mask=offs_cm < M, other=0.0).to(tl.float32)
    ws = tl.load(ws_ptr + offs_cn, mask=offs_cn < N, other=0.0).to(tl.float32)
    acc = acc * xs[:, None] * ws[None, :]
    tl.store(y_ptr + offs_cm[:, None] * sy_m + offs_cn[None, :] * sy_n,
             acc.to(tl.bfloat16), mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def gemm(xq: torch.Tensor, x_scale: torch.Tensor,
         w_packed: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    M, K = xq.shape
    N = w_packed.shape[0]
    y = torch.empty((M, N), device=xq.device, dtype=torch.bfloat16)
    xs = x_scale.reshape(-1).contiguous()   # [M]
    ws = w_scale.reshape(-1).contiguous()   # [N]
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    _w4a8_kernel[grid](
        xq, w_packed, xs, ws, y, M, N, K,
        xq.stride(0), xq.stride(1), w_packed.stride(0), w_packed.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, num_warps=4,
    )
    return y
