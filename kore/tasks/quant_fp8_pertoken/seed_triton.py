"""Seed Triton dynamic per-token fp8 quant for gfx942 (CDNA3).

Exposes ``quant(x) -> (xq, scale)`` where x is [M,N] bf16, xq is [M,N] fp8
e4m3fnuz and scale is [M,1] fp32:
    scale[m] = rowamax[m] / FP8_MAX
    xq[m]    = clamp(x[m] / scale[m], +/-FP8_MAX) -> fp8

One program per row, streamed over column tiles (two passes: rowamax, then
quantize+store) so wide rows work without huge register pressure. A correct
baseline the KORE policy learns to edit/optimize against the AITER fp8 quant
serving bar.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# gfx942 fp8 e4m3fnuz max magnitude.
FP8_MAX = 240.0


@triton.jit
def _quant_kernel(
    x_ptr, y_ptr, s_ptr,
    stride_xm, stride_ym,
    N,
    FP8_MAX: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_xm
    y_row = y_ptr + row * stride_ym

    amax = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x), axis=0))

    amax = tl.maximum(amax, 1e-12)
    scale = amax / FP8_MAX
    tl.store(s_ptr + row, scale)

    inv = 1.0 / scale
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        q = x * inv
        q = tl.minimum(tl.maximum(q, -FP8_MAX), FP8_MAX)
        tl.store(y_row + offs, q.to(y_ptr.dtype.element_ty), mask=mask)


def quant(x: torch.Tensor):
    M, N = x.shape
    xq = torch.empty((M, N), device=x.device, dtype=torch.float8_e4m3fnuz)
    scale = torch.empty((M, 1), device=x.device, dtype=torch.float32)
    BLOCK_N = 1024 if N > 1024 else triton.next_power_of_2(N)
    _quant_kernel[(M,)](
        x, xq, scale,
        x.stride(0), xq.stride(0),
        N,
        FP8_MAX=FP8_MAX,
        BLOCK_N=BLOCK_N, num_warps=8,
    )
    return xq, scale
