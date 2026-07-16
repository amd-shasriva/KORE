"""Seed row-SOFTMAX BACKWARD for gfx950 (CDNA4).

Exposes ``softmax_backward(y, dy) -> dx`` with y, dy, dx all [M, N] bf16, where y
is the saved softmax forward output (probabilities) and dy the upstream gradient:
    s_i  = sum_j (dy_{i,j} * y_{i,j})        (per row, over N)
    dx_{i,j} = y_{i,j} * (dy_{i,j} - s_i)

One program per row, two column-streamed passes (pass 1 reduces the row dot
s = <dy, y>; pass 2 writes dx), fp32 math, so any row width N fits regardless of
BLOCK_N. A correct baseline the KORE policy optimizes: cache the row in LDS to
skip the second load, vectorize, tune BLOCK_N / num_warps.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_bwd_kernel(y_ptr, dy_ptr, dx_ptr, sy, sdy, sdx, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    y_row = y_ptr + row * sy
    dy_row = dy_ptr + row * sdy
    dx_row = dx_ptr + row * sdx

    s = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        y = tl.load(y_row + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(dy_row + offs, mask=mask, other=0.0).to(tl.float32)
        s += tl.sum(y * g, axis=0)

    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        y = tl.load(y_row + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(dy_row + offs, mask=mask, other=0.0).to(tl.float32)
        dx = y * (g - s)
        tl.store(dx_row + offs, dx.to(dx_ptr.dtype.element_ty), mask=mask)


def softmax_backward(y: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    M, N = y.shape
    dx = torch.empty_like(y)
    BLOCK_N = 1024 if N > 1024 else triton.next_power_of_2(N)
    _softmax_bwd_kernel[(M,)](
        y, dy, dx,
        y.stride(0), dy.stride(0), dx.stride(0),
        N, BLOCK_N=BLOCK_N, num_warps=8,
    )
    return dx
