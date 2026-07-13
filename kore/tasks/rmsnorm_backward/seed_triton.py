"""Seed RMSNorm BACKWARD for gfx942 (CDNA3).

Exposes ``rmsnorm_backward(x, w, dy) -> (dx, dw)`` where x,dy are [M,N] bf16,
w is [N] bf16, dx is [M,N] bf16 and dw is [N] bf16:
    r    = rsqrt(mean(x^2) + eps)                   (per row)
    c    = sum_j (dy_j * w_j * x_j)                 (per row)
    dx_j = r*w_j*dy_j - (r^3 * x_j * c) / N
    dw_j = sum_m (dy_{m,j} * x_{m,j} * r_m)         (reduce over tokens via atomics)

One program per row, two column-streamed passes (reduce ss+c, then write dx and
atomic-accumulate dw into an fp32 buffer). A correct baseline the KORE policy
optimizes: replace the atomic dw with a blocked two-stage reduction, cache the
row in LDS to skip the second load, tune BLOCK_N / num_warps.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

EPS = 1e-6


@triton.jit
def _rmsnorm_bwd_kernel(
    x_ptr, w_ptr, dy_ptr, dx_ptr, dw_ptr,
    sx, sdy, sdx,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * sx
    dy_row = dy_ptr + row * sdy
    dx_row = dx_ptr + row * sdx

    ss = 0.0
    c = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(dy_row + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        ss += tl.sum(x * x, axis=0)
        c += tl.sum(g * w * x, axis=0)
    r = 1.0 / tl.sqrt(ss / N + eps)
    r3c_over_n = (r * r * r) * c / N

    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(dy_row + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        dx = r * w * g - r3c_over_n * x
        tl.store(dx_row + offs, dx.to(dx_ptr.dtype.element_ty), mask=mask)
        tl.atomic_add(dw_ptr + offs, g * x * r, mask=mask)


def rmsnorm_backward(x: torch.Tensor, w: torch.Tensor, dy: torch.Tensor, eps: float = EPS):
    M, N = x.shape
    dx = torch.empty_like(x)
    dw = torch.zeros((N,), device=x.device, dtype=torch.float32)
    BLOCK_N = 1024 if N > 1024 else triton.next_power_of_2(N)
    _rmsnorm_bwd_kernel[(M,)](
        x, w, dy, dx, dw,
        x.stride(0), dy.stride(0), dx.stride(0),
        N, eps,
        BLOCK_N=BLOCK_N, num_warps=8,
    )
    return dx, dw.to(w.dtype)
