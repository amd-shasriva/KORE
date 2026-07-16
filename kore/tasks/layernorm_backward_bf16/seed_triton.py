"""Seed LayerNorm BACKWARD for gfx950 (CDNA4).

Exposes ``layernorm_backward(x, gamma, dy, eps=1e-5) -> (dx, dgamma, dbeta)`` where
x, dy are [M,N] bf16, gamma is [N] bf16, dx is [M,N] bf16, dgamma/dbeta are [N] bf16:
    mean = mean_j(x)   rstd = 1/sqrt(var+eps)   xhat = (x-mean)*rstd
    g = dy*gamma
    dx_j     = rstd*(g_j - mean_j(g) - xhat_j*mean_j(g*xhat))
    dgamma_j = sum_m dy_{m,j}*xhat_{m,j}          (reduce over tokens via atomics)
    dbeta_j  = sum_m dy_{m,j}                      (reduce over tokens via atomics)

One program per row, three column-streamed passes (mean/var; then the two row
reductions c1=<g>, c2=<g,xhat>; then write dx and atomic-accumulate dgamma/dbeta
into fp32 buffers), fp32 math. A correct baseline the KORE policy optimizes:
replace the atomics with a blocked two-stage (dgamma,dbeta) reduction, cache the
row in LDS to skip reloads, tune BLOCK_N / num_warps.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

EPS = 1e-5


@triton.jit
def _layernorm_bwd_kernel(
    x_ptr, w_ptr, dy_ptr, dx_ptr, dgamma_ptr, dbeta_ptr,
    sx, sdy, sdx,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * sx
    dy_row = dy_ptr + row * sdy
    dx_row = dx_ptr + row * sdx

    # Pass 1: mean and variance over the row.
    sum_x = 0.0
    sum_x2 = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)
    mean = sum_x / N
    var = sum_x2 / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Pass 2: row reductions c1 = sum(g), c2 = sum(g*xhat), where g = dy*gamma.
    c1 = 0.0
    c2 = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(dy_row + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        xhat = (x - mean) * rstd
        gg = g * w
        c1 += tl.sum(gg, axis=0)
        c2 += tl.sum(gg * xhat, axis=0)
    c1 = c1 / N
    c2 = c2 / N

    # Pass 3: write dx, atomic-accumulate dgamma/dbeta.
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(dy_row + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        xhat = (x - mean) * rstd
        gg = g * w
        dx = rstd * (gg - c1 - xhat * c2)
        tl.store(dx_row + offs, dx.to(dx_ptr.dtype.element_ty), mask=mask)
        tl.atomic_add(dgamma_ptr + offs, g * xhat, mask=mask)
        tl.atomic_add(dbeta_ptr + offs, g, mask=mask)


def layernorm_backward(x: torch.Tensor, gamma: torch.Tensor, dy: torch.Tensor,
                       eps: float = EPS):
    M, N = x.shape
    dx = torch.empty_like(x)
    dgamma = torch.zeros((N,), device=x.device, dtype=torch.float32)
    dbeta = torch.zeros((N,), device=x.device, dtype=torch.float32)
    BLOCK_N = 1024 if N > 1024 else triton.next_power_of_2(N)
    _layernorm_bwd_kernel[(M,)](
        x, gamma, dy, dx, dgamma, dbeta,
        x.stride(0), dy.stride(0), dx.stride(0),
        N, eps,
        BLOCK_N=BLOCK_N, num_warps=8,
    )
    return dx, dgamma.to(gamma.dtype), dbeta.to(gamma.dtype)
