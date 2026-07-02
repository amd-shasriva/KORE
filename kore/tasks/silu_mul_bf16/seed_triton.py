"""Seed Triton bf16 gated-MLP activation for gfx942.

Exposes ``silu_mul(x) -> out`` where x is (M, 2*N):
    gate, up = x[:, :N], x[:, N:]
    out      = silu(gate) * up      (M, N)

2D tiled grid; fp32 math for the SiLU sigmoid. A correct baseline the KORE
policy learns to edit/optimize against the vendor serving bar.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_mul_kernel(
    x_ptr, y_ptr,
    stride_xm, stride_ym,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1)
    offs = col * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    gate = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + row * stride_xm + N + offs, mask=mask, other=0.0).to(tl.float32)
    silu = gate * tl.sigmoid(gate)
    out = silu * up
    tl.store(y_ptr + row * stride_ym + offs, out.to(tl.bfloat16), mask=mask)


def silu_mul(x: torch.Tensor) -> torch.Tensor:
    M, two_n = x.shape
    N = two_n // 2
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    BLOCK_N = 1024
    grid = (M, triton.cdiv(N, BLOCK_N))
    _silu_mul_kernel[grid](
        x, y,
        x.stride(0), y.stride(0),
        N,
        BLOCK_N=BLOCK_N, num_warps=4,
    )
    return y
