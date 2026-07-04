"""GENERATED seed Triton kernel for the gelu_tanh (fp16) activation.

Elementwise gelu_tanh, 2D-tiled, fp32 math, tl.float16 store. A correct-but-naive starting
point the KORE policy learns to optimize against the framework production baseline.
Regenerate via kore/tasks/generate_ops.py — do not hand-edit.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gelu_tanh_kernel(x_ptr, y_ptr, stride_xm, stride_ym, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    offs = col * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    y = 0.5 * x * (1.0 + (2.0 * tl.sigmoid(2.0 * (0.7978845608028654 * (x + 0.044715 * x * x * x))) - 1.0))
    tl.store(y_ptr + row * stride_ym + offs, y.to(tl.float16), mask=mask)


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK_N = 1024
    grid = (M, triton.cdiv(N, BLOCK_N))
    _gelu_tanh_kernel[grid](x, y, x.stride(0), y.stride(0), N, BLOCK_N=BLOCK_N, num_warps=4)
    return y
