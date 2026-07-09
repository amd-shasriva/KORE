"""GENERATED seed Triton kernel for the add_mul_relu (fp32) fusion.

Pointwise FUSION out = f(a, b, c) computed in ONE pass (vs torch-eager multi-kernel).
Regenerate via kore/tasks/generate_ops.py — do not hand-edit.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _add_mul_relu_kernel(a_ptr, b_ptr, c_ptr, o_ptr, sa, sb, sc, so, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    offs = col * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    a = tl.load(a_ptr + row * sa + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + row * sb + offs, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(c_ptr + row * sc + offs, mask=mask, other=0.0).to(tl.float32)
    o = tl.maximum((a + b) * c, 0.0)
    tl.store(o_ptr + row * so + offs, o.to(tl.float32), mask=mask)


def add_mul_relu(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    M, N = a.shape
    o = torch.empty_like(a)
    BLOCK_N = 1024
    grid = (M, triton.cdiv(N, BLOCK_N))
    _add_mul_relu_kernel[grid](a, b, c, o, a.stride(0), b.stride(0), c.stride(0), o.stride(0), N,
                       BLOCK_N=BLOCK_N, num_warps=4)
    return o
