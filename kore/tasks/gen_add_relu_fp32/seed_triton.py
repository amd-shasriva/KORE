"""GENERATED seed Triton kernel for the add_relu (fp32) binary op.

Elementwise add_relu(a, b), 2D-tiled, fp32 math, tl.float32 store. Regenerate via
kore/tasks/generate_ops.py — do not hand-edit.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _add_relu_kernel(a_ptr, b_ptr, o_ptr, stride_am, stride_bm, stride_om, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    offs = col * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(a_ptr + row * stride_am + offs, mask=mask, other=1.0).to(tl.float32)
    y = tl.load(b_ptr + row * stride_bm + offs, mask=mask, other=1.0).to(tl.float32)
    o = tl.maximum(x + y, 0.0)
    tl.store(o_ptr + row * stride_om + offs, o.to(tl.float32), mask=mask)


def add_relu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, N = a.shape
    o = torch.empty_like(a)
    BLOCK_N = 1024
    grid = (M, triton.cdiv(N, BLOCK_N))
    _add_relu_kernel[grid](a, b, o, a.stride(0), b.stride(0), o.stride(0), N,
                       BLOCK_N=BLOCK_N, num_warps=4)
    return o
