"""GENERATED seed Triton kernel for the mul_tanh (bf16) fusion.

Pointwise FUSION out = f(a, b) computed in ONE pass. torch-eager runs this as
separate kernels, so a fused kernel saves HBM round-trips -> real speedup headroom.
Regenerate via kore/tasks/generate_ops.py - do not hand-edit.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _mul_tanh_kernel(a_ptr, b_ptr, o_ptr, sa, sb, so, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    offs = col * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    a = tl.load(a_ptr + row * sa + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + row * sb + offs, mask=mask, other=0.0).to(tl.float32)
    o = 2.0 * tl.sigmoid(2.0 * (a * b)) - 1.0
    tl.store(o_ptr + row * so + offs, o.to(tl.bfloat16), mask=mask)


def mul_tanh(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, N = a.shape
    o = torch.empty_like(a)
    BLOCK_N = 1024
    grid = (M, triton.cdiv(N, BLOCK_N))
    _mul_tanh_kernel[grid](a, b, o, a.stride(0), b.stride(0), o.stride(0), N,
                       BLOCK_N=BLOCK_N, num_warps=4)
    return o
