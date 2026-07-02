"""Seed fused Triton bf16 add-RMSNorm for gfx942.

Exposes ``fused_add_rmsnorm(x, residual, weight, eps) -> (y, added)``:
    added = x + residual          (the new residual carried to the next layer)
    y     = RMSNorm(added) * w

One program per row; fp32 accumulation. A correct baseline the KORE policy
learns to edit/optimize against the vendor serving bar.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_rmsnorm_kernel(
    x_ptr, res_ptr, w_ptr, y_ptr, added_ptr,
    stride_m,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    base = row * stride_m
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(res_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    added = x + r
    tl.store(added_ptr + base + offs, added.to(tl.bfloat16), mask=mask)
    var = tl.sum(added * added, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = added * rstd * w
    tl.store(y_ptr + base + offs, y.to(tl.bfloat16), mask=mask)


def fused_add_rmsnorm(x, residual, weight, eps: float = 1e-6):
    M, N = x.shape
    y = torch.empty_like(x)
    added = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(N)
    _fused_add_rmsnorm_kernel[(M,)](
        x, residual, weight, y, added,
        x.stride(0),
        N, eps,
        BLOCK_N=BLOCK_N, num_warps=8,
    )
    return y, added
