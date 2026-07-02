"""Seed fused Triton bf16 LayerNorm for gfx942. Exposes ``layernorm(x, w, b, eps)``.

One program per row, single fused pass: fp32 mean + variance accumulation,
rsqrt, affine (weight, bias), bf16 store. Masked loads handle non-pow2 N. A
correct, reasonably-tuned baseline the KORE policy learns to edit/optimize
against the AITER CK serving bar.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    stride_xm, stride_ym,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_xm
    y_row = y_ptr + row * stride_ym
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * w + b
    tl.store(y_row + offs, y.to(tl.bfloat16), mask=mask)


def layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
              eps: float = 1e-5) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(N)
    _layernorm_kernel[(M,)](
        x, weight, bias, y,
        x.stride(0), y.stride(0),
        N, eps,
        BLOCK_N=BLOCK_N, num_warps=8,
    )
    return y
