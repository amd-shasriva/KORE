"""Seed fused Triton bf16 RMSNorm for gfx942. Exposes ``rmsnorm(x, w, eps)``.

One program per row, single fused pass: fp32 mean-square accumulation, rsqrt,
scale-and-weight, bf16 store. A correct, reasonably-tuned baseline the KORE
policy learns to edit/optimize against the vendor serving bar.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    x_ptr, w_ptr, y_ptr,
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
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(y_row + offs, y.to(tl.bfloat16), mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(N)
    _rmsnorm_kernel[(M,)](
        x, weight, y,
        x.stride(0), y.stride(0),
        N, eps,
        BLOCK_N=BLOCK_N, num_warps=8,
    )
    return y
