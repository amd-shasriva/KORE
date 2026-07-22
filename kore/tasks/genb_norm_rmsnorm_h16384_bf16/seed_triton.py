"""GENERATED breadth norm_rmsnorm_h16384 seed (bf16) - RMSNorm. One program/row: fp32
mean-square over N, rsqrt, weight, tl.bfloat16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _norm_rmsnorm_h16384_kernel(x_ptr, w_ptr, y_ptr, sm, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * sm + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * sm + offs, (x * rstd * w).to(tl.bfloat16), mask=mask)


def norm_rmsnorm_h16384(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-06) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    _norm_rmsnorm_h16384_kernel[(M,)](x, weight, y, x.stride(0), N, eps,
                       BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return y
