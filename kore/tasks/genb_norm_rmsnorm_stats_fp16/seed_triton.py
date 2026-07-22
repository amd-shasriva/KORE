"""GENERATED breadth norm_rmsnorm_stats seed (fp16) - RMSNorm returning (y, rstd)."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _norm_rmsnorm_stats_kernel(x_ptr, w_ptr, y_ptr, r_ptr, sm, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * sm + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * sm + offs, (x * rstd * w).to(tl.float16), mask=mask)
    tl.store(r_ptr + row, rstd)


def norm_rmsnorm_stats(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-06):
    M, N = x.shape
    y = torch.empty_like(x)
    rstd = torch.empty((M,), device=x.device, dtype=torch.float32)
    _norm_rmsnorm_stats_kernel[(M,)](x, weight, y, rstd, x.stride(0), N, eps,
                       BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return y, rstd
