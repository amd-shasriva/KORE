"""GENERATED breadth norm_add_rmsnorm seed (bf16) - fused add-residual + RMSNorm. Returns
(y, added): added = x + residual is the NEW residual (fresh tensor)."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _norm_add_rmsnorm_kernel(x_ptr, res_ptr, w_ptr, y_ptr, added_ptr, sm, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(res_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    added = x + r
    tl.store(added_ptr + base + offs, added.to(tl.bfloat16), mask=mask)
    var = tl.sum(added * added, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + base + offs, (added * rstd * w).to(tl.bfloat16), mask=mask)


def norm_add_rmsnorm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-06):
    M, N = x.shape
    y = torch.empty_like(x)
    added = torch.empty_like(x)
    _norm_add_rmsnorm_kernel[(M,)](x, residual, weight, y, added, x.stride(0), N, eps,
                       BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return y, added
