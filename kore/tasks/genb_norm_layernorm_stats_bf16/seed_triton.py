"""GENERATED breadth norm_layernorm_stats seed (bf16) - LayerNorm returning (y, mean, rstd)."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _norm_layernorm_stats_kernel(x_ptr, w_ptr, b_ptr, y_ptr, m_ptr, r_ptr, sm, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * sm + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * sm + offs, (xc * rstd * w + b).to(tl.bfloat16), mask=mask)
    tl.store(m_ptr + row, mean)
    tl.store(r_ptr + row, rstd)


def norm_layernorm_stats(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-06):
    M, N = x.shape
    y = torch.empty_like(x)
    mean = torch.empty((M,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((M,), device=x.device, dtype=torch.float32)
    _norm_layernorm_stats_kernel[(M,)](x, weight, bias, y, mean, rstd, x.stride(0), N, eps,
                       BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return y, mean, rstd
