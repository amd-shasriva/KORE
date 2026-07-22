"""GENERATED breadth norm_rmsnorm_quant_fp8 seed (fp8) - RMSNorm + per-row (per-token) fp8
output quant. Returns (q, scale): scale = amax(normed)/448.0; q = normed/scale."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _norm_rmsnorm_quant_fp8_kernel(x_ptr, w_ptr, q_ptr, s_ptr, sm, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * sm + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    normed = x * rstd * w
    amax = tl.max(tl.abs(normed), axis=0)
    scale = tl.where(amax > 0.0, amax / 448.0, 1.0)
    qv = normed / scale
    tl.store(q_ptr + row * sm + offs, qv.to(tl.float8e4nv), mask=mask)
    tl.store(s_ptr + row, scale)


def norm_rmsnorm_quant_fp8(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-06):
    M, N = x.shape
    q = torch.empty((M, N), device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty((M,), device=x.device, dtype=torch.float32)
    _norm_rmsnorm_quant_fp8_kernel[(M,)](x, weight, q, s, x.stride(0), N, eps,
                       BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return q, s
