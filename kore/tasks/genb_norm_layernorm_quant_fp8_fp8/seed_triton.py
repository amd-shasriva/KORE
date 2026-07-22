"""GENERATED breadth norm_layernorm_quant_fp8 seed (fp8) - LayerNorm + per-row fp8 output quant.
Returns (q, scale)."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _norm_layernorm_quant_fp8_kernel(x_ptr, w_ptr, b_ptr, q_ptr, s_ptr, sm, N, eps, BLOCK_N: tl.constexpr):
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
    normed = xc * rstd * w + b
    amax = tl.max(tl.abs(normed), axis=0)
    scale = tl.where(amax > 0.0, amax / 448.0, 1.0)
    qv = normed / scale
    tl.store(q_ptr + row * sm + offs, qv.to(tl.float8e4nv), mask=mask)
    tl.store(s_ptr + row, scale)


def norm_layernorm_quant_fp8(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-06):
    M, N = x.shape
    q = torch.empty((M, N), device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty((M,), device=x.device, dtype=torch.float32)
    _norm_layernorm_quant_fp8_kernel[(M,)](x, weight, bias, q, s, x.stride(0), N, eps,
                       BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return q, s
