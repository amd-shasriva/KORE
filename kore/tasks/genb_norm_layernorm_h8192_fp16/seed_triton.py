"""GENERATED breadth norm_layernorm_h8192 seed (fp16) - LayerNorm (with bias). One
program/row: fp32 mean+var, affine, tl.float16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _norm_layernorm_h8192_kernel(x_ptr, w_ptr, b_ptr, y_ptr, sm, N, eps, BLOCK_N: tl.constexpr):
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
    out = xc * rstd * w + b
    tl.store(y_ptr + row * sm + offs, out.to(tl.float16), mask=mask)


def norm_layernorm_h8192(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-06) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    _norm_layernorm_h8192_kernel[(M,)](x, weight, bias, y, x.stride(0), N, eps,
                       BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return y
