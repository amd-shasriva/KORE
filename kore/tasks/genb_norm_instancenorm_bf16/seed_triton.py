"""GENERATED breadth norm_instancenorm seed (bf16) - InstanceNorm. One program per (n, c):
single-pass fp32 sum & sum-of-squares over the spatial L, normalize + affine, tl.bfloat16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _norm_instancenorm_kernel(x_ptr, w_ptr, b_ptr, y_ptr, C, L, eps, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    c = pid % C
    base = pid * L
    s = 0.0
    ss = 0.0
    for start in range(0, L, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < L
        x = tl.load(x_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
        s += tl.sum(x, axis=0)
        ss += tl.sum(x * x, axis=0)
    mean = s / L
    var = ss / L - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + c).to(tl.float32)
    b = tl.load(b_ptr + c).to(tl.float32)
    for start in range(0, L, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < L
        x = tl.load(x_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + base + offs, ((x - mean) * rstd * w + b).to(tl.bfloat16), mask=m)


def norm_instancenorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-06) -> torch.Tensor:
    N, C, L = x.shape
    xc = x.contiguous()
    y = torch.empty_like(xc)
    _norm_instancenorm_kernel[(N * C,)](xc, weight, bias, y, C, L, eps, BLOCK=1024, num_warps=4)
    return y
