"""GENERATED breadth norm_batchnorm_stats seed (bf16) - BatchNorm (train stats). Kernel 1:
per-channel fp32 mean/var reduced across (N, L) - the cross-batch reduction. Kernel 2:
normalize + affine per element. tl.bfloat16 store. Returns (y, mean, rstd)."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _norm_batchnorm_stats_stats_kernel(x_ptr, m_ptr, r_ptr, N, C, L, eps, BLOCK: tl.constexpr):
    c = tl.program_id(0)
    s = 0.0
    ss = 0.0
    for n in range(0, N):
        base = n * C * L + c * L
        for start in range(0, L, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            m = offs < L
            x = tl.load(x_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
            s += tl.sum(x, axis=0)
            ss += tl.sum(x * x, axis=0)
    cnt = N * L
    mean = s / cnt
    var = ss / cnt - mean * mean
    tl.store(m_ptr + c, mean)
    tl.store(r_ptr + c, 1.0 / tl.sqrt(var + eps))


@triton.jit
def _norm_batchnorm_stats_apply_kernel(x_ptr, m_ptr, r_ptr, w_ptr, b_ptr, y_ptr, C, L, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    c = pid % C
    base = pid * L
    mean = tl.load(m_ptr + c)
    rstd = tl.load(r_ptr + c)
    w = tl.load(w_ptr + c).to(tl.float32)
    b = tl.load(b_ptr + c).to(tl.float32)
    for start in range(0, L, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < L
        x = tl.load(x_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + base + offs, ((x - mean) * rstd * w + b).to(tl.bfloat16), mask=m)


def norm_batchnorm_stats(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-06):
    N, C, L = x.shape
    xc = x.contiguous()
    y = torch.empty_like(xc)
    mean = torch.empty((C,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((C,), device=x.device, dtype=torch.float32)
    _norm_batchnorm_stats_stats_kernel[(C,)](xc, mean, rstd, N, C, L, eps, BLOCK=1024, num_warps=4)
    _norm_batchnorm_stats_apply_kernel[(N * C,)](xc, mean, rstd, weight, bias, y, C, L, BLOCK=1024, num_warps=4)
    return y, mean, rstd
