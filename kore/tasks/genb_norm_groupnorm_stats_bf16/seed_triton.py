"""GENERATED breadth norm_groupnorm_stats seed (bf16) - GroupNorm returning (y, mean, rstd) per group."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _norm_groupnorm_stats_kernel(x_ptr, w_ptr, b_ptr, y_ptr, m_ptr, r_ptr, sm, G, WD, eps, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // G
    grp = pid % G
    t = tl.arange(0, BLOCK)
    mask = t < WD
    col = grp * WD + t
    base = row * sm
    x = tl.load(x_ptr + base + col, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / WD
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / WD
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + col, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + col, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + base + col, (xc * rstd * w + b).to(tl.bfloat16), mask=mask)
    tl.store(m_ptr + pid, mean)
    tl.store(r_ptr + pid, rstd)


def norm_groupnorm_stats(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-06):
    M, C = x.shape
    G = 32
    WD = C // G
    y = torch.empty_like(x)
    mean = torch.empty((M, G), device=x.device, dtype=torch.float32)
    rstd = torch.empty((M, G), device=x.device, dtype=torch.float32)
    _norm_groupnorm_stats_kernel[(M * G,)](x, weight, bias, y, mean, rstd, x.stride(0), G, WD, eps,
                           BLOCK=triton.next_power_of_2(WD), num_warps=4)
    return y, mean, rstd
