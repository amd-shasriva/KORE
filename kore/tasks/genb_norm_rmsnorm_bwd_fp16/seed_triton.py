"""GENERATED breadth norm_rmsnorm_bwd seed (fp16) - RMSNorm BACKWARD. Per-row dx =
rstd*g - (rstd^3/N)*x*sum(g*x), g = dy*w; dweight = sum_rows dy*xhat via atomic add
(the cross-token reduction). Returns (dx, dweight) fp32."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _norm_rmsnorm_bwd_kernel(x_ptr, w_ptr, dy_ptr, dx_ptr, dw_ptr, sm, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = dy * w
    s = tl.sum(g * x, axis=0)
    dx = rstd * g - (rstd * rstd * rstd / N) * x * s
    tl.store(dx_ptr + base + offs, dx, mask=mask)
    tl.atomic_add(dw_ptr + offs, dy * x * rstd, mask=mask)


def norm_rmsnorm_bwd(x: torch.Tensor, weight: torch.Tensor, dy: torch.Tensor, eps: float = 1e-06):
    M, N = x.shape
    dx = torch.empty((M, N), device=x.device, dtype=torch.float32)
    dw = torch.zeros((N,), device=x.device, dtype=torch.float32)
    _norm_rmsnorm_bwd_kernel[(M,)](x, weight, dy, dx, dw, x.stride(0), N, eps,
                       BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return dx, dw
