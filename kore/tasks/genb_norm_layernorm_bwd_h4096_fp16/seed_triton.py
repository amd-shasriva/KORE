"""GENERATED breadth norm_layernorm_bwd_h4096 seed (fp16) - LayerNorm BACKWARD. Per-row dx =
rstd*(g - mean(g) - xhat*mean(g*xhat)), g = dy*w; dweight = sum_rows dy*xhat and
dbias = sum_rows dy via atomic add. Returns (dx, dweight, dbias) fp32."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _norm_layernorm_bwd_h4096_kernel(x_ptr, w_ptr, dy_ptr, dx_ptr, dw_ptr, db_ptr, sm, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    xhat = xc * rstd
    g = dy * w
    sg = tl.sum(g, axis=0) / N
    sgx = tl.sum(g * xhat, axis=0) / N
    dx = rstd * (g - sg - xhat * sgx)
    tl.store(dx_ptr + base + offs, dx, mask=mask)
    tl.atomic_add(dw_ptr + offs, dy * xhat, mask=mask)
    tl.atomic_add(db_ptr + offs, dy, mask=mask)


def norm_layernorm_bwd_h4096(x: torch.Tensor, weight: torch.Tensor, dy: torch.Tensor, eps: float = 1e-06):
    M, N = x.shape
    dx = torch.empty((M, N), device=x.device, dtype=torch.float32)
    dw = torch.zeros((N,), device=x.device, dtype=torch.float32)
    db = torch.zeros((N,), device=x.device, dtype=torch.float32)
    _norm_layernorm_bwd_h4096_kernel[(M,)](x, weight, dy, dx, dw, db, x.stride(0), N, eps,
                       BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return dx, dw, db
