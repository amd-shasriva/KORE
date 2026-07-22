"""GENERATED breadth norm_groupnorm_bwd seed (fp16) - GroupNorm BACKWARD. Per (row, group)
LayerNorm-style dx over the group width; dweight = sum_rows dy*xhat and dbias =
sum_rows dy per channel via atomic add. Returns (dx, dweight, dbias) fp32."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _norm_groupnorm_bwd_kernel(x_ptr, w_ptr, dy_ptr, dx_ptr, dw_ptr, db_ptr, sm, G, WD, eps, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // G
    grp = pid % G
    t = tl.arange(0, BLOCK)
    mask = t < WD
    col = grp * WD + t
    base = row * sm
    x = tl.load(x_ptr + base + col, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + base + col, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + col, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / WD
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / WD
    rstd = 1.0 / tl.sqrt(var + eps)
    xhat = xc * rstd
    g = dy * w
    sg = tl.sum(g, axis=0) / WD
    sgx = tl.sum(g * xhat, axis=0) / WD
    dx = rstd * (g - sg - xhat * sgx)
    tl.store(dx_ptr + base + col, dx, mask=mask)
    tl.atomic_add(dw_ptr + col, dy * xhat, mask=mask)
    tl.atomic_add(db_ptr + col, dy, mask=mask)


def norm_groupnorm_bwd(x: torch.Tensor, weight: torch.Tensor, dy: torch.Tensor, eps: float = 1e-06):
    M, C = x.shape
    G = 32
    WD = C // G
    dx = torch.empty((M, C), device=x.device, dtype=torch.float32)
    dw = torch.zeros((C,), device=x.device, dtype=torch.float32)
    db = torch.zeros((C,), device=x.device, dtype=torch.float32)
    _norm_groupnorm_bwd_kernel[(M * G,)](x, weight, dy, dx, dw, db, x.stride(0), G, WD, eps,
                           BLOCK=triton.next_power_of_2(WD), num_warps=4)
    return dx, dw, db
