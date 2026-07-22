"""GENERATED breadth norm_groupnorm_silu seed (fp16) - GroupNorm + SiLU. One program per
(row, group): fp32 mean+var over the group width, per-channel affine, tl.float16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _norm_groupnorm_silu_kernel(x_ptr, w_ptr, b_ptr, y_ptr, sm, G, WD, eps, BLOCK: tl.constexpr):
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
    v = xc * rstd * w + b
    out = v * tl.sigmoid(v)
    tl.store(y_ptr + base + col, out.to(tl.float16), mask=mask)


def norm_groupnorm_silu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-06) -> torch.Tensor:
    M, C = x.shape
    G = 32
    WD = C // G
    y = torch.empty_like(x)
    _norm_groupnorm_silu_kernel[(M * G,)](x, weight, bias, y, x.stride(0), G, WD, eps,
                           BLOCK=triton.next_power_of_2(WD), num_warps=4)
    return y
