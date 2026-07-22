"""GENERATED breadth norm_rmsnorm_swiglu seed (fp16) - RMSNorm(over 2H) then SwiGLU gate
silu(a)*b on the two halves. One program/row, fp32 reduction, tl.float16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _norm_rmsnorm_swiglu_kernel(x_ptr, w_ptr, y_ptr, sxm, sym, TWOH, H, eps,
                 BLOCK2: tl.constexpr, BLOCKH: tl.constexpr):
    row = tl.program_id(0)
    offs2 = tl.arange(0, BLOCK2)
    m2 = offs2 < TWOH
    x = tl.load(x_ptr + row * sxm + offs2, mask=m2, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / TWOH
    rstd = 1.0 / tl.sqrt(var + eps)
    offh = tl.arange(0, BLOCKH)
    mh = offh < H
    a = tl.load(x_ptr + row * sxm + offh, mask=mh, other=0.0).to(tl.float32)
    wa = tl.load(w_ptr + offh, mask=mh, other=0.0).to(tl.float32)
    b = tl.load(x_ptr + row * sxm + H + offh, mask=mh, other=0.0).to(tl.float32)
    wb = tl.load(w_ptr + H + offh, mask=mh, other=0.0).to(tl.float32)
    an = a * rstd * wa
    bn = b * rstd * wb
    out = (an * tl.sigmoid(an)) * bn
    tl.store(y_ptr + row * sym + offh, out.to(tl.float16), mask=mh)


def norm_rmsnorm_swiglu(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-06) -> torch.Tensor:
    M, TWOH = x.shape
    H = TWOH // 2
    y = torch.empty((M, H), device=x.device, dtype=x.dtype)
    _norm_rmsnorm_swiglu_kernel[(M,)](x, weight, y, x.stride(0), y.stride(0), TWOH, H, eps,
                       BLOCK2=triton.next_power_of_2(TWOH),
                       BLOCKH=triton.next_power_of_2(H), num_warps=8)
    return y
