"""GENERATED breadth ssm_lru seed (fp16). Linear Recurrent Unit: complex
diagonal recurrence h_t = lam*h_{t-1} + b_t, lam = nu*exp(i*theta). Complex is
carried as a trailing size-2 (real, imag) axis. One program per (batch, channel)
keeps the fp32 complex state (hr, hi) and scans over L (the policy parallelizes it
via an associative complex scan). lam's (lr, li) are precomputed. tl.float16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _ssm_lru_kernel(x_ptr, lr_ptr, li_ptr, y_ptr, D, L, srow, sl):
    row = tl.program_id(0)
    d = row % D
    lr = tl.load(lr_ptr + d).to(tl.float32)
    li = tl.load(li_ptr + d).to(tl.float32)
    base = row * srow
    hr = 0.0
    hi = 0.0
    for i in range(0, L):
        br = tl.load(x_ptr + base + i * sl + 0).to(tl.float32)
        bi = tl.load(x_ptr + base + i * sl + 1).to(tl.float32)
        hrn = lr * hr - li * hi + br
        hin = li * hr + lr * hi + bi
        hr = hrn
        hi = hin
        tl.store(y_ptr + base + i * sl + 0, hr.to(tl.float16))
        tl.store(y_ptr + base + i * sl + 1, hi.to(tl.float16))


def ssm_lru(x: torch.Tensor, nu: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    B, D, L, _ = x.shape
    lr = (nu * torch.cos(theta)).contiguous()
    li = (nu * torch.sin(theta)).contiguous()
    xf = x.contiguous().reshape(B * D, L, 2)
    y = torch.empty_like(xf)
    _ssm_lru_kernel[(B * D,)](xf, lr, li, y, D, L, xf.stride(0), xf.stride(1), num_warps=1)
    return y.reshape(x.shape)
