"""GENERATED breadth norm_l2norm seed (fp32) - row L2-normalize x / max(||x||, eps)."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _norm_l2norm_kernel(x_ptr, y_ptr, sm, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * sm + offs, mask=mask, other=0.0).to(tl.float32)
    n = tl.sqrt(tl.sum(x * x, axis=0))
    denom = tl.maximum(n, eps)
    tl.store(y_ptr + row * sm + offs, (x / denom).to(tl.float32), mask=mask)


def norm_l2norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    _norm_l2norm_kernel[(M,)](x, y, x.stride(0), N, eps, BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return y
