"""GENERATED breadth norm_l2norm_bwd seed (fp16) - L2-normalize BACKWARD. Per-row
dx = (dy - y*(y . dy)) / ||x||, y = x/||x||. Returns dx fp32."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _norm_l2norm_bwd_kernel(x_ptr, dy_ptr, dx_ptr, sm, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    n = tl.sqrt(tl.sum(x * x, axis=0))
    y = x / n
    c = tl.sum(y * dy, axis=0)
    tl.store(dx_ptr + base + offs, (dy - y * c) / n, mask=mask)


def norm_l2norm_bwd(x: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    dx = torch.empty((M, N), device=x.device, dtype=torch.float32)
    _norm_l2norm_bwd_kernel[(M,)](x, dy, dx, x.stride(0), N,
                       BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return dx
