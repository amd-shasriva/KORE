"""GENERATED breadth red_softmax_bwd seed (fp32). Given the saved forward
y=softmax(x) [M,N] and upstream dy [M,N] -> dx = y*(dy - sum_j y_j dy_j) per row.
Two fp32 passes (the row dot y.dy, then the elementwise combine). tl.float32 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_softmax_bwd_kernel(y_ptr, dy_ptr, dx_ptr, sy, sd, so, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    dot = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        y = tl.load(y_ptr + row * sy + offs, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + row * sd + offs, mask=mask, other=0.0).to(tl.float32)
        dot += tl.sum(tl.where(mask, y * dy, 0.0), axis=0)
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        y = tl.load(y_ptr + row * sy + offs, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + row * sd + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(dx_ptr + row * so + offs, (y * (dy - dot)).to(tl.float32), mask=mask)


def red_softmax_bwd(y: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    M, N = y.shape
    dx = torch.empty_like(y)
    BLOCK_N = 1024
    _red_softmax_bwd_kernel[(M,)](y, dy, dx, y.stride(0), dy.stride(0), dx.stride(0), N,
                                  BLOCK_N=BLOCK_N, num_warps=8)
    return dx
