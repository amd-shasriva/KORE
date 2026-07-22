"""GENERATED breadth row-sort seed (bf16). x[M,N] -> ascending sorted rows.
NAIVE + CORRECT selection sort: load the row (pad with +inf), N times pull the
running min into the next output slot and mask it out. O(N^2)/row - a partial
starting point; the teacher is expected to replace it with a bitonic sort. tl.bfloat16."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _sort_lastdim_kernel(x_ptr, o_ptr, sx, so, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * sx + offs, mask=mask, other=float("inf")).to(tl.float32)
    for i in range(0, N):
        v = tl.min(x, axis=0)
        j = tl.argmin(x, axis=0)
        tl.store(o_ptr + row * so + i, v.to(tl.bfloat16))
        x = tl.where(offs == j, float("inf"), x)


def sort_lastdim(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty_like(x)
    _sort_lastdim_kernel[(M,)](x, o, x.stride(0), o.stride(0), N,
                               BLOCK_N=triton.next_power_of_2(N), num_warps=4)
    return o
