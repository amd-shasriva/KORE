"""GENERATED breadth red_argmin seed (fp16). x[M,N] -> the min INDEX (int64),
first-occurrence on ties. Streaming fp32 running min + its index across blocks
(strict comparison keeps the earliest winner). int64 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_argmin_kernel(x_ptr, o_ptr, sx, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    best = float("inf")
    best_idx = 0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=float("inf")).to(tl.float32)
        blk = tl.min(x, axis=0)
        blk_idx = start + tl.argmin(x, axis=0)
        take = blk < best
        best_idx = tl.where(take, blk_idx, best_idx)
        best = tl.where(take, blk, best)
    tl.store(o_ptr + row, best_idx.to(tl.int64))


def red_argmin(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty((M,), device=x.device, dtype=torch.int64)
    BLOCK_N = 1024
    _red_argmin_kernel[(M,)](x, o, x.stride(0), N, BLOCK_N=BLOCK_N, num_warps=8)
    return o
