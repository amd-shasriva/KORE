"""GENERATED breadth red_topk2 seed (fp32). x[M,N] -> the top-2 values per row,
descending. Naive but correct STREAMING threshold selection: 2 passes, each
pulling the running max strictly below the previous winner (O(k*N), the policy
replaces it with a real partial/bitonic top-k). tl.float32 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_topk2_kernel(x_ptr, o_ptr, sx, so, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    prev = float("inf")
    for i in range(0, 2):
        cur = -float("inf")
        for start in range(0, N, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            mask = offs < N
            x = tl.load(x_ptr + row * sx + offs, mask=mask, other=-float("inf")).to(tl.float32)
            cand = tl.where(x < prev, x, -float("inf"))
            cur = tl.maximum(cur, tl.max(cand, axis=0))
        tl.store(o_ptr + row * so + i, cur.to(tl.float32))
        prev = cur


def red_topk2(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty((M, 2), device=x.device, dtype=x.dtype)
    BLOCK_N = 1024
    _red_topk2_kernel[(M,)](x, o, x.stride(0), o.stride(0), N, BLOCK_N=BLOCK_N, num_warps=8)
    return o
