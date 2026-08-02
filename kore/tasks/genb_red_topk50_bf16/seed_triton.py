"""GENERATED breadth red_topk50 seed (bf16). x[M,N] -> the top-50 values per row,
descending. Naive but correct STREAMING threshold selection: 50 passes, each
pulling the running max strictly below the previous winner (O(k*N), the policy
replaces it with a real partial/bitonic top-k).

Each pass ALSO counts how many times the previous winner occurs, because the
answer is the top-50 values WITH MULTIPLICITY (``torch.sort(...)[:50]``), not the
50 largest DISTINCT values. Repeats are not an edge case here: bf16 has few
codes per octave, so thousands of the N values in a row collide exactly, and
skipping duplicates walks far down the distribution. tl.bfloat16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_topk50_kernel(x_ptr, o_ptr, sx, so, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    prev = float("inf")      # last value emitted (+inf primes the first pass)
    used = 0.0               # copies of `prev` already emitted
    for i in range(0, 50):
        nxt = -float("inf")  # largest value strictly below `prev`
        mult = 0.0           # how many times `prev` occurs in this row
        for start in range(0, N, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            mask = offs < N
            x = tl.load(x_ptr + row * sx + offs, mask=mask, other=-float("inf")).to(tl.float32)
            cand = tl.where(x < prev, x, -float("inf"))
            nxt = tl.maximum(nxt, tl.max(cand, axis=0))
            mult += tl.sum(tl.where(mask & (x == prev), 1.0, 0.0), axis=0)
        # Emit another copy of `prev` while any remain, else step down.
        cur = tl.where(used < mult, prev, nxt)
        used = tl.where(cur == prev, used + 1.0, 1.0)
        tl.store(o_ptr + row * so + i, cur.to(tl.bfloat16))
        prev = cur


def red_topk50(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty((M, 50), device=x.device, dtype=x.dtype)
    BLOCK_N = 1024
    _red_topk50_kernel[(M,)](x, o, x.stride(0), o.stride(0), N, BLOCK_N=BLOCK_N, num_warps=8)
    return o
