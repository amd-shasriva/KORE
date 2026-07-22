"""GENERATED breadth top-k values seed (fp16). x[M,N] -> top-k values per row.
Naive: load the row, iteratively pull the running max K times (tl.max/argmax with
masking of the just-taken lane). O(K*N) - cheap - and CORRECT for the returned
VALUES (descending, ties are value-identical). tl.float16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _topk_values_kernel(x_ptr, o_ptr, sx, so, N, K, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * sx + offs, mask=mask, other=-float("inf")).to(tl.float32)
    for k in range(0, K):
        v = tl.max(x, axis=0)
        j = tl.argmax(x, axis=0)
        tl.store(o_ptr + row * so + k, v.to(tl.float16))
        x = tl.where(offs == j, -float("inf"), x)


def topk_values(x: torch.Tensor, k: int = 8) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty((M, k), device=x.device, dtype=x.dtype)
    _topk_values_kernel[(M,)](x, o, x.stride(0), o.stride(0), N, k,
                              BLOCK_N=triton.next_power_of_2(N), num_warps=4)
    return o
