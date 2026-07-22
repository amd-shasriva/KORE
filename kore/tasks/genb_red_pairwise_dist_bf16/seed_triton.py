"""GENERATED breadth red_pairwise_dist seed (bf16). a[M,N], b[M,N] -> the per-row
Euclidean distance ||a-b||_2. Single streaming fp32 sum of squared differences,
then sqrt. tl.bfloat16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_pairwise_dist_kernel(a_ptr, b_ptr, o_ptr, sa, sb, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    acc = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        a = tl.load(a_ptr + row * sa + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + row * sb + offs, mask=mask, other=0.0).to(tl.float32)
        d = a - b
        acc += tl.sum(tl.where(mask, d * d, 0.0), axis=0)
    tl.store(o_ptr + row, tl.sqrt(acc).to(tl.bfloat16))


def red_pairwise_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, N = a.shape
    o = torch.empty((M,), device=a.device, dtype=a.dtype)
    BLOCK_N = 1024
    _red_pairwise_dist_kernel[(M,)](a, b, o, a.stride(0), b.stride(0), N,
                                    BLOCK_N=BLOCK_N, num_warps=8)
    return o
