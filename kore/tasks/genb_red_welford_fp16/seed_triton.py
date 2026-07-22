"""GENERATED breadth red_welford seed (fp16). x[M,N] -> (mean, biased var) per
row via a single-pass chunked Welford merge in fp32 (numerically stable running
mean + M2). tl.float16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_welford_kernel(x_ptr, mean_ptr, var_ptr, sx, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    count = 0.0
    mean = 0.0
    m2 = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)
        cnt = tl.sum(tl.where(mask, 1.0, 0.0), axis=0)
        bsum = tl.sum(tl.where(mask, x, 0.0), axis=0)
        bmean = bsum / cnt
        bm2 = tl.sum(tl.where(mask, (x - bmean) * (x - bmean), 0.0), axis=0)
        new_count = count + cnt
        delta = bmean - mean
        mean = mean + delta * cnt / new_count
        m2 = m2 + bm2 + delta * delta * count * cnt / new_count
        count = new_count
    tl.store(mean_ptr + row, mean.to(tl.float16))
    tl.store(var_ptr + row, (m2 / count).to(tl.float16))


def red_welford(x: torch.Tensor):
    M, N = x.shape
    mean = torch.empty((M,), device=x.device, dtype=x.dtype)
    var = torch.empty((M,), device=x.device, dtype=x.dtype)
    BLOCK_N = 1024
    _red_welford_kernel[(M,)](x, mean, var, x.stride(0), N, BLOCK_N=BLOCK_N, num_warps=8)
    return mean, var
