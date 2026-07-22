"""GENERATED breadth red_var seed (bf16). x[M,N] -> a per-row variance/std.
Numerically-stable TWO-pass (Welford-equivalent) reduction: pass 1 the fp32 mean,
pass 2 the fp32 sum of CENTERED squares (avoids the catastrophic cancellation of
E[x^2]-E[x]^2 for large means). tl.bfloat16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_var_kernel(x_ptr, o_ptr, sx, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    s = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)
        s += tl.sum(tl.where(mask, x, 0.0), axis=0)
    mean = s / N
    ss = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)
        d = x - mean
        ss += tl.sum(tl.where(mask, d * d, 0.0), axis=0)
    v = ss / N
    tl.store(o_ptr + row, v.to(tl.bfloat16))


def red_var(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty((M,), device=x.device, dtype=x.dtype)
    BLOCK_N = 1024
    _red_var_kernel[(M,)](x, o, x.stride(0), N, BLOCK_N=BLOCK_N, num_warps=8)
    return o
