"""GENERATED breadth red_layernorm seed (bf16). x[M,N] -> (x-mean)/sqrt(var+eps)
over the last dim (no affine), eps=1e-05. Three fp32 passes (mean, centered var,
write); the centered variance is numerically stable. tl.bfloat16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_layernorm_kernel(x_ptr, o_ptr, sx, so, N, EPS, BLOCK_N: tl.constexpr):
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
    rstd = 1.0 / tl.sqrt(ss / N + EPS)
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(o_ptr + row * so + offs, ((x - mean) * rstd).to(tl.bfloat16), mask=mask)


def red_layernorm(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty_like(x)
    BLOCK_N = 1024
    _red_layernorm_kernel[(M,)](x, o, x.stride(0), o.stride(0), N, 1e-05,
                                BLOCK_N=BLOCK_N, num_warps=8)
    return o
