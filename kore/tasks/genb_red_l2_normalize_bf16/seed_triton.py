"""GENERATED breadth red_l2_normalize seed (bf16). x[M,N] -> a per-row rescaled output.
Two fp32 passes: pass 1 sums squares, pass 2 rescales x by the (rms/l2) factor. tl.bfloat16."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_l2_normalize_kernel(x_ptr, o_ptr, sx, so, N, EPS, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    acc = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(tl.where(mask, x * x, 0.0), axis=0)
    scale = 1.0 / tl.maximum(tl.sqrt(acc), EPS)
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(o_ptr + row * so + offs, (x * scale).to(tl.bfloat16), mask=mask)


def red_l2_normalize(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty_like(x)
    BLOCK_N = 1024
    _red_l2_normalize_kernel[(M,)](x, o, x.stride(0), o.stride(0), N, 1e-12, BLOCK_N=BLOCK_N, num_warps=8)
    return o
