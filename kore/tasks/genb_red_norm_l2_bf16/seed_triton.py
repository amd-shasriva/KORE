"""GENERATED breadth red_norm_l2 seed (bf16). x[M,N] -> a per-row additive reduction
(fp32 accumulate) with a scalar post-op. tl.bfloat16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_norm_l2_kernel(x_ptr, o_ptr, sx, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    acc = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(tl.where(mask, x * x, 0.0), axis=0)
    v = tl.sqrt(acc)
    tl.store(o_ptr + row, v.to(tl.bfloat16))


def red_norm_l2(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty((M,), device=x.device, dtype=x.dtype)
    BLOCK_N = 1024
    _red_norm_l2_kernel[(M,)](x, o, x.stride(0), N, BLOCK_N=BLOCK_N, num_warps=8)
    return o
