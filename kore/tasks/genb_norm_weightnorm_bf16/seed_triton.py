"""GENERATED breadth norm_weightnorm seed (bf16) - weight normalization g * v / ||v|| per row."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _norm_weightnorm_kernel(v_ptr, g_ptr, y_ptr, sm, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    v = tl.load(v_ptr + row * sm + offs, mask=mask, other=0.0).to(tl.float32)
    n = tl.sqrt(tl.sum(v * v, axis=0))
    g = tl.load(g_ptr + row).to(tl.float32)
    tl.store(y_ptr + row * sm + offs, (v * (g / n)).to(tl.bfloat16), mask=mask)


def norm_weightnorm(v: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    M, N = v.shape
    y = torch.empty_like(v)
    _norm_weightnorm_kernel[(M,)](v, g, y, v.stride(0), N, BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return y
