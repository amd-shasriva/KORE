"""GENERATED breadth argmax-lastdim seed (fp16). x[M,N] -> max value per row.
One program/row: fp32 masked load, tl.max reduction (== value at argmax), tl.float16
store. SNR-safe (returns the VALUE, not the index)."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _argmax_lastdim_kernel(x_ptr, o_ptr, sx, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * sx + offs, mask=mask, other=-float("inf")).to(tl.float32)
    v = tl.max(x, axis=0)
    tl.store(o_ptr + row, v.to(tl.float16))


def argmax_lastdim(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty((M,), device=x.device, dtype=x.dtype)
    _argmax_lastdim_kernel[(M,)](x, o, x.stride(0), N,
                                 BLOCK_N=triton.next_power_of_2(N), num_warps=4)
    return o
