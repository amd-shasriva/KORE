"""GENERATED breadth red_rms seed (fp16). x[M,N] -> a per-row additive reduction
(fp32 accumulate) with a scalar post-op. tl.float16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_rms_kernel(x_ptr, o_ptr, sx, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    acc = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(tl.where(mask, x * x, 0.0), axis=0)
    v = tl.sqrt(acc / N)
    tl.store(o_ptr + row, v.to(tl.float16))


def red_rms(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty((M,), device=x.device, dtype=x.dtype)
    BLOCK_N = 1024
    _red_rms_kernel[(M,)](x, o, x.stride(0), N, BLOCK_N=BLOCK_N, num_warps=8)
    return o
