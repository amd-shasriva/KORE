"""GENERATED breadth red_z_loss seed (fp16). x[M,N] -> a per-row log-sum-exp
reduction. Streaming max-subtracted log-sum-exp in fp32 (numerically stable for
large inputs): one online pass tracks the running max m and rescaled sum s, then
lse = m + log(s). tl.float16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_z_loss_kernel(x_ptr, o_ptr, sx, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    m = -float("inf")
    s = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=-float("inf")).to(tl.float32)
        blk = tl.max(x, axis=0)
        new_m = tl.maximum(m, blk)
        s = s * tl.exp(m - new_m) + tl.sum(tl.where(mask, tl.exp(x - new_m), 0.0), axis=0)
        m = new_m
    lse = m + tl.log(s)
    v = 0.0001 * lse * lse
    tl.store(o_ptr + row, v.to(tl.float16))


def red_z_loss(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty((M,), device=x.device, dtype=x.dtype)
    BLOCK_N = 1024
    _red_z_loss_kernel[(M,)](x, o, x.stride(0), N, BLOCK_N=BLOCK_N, num_warps=8)
    return o
