"""GENERATED breadth red_softmax_dim0 seed (bf16). x[M,N] -> softmax over dim 0
(the ROW axis). One program per column-block; a streaming running max + rescaled
exp-sum over the M rows (fp32, stable), then a second pass writes the normalized
column. tl.bfloat16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_softmax_dim0_kernel(x_ptr, o_ptr, sr, sc, M, N, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    cmask = cols < N
    m = tl.zeros([BLOCK_N], dtype=tl.float32) - float("inf")
    s = tl.zeros([BLOCK_N], dtype=tl.float32)
    for r in range(0, M):
        x = tl.load(x_ptr + r * sr + cols * sc, mask=cmask, other=-float("inf")).to(tl.float32)
        new_m = tl.maximum(m, x)
        s = s * tl.exp(m - new_m) + tl.exp(x - new_m)
        m = new_m
    for r in range(0, M):
        x = tl.load(x_ptr + r * sr + cols * sc, mask=cmask, other=0.0).to(tl.float32)
        tl.store(o_ptr + r * sr + cols * sc, (tl.exp(x - m) / s).to(tl.bfloat16), mask=cmask)


def red_softmax_dim0(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty_like(x)
    BLOCK_N = 256
    grid = (triton.cdiv(N, BLOCK_N),)
    _red_softmax_dim0_kernel[grid](x, o, x.stride(0), x.stride(1), M, N,
                                   BLOCK_N=BLOCK_N, num_warps=4)
    return o
