"""GENERATED breadth red_entropy seed (fp32). logits[M,N] -> Shannon entropy of
softmax(logits) per row: H = lse - sum_j p_j x_j (fp32, max-subtracted, stable).
Pass 1 the streaming lse; pass 2 the probability-weighted logit sum. tl.float32 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_entropy_kernel(x_ptr, o_ptr, sx, N, BLOCK_N: tl.constexpr):
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
    dot = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)
        p = tl.exp(x - m) / s
        dot += tl.sum(tl.where(mask, p * x, 0.0), axis=0)
    tl.store(o_ptr + row, (lse - dot).to(tl.float32))


def red_entropy(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty((M,), device=x.device, dtype=x.dtype)
    BLOCK_N = 1024
    _red_entropy_kernel[(M,)](x, o, x.stride(0), N, BLOCK_N=BLOCK_N, num_warps=8)
    return o
