"""GENERATED vendor-baselined row-softmax seed (bf16) vs torch/MIOpen softmax.
Online (streaming) softmax: pass 1 running max+sum, pass 2 normalize+store, so any
row width N fits regardless of BLOCK_N. Regenerate via generate_vendor_ops.py."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _softmax_kernel(x_ptr, y_ptr, sm, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    m = -float("inf")
    s = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=-float("inf")).to(tl.float32)
        blk_max = tl.max(x, axis=0)
        new_m = tl.maximum(m, blk_max)
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m), axis=0)
        m = new_m
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(y_ptr + base + offs, (tl.exp(x - m) / s).to(tl.bfloat16), mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    _softmax_kernel[(M,)](x, y, x.stride(0), N, BLOCK_N=1024, num_warps=8)
    return y
