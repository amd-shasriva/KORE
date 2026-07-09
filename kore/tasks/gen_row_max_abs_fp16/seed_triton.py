"""GENERATED seed Triton kernel for the row_max_abs (fp16) row reduction.

Per-row reduction [M,N]->[M], fp32 accumulate, tl.float16 store. Regenerate via
kore/tasks/generate_ops.py — do not hand-edit.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _row_max_abs_kernel(x_ptr, y_ptr, stride_xm, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32) + (0.0)
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=(0.0)).to(tl.float32)
        acc = tl.maximum(acc, tl.abs(x))
    v = tl.max(acc, axis=0)
    v = v
    tl.store(y_ptr + row, v.to(tl.float16))


def row_max_abs(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty((M,), device=x.device, dtype=x.dtype)
    BLOCK_N = 1024
    _row_max_abs_kernel[(M,)](x, y, x.stride(0), N, BLOCK_N=BLOCK_N, num_warps=4)
    return y
