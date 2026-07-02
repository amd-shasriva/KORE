"""Seed Triton bf16 row-softmax for gfx942. Exposes ``softmax(x) -> y``.

One program per row, online (streaming) softmax over column tiles so arbitrarily
wide rows (e.g. N=32768) work without huge register/LDS pressure:
    pass 1: running max + running (rescaled) sum of exp
    pass 2: normalize and bf16 store
fp32 math throughout. A correct, reasonably-tuned baseline the KORE policy
learns to edit/optimize against the fused framework softmax serving bar.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    x_ptr, y_ptr,
    stride_xm, stride_ym,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_xm
    y_row = y_ptr + row * stride_ym

    m = -float("inf")
    l = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=-float("inf")).to(tl.float32)
        block_max = tl.max(x, axis=0)
        new_m = tl.maximum(m, block_max)
        l = l * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m), axis=0)
        m = new_m

    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=-float("inf")).to(tl.float32)
        y = tl.exp(x - m) / l
        tl.store(y_row + offs, y.to(tl.bfloat16), mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK_N = 1024 if N > 1024 else triton.next_power_of_2(N)
    _softmax_kernel[(M,)](
        x, y,
        x.stride(0), y.stride(0),
        N,
        BLOCK_N=BLOCK_N, num_warps=8,
    )
    return y
