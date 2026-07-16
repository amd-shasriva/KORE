"""Seed Triton bf16 MoE weighted combine (moe_sum reduce) for gfx950.

Exposes ``moe_sum(y, topk_weight)`` with y ``[M, topk, D]`` bf16, topk_weight
``[M, topk]`` fp32 -> out ``[M, D]`` bf16, computing
``out[m] = sum_k topk_weight[m,k] * y[m,k,:]``. One program per (token, D-tile):
fp32 accumulate over the top-k slots, bf16 store. A correct, simple seed the KORE
policy optimizes against the framework weighted-reduce bar.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _moe_sum_kernel(
    y_ptr, w_ptr, o_ptr,
    sy_m, sy_k, sy_d, sw_m, sw_k, so_m,
    D,
    TOPK: tl.constexpr, BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    dblk = tl.program_id(1)
    offs = dblk * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < D
    acc = tl.zeros([BLOCK_D], tl.float32)
    for k in range(0, TOPK):
        w = tl.load(w_ptr + row * sw_m + k * sw_k).to(tl.float32)
        yv = tl.load(y_ptr + row * sy_m + k * sy_k + offs * sy_d,
                     mask=mask, other=0.0).to(tl.float32)
        acc += w * yv
    tl.store(o_ptr + row * so_m + offs, acc.to(tl.bfloat16), mask=mask)


def moe_sum(y: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
    M, topk, D = y.shape
    out = torch.empty((M, D), device=y.device, dtype=torch.bfloat16)
    BLOCK_D = 1024
    grid = (M, triton.cdiv(D, BLOCK_D))
    _moe_sum_kernel[grid](
        y, topk_weight.contiguous(), out,
        y.stride(0), y.stride(1), y.stride(2),
        topk_weight.stride(0), topk_weight.stride(1), out.stride(0),
        D,
        TOPK=topk, BLOCK_D=BLOCK_D, num_warps=4,
    )
    return out
