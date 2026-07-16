"""Seed Triton bf16 MoE dispatch/permute (indexed gather) for gfx950.

Exposes ``moe_permute(hidden, sort_idx)`` with hidden ``[M, D]`` bf16, sort_idx
``[M]`` int -> out ``[M, D]`` bf16 where ``out[i] = hidden[sort_idx[i]]``. One
program per (destination row, D-tile): reads the source row id and copies that
row's tile into the destination row. Memory-bound; a correct, simple seed the
KORE policy optimizes against the framework gather bar.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _permute_kernel(
    x_ptr, idx_ptr, o_ptr,
    sx_m, so_m, D,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    dblk = tl.program_id(1)
    src = tl.load(idx_ptr + row)
    offs = dblk * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < D
    x = tl.load(x_ptr + src * sx_m + offs, mask=mask, other=0.0)
    tl.store(o_ptr + row * so_m + offs, x, mask=mask)


def moe_permute(hidden: torch.Tensor, sort_idx: torch.Tensor) -> torch.Tensor:
    M, D = hidden.shape
    out = torch.empty((M, D), device=hidden.device, dtype=hidden.dtype)
    BLOCK_D = 1024
    grid = (M, triton.cdiv(D, BLOCK_D))
    _permute_kernel[grid](
        hidden, sort_idx.contiguous(), out,
        hidden.stride(0), out.stride(0), D,
        BLOCK_D=BLOCK_D, num_warps=4,
    )
    return out
