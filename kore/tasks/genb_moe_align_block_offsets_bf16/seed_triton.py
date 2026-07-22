"""GENERATED breadth MoE seed: moe_align_block_offsets (bf16).

block-aligned per-expert offsets (moe_align_block_size). Naive, COMPILING, CORRECT starting point: host-side routing/permute
selection (torch) with a Triton kernel for the dominant primitive. The policy is
expected to fuse the routing + grouped GEMM + activation + combine into one kernel.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl



@triton.jit
def _excl_cumsum_kernel(cnt_ptr, off_ptr, E):
    acc = tl.zeros([], dtype=tl.int32)
    tl.store(off_ptr + 0, acc)
    for e in range(0, E):
        c = tl.load(cnt_ptr + e).to(tl.int32)
        acc = acc + c
        tl.store(off_ptr + e + 1, acc)

def moe_align_block_offsets(expert_ids, E, block):
    cnt = torch.bincount(expert_ids.to(torch.long), minlength=E)
    padded = (((cnt + block - 1) // block) * block).to(torch.int32)
    off = torch.zeros((E + 1,), device=expert_ids.device, dtype=torch.int32)
    _excl_cumsum_kernel[(1,)](padded, off, E)
    return off
