"""GENERATED breadth MoE seed: moe_unpermute (bf16).

MoE un-permute (scatter tokens back to original order). Naive, COMPILING, CORRECT starting point: host-side routing/permute
selection (torch) with a Triton kernel for the dominant primitive. The policy is
expected to fuse the routing + grouped GEMM + activation + combine into one kernel.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl



@triton.jit
def _gather_kernel(src_ptr, idx_ptr, dst_ptr, D, ss, sd, BD: tl.constexpr):
    row = tl.program_id(0)
    src_row = tl.load(idx_ptr + row).to(tl.int64)
    for d0 in range(0, D, BD):
        off = d0 + tl.arange(0, BD)
        m = off < D
        v = tl.load(src_ptr + src_row * ss + off, mask=m)
        tl.store(dst_ptr + row * sd + off, v, mask=m)

def moe_unpermute(permuted, sort_idx):
    M, D = permuted.shape
    inv = torch.argsort(sort_idx.to(torch.long)).to(torch.int64).contiguous()
    out = torch.empty_like(permuted)
    _gather_kernel[(M,)](permuted, inv, out, D, permuted.stride(0), out.stride(0), BD=256)
    return out
