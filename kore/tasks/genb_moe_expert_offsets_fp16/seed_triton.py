"""GENERATED breadth MoE seed: moe_expert_offsets (fp16).

per-expert exclusive-scan offsets into the sorted token buffer. Naive, COMPILING, CORRECT starting point: host-side routing/permute
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

def moe_expert_offsets(expert_ids, E):
    cnt = torch.bincount(expert_ids.to(torch.long), minlength=E).to(torch.int32)
    off = torch.zeros((E + 1,), device=expert_ids.device, dtype=torch.int32)
    _excl_cumsum_kernel[(1,)](cnt, off, E)
    return off
