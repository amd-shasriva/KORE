"""GENERATED breadth MoE seed: moe_expert_histogram (fp16).

per-expert token histogram (routing counts). Naive, COMPILING, CORRECT starting point: host-side routing/permute
selection (torch) with a Triton kernel for the dominant primitive. The policy is
expected to fuse the routing + grouped GEMM + activation + combine into one kernel.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl



@triton.jit
def _hist_kernel(ids_ptr, cnt_ptr, M, BM: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BM + tl.arange(0, BM)
    m = off < M
    e = tl.load(ids_ptr + off, mask=m, other=0).to(tl.int32)
    tl.atomic_add(cnt_ptr + e, tl.where(m, 1, 0), mask=m)

def moe_expert_histogram(expert_ids, E):
    M = expert_ids.shape[0]
    cnt = torch.zeros((E,), device=expert_ids.device, dtype=torch.int32)
    ids = expert_ids.to(torch.int32).contiguous()
    BM = 256
    _hist_kernel[(triton.cdiv(M, BM),)](ids, cnt, M, BM=BM)
    return cnt
