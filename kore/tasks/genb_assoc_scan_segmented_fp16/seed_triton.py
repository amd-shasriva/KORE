"""GENERATED breadth assoc_scan_segmented seed (fp16). Gated linear recurrence
h_t = a_t*h_{t-1} + b_t (h_{-1}=0) over the last dim. One program per flattened row;
sequential fp32 scan (naive but correct; the associative operator (a,b) makes this a
parallel prefix scan the policy is expected to build). Inputs a[...,L], b[...,L]. tl.float16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _assoc_scan_segmented_kernel(a_ptr, b_ptr, h_ptr, L, srow):
    row = tl.program_id(0)
    base = row * srow
    h = 0.0
    for i in range(0, L):
        av = tl.load(a_ptr + base + i).to(tl.float32)
        bv = tl.load(b_ptr + base + i).to(tl.float32)
        h = av * h + bv
        tl.store(h_ptr + base + i, h.to(tl.float16))


def assoc_scan_segmented(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    L = a.shape[-1]
    af = a.contiguous().reshape(-1, L)
    bf = b.contiguous().reshape(-1, L)
    h = torch.empty_like(af)
    _assoc_scan_segmented_kernel[(af.shape[0],)](af, bf, h, L, af.stride(0), num_warps=1)
    return h.reshape(a.shape)
