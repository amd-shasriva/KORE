"""GENERATED breadth MoE seed: moe_sum_combine (bf16).

top-k weighted combine (moe_sum reduce). Naive, COMPILING, CORRECT starting point: host-side routing/permute
selection (torch) with a Triton kernel for the dominant primitive. The policy is
expected to fuse the routing + grouped GEMM + activation + combine into one kernel.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl



@triton.jit
def _combine_kernel(y_ptr, w_ptr, out_ptr, topk, D,
                    sy0, sy1, sy2, sw0, so0, so1, BD: tl.constexpr):
    row = tl.program_id(0)
    for d0 in range(0, D, BD):
        off = d0 + tl.arange(0, BD)
        m = off < D
        acc = tl.zeros([BD], dtype=tl.float32)
        for k in range(0, topk):
            wv = tl.load(w_ptr + row * sw0 + k).to(tl.float32)
            yv = tl.load(y_ptr + row * sy0 + k * sy1 + off * sy2, mask=m, other=0.0).to(tl.float32)
            acc += wv * yv
        tl.store(out_ptr + row * so0 + off * so1, acc.to(out_ptr.dtype.element_ty), mask=m)

def moe_sum_combine(y, tw):
    M, topk, D = y.shape
    y = y.contiguous()
    tw = tw.contiguous()
    out = torch.empty((M, D), device=y.device, dtype=y.dtype)
    _combine_kernel[(M,)](y, tw, out, topk, D, y.stride(0), y.stride(1), y.stride(2),
                          tw.stride(0), out.stride(0), out.stride(1), BD=256)
    return out
