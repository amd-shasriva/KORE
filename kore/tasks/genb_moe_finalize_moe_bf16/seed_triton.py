"""GENERATED breadth MoE seed: moe_finalize_moe (bf16).

MoE finalize: gather from the permuted buffer + weighted combine. Naive, COMPILING, CORRECT starting point: host-side routing/permute
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

def moe_finalize_moe(y_perm, row_map, tw):
    M, topk = row_map.shape
    D = y_perm.shape[1]
    yg = y_perm.index_select(0, row_map.reshape(-1).to(torch.long)).reshape(M, topk, D).contiguous()
    tw = tw.contiguous()
    out = torch.empty((M, D), device=y_perm.device, dtype=y_perm.dtype)
    _combine_kernel[(M,)](yg, tw, out, topk, D, yg.stride(0), yg.stride(1), yg.stride(2),
                          tw.stride(0), out.stride(0), out.stride(1), BD=256)
    return out
