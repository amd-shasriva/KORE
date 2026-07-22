"""GENERATED breadth MoE seed: moe_topk_then_softmax (bf16).

top-k-then-softmax MoE router -> dense [M,E]. Naive, COMPILING, CORRECT starting point: host-side routing/permute
selection (torch) with a Triton kernel for the dominant primitive. The policy is
expected to fuse the routing + grouped GEMM + activation + combine into one kernel.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl



@triton.jit
def _scatter_dense_kernel(w_ptr, id_ptr, out_ptr, topk, sw, sid, so, TB: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, TB)
    km = k < topk
    ids = tl.load(id_ptr + row * sid + k, mask=km, other=0).to(tl.int64)
    ws = tl.load(w_ptr + row * sw + k, mask=km, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * so + ids, ws, mask=km)


def _scatter(tw, ti, M, E):
    """Scatter top-k (weights, ids) into a dense [M, E] fp32 routing map."""
    out = torch.zeros((M, E), device=tw.device, dtype=torch.float32)
    tw = tw.contiguous().float()
    ti = ti.contiguous().to(torch.int32)
    topk = tw.shape[1]
    _scatter_dense_kernel[(M,)](tw, ti, out, topk, tw.stride(0), ti.stride(0),
                                out.stride(0), TB=triton.next_power_of_2(topk))
    return out

def moe_topk_then_softmax(gate, topk):
    M, E = gate.shape
    tv, ti = torch.topk(gate.float(), topk, dim=-1)
    w = torch.softmax(tv, dim=-1)
    return _scatter(w, ti, M, E)
