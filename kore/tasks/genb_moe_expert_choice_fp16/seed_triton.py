"""GENERATED breadth MoE seed: moe_expert_choice (fp16).

expert-choice router (each expert picks top-C tokens). Naive, COMPILING, CORRECT starting point: host-side routing/permute
selection (torch) with a Triton kernel for the dominant primitive. The policy is
expected to fuse the routing + grouped GEMM + activation + combine into one kernel.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl



@triton.jit
def _ec_scatter_kernel(tv_ptr, ti_ptr, out_ptr, cap, stv, sti, sor, soc, CB: tl.constexpr):
    e = tl.program_id(0)
    c = tl.arange(0, CB)
    cm = c < cap
    toks = tl.load(ti_ptr + e * sti + c, mask=cm, other=0).to(tl.int64)
    vals = tl.load(tv_ptr + e * stv + c, mask=cm, other=0.0).to(tl.float32)
    tl.store(out_ptr + toks * sor + e * soc, vals, mask=cm)

def moe_expert_choice(gate, cap):
    M, E = gate.shape
    cap = min(int(cap), M)
    sm = torch.softmax(gate.float(), dim=-1)
    tv, ti = torch.topk(sm.t().contiguous(), cap, dim=-1)
    out = torch.zeros((M, E), device=gate.device, dtype=torch.float32)
    tv = tv.contiguous().float()
    ti = ti.contiguous().to(torch.int32)
    _ec_scatter_kernel[(E,)](tv, ti, out, cap, tv.stride(0), ti.stride(0),
                             out.stride(0), out.stride(1), CB=triton.next_power_of_2(cap))
    return out
