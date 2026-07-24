"""GENERATED breadth MoE seed: moe_expert_choice (bf16).

expert-choice router (each expert picks top-C tokens). Naive, COMPILING, CORRECT starting point: host-side routing/permute
selection (torch) with a Triton kernel for the dominant primitive. The policy is
expected to fuse the routing + grouped GEMM + activation + combine into one kernel.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl



@triton.jit
def _row_softmax_kernel(gate_ptr, work_ptr, E, sgm, sge, EB: tl.constexpr):
    row = tl.program_id(0)
    e = tl.arange(0, EB)
    mask = e < E
    raw = tl.load(gate_ptr + row * sgm + e * sge,
                  mask=mask, other=-float("inf")).to(tl.float32)
    row_max = tl.max(raw, axis=0)
    ex = tl.exp(raw - row_max)
    probs = ex / tl.sum(tl.where(mask, ex, 0.0), axis=0)
    tl.store(work_ptr + row * E + e, probs, mask=mask)


@triton.jit
def _expert_choice_kernel(work_ptr, out_ptr, M, E, cap, BLOCK: tl.constexpr):
    expert = tl.program_id(0)
    for c in range(0, cap):
        best = -float("inf")
        best_token = 0
        for start in range(0, M, BLOCK):
            tok = start + tl.arange(0, BLOCK)
            mask = tok < M
            values = tl.load(work_ptr + tok * E + expert,
                             mask=mask, other=-float("inf"))
            block_best = tl.max(values, axis=0)
            block_idx = tl.argmax(values, axis=0) + start
            take = block_best > best
            best = tl.where(take, block_best, best)
            best_token = tl.where(take, block_idx, best_token)
        tl.store(out_ptr + best_token * E + expert, best)
        tl.store(work_ptr + best_token * E + expert, -float("inf"))


def _expert_choice(gate, cap):
    gate = gate.contiguous()
    M, E = gate.shape
    cap = min(int(cap), M)
    work = torch.empty((M, E), device=gate.device, dtype=torch.float32)
    out = torch.zeros((M, E), device=gate.device, dtype=torch.float32)
    _row_softmax_kernel[(M,)](
        gate, work, E, gate.stride(0), gate.stride(1),
        EB=triton.next_power_of_2(E))
    _expert_choice_kernel[(E,)](work, out, M, E, cap, BLOCK=256)
    return out

def moe_expert_choice(gate, cap):
    return _expert_choice(gate, cap)
