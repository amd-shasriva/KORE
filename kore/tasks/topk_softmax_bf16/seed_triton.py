"""Seed Triton bf16 MoE router (top-k softmax) kernel for gfx942.

Exposes ``topk_softmax(gate, topk)`` with gate ``[M, E]`` bf16, returning
``(topk_weights[M,topk] fp32, topk_ids[M,topk] int32)``. One program per token:
fp32 softmax over the E experts, then ``topk`` iterations of masked argmax to
select the highest-probability experts, then renormalize the selected weights to
sum to 1. Correct, simple baseline the KORE policy optimizes against AITER's
``topk_softmax``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _topk_softmax_kernel(
    gate_ptr, w_ptr, id_ptr,
    sg_m, sw_m, sid_m,
    E, topk,
    EMAX: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, EMAX)
    mask = offs < E
    g = tl.load(gate_ptr + row * sg_m + offs, mask=mask, other=-float("inf")).to(tl.float32)

    m = tl.max(g, axis=0)
    ex = tl.exp(g - m)
    ex = tl.where(mask, ex, 0.0)
    denom = tl.sum(ex, axis=0)
    probs = ex / denom
    probs = tl.where(mask, probs, -1.0)

    # pass 1: sum of the top-k probabilities (for renormalization)
    pw = probs
    wsum = 0.0
    for _ in range(0, topk):
        bv = tl.max(pw, axis=0)
        bi = tl.argmax(pw, axis=0)
        wsum += bv
        pw = tl.where(offs == bi, -1.0, pw)

    # pass 2: re-select deterministically and store normalized weights + ids
    pw = probs
    for k in range(0, topk):
        bv = tl.max(pw, axis=0)
        bi = tl.argmax(pw, axis=0)
        tl.store(id_ptr + row * sid_m + k, bi.to(tl.int32))
        tl.store(w_ptr + row * sw_m + k, bv / wsum)
        pw = tl.where(offs == bi, -1.0, pw)


def topk_softmax(gate: torch.Tensor, topk: int):
    M, E = gate.shape
    topk_weights = torch.empty((M, topk), device=gate.device, dtype=torch.float32)
    topk_ids = torch.empty((M, topk), device=gate.device, dtype=torch.int32)
    EMAX = triton.next_power_of_2(E)
    grid = (M,)
    _topk_softmax_kernel[grid](
        gate, topk_weights, topk_ids,
        gate.stride(0), topk_weights.stride(0), topk_ids.stride(0),
        E, topk,
        EMAX=EMAX, num_warps=4,
    )
    return topk_weights, topk_ids
