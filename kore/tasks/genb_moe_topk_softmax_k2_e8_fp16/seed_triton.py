"""GENERATED breadth MoE seed: moe_topk_softmax_k2_e8 (fp16).

top-k softmax MoE router -> dense [M,E] routing weights. Naive, COMPILING, CORRECT starting point: host-side routing/permute
selection (torch) with a Triton kernel for the dominant primitive. The policy is
expected to fuse the routing + grouped GEMM + activation + combine into one kernel.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl



@triton.jit
def _route_topk_kernel(gate_ptr, dense_ptr, tw_ptr, ti_ptr, E, sgm, sge,
                       TOPK: tl.constexpr, SOFTMAX: tl.constexpr,
                       SIGMOID: tl.constexpr, TOPK_SOFTMAX: tl.constexpr,
                       RENORM: tl.constexpr, EB: tl.constexpr):
    row = tl.program_id(0)
    e = tl.arange(0, EB)
    mask = e < E
    raw = tl.load(gate_ptr + row * sgm + e * sge,
                  mask=mask, other=-float("inf")).to(tl.float32)
    row_max = tl.max(raw, axis=0)
    if SOFTMAX:
        ex = tl.exp(raw - row_max)
        scores = ex / tl.sum(tl.where(mask, ex, 0.0), axis=0)
    elif SIGMOID:
        scores = tl.sigmoid(raw)
    else:
        scores = raw
    candidates = tl.where(mask, scores, -float("inf"))
    # The renormalizer used to write the k winners to `dense`, read the row back,
    # divide and write it again.  A load has no ordering against earlier stores
    # to the same addresses from the same program, and the second vector store
    # races the scalar ones, so some winners kept their UN-normalized value --
    # every renormalizing router was wrong on ~87% of rows.  Selecting twice
    # instead, with the row held in registers, removes the round trip entirely:
    # pass one only accumulates `total`, pass two writes values already divided.
    scan = candidates
    total = 0.0
    for j in range(0, TOPK):
        picked = tl.max(scan, axis=0)
        if TOPK_SOFTMAX:
            total += tl.exp(picked - row_max)
        else:
            total += picked
        scan = tl.where(e == tl.argmax(scan, axis=0), -float("inf"), scan)
    inv = 1.0 / total if (RENORM or TOPK_SOFTMAX) else 1.0
    dense_row = tl.zeros([EB], dtype=tl.float32)
    for j in range(0, TOPK):
        pick = tl.argmax(candidates, axis=0)
        picked = tl.max(candidates, axis=0)
        if TOPK_SOFTMAX:
            value = tl.exp(picked - row_max) * inv
        else:
            value = picked * inv
        dense_row = tl.where(e == pick, value, dense_row)
        tl.store(tw_ptr + row * TOPK + j, value)
        tl.store(ti_ptr + row * TOPK + j, pick)
        candidates = tl.where(e == pick, -float("inf"), candidates)
    tl.store(dense_ptr + row * E + e, dense_row, mask=mask)


def _route_topk(gate, topk, mode, renorm):
    gate = gate.contiguous()
    M, E = gate.shape
    dense = torch.zeros((M, E), device=gate.device, dtype=torch.float32)
    tw = torch.empty((M, topk), device=gate.device, dtype=torch.float32)
    ti = torch.empty((M, topk), device=gate.device, dtype=torch.int32)
    EB = triton.next_power_of_2(E)
    _route_topk_kernel[(M,)](
        gate, dense, tw, ti, E, gate.stride(0), gate.stride(1),
        TOPK=topk, SOFTMAX=mode == "softmax", SIGMOID=mode == "sigmoid",
        TOPK_SOFTMAX=mode == "topk_softmax", RENORM=renorm, EB=EB)
    return dense, tw, ti

def moe_topk_softmax_k2_e8(gate, topk):
    dense, _, _ = _route_topk(gate, topk, 'softmax', True)
    return dense
