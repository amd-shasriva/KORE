"""GENERATED breadth MoE seed: moe_biased_grouped_topk (fp16).

DeepSeek-V3 biased grouped top-k router -> dense [M,E]. Naive, COMPILING, CORRECT starting point: host-side routing/permute
selection (torch) with a Triton kernel for the dominant primitive. The policy is
expected to fuse the routing + grouped GEMM + activation + combine into one kernel.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl



@triton.jit
def _group_score_kernel(gate_ptr, bias_ptr, score_ptr, E, n_groups, group_size,
                        sgm, sge, BIASED: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    group = tl.program_id(1)
    k = tl.arange(0, BLOCK)
    mask = k < group_size
    e = group * group_size + k
    raw = tl.load(gate_ptr + row * sgm + e * sge,
                  mask=mask, other=-float("inf")).to(tl.float32)
    if BIASED:
        bias = tl.load(bias_ptr + e, mask=mask, other=0.0).to(tl.float32)
        scores = tl.sigmoid(raw) + bias
        first_idx = tl.argmax(scores, axis=0)
        first = tl.max(scores, axis=0)
        second = tl.max(tl.where(k == first_idx, -float("inf"), scores), axis=0)
        group_score = first + tl.where(group_size > 1, second, 0.0)
    else:
        group_score = tl.max(raw, axis=0)
    tl.store(score_ptr + row * n_groups + group, group_score)


@triton.jit
def _route_grouped_kernel(gate_ptr, bias_ptr, group_ptr, dense_ptr,
                          E, n_groups, group_size, sgm, sge,
                          TOPK: tl.constexpr, TOPK_GROUP: tl.constexpr,
                          BIASED: tl.constexpr, RENORM: tl.constexpr,
                          EB: tl.constexpr, GB: tl.constexpr):
    row = tl.program_id(0)
    e = tl.arange(0, EB)
    emask = e < E
    raw = tl.load(gate_ptr + row * sgm + e * sge,
                  mask=emask, other=-float("inf")).to(tl.float32)
    if BIASED:
        bias = tl.load(bias_ptr + e, mask=emask, other=0.0).to(tl.float32)
        weights = tl.sigmoid(raw)
        select_scores = weights + bias
    else:
        row_max = tl.max(raw, axis=0)
        ex = tl.exp(raw - row_max)
        weights = ex / tl.sum(tl.where(emask, ex, 0.0), axis=0)
        select_scores = weights

    groups = e // group_size
    goffs = tl.arange(0, GB)
    gmask = goffs < n_groups
    group_scores = tl.load(group_ptr + row * n_groups + goffs,
                           mask=gmask, other=-float("inf"))
    allowed = e < 0
    for j in range(0, TOPK_GROUP):
        picked_group = tl.argmax(group_scores, axis=0)
        allowed = allowed | (groups == picked_group)
        group_scores = tl.where(goffs == picked_group, -float("inf"), group_scores)

    candidates = tl.where(emask & allowed, select_scores, -float("inf"))
    tl.store(dense_ptr + row * E + e, 0.0, mask=emask)
    total = 0.0
    for j in range(0, TOPK):
        pick = tl.argmax(candidates, axis=0)
        value = tl.sum(tl.where(e == pick, weights, 0.0), axis=0)
        total += value
        tl.store(dense_ptr + row * E + pick, value)
        candidates = tl.where(e == pick, -float("inf"), candidates)
    if RENORM:
        vals = tl.load(dense_ptr + row * E + e, mask=emask, other=0.0)
        tl.store(dense_ptr + row * E + e,
                 tl.where(vals != 0.0, vals / tl.maximum(total, 1.0e-12), 0.0),
                 mask=emask)


def _route_grouped(gate, bias, topk, n_groups, topk_group, biased, renorm):
    gate = gate.contiguous()
    if biased:
        bias = bias.contiguous()
    M, E = gate.shape
    group_size = E // n_groups
    group_scores = torch.empty((M, n_groups), device=gate.device, dtype=torch.float32)
    dense = torch.zeros((M, E), device=gate.device, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(group_size)
    _group_score_kernel[(M, n_groups)](
        gate, bias if biased else gate, group_scores, E, n_groups, group_size,
        gate.stride(0), gate.stride(1), BIASED=biased, BLOCK=BLOCK)
    _route_grouped_kernel[(M,)](
        gate, bias if biased else gate, group_scores, dense,
        E, n_groups, group_size, gate.stride(0), gate.stride(1),
        TOPK=topk, TOPK_GROUP=topk_group, BIASED=biased, RENORM=renorm,
        EB=triton.next_power_of_2(E), GB=triton.next_power_of_2(n_groups))
    return dense

def moe_biased_grouped_topk(gate, bias, topk, n_groups, topk_group):
    return _route_grouped(gate, bias, topk, n_groups, topk_group,
                          True, True)
