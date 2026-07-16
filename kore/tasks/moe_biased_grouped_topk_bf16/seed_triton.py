"""Seed Triton DeepSeek-V3 biased grouped top-k MoE router for gfx950.

Exposes ``biased_grouped_topk(gate, correction_bias, topk, n_groups, topk_group)``
with gate ``[M, E]`` bf16, correction_bias ``[E]`` fp32, returning
``(topk_weights[M,topk] fp32, topk_ids[M,topk] int32)``.

One program per token (routers are tiny / memory-bound):
  scores      = sigmoid(gate)
  scores_bias = scores + correction_bias
  per group: top-2 sum of scores_bias -> group score
  keep the topk_group groups with the highest group score
  top-k experts (by scores_bias) among the kept groups
  weights     = the sigmoid scores at the chosen experts, renormalized to sum 1

``topk``, ``n_groups``, ``topk_group`` are constexpr (drive the unrolled loops).
A correct, simple seed the KORE policy optimizes against AITER biased_grouped_topk.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _bgt_kernel(
    gate_ptr, bias_ptr, w_ptr, id_ptr,
    sg_m, sw_m, sid_m,
    E, grp,
    EMAX: tl.constexpr, GMAX: tl.constexpr,
    TOPK: tl.constexpr, NGROUPS: tl.constexpr, TOPK_GROUP: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, EMAX)
    emask = offs < E
    g = tl.load(gate_ptr + row * sg_m + offs, mask=emask, other=-float("inf")).to(tl.float32)
    bias = tl.load(bias_ptr + offs, mask=emask, other=0.0).to(tl.float32)

    sig = tl.sigmoid(g)                                  # 0.0 at invalid lanes (g=-inf)
    sb = sig + bias
    sb = tl.where(emask, sb, -float("inf"))
    gid = offs // grp                                    # expert -> group id

    garange = tl.arange(0, GMAX)
    gvalid = garange < NGROUPS

    # group score = sum of the top-2 scores_bias within each group
    kept = tl.zeros([GMAX], tl.float32)                  # 1.0 if group is kept
    gscore = tl.full([GMAX], -float("inf"), tl.float32)
    for gg in range(0, NGROUPS):
        gsel = emask & (gid == gg)
        v = tl.where(gsel, sb, -float("inf"))
        m1 = tl.max(v, axis=0)
        a1 = tl.argmax(v, axis=0)
        v2 = tl.where(offs == a1, -float("inf"), v)
        m2 = tl.max(v2, axis=0)
        m2 = tl.where(m2 == -float("inf"), 0.0, m2)      # grp==1 -> only one member
        gs = m1 + m2
        gscore = tl.where(garange == gg, gs, gscore)

    # keep the TOPK_GROUP highest-scoring groups
    gtmp = tl.where(gvalid, gscore, -float("inf"))
    for _ in range(0, TOPK_GROUP):
        bg = tl.argmax(gtmp, axis=0)
        kept = tl.where(garange == bg, 1.0, kept)
        gtmp = tl.where(garange == bg, -float("inf"), gtmp)

    # broadcast the group-keep flag down to expert lanes
    lane_keep = tl.zeros([EMAX], tl.float32)
    for gg in range(0, NGROUPS):
        kg = tl.sum(tl.where(garange == gg, kept, 0.0), axis=0)   # scalar 0/1
        lane_keep = tl.where(gid == gg, kg, lane_keep)
    masked = tl.where((lane_keep > 0.5) & emask, sb, -float("inf"))

    # pass 1: sum of the top-k sigmoid scores (for renormalization)
    pw = masked
    wsum = 0.0
    for _ in range(0, TOPK):
        bi = tl.argmax(pw, axis=0)
        wsum += tl.sum(tl.where(offs == bi, sig, 0.0), axis=0)
        pw = tl.where(offs == bi, -float("inf"), pw)

    # pass 2: re-select deterministically, store normalized sigmoid weights + ids
    pw = masked
    for k in range(0, TOPK):
        bi = tl.argmax(pw, axis=0)
        sw = tl.sum(tl.where(offs == bi, sig, 0.0), axis=0)
        tl.store(id_ptr + row * sid_m + k, bi.to(tl.int32))
        tl.store(w_ptr + row * sw_m + k, sw / wsum)
        pw = tl.where(offs == bi, -float("inf"), pw)


def biased_grouped_topk(gate: torch.Tensor, correction_bias: torch.Tensor,
                        topk: int, n_groups: int, topk_group: int):
    M, E = gate.shape
    grp = E // n_groups
    topk_weights = torch.empty((M, topk), device=gate.device, dtype=torch.float32)
    topk_ids = torch.empty((M, topk), device=gate.device, dtype=torch.int32)
    EMAX = triton.next_power_of_2(E)
    GMAX = triton.next_power_of_2(n_groups)
    grid = (M,)
    _bgt_kernel[grid](
        gate, correction_bias.contiguous().to(torch.float32), topk_weights, topk_ids,
        gate.stride(0), topk_weights.stride(0), topk_ids.stride(0),
        E, grp,
        EMAX=EMAX, GMAX=GMAX,
        TOPK=topk, NGROUPS=n_groups, TOPK_GROUP=topk_group,
        num_warps=4,
    )
    return topk_weights, topk_ids
