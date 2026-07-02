"""Seed Triton bf16 fused-MoE (top-k grouped GEMM + SiLU-mul) kernel for gfx942.

Exposes ``fused_moe(hidden_states, w1, w2, topk_weight, topk_ids)`` with natural
weight layout w1 ``[E, 2*inter, model_dim]``, w2 ``[E, model_dim, inter]``.

One program per (token, expert-slot): it loads the token's hidden vector, streams
the expert's gate/up rows in BLOCK_I tiles (gate_up = x @ w1[e].T), applies
SiLU-mul, immediately contracts with the corresponding w2 columns
(y += w2[e][:, tile] @ h_tile), scales by the router weight, and atomically
accumulates into the token's output row. Fully fp32 accumulation, bf16 IO.
Experts with zero assigned tokens are never visited, so the 0-token edge is free.
A correct, deliberately simple baseline the KORE policy learns to optimize against
the AITER fused_moe serving bar.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _moe_kernel(
    x_ptr, w1_ptr, w2_ptr, tw_ptr, tid_ptr, o_ptr,
    sx_m, sw1_e, sw1_n, sw2_e, sw2_d, so_m, stk,
    D, I, topk,
    DMAX: tl.constexpr, BLOCK_I: tl.constexpr,
):
    pid = tl.program_id(0)
    token = pid // topk
    kk = pid % topk

    e = tl.load(tid_ptr + token * stk + kk)
    w = tl.load(tw_ptr + token * stk + kk)

    offs_d = tl.arange(0, DMAX)
    dmask = offs_d < D
    x = tl.load(x_ptr + token * sx_m + offs_d, mask=dmask, other=0.0).to(tl.float32)  # [DMAX]

    out = tl.zeros([DMAX], tl.float32)
    for it in range(0, I, BLOCK_I):
        ii = it + tl.arange(0, BLOCK_I)
        imask = ii < I
        # gate rows w1[e, ii, :] and up rows w1[e, I+ii, :]
        wg = tl.load(
            w1_ptr + e * sw1_e + ii[:, None] * sw1_n + offs_d[None, :],
            mask=imask[:, None] & dmask[None, :], other=0.0,
        ).to(tl.float32)                                   # [BLOCK_I, DMAX]
        g = tl.sum(x[None, :] * wg, axis=1)                # [BLOCK_I]
        wu = tl.load(
            w1_ptr + e * sw1_e + (I + ii)[:, None] * sw1_n + offs_d[None, :],
            mask=imask[:, None] & dmask[None, :], other=0.0,
        ).to(tl.float32)
        u = tl.sum(x[None, :] * wu, axis=1)                # [BLOCK_I]
        silu = g * (1.0 / (1.0 + tl.exp(-g)))
        h = tl.where(imask, silu * u, 0.0)                 # [BLOCK_I]
        # contract with w2[e][:, ii] : [DMAX, BLOCK_I]
        w2 = tl.load(
            w2_ptr + e * sw2_e + offs_d[:, None] * sw2_d + ii[None, :],
            mask=dmask[:, None] & imask[None, :], other=0.0,
        ).to(tl.float32)
        out += tl.sum(w2 * h[None, :], axis=1)             # [DMAX]

    out = out * w
    tl.atomic_add(o_ptr + token * so_m + offs_d, out, mask=dmask)


def fused_moe(hidden_states, w1, w2, topk_weight, topk_ids):
    M, D = hidden_states.shape
    E, twoI, _ = w1.shape
    I = twoI // 2
    topk = topk_ids.shape[1]
    out = torch.zeros((M, D), device=hidden_states.device, dtype=torch.float32)
    DMAX = triton.next_power_of_2(D)
    BLOCK_I = 16
    grid = (M * topk,)
    _moe_kernel[grid](
        hidden_states, w1, w2, topk_weight.contiguous(), topk_ids.contiguous(), out,
        hidden_states.stride(0), w1.stride(0), w1.stride(1),
        w2.stride(0), w2.stride(1), out.stride(0), topk_ids.stride(0),
        D, I, topk,
        DMAX=DMAX, BLOCK_I=BLOCK_I,
        num_warps=4, num_stages=2,
    )
    return out.to(hidden_states.dtype)
