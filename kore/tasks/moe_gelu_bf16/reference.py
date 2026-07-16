"""Reference + inputs for the bf16 fused-MoE with a GeGLU (tanh-GELU) expert MLP.

Per token, the router selects ``topk`` experts (ids + weights). For each selected
expert e the token runs a gated MLP with a tanh-GELU gate:

    gate_up = x @ w1[e].T          # [2*inter]  (gate = first half, up = second)
    h       = gelu_tanh(gate) * up # [inter]
    y_e     = h  @ w2[e].T         # [model_dim]

and the token output is the weighted sum over its top-k experts. This is the
GeGLU variant of the live SiLU fused-MoE (Llama-4 Scout / gpt-oss ship gated
GELU). Correctness oracle: the exact fp32 computation (shared
``_moe_common.gated_mlp_fp32`` with act="gelu"). Perf baseline: AITER
``fused_moe`` with ``ActivationType.Gelu`` (weights pre-shuffled outside timing).

Weight layout (aiter-native): w1 ``[E, 2*inter, model_dim]``, w2 ``[E, model_dim, inter]``.
Router: the unbalanced jagged trace with a guaranteed 0-token last expert.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kore.tasks._moe_common import gated_mlp_fp32, make_routing  # noqa: E402

ENTRY = "fused_moe"
ATOL = 3e-2
RTOL = 3e-2


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"M": 256, "E": 16, "topk": 1, "D": 5120, "I": 8192}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def get_inputs(shape: dict, device="cuda", seed: int = 0):
    """Returns (hidden [M,D] bf16, w1 [E,2I,D] bf16, w2 [E,D,I] bf16,
    topk_weight [M,topk] fp32, topk_ids [M,topk] int32)."""
    import torch

    g = torch.Generator(device=device).manual_seed(seed)
    M, E, topk = shape["M"], shape["E"], shape["topk"]
    D, I = shape["D"], shape["I"]
    hidden = torch.randn((M, D), generator=g, device=device, dtype=torch.float32).to(torch.bfloat16)
    # small init so gelu*up*down stays well-scaled in bf16
    w1 = (torch.randn((E, 2 * I, D), generator=g, device=device, dtype=torch.float32) * 0.05).to(torch.bfloat16)
    w2 = (torch.randn((E, D, I), generator=g, device=device, dtype=torch.float32) * 0.05).to(torch.bfloat16)
    tw, ti = make_routing(M, E, topk, device, g, renorm=True)
    return hidden, w1, w2, tw, ti


def reference_output(shape, inputs):
    """Exact fp32 top-k GeGLU fused-MoE oracle -> bf16 [M, model_dim]."""
    hidden, w1, w2, tw, ti = inputs
    return gated_mlp_fp32(hidden, w1, w2, tw, ti, act="gelu")


def candidate_output(fn, shape, inputs):
    hidden, w1, w2, tw, ti = inputs
    return fn(hidden, w1, w2, tw, ti)


def baseline_output(shape, inputs):
    """Perf-only vendor bar: dense bf16 GeGLU MoE via per-expert hipBLASLt matmuls.

    The installed AITER ``fused_moe`` with ActivationType.Gelu fails to JIT-build on this
    node, so per VERIFICATION_CHECKLIST.md we KEEP the verified fp32 GeGLU oracle and time
    against a dense per-expert top-k GeGLU computed with torch bf16 matmuls (hipBLASLt) -- a
    real vendor-library bar the fused Triton kernel must beat. Grouped by expert so weight
    materialization stays O(E), not O(M)."""
    import torch
    import torch.nn.functional as F

    hidden, w1, w2, tw, ti = inputs
    M, D = hidden.shape
    topk = ti.shape[1]
    rows = hidden.repeat_interleave(topk, dim=0)              # [M*topk, D] bf16
    flat_ids = ti.reshape(-1).long()
    flat_w = tw.reshape(-1).float()
    y = torch.zeros((M * topk, D), dtype=torch.bfloat16, device=hidden.device)
    for e in torch.unique(flat_ids).tolist():
        idx = (flat_ids == e).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        gate_up = torch.matmul(rows[idx], w1[e].t())          # [n, 2I] bf16 (hipBLASLt)
        i2 = gate_up.shape[1] // 2
        h = F.gelu(gate_up[:, :i2].float(), approximate="tanh").to(torch.bfloat16) * gate_up[:, i2:]
        y[idx] = torch.matmul(h, w2[e].t())                   # [n, D] bf16 (hipBLASLt)
    y = (y.float().reshape(M, topk, D) * flat_w.reshape(M, topk, 1)).sum(dim=1)
    return y.to(torch.bfloat16)
