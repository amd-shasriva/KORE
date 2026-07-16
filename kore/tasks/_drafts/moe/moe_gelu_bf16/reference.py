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
from _moe_common import gated_mlp_fp32, make_routing, vendor_fused_moe  # noqa: E402

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
    """REAL vendor bar: AITER CK fused MoE with ActivationType.Gelu."""
    hidden, w1, w2, tw, ti = inputs
    return vendor_fused_moe(hidden, w1, w2, tw, ti, activation="gelu")
