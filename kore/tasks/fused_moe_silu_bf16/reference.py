"""Reference + inputs for the bf16 fused-MoE (top-k grouped GEMM + SiLU-mul) task.

Per token, the router has already selected ``topk`` experts (ids + weights). For
each selected expert e the token runs a gated MLP:

    gate_up = x @ w1[e].T          # [2*inter]  (gate = first half, up = second)
    h       = silu(gate) * up      # [inter]
    y_e     = h  @ w2[e].T         # [model_dim]

and the token output is the weighted sum over its top-k experts:
``y = sum_k topk_weight[:,k] * y_{e_k}``.

Correctness oracle: the exact fp32 computation above (mirrors aiter's own
``torch_moe`` reference). Perf baseline (driver ``--impl reference``): AITER
production ``fused_moe`` (CK 2-stage, SiLU), with weights pre-shuffled once at
load time (outside the timed region), matching how serving deploys MoE weights.

Weight layout (natural, matches aiter API): w1 ``[E, 2*inter, model_dim]``,
w2 ``[E, model_dim, inter]``.

Token distribution: the router assignment is deliberately *unbalanced* (a jagged
32-expert trace in the spirit of DATASET_SPEC §1.6) and always leaves the last
expert with **zero** tokens (the mandatory 0-token / giant-expert MoE edge).
"""

from __future__ import annotations

import torch

# Representative jagged per-expert weighting (DATASET_SPEC §1.6 unbalanced
# 32-expert trace character): one giant expert, several mid, a long tail, and a
# final dead expert. Used as a router bias so token->expert counts are unbalanced
# and expert E-1 receives 0 tokens.
_TRACE32 = [
    16053, 105, 1843, 2724, 327, 88, 4102, 51, 9210, 61, 3020, 44, 1502, 990,
    233, 77, 6740, 120, 410, 58, 2210, 39, 812, 175, 5030, 66, 1360, 92, 3550,
    47, 1180, 0,
]


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"M": 256, "E": 32, "topk": 8, "D": 1024, "I": 768}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def _routing(M, E, topk, device, g):
    """Unbalanced router assignment with a guaranteed 0-token last expert.

    Returns (topk_weight[M,topk] fp32, topk_ids[M,topk] int32)."""
    if E <= len(_TRACE32):
        counts = torch.tensor(_TRACE32[:E], dtype=torch.float32, device=device)
    else:
        counts = torch.ones(E, dtype=torch.float32, device=device)
    counts[-1] = 0.0                              # dead expert (0-token edge)
    bias = torch.log(counts + 1e-6)
    bias[counts == 0] = float("-inf")            # never select the dead expert
    gate = torch.randn((M, E), generator=g, device=device, dtype=torch.float32) + bias
    probs = torch.softmax(gate, dim=-1)
    tw, ti = torch.topk(probs, topk, dim=-1)
    tw = (tw / tw.sum(dim=-1, keepdim=True)).to(torch.float32)
    return tw, ti.to(torch.int32)


def get_inputs(shape: dict, dtype=torch.bfloat16, device="cuda", seed: int = 0):
    """Returns (hidden_states, w1, w2, topk_weight, topk_ids)."""
    g = torch.Generator(device=device).manual_seed(seed)
    M, E, topk, D, I = shape["M"], shape["E"], shape["topk"], shape["D"], shape["I"]
    hidden = torch.randn((M, D), generator=g, device=device, dtype=torch.float32).to(dtype)
    # small init so silu*up*down stays well-scaled in bf16
    w1 = (torch.randn((E, 2 * I, D), generator=g, device=device, dtype=torch.float32) * 0.05).to(dtype)
    w2 = (torch.randn((E, D, I), generator=g, device=device, dtype=torch.float32) * 0.05).to(dtype)
    tw, ti = _routing(M, E, topk, device, g)
    return hidden, w1, w2, tw, ti


def moe_ref(hidden, w1, w2, topk_weight, topk_ids) -> torch.Tensor:
    """Exact fp32 top-k fused-MoE oracle -> bf16, output [M, model_dim]."""
    M, D = hidden.shape
    E = w1.shape[0]
    I = w2.shape[2]
    x = hidden.float()
    w1f = w1.float()
    w2f = w2.float()
    out = torch.zeros((M, D), device=hidden.device, dtype=torch.float32)
    ids = topk_ids.long()
    for e in range(E):
        mask = ids == e                          # [M, topk]
        tok = mask.any(dim=1)
        if not bool(tok.any()):
            continue                             # 0-token expert -> skipped
        idx = tok.nonzero(as_tuple=True)[0]
        xe = x[idx]                              # [n, D]
        gate_up = xe @ w1f[e].t()               # [n, 2I]
        gate, up = gate_up[:, :I], gate_up[:, I:]
        h = torch.nn.functional.silu(gate) * up  # [n, I]
        ye = h @ w2f[e].t()                      # [n, D]
        # weight for expert e per selected token
        w_e = (topk_weight * mask.float()).sum(dim=1)[idx]  # [n]
        out[idx] += ye * w_e[:, None]
    return out.to(hidden.dtype)
