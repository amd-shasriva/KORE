"""Reference + inputs for the bf16 MoE router (top-k softmax) task.

The router takes gate logits ``[M, E]`` (one score per expert per token),
softmaxes over experts, selects the top-k experts, and renormalizes their
weights to sum to 1 (the standard vLLM/SGLang ``topk_softmax`` used before a
fused-MoE dispatch).

Correctness oracle: exact fp32 softmax -> top-k -> renormalize, materialized as a
dense ``[M, E]`` weight tensor (top-k weights scattered to their expert columns,
zeros elsewhere). The dense form makes correctness order-independent: it checks
both *which* experts were selected and their weights in one SNR number. Perf
baseline (driver ``--impl reference``): AITER ``topk_softmax``.
"""

from __future__ import annotations

import torch


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"M": 4096, "E": 32, "topk": 8}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def get_inputs(shape: dict, dtype=torch.bfloat16, device="cuda", seed: int = 0):
    """Returns gate logits ``[M, E]`` in ``dtype`` (topk read from the shape)."""
    g = torch.Generator(device=device).manual_seed(seed)
    M, E = shape["M"], shape["E"]
    gate = torch.randn((M, E), generator=g, device=device, dtype=torch.float32).to(dtype)
    return gate


def to_dense(topk_weights, topk_ids, E) -> torch.Tensor:
    M = topk_weights.shape[0]
    dense = torch.zeros((M, E), device=topk_weights.device, dtype=torch.float32)
    dense.scatter_(1, topk_ids.long(), topk_weights.float())
    return dense


def topk_softmax_ref(gate, topk) -> torch.Tensor:
    """Exact fp32 softmax -> top-k -> renorm, returned as dense [M, E]."""
    E = gate.shape[1]
    sm = torch.softmax(gate.float(), dim=-1)
    tw, ti = torch.topk(sm, topk, dim=-1)
    tw = tw / tw.sum(dim=-1, keepdim=True)
    return to_dense(tw, ti, E)
