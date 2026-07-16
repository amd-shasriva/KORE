"""Reference + inputs for the bf16 MoE router: top-k softmax WITHOUT renorm.

The router softmaxes the gate logits ``[M, E]`` over experts, selects the top-k,
and keeps their RAW softmax probabilities as weights (no renormalization to sum
1). This is the un-renormalized variant of the live ``topk_softmax_bf16`` router
(some MoE configs, e.g. certain top-1 / soft routing, skip the renorm).

Correctness oracle: exact fp32 softmax -> top-k, materialized dense ``[M, E]``
(top-k probs scattered to their expert columns, zeros elsewhere), so grading is
order-independent -- it checks WHICH experts were selected AND their weights in
one SNR number. Perf baseline: AITER ``topk_softmax(..., renormalize=False)``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kore.tasks._moe_common import to_dense, topk_softmax_dense_fp32, vendor_topk_softmax_dense  # noqa: E402

ENTRY = "topk_softmax"
ATOL = 1e-2
RTOL = 1e-2


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"M": 4096, "E": 256, "topk": 8}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def get_inputs(shape: dict, device="cuda", seed: int = 0):
    """Returns (gate [M,E] bf16,) (topk read from the shape)."""
    import torch

    g = torch.Generator(device=device).manual_seed(seed)
    M, E = shape["M"], shape["E"]
    gate = torch.randn((M, E), generator=g, device=device, dtype=torch.float32).to(torch.bfloat16)
    return (gate,)


def reference_output(shape, inputs):
    """Exact fp32 softmax -> top-k (no renorm), dense [M, E]."""
    (gate,) = inputs
    return topk_softmax_dense_fp32(gate, shape["topk"], renorm=False)


def candidate_output(fn, shape, inputs):
    (gate,) = inputs
    w, ids = fn(gate, shape["topk"])
    return to_dense(w, ids, shape["E"])


def baseline_output(shape, inputs):
    """REAL vendor bar: AITER topk_softmax with renormalize=False, scattered dense."""
    (gate,) = inputs
    return vendor_topk_softmax_dense(gate, shape["topk"], shape["E"], renorm=False)
