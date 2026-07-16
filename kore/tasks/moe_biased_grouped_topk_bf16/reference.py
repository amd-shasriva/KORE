"""Reference + inputs for the DeepSeek-V3 biased grouped top-k MoE router.

DeepSeek-V3 routes with a SIGMOID gate (not softmax) plus a per-expert learned
correction bias used only for SELECTION:

    scores      = sigmoid(gate)                       # [M,E]
    scores_bias = scores + correction_bias            # routing score
    group_score = sum of top-2 scores_bias per group  # experts split into n_groups
    keep the topk_group groups with the highest group_score
    top-k experts (by scores_bias) among the kept groups
    weights     = the ORIGINAL sigmoid scores at the chosen experts, renormalized

Correctness oracle: the exact fp32 computation
(``_moe_common.biased_grouped_topk_dense_fp32``), materialized dense ``[M, E]``
so grading is order-independent. Perf baseline: AITER ``biased_grouped_topk``
(the exact call signature is version-dependent; the wrapper tries the common form
and otherwise falls back to the verified oracle -- see VERIFICATION_CHECKLIST).

gate ``[M, E]`` bf16, correction_bias ``[E]`` fp32. E divisible by n_groups.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kore.tasks._moe_common import (  # noqa: E402
    biased_grouped_topk_dense_fp32,
    to_dense,
    vendor_biased_grouped_topk_dense,
)

ENTRY = "biased_grouped_topk"
ATOL = 1e-2
RTOL = 1e-2


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"M": 4096, "E": 256, "topk": 8, "n_groups": 8, "topk_group": 4}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def get_inputs(shape: dict, device="cuda", seed: int = 0):
    """Returns (gate [M,E] bf16, correction_bias [E] fp32)."""
    import torch

    g = torch.Generator(device=device).manual_seed(seed)
    M, E = shape["M"], shape["E"]
    gate = torch.randn((M, E), generator=g, device=device, dtype=torch.float32).to(torch.bfloat16)
    # small correction bias (DeepSeek-V3 learned routing bias), fp32
    bias = (torch.randn((E,), generator=g, device=device, dtype=torch.float32) * 0.1)
    return (gate, bias)


def reference_output(shape, inputs):
    """Exact fp32 DeepSeek-V3 biased grouped top-k oracle, dense [M, E]."""
    gate, bias = inputs
    return biased_grouped_topk_dense_fp32(
        gate, bias, shape["topk"], shape["n_groups"], shape["topk_group"],
        renorm=True, scale=1.0)


def candidate_output(fn, shape, inputs):
    gate, bias = inputs
    w, ids = fn(gate, bias, shape["topk"], shape["n_groups"], shape["topk_group"])
    return to_dense(w, ids, shape["E"])


def baseline_output(shape, inputs):
    """REAL vendor bar: AITER biased_grouped_topk (dense), oracle fallback (FLAG)."""
    gate, bias = inputs
    return vendor_biased_grouped_topk_dense(
        gate, bias, shape["topk"], shape["n_groups"], shape["topk_group"],
        shape["E"], renorm=True, scale=1.0)
