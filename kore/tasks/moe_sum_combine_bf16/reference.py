"""Reference + inputs for the bf16 MoE weighted combine (moe_sum reduce).

After each token's top-k experts have produced their per-slot outputs
``y[m, k, :]``, MoE combines them back into a single per-token vector by the
router weights:

    out[m] = sum_k topk_weight[m, k] * y[m, k, :]

(blueprint M13). This is the "unpermute + weighted reduce" half of MoE (the
counterpart to the dispatch/permute). Correctness oracle: the exact fp32
weighted reduce (``_moe_common.moe_sum_fp32``). Perf baseline: the framework
weighted reduce (a fused ROCm reduction).

y ``[M, topk, D]`` bf16, topk_weight ``[M, topk]`` fp32, output ``[M, D]`` bf16.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kore.tasks._moe_common import moe_sum_fp32, vendor_moe_sum  # noqa: E402

ENTRY = "moe_sum"
ATOL = 2e-2
RTOL = 2e-2


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"M": 8192, "topk": 8, "D": 7168}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def get_inputs(shape: dict, device="cuda", seed: int = 0):
    """Returns (y [M,topk,D] bf16, topk_weight [M,topk] fp32).

    Weights are normalized per token (sum 1 over the top-k slots), like a real
    renormalized router; y is scaled 1/sqrt(D) so the combine stays well-scaled."""
    import torch

    g = torch.Generator(device=device).manual_seed(seed)
    M, topk, D = shape["M"], shape["topk"], shape["D"]
    sc = 1.0 / (D ** 0.5)
    y = (torch.randn((M, topk, D), generator=g, device=device, dtype=torch.float32) * sc).to(torch.bfloat16)
    w = torch.rand((M, topk), generator=g, device=device, dtype=torch.float32) + 1e-3
    w = (w / w.sum(dim=-1, keepdim=True)).to(torch.float32)
    return (y, w)


def reference_output(shape, inputs):
    """Exact fp32 weighted-combine oracle -> bf16 [M, D]."""
    y, w = inputs
    return moe_sum_fp32(y, w)


def candidate_output(fn, shape, inputs):
    y, w = inputs
    return fn(y, w)


def baseline_output(shape, inputs):
    """REAL bar: framework weighted reduce (fused ROCm reduction)."""
    y, w = inputs
    return vendor_moe_sum(y, w)
