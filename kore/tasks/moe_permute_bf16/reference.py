"""Reference + inputs for the bf16 MoE dispatch/permute (scatter tokens to experts).

MoE needs the routed tokens grouped by expert before the grouped GEMM. Given the
token->expert assignment, a stable argsort produces ``sort_idx`` (each expert's
tokens become one contiguous, ascending-expert block; ties keep original token
order). The permute kernel then gathers ``permuted[i] = hidden[sort_idx[i]]`` --
the memory-bound indexed copy that physically scatters tokens to their experts
(the index/sort itself is a separate tiny op, e.g. aiter ``moe_sorting``).

Here ``sort_idx`` is precomputed in ``get_inputs`` from the unbalanced jagged
router (top-1, so M tokens -> M permuted rows), and the graded op is the gather.
Correctness oracle: the exact fp32 gather ``hidden[sort_idx]`` (unambiguous, fully
order-defined). Perf baseline: the framework indexed gather (a fused ROCm gather
kernel). hidden ``[M, D]`` bf16, sort_idx ``[M]`` int32, output ``[M, D]`` bf16.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kore.tasks._moe_common import make_routing, vendor_permute  # noqa: E402

ENTRY = "moe_permute"
ATOL = 0.0
RTOL = 0.0


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"M": 8192, "E": 256, "D": 7168}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def get_inputs(shape: dict, device="cuda", seed: int = 0):
    """Returns (hidden [M,D] bf16, sort_idx [M] int32).

    ``sort_idx`` = stable argsort of a real (unbalanced, top-1) token->expert
    assignment, so it groups tokens into contiguous ascending-expert blocks."""
    import torch

    g = torch.Generator(device=device).manual_seed(seed)
    M, E, D = shape["M"], shape["E"], shape["D"]
    hidden = torch.randn((M, D), generator=g, device=device, dtype=torch.float32).to(torch.bfloat16)
    _, ti = make_routing(M, E, 1, device, g, renorm=False)
    expert_ids = ti[:, 0].contiguous()
    sort_idx = torch.argsort(expert_ids, stable=True).to(torch.int32)
    return (hidden, sort_idx)


def reference_output(shape, inputs):
    """Exact gather oracle: permuted[i] = hidden[sort_idx[i]] -> bf16 [M, D]."""
    hidden, sort_idx = inputs
    return hidden[sort_idx.long()]


def candidate_output(fn, shape, inputs):
    hidden, sort_idx = inputs
    return fn(hidden, sort_idx)


def baseline_output(shape, inputs):
    """REAL bar: framework indexed gather (fused ROCm gather kernel)."""
    hidden, sort_idx = inputs
    return vendor_permute(hidden, sort_idx)
