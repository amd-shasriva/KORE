"""Reference + inputs for the bf16 grouped (segmented) expert GEMM.

Each token is routed to exactly ONE expert (top-1). The op is the per-expert
GEMM ``out[m] = hidden[m] @ w[e]^T`` where ``e = expert_ids[m]`` -- i.e. group
the tokens by expert and run one GEMM per group (the MoE gate_up projection
stage, blueprint G11 / M6). Token->expert counts are the unbalanced jagged trace
with a guaranteed 0-token last expert (that expert's GEMM is simply skipped).

Correctness oracle: exact fp32 per-group matmul (``_moe_common.grouped_gemm_fp32``),
output preserved in the ORIGINAL token order so the result is unambiguous (no
sort). Perf baseline: one ``torch.matmul`` per non-empty expert (hipBLASLt) -- the
honest production dense grouped bar; the candidate wins by fusing the launches.

hidden ``[M, K]`` bf16, w ``[E, N, K]`` bf16 (per-expert weight), expert_ids
``[M]`` int32, output ``[M, N]`` bf16.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kore.tasks._moe_common import grouped_gemm_fp32, make_routing, vendor_grouped_gemm_bf16  # noqa: E402

ENTRY = "grouped_gemm"
ATOL = 2e-2
RTOL = 2e-2


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"M": 4096, "E": 32, "N": 2048, "K": 4096}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def get_inputs(shape: dict, device="cuda", seed: int = 0):
    """Returns (hidden [M,K] bf16, w [E,N,K] bf16, expert_ids [M] int32)."""
    import torch

    g = torch.Generator(device=device).manual_seed(seed)
    M, E, N, K = shape["M"], shape["E"], shape["N"], shape["K"]
    sc = 1.0 / (K ** 0.5)
    hidden = (torch.randn((M, K), generator=g, device=device, dtype=torch.float32) * sc).to(torch.bfloat16)
    w = (torch.randn((E, N, K), generator=g, device=device, dtype=torch.float32) * sc).to(torch.bfloat16)
    # top-1 routing -> one expert per token; last expert guaranteed 0 tokens.
    _, ti = make_routing(M, E, 1, device, g, renorm=False)
    expert_ids = ti[:, 0].contiguous().to(torch.int32)
    return (hidden, w, expert_ids)


def reference_output(shape, inputs):
    """Exact fp32 per-group matmul oracle -> bf16 [M, N] (original token order)."""
    hidden, w, expert_ids = inputs
    return grouped_gemm_fp32(hidden, w, expert_ids)


def candidate_output(fn, shape, inputs):
    hidden, w, expert_ids = inputs
    return fn(hidden, w, expert_ids)


def baseline_output(shape, inputs):
    """REAL vendor bar: one torch.matmul (hipBLASLt) per non-empty expert."""
    hidden, w, expert_ids = inputs
    return vendor_grouped_gemm_bf16(hidden, w, expert_ids)
