"""Reference + inputs for the bf16 batched expert GEMM ``C[e] = A[e] @ B[e]^T``.

The balanced expert-parallel case: E experts, each processing exactly ``m``
tokens, so the per-expert GEMMs stack into one batched GEMM. A ``[E, m, K]``
(the per-expert token activations), B ``[E, N, K]`` (the per-expert weight, e.g.
a gate/up/down projection), output ``[E, m, N]``. Correctness oracle: exact fp32
batched matmul (``_moe_common.batched_gemm_fp32``). Perf baseline: AITER
``batched_gemm_bf16`` (CK), which the wrapper falls back to ``torch.bmm`` ->
hipBLASLt for.

Scaled 1/sqrt(K) init keeps the accumulated magnitude ~O(1) for stable bf16.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kore.tasks._moe_common import batched_gemm_fp32, vendor_batched_gemm_bf16  # noqa: E402

ENTRY = "batched_gemm"
ATOL = 2e-2
RTOL = 2e-2


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"E": 8, "m": 512, "N": 512, "K": 512}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def get_inputs(shape: dict, device="cuda", seed: int = 0):
    """Returns (a [E,m,K] bf16, b [E,N,K] bf16)."""
    import torch

    g = torch.Generator(device=device).manual_seed(seed)
    E, m, N, K = shape["E"], shape["m"], shape["N"], shape["K"]
    sc = 1.0 / (K ** 0.5)
    a = (torch.randn((E, m, K), generator=g, device=device, dtype=torch.float32) * sc).to(torch.bfloat16)
    b = (torch.randn((E, N, K), generator=g, device=device, dtype=torch.float32) * sc).to(torch.bfloat16)
    return (a, b)


def reference_output(shape, inputs):
    """Exact fp32 batched A@B^T oracle -> bf16 [E, m, N]."""
    a, b = inputs
    return batched_gemm_fp32(a, b)


def candidate_output(fn, shape, inputs):
    a, b = inputs
    return fn(a, b)


def baseline_output(shape, inputs):
    """REAL vendor bar: AITER batched bf16 GEMM (torch.bmm -> hipBLASLt fallback)."""
    a, b = inputs
    return vendor_batched_gemm_bf16(a, b)
