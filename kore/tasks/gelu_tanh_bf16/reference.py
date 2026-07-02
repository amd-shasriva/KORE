"""Reference + inputs for the bf16 tanh-approx GELU activation task.

Elementwise tanh-approximation GELU (the activation used by GPT-2/Gemma-style
MLPs and the ``approximate='tanh'`` path):
    gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
bf16 in/out with fp32 math.

Correctness oracle: exact torch-fp32 ``F.gelu(x, approximate='tanh')``.
Perf baseline (driver --impl reference): ``F.gelu(approximate='tanh')`` which on
ROCm lowers to a fused elementwise kernel. AITER exposes only *gated* GELU
(``gelu_and_mul`` / ``gelu_tanh_and_mul``), not a standalone activation, so the
framework path is the honest production baseline.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"M": 4096, "N": 14336}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def get_inputs(shape: dict, dtype=torch.bfloat16, device="cuda", seed: int = 0):
    """Returns (x,): x[M,N] bf16."""
    g = torch.Generator(device=device).manual_seed(seed)
    M, N = shape["M"], shape["N"]
    x = torch.randn((M, N), generator=g, device=device, dtype=torch.float32).to(dtype)
    return (x,)


def gelu_tanh_ref(x: torch.Tensor) -> torch.Tensor:
    """Exact fp32 oracle: tanh-approximation GELU."""
    return F.gelu(x.float(), approximate="tanh").to(x.dtype)
