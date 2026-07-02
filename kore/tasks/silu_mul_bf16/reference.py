"""Reference + inputs for the gated-MLP activation SiLU(gate)*up.

The fused MLP activation used by Llama/Mistral-style gated FFNs: a single
input of width ``2*N`` is split into (gate, up); output is ``silu(gate) * up``
of width ``N``.

Correctness oracle: exact torch-fp32 silu*mul.
Perf baseline (driver --impl reference): AITER ``silu_and_mul``.
"""

from __future__ import annotations

import torch


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"M": 4096, "N": 14336}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def get_inputs(shape: dict, dtype=torch.bfloat16, device="cuda", seed: int = 0):
    """Returns a single (M, 2*N) tensor: columns [0:N)=gate, [N:2N)=up."""
    g = torch.Generator(device=device).manual_seed(seed)
    M, N = shape["M"], shape["N"]
    x = torch.randn((M, 2 * N), generator=g, device=device, dtype=torch.float32).to(dtype)
    return (x,)


def silu_mul_ref(x: torch.Tensor) -> torch.Tensor:
    """Exact fp32 oracle. Input (M, 2N) -> output (M, N)."""
    n = x.shape[-1] // 2
    xf = x.float()
    gate, up = xf[..., :n], xf[..., n:]
    return (torch.nn.functional.silu(gate) * up).to(x.dtype)
