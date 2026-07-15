"""Reference + inputs for the AMD-correct bf16 RMSNorm task.

Correctness oracle: exact torch-fp32 RMSNorm (mathematically exact).
Perf baseline (in driver, --impl reference): AITER ``rms_norm`` (CK) - the
kernel the production serving stack actually calls, NOT unfused torch.
"""

from __future__ import annotations

import torch

EPS = 1e-6


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"M": 4096, "N": 4096}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def get_inputs(shape: dict, dtype=torch.bfloat16, device="cuda", seed: int = 0):
    g = torch.Generator(device=device).manual_seed(seed)
    M, N = shape["M"], shape["N"]
    x = torch.randn((M, N), generator=g, device=device, dtype=torch.float32).to(dtype)
    w = torch.randn((N,), generator=g, device=device, dtype=torch.float32).to(dtype)
    return x, w


def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Exact fp32 oracle."""
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    y = xf * torch.rsqrt(var + eps)
    return (y * weight.float()).to(x.dtype)
