"""Reference + inputs for fused add-RMSNorm (a hot transformer-block kernel).

Computes, in one fused op:
    added = x + residual
    y     = RMSNorm(added) * weight
This is the residual-add + norm that occurs at every transformer sub-layer.

Correctness oracle: exact torch-fp32 (both ``y`` and the updated residual).
Perf baseline (driver --impl reference): AITER ``fused_add_rms_norm_cu`` (in
place), the kernel the serving stack actually calls.
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
    residual = torch.randn((M, N), generator=g, device=device, dtype=torch.float32).to(dtype)
    w = torch.randn((N,), generator=g, device=device, dtype=torch.float32).to(dtype)
    return x, residual, w


def fused_add_rmsnorm_ref(x, residual, weight, eps: float = EPS):
    """Exact fp32 oracle. Returns (y, added) both cast back to input dtype."""
    added = x.float() + residual.float()
    var = added.pow(2).mean(dim=-1, keepdim=True)
    y = added * torch.rsqrt(var + eps) * weight.float()
    return y.to(x.dtype), added.to(x.dtype)
