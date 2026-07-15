"""Reference + inputs for the affine bf16 LayerNorm task.

Standard LayerNorm over the last dim with learnable weight + bias:
    y = (x - mean) / sqrt(var + eps) * weight + bias
(mean/var are the per-row mean and *biased* variance, matching torch and the
AITER CK kernel). bf16 in/out, fp32 reductions.

Correctness oracle: exact torch-fp32 ``F.layer_norm``.
Perf baseline (driver --impl reference): AITER ``layer_norm`` (CK) - the kernel
the production serving stack calls.
"""

from __future__ import annotations

import torch

EPS = 1e-5


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"M": 4096, "N": 4096}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def get_inputs(shape: dict, dtype=torch.bfloat16, device="cuda", seed: int = 0):
    """Returns (x, weight, bias): x[M,N], weight[N], bias[N]."""
    g = torch.Generator(device=device).manual_seed(seed)
    M, N = shape["M"], shape["N"]
    x = torch.randn((M, N), generator=g, device=device, dtype=torch.float32).to(dtype)
    w = torch.randn((N,), generator=g, device=device, dtype=torch.float32).to(dtype)
    b = torch.randn((N,), generator=g, device=device, dtype=torch.float32).to(dtype)
    return x, w, b


def layernorm_ref(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                  eps: float = EPS) -> torch.Tensor:
    """Exact fp32 oracle."""
    xf = x.float()
    mean = xf.mean(dim=-1, keepdim=True)
    var = (xf - mean).pow(2).mean(dim=-1, keepdim=True)
    y = (xf - mean) * torch.rsqrt(var + eps)
    return (y * weight.float() + bias.float()).to(x.dtype)
