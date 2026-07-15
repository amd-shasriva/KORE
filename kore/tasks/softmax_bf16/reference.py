"""Reference + inputs for the bf16 row-softmax task.

Numerically-stable softmax over the last dim (subtract row-max, exp, normalize),
bf16 in/out with fp32 math - the softmax used in attention scores / logits.

Correctness oracle: exact torch-fp32 ``softmax``.
Perf baseline (driver --impl reference): ``torch.softmax`` which on ROCm lowers
to a fused MIOpen/rocm softmax kernel (AITER exposes only ``topk_softmax`` for
MoE routing, no standalone dense row-softmax), so the framework path is the
honest production baseline.
"""

from __future__ import annotations

import torch


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"M": 4096, "N": 4096}
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


def softmax_ref(x: torch.Tensor) -> torch.Tensor:
    """Exact fp32 oracle: row softmax over the last dim."""
    xf = x.float()
    xf = xf - xf.max(dim=-1, keepdim=True).values
    e = torch.exp(xf)
    return (e / e.sum(dim=-1, keepdim=True)).to(x.dtype)
