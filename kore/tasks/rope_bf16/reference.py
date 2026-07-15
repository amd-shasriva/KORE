"""Reference + inputs for the bf16 RoPE (rotary position embedding) task.

NEOX-style rotary embedding applied over the full head dim ``D`` of a
(S, B, H, D) tensor. For rotation angles ``theta`` (shape (S, D//2)):
    cos = cat(cos(theta), cos(theta)), sin = cat(sin(theta), sin(theta))
    out = x * cos + rotate_neox(x) * sin
where ``rotate_neox(x) = cat(-x[..., D//2:], x[..., :D//2])``. This matches the
AITER HIP ``rope_fwd`` kernel with rotate_style=NEOX, reuse_freqs_front_part=True,
nope_first=False (verified on-box).

Correctness oracle: the exact torch-fp32 NEOX rope above.
Perf baseline (driver --impl reference): AITER ``rope_fwd`` - the vendor HIP rope
the serving stack calls. ``freqs`` (the raw angles) are fed to both.
"""

from __future__ import annotations

import torch

ROPE_BASE = 10000.0


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"S": 2048, "B": 2, "H": 32, "D": 128}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def get_inputs(shape: dict, dtype=torch.bfloat16, device="cuda", seed: int = 0):
    """Returns (x, freqs): x[S,B,H,D] bf16, freqs[S,1,1,D//2] fp32 (angles)."""
    g = torch.Generator(device=device).manual_seed(seed)
    S, B, H, D = shape["S"], shape["B"], shape["H"], shape["D"]
    x = torch.randn((S, B, H, D), generator=g, device=device, dtype=torch.float32).to(dtype)
    inv_freq = 1.0 / (ROPE_BASE ** (torch.arange(0, D, 2, device=device, dtype=torch.float32) / D))
    t = torch.arange(S, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # (S, D//2)
    freqs = freqs.view(S, 1, 1, D // 2).contiguous()
    return x, freqs


def rope_ref(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Exact fp32 NEOX oracle. x[S,B,H,D], freqs[S,1,1,D//2] -> [S,B,H,D]."""
    xf = x.float()
    D = xf.shape[-1]
    cos = torch.cos(freqs).float()
    sin = torch.sin(freqs).float()
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    x1 = xf[..., : D // 2]
    x2 = xf[..., D // 2:]
    rot = torch.cat([-x2, x1], dim=-1)
    return (xf * cos + rot * sin).to(x.dtype)
