"""Reference + inputs for FUSED SiLU-gate-mul -> dynamic per-token fp8 quant.

The MoE / FFN down-projection prologue: ``silu(x_gate) * x_up`` followed by a
dynamic per-token fp8 quant, fused into ONE kernel instead of AITER's
``silu_and_mul`` + ``dynamic_per_token_scaled_quant`` (two kernels + an HBM round
trip on the [M,inter] intermediate). Input x is [M, 2*inter] (gate || up halves
concatenated on the last dim); output is [M, inter] fp8 + [M,1] fp32 scales.

fp8 e4m3 encoding is arch-selected via ``FP8_DTYPE`` (OCP ``e4m3fn`` on
gfx950/CDNA4 MI350X/MI355X - native; FNUZ ``e4m3fnuz`` on gfx942/CDNA3).
Oracle: fp32 silu_mul, then the exact torch per-token quant. Correctness gate
(see driver): (a) SNR of dequant ``xq*scale`` vs the TRUE fp32 silu_mul value,
(b) candidate scales/codes match the oracle to fp8 rounding.
Perf baseline (driver --impl reference): AITER silu_and_mul + dynamic_per_token
quant (the two-kernel serving bar the fused kernel must beat).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from kore.tasks.aiter_ref import FP8_DTYPE, FP8_MAX


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"M": 4096, "N": 8192}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def get_inputs(shape: dict, dtype=torch.bfloat16, device="cuda", seed: int = 0):
    """Returns (x[M,N],): N = 2*inter (gate half || up half)."""
    g = torch.Generator(device=device).manual_seed(seed)
    M, N = shape["M"], shape["N"]
    x = torch.randn((M, N), generator=g, device=device, dtype=torch.float32).to(dtype)
    return (x,)


def silu_mul_ref(x: torch.Tensor) -> torch.Tensor:
    """fp32 silu(gate)*up -> [M, inter] fp32 (the pre-quant value; SNR reference)."""
    inter = x.shape[-1] // 2
    g, u = x[:, :inter].float(), x[:, inter:].float()
    return F.silu(g) * u


def fused_ref(x: torch.Tensor):
    """Exact oracle. Returns (xq fp8 [M,inter], scale fp32 [M,1])."""
    y = silu_mul_ref(x)
    amax = y.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / FP8_MAX
    xq = (y / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return xq, scale


def dequant(xq: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return xq.float() * scale.float()
