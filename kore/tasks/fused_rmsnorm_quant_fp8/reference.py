"""Reference + inputs for FUSED RMSNorm -> dynamic per-token fp8 quant.

The serving prologue for an fp8 (W8A8) GEMM: ``RMSNorm(x) * w`` followed by a
dynamic per-token fp8 quant, fused into ONE kernel instead of two AITER kernels
plus a full-tensor HBM round trip. gfx942 / CDNA3 fp8 e4m3 is the **FNUZ**
variant (``torch.float8_e4m3fnuz``); the OCP ``e4m3fn`` variant silently
mismatches AITER.

Oracle: fp32 RMSNorm, then the exact torch per-token quant (codes + scales)
    scale[m] = rowamax(y[m]) / FP8_MAX,  xq[m] = round(y[m]/scale[m])  (clamped)
so ``y[m] ~= xq[m] * scale[m]``. Correctness gate (see driver): (a) SNR of the
dequantized output ``xq*scale`` vs the TRUE fp32 normed activation, and (b) the
candidate scales/codes match the oracle to fp8 rounding.
Perf baseline (driver --impl reference): AITER ``rms_norm`` + ``dynamic_per_token
_scaled_quant`` (the two-kernel serving bar the fused kernel must beat).
"""

from __future__ import annotations

import torch

from kore.tasks.aiter_ref import FP8_DTYPE, FP8_MAX

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
    """Returns (x[M,N] bf16 activation, w[N] bf16 RMSNorm weight)."""
    g = torch.Generator(device=device).manual_seed(seed)
    M, N = shape["M"], shape["N"]
    x = torch.randn((M, N), generator=g, device=device, dtype=torch.float32).to(dtype)
    gw = torch.Generator(device=device).manual_seed(seed + 1)
    w = (torch.randn((N,), generator=gw, device=device, dtype=torch.float32) * 0.1 + 1.0).to(dtype)
    return (x, w)


def rmsnorm_ref(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """fp32 RMSNorm(x)*w -> [M,N] fp32 (the pre-quant activation; SNR reference)."""
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * w.float()


def fused_ref(x: torch.Tensor, w: torch.Tensor):
    """Exact oracle. Returns (xq fp8 [M,N], scale fp32 [M,1])."""
    y = rmsnorm_ref(x, w)
    amax = y.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / FP8_MAX
    xq = (y / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return xq, scale


def dequant(xq: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return xq.float() * scale.float()
