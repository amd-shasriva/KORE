"""Reference + inputs for the dynamic per-token fp8 quantization task.

Dynamic (activation) per-token quant used before fp8 W8A8 GEMM: each row of a
bf16 activation is quantized to fp8 (arch ``FP8_DTYPE``) with its own scale
    scale[m] = rowamax[m] / FP8_MAX
    xq[m]    = round(x[m] / scale[m])  (clamped to +/-FP8_MAX)
so that ``x[m] ~= xq[m] * scale[m]``.

The fp8 e4m3 encoding is arch-selected: OCP ``e4m3fn`` on gfx950/CDNA4
(MI350X/MI355X — native), FNUZ ``e4m3fnuz`` on gfx942/CDNA3.

Correctness oracle: the exact torch quant above (codes + scales). The gate
checks (a) SNR of the dequantized activation vs the original, and (b) that the
candidate's scales and fp8 codes match the oracle.
Perf baseline (driver --impl reference): AITER ``dynamic_per_token_scaled_quant``.
"""

from __future__ import annotations

import torch

from kore.tasks.aiter_ref import FP8_DTYPE, FP8_MAX


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"M": 4096, "N": 4096}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def get_inputs(shape: dict, dtype=torch.bfloat16, device="cuda", seed: int = 0):
    """Returns (x,): x[M,N] bf16 activation."""
    g = torch.Generator(device=device).manual_seed(seed)
    M, N = shape["M"], shape["N"]
    x = torch.randn((M, N), generator=g, device=device, dtype=torch.float32).to(dtype)
    return (x,)


def per_token_quant_ref(x: torch.Tensor):
    """Exact oracle. Returns (xq fp8 [M,N], scale fp32 [M,1])."""
    xf = x.float()
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / FP8_MAX
    xq = (xf / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return xq, scale


def dequant(xq: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return xq.float() * scale.float()
