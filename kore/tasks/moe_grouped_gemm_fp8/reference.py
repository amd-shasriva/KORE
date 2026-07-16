"""Reference + inputs for the fp8 grouped (segmented) expert GEMM (a8w8).

The fp8-quantized MoE gate_up projection. Each token is routed to ONE expert
(top-1); for expert e = expert_ids[m]:

    out[m] = (xq[m] * x_scale[m]) @ (wq[e] * w_scale[e]).T

with per-token activation scales ``x_scale[M,1]`` and per-channel weight scales
``w_scale[E,N,1]`` (a8w8). Token->expert counts are the unbalanced jagged trace
with a guaranteed 0-token last expert. Correctness oracle: fp32 of the
DEQUANTIZED per-group matmul (``_moe_common.grouped_gemm_fp8_fp32``) -- the fp8
rounding is shared by candidate + reference, so the SNR gate measures the
kernel's accumulation/scale-fold fidelity, not the quantization error. Perf
baseline: per-expert AITER ``gemm_a8w8`` (CK).

fp8 e4m3 is arch-selected (``FP8_DTYPE``): OCP ``e4m3fn`` on gfx950/CDNA4 (max
448), FNUZ ``e4m3fnuz`` on gfx942/CDNA3. xq ``[M,K]`` fp8, wq ``[E,N,K]`` fp8,
output ``[M,N]`` bf16.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kore.tasks._moe_common import grouped_gemm_fp8_fp32, make_routing, vendor_grouped_gemm_fp8  # noqa: E402

ENTRY = "grouped_gemm_fp8"
ATOL = 3e-2
RTOL = 3e-2


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"M": 4096, "E": 32, "N": 2048, "K": 4096}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def get_inputs(shape: dict, device="cuda", seed: int = 0):
    """Returns (xq [M,K] fp8, wq [E,N,K] fp8, x_scale [M,1] fp32, w_scale [E,N,1] fp32,
    expert_ids [M] int32)."""
    import torch

    from kore.tasks.aiter_ref import FP8_DTYPE, FP8_MAX
    g = torch.Generator(device=device).manual_seed(seed)
    M, E, N, K = shape["M"], shape["E"], shape["N"], shape["K"]
    xf = torch.randn((M, K), generator=g, device=device, dtype=torch.float32)
    wf = torch.randn((E, N, K), generator=g, device=device, dtype=torch.float32) * (1.0 / (K ** 0.5))
    # per-token activation quant
    xs = (xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / FP8_MAX).to(torch.float32)
    xq = (xf / xs).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    # per-expert per-output-channel weight quant
    ws = (wf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / FP8_MAX).to(torch.float32)  # [E,N,1]
    wq = (wf / ws).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    _, ti = make_routing(M, E, 1, device, g, renorm=False)
    expert_ids = ti[:, 0].contiguous().to(torch.int32)
    return (xq, wq, xs, ws, expert_ids)


def reference_output(shape, inputs):
    """Exact fp32-of-dequant per-group matmul oracle -> bf16 [M, N]."""
    xq, wq, xs, ws, expert_ids = inputs
    return grouped_gemm_fp8_fp32(xq, wq, xs, ws, expert_ids)


def candidate_output(fn, shape, inputs):
    xq, wq, xs, ws, expert_ids = inputs
    return fn(xq, wq, xs, ws, expert_ids)


def baseline_output(shape, inputs):
    """REAL vendor bar: per-expert AITER fp8 a8w8 GEMM (CK)."""
    xq, wq, xs, ws, expert_ids = inputs
    return vendor_grouped_gemm_fp8(xq, wq, xs, ws, expert_ids)
