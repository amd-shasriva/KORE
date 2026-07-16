"""Reference + inputs for fp8 a8w8 GEMM with a FUSED bias + fp8 REQUANT epilogue.

Serving fp8 GEMM whose output is itself fp8 (feeding the next fp8 op), fusing the
dequant/requant HBM round trip into the epilogue:
    acc   = (A_deq) @ (W_deq)^T + bias         (fp32; per-token A scale, per-channel W scale)
    Y_fp8 = clamp(round(acc / out_scale), +/- FP8_MAX)   (static-scale fp8 requant)
The candidate returns the fp8 output ``Y_fp8``; the grade compares the DEQUANTIZED views
``Y_fp8 * out_scale`` (both candidate and reference share the fp8 output rounding, so the
SNR gate measures the matmul + bias + requant fidelity, not the output quantization).

fp8 is arch-selected via the live ``kore.tasks.aiter_ref`` (OCP e4m3fn, max 448, on
gfx950). ``out_scale`` is a calibrated STATIC per-tensor output scale (amax of the fp32
epilogue / FP8_MAX), computed deterministically from the inputs in get_inputs.

Correctness oracle: the exact fp32 epilogue above; every scale (x_scale, w_scale,
out_scale) applied EXACTLY once.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _quant_common import fp8_dtype_max, gemm_fp8_requant_fp32, quant_rowwise_fp8  # noqa: E402

ENTRY = "gemm"
ATOL = 5e-1
RTOL = 5e-2


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"M": 4096, "N": 4096, "K": 4096}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def get_inputs(shape: dict, device="cuda", seed: int = 0):
    """Returns (xq[M,K] fp8, wq[N,K] fp8, x_scale[M,1], w_scale[1,N], bias[N], out_scale[])."""
    import torch

    g = torch.Generator(device=device).manual_seed(seed)
    M, N, K = shape["M"], shape["N"], shape["K"]
    _, fmax = fp8_dtype_max()
    a = torch.randn((M, K), generator=g, device=device, dtype=torch.float32)
    w = torch.randn((N, K), generator=g, device=device, dtype=torch.float32)
    bias = torch.randn((N,), generator=g, device=device, dtype=torch.float32) * 0.1
    xq, x_scale = quant_rowwise_fp8(a)                       # [M,K], [M,1]
    wq, w_scale_col = quant_rowwise_fp8(w)                   # [N,K], [N,1]
    w_scale = w_scale_col.reshape(1, N).contiguous()         # CK [1,N]
    # Calibrated STATIC output scale: amax of the fp32 epilogue / FP8_MAX (a one-time
    # calibration, deterministic from the inputs; not peeking at the candidate).
    acc = (xq.float() * x_scale.float()) @ (wq.float() * w_scale.float().reshape(-1, 1)).t()
    acc = acc + bias.reshape(1, -1)
    out_scale = (acc.abs().amax().clamp(min=1e-12) / fmax).to(torch.float32).reshape(())
    return (xq, wq, x_scale, w_scale, bias, out_scale)


def reference_output(shape, inputs):
    """Exact fp32 dequant-matmul + bias + fp8 requant, returned DEQUANTIZED -> bf16 [M,N]."""
    xq, wq, x_scale, w_scale, bias, out_scale = inputs
    return gemm_fp8_requant_fp32(xq, wq, x_scale, w_scale, bias, out_scale)


def candidate_output(fn, shape, inputs):
    """Candidate returns the fp8 output; compare on its dequantized (bf16) view."""
    import torch

    xq, wq, x_scale, w_scale, bias, out_scale = inputs
    yq = fn(xq, wq, x_scale, w_scale, bias, out_scale)       # fp8 [M,N]
    return (yq.float() * out_scale.float()).to(torch.bfloat16)


def baseline_output(shape, inputs):
    """REAL vendor bar: AITER ``gemm_a8w8`` (fp8 GEMM) + an UNFUSED torch requant epilogue.

    The vendor GEMM is the real serving kernel; the fused-epilogue candidate beats this
    multi-kernel path by fusing bias + fp8 requant (removing an HBM round trip). Returned
    dequantized (bf16) to match the oracle."""
    import torch

    from _quant_common import fp8_dtype_max
    from kore.tasks.aiter_ref import aiter_gemm_a8w8

    xq, wq, x_scale, w_scale, bias, out_scale = inputs
    fp8, fmax = fp8_dtype_max()
    y = aiter_gemm_a8w8(xq, wq, x_scale, w_scale, out_dtype=torch.bfloat16)   # [M,N] bf16
    acc = y.float() + bias.reshape(1, -1)
    yq = (acc / out_scale.float()).clamp(-fmax, fmax).to(fp8)
    return (yq.float() * out_scale.float()).to(torch.bfloat16)
