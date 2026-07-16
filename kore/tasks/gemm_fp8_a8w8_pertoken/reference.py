"""Reference + inputs for fp8 (a8w8) GEMM with DYNAMIC PER-TOKEN activation scale.

The serving-critical quantized GEMM: the activation A[M,K] is quantized to fp8 with a
per-ROW (per-token) scale ``x_scale[M,1]`` computed dynamically from each row's amax, and
the weight W[N,K] is quantized to fp8 with a per-OUTPUT-CHANNEL scale ``w_scale[1,N]``.
Computes ``Y = (A_deq) @ (W_deq)^T`` in bf16.

fp8 is arch-selected via the live ``kore.tasks.aiter_ref``: OCP ``float8_e4m3fn`` (max
448) on gfx950/CDNA4, FNUZ ``float8_e4m3fnuz`` (max 240) on gfx942/CDNA3. Candidate +
oracle both consume ``FP8_DTYPE``.

Layout matches AITER ``gemm_a8w8`` (CK): XQ[M,K], WQ[N,K] (so the op does ``X @ W^T``),
x_scale[M,1], w_scale[1,N], both fp32.

Correctness oracle: exact torch-fp32 matmul of the DEQUANTIZED fp8 operands, applying
each scale EXACTLY once. The fp8 rounding is shared by candidate + reference, so the SNR
gate measures the kernel's numerical fidelity, not the quantization itself.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kore.tasks._quant_common import fp8_dtype_max, matmul_a8w8_fp32, quant_rowwise_fp8  # noqa: E402

ENTRY = "gemm"
ATOL = 5e-1   # magnitude-scaled; SNR is the real gate (fp8 GEMM)
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
    """Returns (xq[M,K] fp8, wq[N,K] fp8, x_scale[M,1] fp32, w_scale[1,N] fp32)."""
    import torch

    g = torch.Generator(device=device).manual_seed(seed)
    M, N, K = shape["M"], shape["N"], shape["K"]
    if bool(shape.get("TA", 0)):
        # Transposed A: xq is a NON-CONTIGUOUS [M,K] view of a [K,M] fp8 buffer, so the
        # kernel must honor strides (layout edge L1). Per-token scale is still over K.
        fp8, fmax = fp8_dtype_max()
        buf = torch.randn((K, M), generator=g, device=device, dtype=torch.float32)
        sc = buf.abs().amax(dim=0, keepdim=True).clamp(min=1e-12) / fmax     # [1,M]
        xq = (buf / sc).clamp(-fmax, fmax).to(fp8).t()                       # [M,K] view
        x_scale = sc.reshape(M, 1).to(torch.float32)
    else:
        a = torch.randn((M, K), generator=g, device=device, dtype=torch.float32)
        xq, x_scale = quant_rowwise_fp8(a)                                   # [M,K], [M,1]
    w = torch.randn((N, K), generator=g, device=device, dtype=torch.float32)
    wq, w_scale_col = quant_rowwise_fp8(w)                                    # [N,K], [N,1]
    w_scale = w_scale_col.reshape(1, N).contiguous()                         # CK [1,N]
    return (xq, wq, x_scale, w_scale)


def reference_output(shape, inputs):
    """Exact fp32 dequant-matmul oracle -> bf16 [M,N]."""
    xq, wq, x_scale, w_scale = inputs
    return matmul_a8w8_fp32(xq, wq, x_scale, w_scale)


def candidate_output(fn, shape, inputs):
    xq, wq, x_scale, w_scale = inputs
    return fn(xq, wq, x_scale, w_scale)


def baseline_output(shape, inputs):
    """REAL vendor bar: AITER ``gemm_a8w8`` (CK fp8 per-token/per-channel scaled GEMM)."""
    import torch

    from kore.tasks.aiter_ref import aiter_gemm_a8w8

    xq, wq, x_scale, w_scale = inputs
    return aiter_gemm_a8w8(xq, wq, x_scale, w_scale, out_dtype=torch.bfloat16)
