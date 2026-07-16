"""Reference + inputs for W4A8 GEMM (int4 per-channel weight, fp8 per-token activation).

Mixed-precision serving GEMM (QServe/QoQ style): the activation A[M,K] is quantized to
fp8 with a per-ROW (per-token) scale ``x_scale[M,1]``; the weight W[N,K] is quantized to
symmetric per-OUTPUT-CHANNEL int4 (codes 0..15 <-> values -8..7) with a per-row scale
``w_scale[N,1]``, packed 2 nibbles/byte along K. Computes ``Y = (A_deq) @ (W_deq)^T`` in
bf16, where ``A_deq = xq * x_scale`` and ``W_deq[n,k] = (code(n,k)-8) * w_scale[n]``.

fp8 is arch-selected via the live ``kore.tasks.aiter_ref`` (OCP e4m3fn on gfx950).

Correctness oracle: exact fp32 matmul of the dequantized operands. The two scales live on
DIFFERENT axes (activation per-row, weight per-row-of-W), so the classic bug is applying
one on the wrong axis; the oracle pins ``Y[m,n] = x_scale[m]*w_scale[n]*sum_k xq[m,k]*(code-8)``.
The fp8 + int4 rounding is shared by candidate + reference. get_inputs GUARDS K even.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kore.tasks._quant_common import matmul_w4a8_fp32, quant_pack_int4_perchannel, quant_rowwise_fp8  # noqa: E402

ENTRY = "gemm"
ATOL = 5e-1
RTOL = 5e-2


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"M": 16, "N": 4096, "K": 4096}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def get_inputs(shape: dict, device="cuda", seed: int = 0):
    """Returns (xq[M,K] fp8, x_scale[M,1] fp32, w_packed[N,K//2] u8, w_scale[N,1] fp32)."""
    import torch

    M, N, K = shape["M"], shape["N"], shape["K"]
    assert K % 2 == 0, f"int4 nibble packing needs K even (got K={K}); K=4095 illegal"
    g = torch.Generator(device=device).manual_seed(seed)
    a = torch.randn((M, K), generator=g, device=device, dtype=torch.float32)
    w = torch.randn((N, K), generator=g, device=device, dtype=torch.float32)
    xq, x_scale = quant_rowwise_fp8(a)                       # [M,K], [M,1]
    w_packed, w_scale = quant_pack_int4_perchannel(w)        # [N,K//2], [N,1]
    return (xq, x_scale, w_packed, w_scale)


def reference_output(shape, inputs):
    """Exact fp32 dequant-matmul oracle -> bf16 [M,N]."""
    xq, x_scale, w_packed, w_scale = inputs
    return matmul_w4a8_fp32(xq, x_scale, w_packed, w_scale, shape["K"])


def candidate_output(fn, shape, inputs):
    xq, x_scale, w_packed, w_scale = inputs
    return fn(xq, x_scale, w_packed, w_scale)


def baseline_output(shape, inputs):
    """REAL vendor bar: dequant fp8 activation + int4 weight to bf16 + hipBLASLt matmul."""
    import torch

    from kore.tasks._quant_common import unpack_dequant_int4_perchannel
    from kore.tasks.aiter_ref import hipblaslt_gemm_bf16

    xq, x_scale, w_packed, w_scale = inputs
    a_deq = (xq.float() * x_scale.float()).to(torch.bfloat16)
    w_deq = unpack_dequant_int4_perchannel(w_packed, w_scale, shape["K"]).to(torch.bfloat16)
    return hipblaslt_gemm_bf16(a_deq, w_deq.t().contiguous())
