"""Reference + inputs for MXFP4 a4w4 GEMM (both operands OCP microscaling FP4).

MI350X/gfx950 headline capability: OCP Microscaling FP4. BOTH the activation A[M,K] and
the weight W[N,K] are quantized to 4-bit E2M1 codes with a shared E8M0 (power-of-two)
scale per 32 consecutive K elements (the OCP MX spec):
    A_deq[m,k] = e2m1(a_code) * 2^(a_e8m0[m, k//32] - 127)
    W_deq[n,k] = e2m1(w_code) * 2^(w_e8m0[n, k//32] - 127)
    Y = A_deq @ W_deq^T                (bf16 out)
Codes are packed 2 nibbles/byte along K (even-K low nibble, odd-K high nibble).

This is the both-operands-fp4 complement of the live weight-only ``gemm_mxfp4`` (which
keeps the activation in bf16). It is native on gfx950/CDNA4 (scaled MFMA), emulated only
on gfx942.

Correctness oracle: exact fp32 matmul of the dequantized mxfp4 A and W (each block scale
applied EXACTLY once). The mxfp4 rounding is shared by candidate + reference, so the SNR
gate measures the kernel's MFMA-accumulation fidelity, not the quantization. get_inputs
GUARDS K%32==0 (K=4095 is illegal for the MX scale groups and raises).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kore.tasks._quant_common import MX_BLOCK, matmul_mxfp4_a4w4_fp32, quant_pack_mxfp4  # noqa: E402

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
    """Returns (a_packed[M,K//2] u8, a_e8m0[M,K//32] u8,
                w_packed[N,K//2] u8, w_e8m0[N,K//32] u8)."""
    import torch

    M, N, K = shape["M"], shape["N"], shape["K"]
    # MX scale-group guard (edge S6): K MUST be a multiple of 32. K=4095 is illegal.
    assert K % MX_BLOCK == 0, f"MXFP4 needs K %% {MX_BLOCK} == 0 (got K={K}); K=4095 illegal"
    g = torch.Generator(device=device).manual_seed(seed)
    a = torch.randn((M, K), generator=g, device=device, dtype=torch.float32)
    w = torch.randn((N, K), generator=g, device=device, dtype=torch.float32)
    a_packed, a_e8m0 = quant_pack_mxfp4(a)
    w_packed, w_e8m0 = quant_pack_mxfp4(w)
    return (a_packed, a_e8m0, w_packed, w_e8m0)


def reference_output(shape, inputs):
    """Exact fp32 dequant-matmul oracle on mxfp4 A and W -> bf16 [M,N]."""
    a_packed, a_e8m0, w_packed, w_e8m0 = inputs
    K = shape["K"]
    return matmul_mxfp4_a4w4_fp32(a_packed, a_e8m0, w_packed, w_e8m0, K)


def candidate_output(fn, shape, inputs):
    a_packed, a_e8m0, w_packed, w_e8m0 = inputs
    return fn(a_packed, a_e8m0, w_packed, w_e8m0)


def baseline_output(shape, inputs):
    """REAL vendor bar: dequant both operands to bf16 + hipBLASLt matmul.

    torch.matmul on bf16 lowers to hipBLASLt on ROCm (the production dense-GEMM lib), so
    this is a real vendor baseline the fp4 kernel beats on HBM traffic (~4x less) and via
    the native CDNA4 MX matrix path. The native fp4 vendor symbol is ``aiter.gemm_a4w4``
    (1x32 e8m0); switch the baseline to it once its layout/signature is confirmed on the
    node (see ../VERIFICATION_CHECKLIST.md FLAG)."""
    import torch

    from kore.tasks._quant_common import unpack_dequant_mxfp4
    from kore.tasks.aiter_ref import hipblaslt_gemm_bf16

    a_packed, a_e8m0, w_packed, w_e8m0 = inputs
    K = shape["K"]
    a_deq = unpack_dequant_mxfp4(a_packed, a_e8m0, K).to(torch.bfloat16)
    w_deq = unpack_dequant_mxfp4(w_packed, w_e8m0, K).to(torch.bfloat16)
    return hipblaslt_gemm_bf16(a_deq, w_deq.t().contiguous())
