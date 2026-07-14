"""Seed FUSED SiLU-gate-mul -> per-token fp8 quant (arch-aware fp8: OCP e4m3fn on
gfx950/CDNA4 MI350X/MI355X, FNUZ on gfx942/CDNA3).

Exposes ``quant(x) -> (xq, scale)`` where x is [M, 2*inter] bf16 (gate || up),
xq is [M, inter] fp8 (arch dtype) and scale is [M,1] fp32:
    y        = silu(x[:, :inter]) * x[:, inter:]      (fp32 math)
    scale[m] = rowamax(y[m]) / FP8_MAX
    xq[m]    = clamp(y[m] / scale[m], +/-FP8_MAX) -> fp8

One program per row, two column-streamed passes (rowamax of silu_mul, then
quantize+store) so wide rows fit in registers. Fuses AITER's silu_and_mul +
dynamic_per_token quant (two kernels + an HBM round trip) into one: a correct
baseline the KORE policy learns to optimize (e.g. cache silu_mul in LDS to skip
the recompute, vectorize the gate||up loads, tune BLOCK_N / num_warps).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# gfx950/CDNA4 (MI350X/MI355X) fp8 e4m3 is the OCP variant e4m3fn (max 448.0) --
# the CDNA4-native format the reference oracle also quantizes to on this arch.
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)


@triton.jit
def _silu_mul_quant_kernel(
    x_ptr, y_ptr, s_ptr,
    stride_xm, stride_ym,
    INTER,
    FP8_MAX: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_xm
    y_row = y_ptr + row * stride_ym

    amax = 1e-12
    for start in range(0, INTER, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < INTER
        g = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(x_row + INTER + offs, mask=mask, other=0.0).to(tl.float32)
        val = (g * tl.sigmoid(g)) * u
        amax = tl.maximum(amax, tl.max(tl.abs(val), axis=0))
    scale = amax / FP8_MAX
    tl.store(s_ptr + row, scale)

    inv = 1.0 / scale
    for start in range(0, INTER, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < INTER
        g = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(x_row + INTER + offs, mask=mask, other=0.0).to(tl.float32)
        val = (g * tl.sigmoid(g)) * u
        q = tl.minimum(tl.maximum(val * inv, -FP8_MAX), FP8_MAX)
        tl.store(y_row + offs, q.to(y_ptr.dtype.element_ty), mask=mask)


def quant(x: torch.Tensor):
    M, N = x.shape
    INTER = N // 2
    xq = torch.empty((M, INTER), device=x.device, dtype=FP8_DTYPE)
    scale = torch.empty((M, 1), device=x.device, dtype=torch.float32)
    BLOCK_N = 1024 if INTER > 1024 else triton.next_power_of_2(INTER)
    _silu_mul_quant_kernel[(M,)](
        x, xq, scale,
        x.stride(0), xq.stride(0),
        INTER,
        FP8_MAX=FP8_MAX,
        BLOCK_N=BLOCK_N, num_warps=8,
    )
    return xq, scale
