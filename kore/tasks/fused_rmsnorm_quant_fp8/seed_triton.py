"""Seed FUSED RMSNorm -> per-token fp8 quant (arch-aware fp8: OCP e4m3fn on
gfx950/CDNA4 MI350X/MI355X, FNUZ on gfx942/CDNA3).

Exposes ``quant(x, w) -> (xq, scale)`` where x is [M,N] bf16, w is [N] bf16, xq
is [M,N] fp8 (arch dtype) and scale is [M,1] fp32:
    y        = x * rsqrt(mean(x^2) + eps) * w        (fp32 math)
    scale[m] = rowamax(y[m]) / FP8_MAX
    xq[m]    = clamp(y[m] / scale[m], +/-FP8_MAX) -> fp8

One program per row, three column-streamed passes (sum-of-squares -> rms, then
rowamax of the normed value, then quantize+store) so wide rows fit in registers.
This fuses AITER's rms_norm + dynamic_per_token quant (two kernels + an HBM round
trip) into a single kernel: a correct baseline the KORE policy learns to optimize
(e.g. cache the normed value in LDS to avoid the third recompute pass, widen
BLOCK_N, tune num_warps).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# gfx950/CDNA4 (MI350X/MI355X) fp8 e4m3 is the OCP variant e4m3fn (max 448.0) --
# the CDNA4-native format the reference oracle also quantizes to on this arch.
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)
EPS = 1e-6


@triton.jit
def _rmsnorm_quant_kernel(
    x_ptr, w_ptr, y_ptr, s_ptr,
    stride_xm, stride_ym,
    N, eps,
    FP8_MAX: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_xm
    y_row = y_ptr + row * stride_ym

    ss = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        ss += tl.sum(x * x, axis=0)
    rms = 1.0 / tl.sqrt(ss / N + eps)

    amax = 1e-12
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        normed = x * rms * w
        amax = tl.maximum(amax, tl.max(tl.abs(normed), axis=0))
    scale = amax / FP8_MAX
    tl.store(s_ptr + row, scale)

    inv = 1.0 / scale
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        normed = x * rms * w
        q = tl.minimum(tl.maximum(normed * inv, -FP8_MAX), FP8_MAX)
        tl.store(y_row + offs, q.to(y_ptr.dtype.element_ty), mask=mask)


def quant(x: torch.Tensor, w: torch.Tensor, eps: float = EPS):
    M, N = x.shape
    xq = torch.empty((M, N), device=x.device, dtype=FP8_DTYPE)
    scale = torch.empty((M, 1), device=x.device, dtype=torch.float32)
    BLOCK_N = 1024 if N > 1024 else triton.next_power_of_2(N)
    _rmsnorm_quant_kernel[(M,)](
        x, w, xq, scale,
        x.stride(0), xq.stride(0),
        N, eps,
        FP8_MAX=FP8_MAX,
        BLOCK_N=BLOCK_N, num_warps=8,
    )
    return xq, scale
