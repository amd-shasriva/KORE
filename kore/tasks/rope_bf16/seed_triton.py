"""Seed Triton bf16 NEOX RoPE for gfx942. Exposes ``rope(x, freqs) -> out``.

x is (S, B, H, D), freqs is (S, 1, 1, D//2) of rotation angles. One program per
(s, b, h) row; the half-width rotate-NEOX identity is applied directly:
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
with cos/sin = cos/sin(freqs[s]) (fp32 math), bf16 store. A correct baseline the
KORE policy learns to edit/optimize against the AITER HIP rope serving bar.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    x_ptr, f_ptr, y_ptr,
    B, H, D,
    stride_xs, stride_xb, stride_xh, stride_xd,
    stride_fs,
    HALF: tl.constexpr,
):
    pid = tl.program_id(0)
    h = pid % H
    tmp = pid // H
    b = tmp % B
    s = tmp // B

    base = s * stride_xs + b * stride_xb + h * stride_xh
    offs = tl.arange(0, HALF)
    x1 = tl.load(x_ptr + base + offs * stride_xd).to(tl.float32)
    x2 = tl.load(x_ptr + base + (offs + HALF) * stride_xd).to(tl.float32)

    theta = tl.load(f_ptr + s * stride_fs + offs).to(tl.float32)
    cos = tl.cos(theta)
    sin = tl.sin(theta)

    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    tl.store(y_ptr + base + offs * stride_xd, o1.to(tl.bfloat16))
    tl.store(y_ptr + base + (offs + HALF) * stride_xd, o2.to(tl.bfloat16))


def rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    S, B, H, D = x.shape
    y = torch.empty_like(x)
    f = freqs.reshape(S, D // 2)
    grid = (S * B * H,)
    _rope_kernel[grid](
        x, f, y,
        B, H, D,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        f.stride(0),
        HALF=D // 2,
        num_warps=4,
    )
    return y
