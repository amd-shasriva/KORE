"""GENERATED vendor-baselined NEOX RoPE seed (bf16) vs aiter.rope_fwd.
x[S,B,H,D], freqs[S,1,1,D//2] angles. One program per (s,b,h) row; half-width
rotate-NEOX identity (o1=x1*cos-x2*sin, o2=x2*cos+x1*sin), fp32 math, tl.bfloat16 store.
Regenerate via generate_vendor_ops.py."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _rope_kernel(x_ptr, f_ptr, y_ptr, B, H, D,
                 sxs, sxb, sxh, sxd, sfs, HALF: tl.constexpr):
    pid = tl.program_id(0)
    h = pid % H
    tmp = pid // H
    b = tmp % B
    s = tmp // B
    base = s * sxs + b * sxb + h * sxh
    offs = tl.arange(0, HALF)
    x1 = tl.load(x_ptr + base + offs * sxd).to(tl.float32)
    x2 = tl.load(x_ptr + base + (offs + HALF) * sxd).to(tl.float32)
    theta = tl.load(f_ptr + s * sfs + offs).to(tl.float32)
    cos = tl.cos(theta)
    sin = tl.sin(theta)
    tl.store(y_ptr + base + offs * sxd, (x1 * cos - x2 * sin).to(tl.bfloat16))
    tl.store(y_ptr + base + (offs + HALF) * sxd, (x2 * cos + x1 * sin).to(tl.bfloat16))


def rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    S, B, H, D = x.shape
    y = torch.empty_like(x)
    f = freqs.reshape(S, D // 2)
    _rope_kernel[(S * B * H,)](x, f, y, B, H, D,
                               x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                               f.stride(0), HALF=D // 2, num_warps=4)
    return y
