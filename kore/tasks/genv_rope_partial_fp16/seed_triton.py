"""GENERATED partial-rotary RoPE seed (fp16).
x[S,B,H,D]; rotate only the first rotary_dim = D//2 lanes (NEOX half-split within
that band: pair i with i+rot/2), pass the remaining lanes through unchanged.
freqs[S,1,1,rot//2]. fp32 math, tl.float16 store. Regenerate via generate_vendor_ops.py."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _rope_partial_kernel(x_ptr, f_ptr, y_ptr, B, H, D,
                         sxs, sxb, sxh, sxd, sfs, ROT: tl.constexpr, QUART: tl.constexpr):
    pid = tl.program_id(0)
    h = pid % H
    tmp = pid // H
    b = tmp % B
    s = tmp // B
    base = s * sxs + b * sxb + h * sxh
    offs = tl.arange(0, QUART)                       # rot//2 rotation angles
    x1 = tl.load(x_ptr + base + offs * sxd).to(tl.float32)
    x2 = tl.load(x_ptr + base + (offs + QUART) * sxd).to(tl.float32)
    theta = tl.load(f_ptr + s * sfs + offs).to(tl.float32)
    cos = tl.cos(theta)
    sin = tl.sin(theta)
    tl.store(y_ptr + base + offs * sxd, (x1 * cos - x2 * sin).to(tl.float16))
    tl.store(y_ptr + base + (offs + QUART) * sxd, (x2 * cos + x1 * sin).to(tl.float16))
    poffs = ROT + tl.arange(0, ROT)                  # pass-through lanes [rot, D)
    xp = tl.load(x_ptr + base + poffs * sxd, mask=poffs < D, other=0.0)
    tl.store(y_ptr + base + poffs * sxd, xp, mask=poffs < D)


def rope_partial(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    S, B, H, D = x.shape
    rot = D // 2
    y = torch.empty_like(x)
    f = freqs.reshape(S, rot // 2)
    _rope_partial_kernel[(S * B * H,)](x, f, y, B, H, D,
                                       x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                                       f.stride(0), ROT=rot, QUART=rot // 2, num_warps=4)
    return y
