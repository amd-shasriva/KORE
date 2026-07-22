"""GENERATED breadth smp_rope_llama3 seed (fp32). Llama-3 frequency-smoothing RoPE. Naive but correct; the
data-dependent selection runs host-side in torch (the policy fuses it)."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _rope_kernel(x_ptr, c_ptr, s_ptr, o_ptr, sx, sc, ss, so, HALF, BLOCK_H: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < HALF
    x1 = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + row * sx + HALF + offs, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(c_ptr + row * sc + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(s_ptr + row * ss + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(o_ptr + row * so + offs, (x1 * c - x2 * s).to(tl.float32), mask=mask)
    tl.store(o_ptr + row * so + HALF + offs, (x2 * c + x1 * s).to(tl.float32), mask=mask)


def smp_rope_llama3(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    xf = x.float().contiguous()
    M, D = xf.shape
    half = D // 2
    device = xf.device
    i = torch.arange(half, device=device, dtype=torch.float32)
    inv0 = 10000.0 ** (-(2.0 * i) / D)
    wl = 6.283185307179586 / inv0
    low_wl = 8192.0 / 1.0
    high_wl = 8192.0 / 4.0
    inv_low = inv0 / 8.0
    smooth = (8192.0 / wl - 1.0) / (4.0 - 1.0)
    inv_sm = (1.0 - smooth) * inv0 / 8.0 + smooth * inv0
    inv = torch.where(wl > low_wl, inv_low, torch.where(wl < high_wl, inv0, inv_sm))
    mscale = 1.0
    ang = pos.float()[:, None] * inv[None, :]
    c = (torch.cos(ang) * mscale).contiguous()
    s = (torch.sin(ang) * mscale).contiguous()
    o = torch.empty((M, D), device=device, dtype=torch.float32)
    BLOCK_H = triton.next_power_of_2(half)
    _rope_kernel[(M,)](xf, c, s, o, xf.stride(0), c.stride(0), s.stride(0), o.stride(0), half, BLOCK_H=BLOCK_H, num_warps=4)
    return o
