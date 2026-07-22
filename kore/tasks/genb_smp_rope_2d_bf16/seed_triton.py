"""GENERATED breadth smp_rope_2d seed (bf16). 2D RoPE: two head-dim halves rotated by two position coordinates. Naive but correct; the
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
    tl.store(o_ptr + row * so + offs, (x1 * c - x2 * s).to(tl.bfloat16), mask=mask)
    tl.store(o_ptr + row * so + HALF + offs, (x2 * c + x1 * s).to(tl.bfloat16), mask=mask)


def smp_rope_2d(x: torch.Tensor, pos_h: torch.Tensor, pos_w: torch.Tensor) -> torch.Tensor:
    xf = x.float().contiguous()
    M, D = xf.shape
    device = xf.device
    Dh = D // 2
    half = Dh // 2
    i = torch.arange(half, device=device, dtype=torch.float32)
    inv = 10000.0 ** (-(2.0 * i) / Dh)
    ah = pos_h.float()[:, None] * inv[None, :]
    aw = pos_w.float()[:, None] * inv[None, :]
    ch = torch.cos(ah).contiguous(); sh = torch.sin(ah).contiguous()
    cw = torch.cos(aw).contiguous(); sw = torch.sin(aw).contiguous()
    xh = xf[:, :Dh].contiguous(); xw = xf[:, Dh:].contiguous()
    oh = torch.empty((M, Dh), device=device, dtype=torch.bfloat16)
    ow = torch.empty((M, Dh), device=device, dtype=torch.bfloat16)
    BLOCK_H = triton.next_power_of_2(half)
    _rope_kernel[(M,)](xh, ch, sh, oh, xh.stride(0), ch.stride(0), sh.stride(0), oh.stride(0), half, BLOCK_H=BLOCK_H, num_warps=4)
    _rope_kernel[(M,)](xw, cw, sw, ow, xw.stride(0), cw.stride(0), sw.stride(0), ow.stride(0), half, BLOCK_H=BLOCK_H, num_warps=4)
    return torch.cat([oh, ow], dim=-1)
