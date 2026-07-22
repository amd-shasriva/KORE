"""GENERATED breadth smp_rope_partial seed (bf16). partial rotary: rotate the first rot_pct of head_dim, pass the rest. Naive but correct; the
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


def smp_rope_partial(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    xf = x.float().contiguous()
    M, D = xf.shape
    device = xf.device
    rot = int(D * 0.5)
    rot -= rot % 2
    half = rot // 2
    i = torch.arange(half, device=device, dtype=torch.float32)
    inv = 10000.0 ** (-(2.0 * i) / rot)
    ang = pos.float()[:, None] * inv[None, :]
    c = torch.cos(ang).contiguous()
    s = torch.sin(ang).contiguous()
    o = xf.to(torch.bfloat16).clone()
    xr = xf[:, :rot].contiguous()
    orot = torch.empty((M, rot), device=device, dtype=torch.bfloat16)
    BLOCK_H = triton.next_power_of_2(half)
    _rope_kernel[(M,)](xr, c, s, orot, xr.stride(0), c.stride(0), s.stride(0), orot.stride(0), half, BLOCK_H=BLOCK_H, num_warps=4)
    o[:, :rot] = orot
    return o
