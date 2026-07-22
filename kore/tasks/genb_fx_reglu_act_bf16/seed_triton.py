from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _fx_reglu_act_kernel(x_ptr, y_ptr, sm, sy, H, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < H
    g = tl.load(x_ptr + row * sm + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(x_ptr + row * sm + H + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * sy + offs, ((tl.maximum(g, 0.0)) * u).to(tl.bfloat16), mask=mask)


def fx_reglu_act(x):
    M, W = x.shape
    H = W // 2
    y = torch.empty((M, H), device=x.device, dtype=x.dtype)
    _fx_reglu_act_kernel[(M,)](x, y, x.stride(0), y.stride(0), H, BLOCK=triton.next_power_of_2(H), num_warps=8)
    return y
