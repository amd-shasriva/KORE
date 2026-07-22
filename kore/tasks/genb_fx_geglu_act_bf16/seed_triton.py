from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _fx_geglu_act_kernel(x_ptr, y_ptr, sm, sy, H, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < H
    g = tl.load(x_ptr + row * sm + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(x_ptr + row * sm + H + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * sy + offs, ((0.5 * g * (1.0 + tl.math.tanh(0.7978845608028654 * (g + 0.044715 * g * g * g)))) * u).to(tl.bfloat16), mask=mask)


def fx_geglu_act(x):
    M, W = x.shape
    H = W // 2
    y = torch.empty((M, H), device=x.device, dtype=x.dtype)
    _fx_geglu_act_kernel[(M,)](x, y, x.stride(0), y.stride(0), H, BLOCK=triton.next_power_of_2(H), num_warps=8)
    return y
