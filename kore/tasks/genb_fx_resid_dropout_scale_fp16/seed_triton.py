from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _fx_resid_dropout_scale_kernel(x_ptr, r_ptr, msk_ptr, y_ptr, sm, N, inv_keep, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    d = tl.load(msk_ptr + row).to(tl.float32)
    sc = x * d * inv_keep
    tl.store(y_ptr + base + offs, (r + sc).to(tl.float16), mask=mask)


def fx_resid_dropout_scale(x, residual, mask, inv_keep: float = 1.1111111111111112):
    M, N = x.shape
    y = torch.empty_like(x)
    _fx_resid_dropout_scale_kernel[(M,)](x, residual, mask, y, x.stride(0), N, inv_keep, BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return y
