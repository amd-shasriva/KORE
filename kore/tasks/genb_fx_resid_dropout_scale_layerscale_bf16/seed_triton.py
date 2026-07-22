from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _fx_resid_dropout_scale_layerscale_kernel(x_ptr, r_ptr, msk_ptr, g_ptr, y_ptr, sm, N, inv_keep, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    d = tl.load(msk_ptr + row).to(tl.float32)
    sc = x * d * inv_keep
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sc = sc * g
    tl.store(y_ptr + base + offs, (r + sc).to(tl.bfloat16), mask=mask)


def fx_resid_dropout_scale_layerscale(x, residual, mask, gamma, inv_keep: float = 1.1111111111111112):
    M, N = x.shape
    y = torch.empty_like(x)
    _fx_resid_dropout_scale_layerscale_kernel[(M,)](x, residual, mask, gamma, y, x.stride(0), N, inv_keep, BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return y
