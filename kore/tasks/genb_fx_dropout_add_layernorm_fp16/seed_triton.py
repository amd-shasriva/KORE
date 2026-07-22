from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _fx_dropout_add_layernorm_kernel(x_ptr, res_ptr, msk_ptr, w_ptr, lnb_ptr, y_ptr, added_ptr, sm, N, eps, inv_keep, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(res_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    d = tl.load(msk_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    added = r + (x) * d * inv_keep
    tl.store(added_ptr + base + offs, added.to(tl.float16), mask=mask)
    mean = tl.sum(added, axis=0) / N
    xc = tl.where(mask, added - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    lb = tl.load(lnb_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + base + offs, (xc * rstd * w + lb).to(tl.float16), mask=mask)


def fx_dropout_add_layernorm(x, residual, mask, weight, lnbias, eps: float = 1e-06, inv_keep: float = 1.1111111111111112):
    M, N = x.shape
    y = torch.empty_like(x)
    added = torch.empty_like(x)
    _fx_dropout_add_layernorm_kernel[(M,)](x, residual, mask, weight, lnbias, y, added, x.stride(0), N, eps, inv_keep,
                       BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return y, added
