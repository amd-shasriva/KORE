from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _fx_add_rmsnorm_quant_fp8_kernel(x_ptr, res_ptr, w_ptr, q_ptr, s_ptr, added_ptr, sm, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(res_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    added = x + r
    tl.store(added_ptr + base + offs, added.to(tl.bfloat16), mask=mask)
    var = tl.sum(added * added, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    normed = added * rstd * w
    amax = tl.max(tl.abs(normed), axis=0)
    scale = tl.where(amax > 0.0, amax / 448.0, 1.0)
    qv = normed / scale
    tl.store(q_ptr + base + offs, qv.to(tl.float8e4nv), mask=mask)
    tl.store(s_ptr + row, scale)


def fx_add_rmsnorm_quant_fp8(x, residual, weight, eps: float = 1e-06):
    M, N = x.shape
    q = torch.empty((M, N), device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty((M,), device=x.device, dtype=torch.float32)
    added = torch.empty_like(x)
    _fx_add_rmsnorm_quant_fp8_kernel[(M,)](x, residual, weight, q, s, added, x.stride(0), N, eps,
                       BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return q, s, added
