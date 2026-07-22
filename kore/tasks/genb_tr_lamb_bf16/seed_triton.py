"""GENERATED breadth tr_lamb seed (bf16). LAMB: Adam ratio + layer-wise trust ratio."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _tr_lamb_kernel(p_ptr, u_ptr, numel, coef, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    p = tl.load(p_ptr + offs, mask=mask).to(tl.float32)
    u = tl.load(u_ptr + offs, mask=mask).to(tl.float32)
    p = p - coef * u
    tl.store(p_ptr + offs, p.to(tl.bfloat16), mask=mask)


def tr_lamb(param, grad, exp_avg, exp_avg_sq, lr, b1, b2, eps, wd, step):
    m = b1 * exp_avg.float() + (1.0 - b1) * grad.float()
    v = b2 * exp_avg_sq.float() + (1.0 - b2) * grad.float() ** 2
    mhat = m / (1.0 - b1 ** step)
    vhat = v / (1.0 - b2 ** step)
    upd = mhat / (vhat.sqrt() + eps) + wd * param.float()
    trust = param.float().norm() / upd.norm().clamp(min=1e-30)
    coef = float(lr * trust)
    upd = upd
    exp_avg.copy_(m.to(exp_avg.dtype)); exp_avg_sq.copy_(v.to(exp_avg_sq.dtype))
    numel = param.numel()
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _tr_lamb_kernel[grid](param, upd.contiguous(), numel, coef, BLOCK=BLOCK, num_warps=4)
    return param, exp_avg, exp_avg_sq
