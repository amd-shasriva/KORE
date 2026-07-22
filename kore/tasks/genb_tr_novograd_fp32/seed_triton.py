"""GENERATED breadth tr_novograd seed (fp32). NovoGrad: layer-wise 2nd moment."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _tr_novograd_kernel(p_ptr, u_ptr, numel, coef, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    p = tl.load(p_ptr + offs, mask=mask).to(tl.float32)
    u = tl.load(u_ptr + offs, mask=mask).to(tl.float32)
    p = p - coef * u
    tl.store(p_ptr + offs, p.to(tl.float32), mask=mask)


def tr_novograd(param, grad, exp_avg, exp_avg_sq, lr, b1, b2, eps, wd):
    v = b2 * exp_avg_sq.float() + (1.0 - b2) * (grad.float() ** 2).sum()
    ghat = grad.float() / (v.sqrt() + eps) + wd * param.float()
    m = b1 * exp_avg.float() + ghat
    upd = m
    coef = float(lr)
    upd = upd
    exp_avg.copy_(m.to(exp_avg.dtype)); exp_avg_sq = v.to(exp_avg_sq.dtype)
    numel = param.numel()
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _tr_novograd_kernel[grid](param, upd.contiguous(), numel, coef, BLOCK=BLOCK, num_warps=4)
    return param, exp_avg, exp_avg_sq
