"""GENERATED breadth tr_lars seed (fp16). LARS: layer-wise adaptive rate scaling."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _tr_lars_kernel(p_ptr, u_ptr, numel, coef, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    p = tl.load(p_ptr + offs, mask=mask).to(tl.float32)
    u = tl.load(u_ptr + offs, mask=mask).to(tl.float32)
    p = p - coef * u
    tl.store(p_ptr + offs, p.to(tl.float16), mask=mask)


def tr_lars(param, grad, momentum_buffer, lr, momentum, wd, trust_coef, eps):
    d = grad.float() + wd * param.float()
    local_lr = trust_coef * param.float().norm() / (d.norm() + eps)
    buf = momentum * momentum_buffer.float() + lr * local_lr * d
    upd = buf
    coef = 1.0
    upd = upd
    momentum_buffer.copy_(buf.to(momentum_buffer.dtype))
    numel = param.numel()
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _tr_lars_kernel[grid](param, upd.contiguous(), numel, coef, BLOCK=BLOCK, num_warps=4)
    return param, momentum_buffer
