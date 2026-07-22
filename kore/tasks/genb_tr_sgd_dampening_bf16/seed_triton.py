"""GENERATED breadth tr_sgd_dampening seed (bf16). Fused SGD(+momentum/nesterov/wd) step."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _tr_sgd_dampening_kernel(param_ptr, grad_ptr, buf_ptr, lr, momentum, dampening, wd, nesterov, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    param = tl.load(param_ptr + offs, mask=mask).to(tl.float32)
    grad = tl.load(grad_ptr + offs, mask=mask).to(tl.float32)
    buf = tl.load(buf_ptr + offs, mask=mask).to(tl.float32)
    grad = grad + wd * param
    buf = momentum * buf + (1.0 - dampening) * grad
    d = tl.where(nesterov, grad + momentum * buf, buf)
    param = param - lr * d
    tl.store(param_ptr + offs, param.to(tl.bfloat16), mask=mask)
    tl.store(buf_ptr + offs, buf.to(tl.bfloat16), mask=mask)


def tr_sgd_dampening(param, grad, buf, lr, momentum, dampening, wd, nesterov):
    numel = param.numel()
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _tr_sgd_dampening_kernel[grid](param, grad, buf, lr, momentum, dampening, wd, nesterov, numel, BLOCK=BLOCK, num_warps=4)
    return param, buf
