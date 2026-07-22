"""GENERATED breadth tr_foreach_sgd seed (fp32). Fused multi-tensor SGD+momentum over a blob."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _tr_foreach_sgd_kernel(param_ptr, grad_ptr, buf_ptr, lr, momentum, wd, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    param = tl.load(param_ptr + offs, mask=mask).to(tl.float32)
    grad = tl.load(grad_ptr + offs, mask=mask).to(tl.float32)
    buf = tl.load(buf_ptr + offs, mask=mask).to(tl.float32)
    grad = grad + wd * param
    buf = momentum * buf + grad
    param = param - lr * buf
    tl.store(param_ptr + offs, param.to(tl.float32), mask=mask)
    tl.store(buf_ptr + offs, buf.to(tl.float32), mask=mask)


def tr_foreach_sgd(param, grad, buf, lr, momentum, wd):
    numel = param.numel()
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _tr_foreach_sgd_kernel[grid](param, grad, buf, lr, momentum, wd, numel, BLOCK=BLOCK, num_warps=4)
    return param, buf
