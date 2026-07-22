"""GENERATED breadth fused_lion seed (fp16). One Lion step, fused, UPDATING
param + exp_avg IN PLACE. update = sign(beta1*m + (1-beta1)*g); param -= lr*(update
+ wd*param); then m = beta2*m + (1-beta2)*g (both use the OLD m). Elementwise fp32."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _fused_lion_kernel(p_ptr, g_ptr, m_ptr, numel, lr, beta1, beta2, wd, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    p = tl.load(p_ptr + offs, mask=mask).to(tl.float32)
    g = tl.load(g_ptr + offs, mask=mask).to(tl.float32)
    m = tl.load(m_ptr + offs, mask=mask).to(tl.float32)
    c = beta1 * m + (1.0 - beta1) * g
    upd = tl.where(c > 0.0, 1.0, tl.where(c < 0.0, -1.0, 0.0))
    p = p - lr * (upd + wd * p)
    m = beta2 * m + (1.0 - beta2) * g
    tl.store(p_ptr + offs, p.to(tl.float16), mask=mask)
    tl.store(m_ptr + offs, m.to(tl.float16), mask=mask)


def fused_lion(param, grad, exp_avg, lr, beta1, beta2, wd):
    numel = param.numel()
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _fused_lion_kernel[grid](param, grad, exp_avg, numel, lr, beta1, beta2, wd,
                             BLOCK=BLOCK, num_warps=4)
    return param, exp_avg
