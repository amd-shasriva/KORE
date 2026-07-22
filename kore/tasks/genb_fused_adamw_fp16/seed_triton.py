"""GENERATED breadth fused_adamw seed (fp16). One decoupled AdamW step, fused,
UPDATING param + exp_avg + exp_avg_sq IN PLACE. Elementwise fp32 math; the bias
corrections (1-beta**step) are precomputed host-side and passed as step_size /
bc2_sqrt. Matches torch.optim.AdamW. Regenerate/optimize from this naive seed."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _fused_adamw_kernel(p_ptr, g_ptr, m_ptr, v_ptr, numel,
                        lr, wd, beta1, beta2, eps, step_size, bc2_sqrt, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    p = tl.load(p_ptr + offs, mask=mask).to(tl.float32)
    g = tl.load(g_ptr + offs, mask=mask).to(tl.float32)
    m = tl.load(m_ptr + offs, mask=mask).to(tl.float32)
    v = tl.load(v_ptr + offs, mask=mask).to(tl.float32)
    p = p * (1.0 - lr * wd)
    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * g * g
    denom = tl.sqrt(v) / bc2_sqrt + eps
    p = p - step_size * m / denom
    tl.store(p_ptr + offs, p.to(tl.float16), mask=mask)
    tl.store(m_ptr + offs, m.to(tl.float16), mask=mask)
    tl.store(v_ptr + offs, v.to(tl.float16), mask=mask)


def fused_adamw(param, grad, exp_avg, exp_avg_sq, lr, beta1, beta2, eps, wd, step):
    numel = param.numel()
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    step_size = lr / bc1
    bc2_sqrt = bc2 ** 0.5
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _fused_adamw_kernel[grid](param, grad, exp_avg, exp_avg_sq, numel,
                              lr, wd, beta1, beta2, eps, step_size, bc2_sqrt,
                              BLOCK=BLOCK, num_warps=4)
    return param, exp_avg, exp_avg_sq
