"""GENERATED breadth tr_adafactor seed. Factored (row/col) second-moment estimate + update-clipping in torch; the final decoupled-decay scaled update runs in a Triton elementwise kernel. Returns (param, row_var, col_var)."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _tr_adafactor_kernel(p_ptr, u_ptr, numel, decay, coef, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    p = tl.load(p_ptr + offs, mask=mask).to(tl.float32)
    u = tl.load(u_ptr + offs, mask=mask).to(tl.float32)
    p = p * decay - coef * u
    tl.store(p_ptr + offs, p.to(tl.bfloat16), mask=mask)


def tr_adafactor(param, grad, row_var, col_var, lr, beta2_decay, eps1, eps2, d, weight_decay, step):
    sf = float(step)
    omb2 = sf ** beta2_decay
    rho_t = min(lr, 1.0 / (sf ** 0.5))
    g = grad.float()
    alpha = max(eps2, param.float().norm().item() / (param.numel() ** 0.5)) * rho_t
    rv = row_var.float() + omb2 * ((g * g).mean(dim=-1, keepdim=True) - row_var.float())
    cv = col_var.float() + omb2 * ((g * g).mean(dim=-2, keepdim=True) - col_var.float())
    var = (rv @ cv) / rv.mean(dim=-2, keepdim=True).clamp(min=eps1)
    upd = var.clamp(min=eps1 * eps1).rsqrt() * g
    denom = max(1.0, upd.norm().item() / ((upd.numel() ** 0.5) * d))
    row_var.copy_(rv.to(row_var.dtype)); col_var.copy_(cv.to(col_var.dtype))
    numel = param.numel()
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _tr_adafactor_kernel[grid](param, upd.contiguous(), numel, 1.0 - lr * weight_decay, alpha / denom, BLOCK=BLOCK, num_warps=4)
    return param, row_var, col_var
