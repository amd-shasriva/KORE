"""GENERATED breadth tr_adafactor seed. Triton computes the factored
row/column second moments, update RMS clipping, and parameter write directly.
Returns the in-place (param, row_var, col_var) state."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _adafactor_row_kernel(g_ptr, row_ptr, row_fp_ptr, M, N, omb2,
                          BLOCK: tl.constexpr):
    row = tl.program_id(0)
    acc = 0.0
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        g = tl.load(g_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(g * g, axis=0)
    old = tl.load(row_ptr + row).to(tl.float32)
    new = old + omb2 * (acc / N - old)
    tl.store(row_fp_ptr + row, new)
    tl.store(row_ptr + row, new.to(row_ptr.dtype.element_ty))


@triton.jit
def _adafactor_col_kernel(g_ptr, col_ptr, col_fp_ptr, M, N, omb2,
                          BLOCK: tl.constexpr):
    col = tl.program_id(0)
    acc = 0.0
    for start in range(0, M, BLOCK):
        rows = start + tl.arange(0, BLOCK)
        mask = rows < M
        g = tl.load(g_ptr + rows * N + col, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(g * g, axis=0)
    old = tl.load(col_ptr + col).to(tl.float32)
    new = old + omb2 * (acc / M - old)
    tl.store(col_fp_ptr + col, new)
    tl.store(col_ptr + col, new.to(col_ptr.dtype.element_ty))


@triton.jit
def _adafactor_update_kernel(p_ptr, g_ptr, row_ptr, col_ptr, M, N,
                             lr, rho_t, eps1, eps2, d, weight_decay,
                             BLOCK: tl.constexpr):
    numel = M * N
    p2 = 0.0
    row_sum = 0.0
    for start in range(0, numel, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < numel
        p = tl.load(p_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        p2 += tl.sum(p * p, axis=0)
    for start in range(0, M, BLOCK):
        rows = start + tl.arange(0, BLOCK)
        mask = rows < M
        rv = tl.load(row_ptr + rows, mask=mask, other=0.0).to(tl.float32)
        row_sum += tl.sum(rv, axis=0)
    row_mean = tl.maximum(row_sum / M, eps1)

    u2 = 0.0
    for start in range(0, numel, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < numel
        rows = offs // N
        cols = offs % N
        rv = tl.load(row_ptr + rows, mask=mask, other=0.0).to(tl.float32)
        cv = tl.load(col_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        var = tl.maximum(rv * cv / row_mean, eps1 * eps1)
        upd = g / tl.sqrt(var)
        u2 += tl.sum(tl.where(mask, upd * upd, 0.0), axis=0)

    param_rms = tl.sqrt(p2 / numel)
    alpha = tl.maximum(eps2, param_rms) * rho_t
    denom = tl.maximum(1.0, tl.sqrt(u2 / numel) / d)
    coef = alpha / denom
    decay = 1.0 - lr * weight_decay
    for start in range(0, numel, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < numel
        rows = offs // N
        cols = offs % N
        p = tl.load(p_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        rv = tl.load(row_ptr + rows, mask=mask, other=0.0).to(tl.float32)
        cv = tl.load(col_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        upd = g / tl.sqrt(tl.maximum(rv * cv / row_mean, eps1 * eps1))
        tl.store(p_ptr + offs, (p * decay - coef * upd).to(tl.float32), mask=mask)


def tr_adafactor(param, grad, row_var, col_var, lr, beta2_decay, eps1, eps2, d,
         weight_decay, step):
    M, N = param.shape
    grad = grad.contiguous()
    sf = float(step)
    omb2 = sf ** beta2_decay
    rho_t = min(lr, 1.0 / (sf ** 0.5))
    row_fp = torch.empty((M,), device=param.device, dtype=torch.float32)
    col_fp = torch.empty((N,), device=param.device, dtype=torch.float32)
    _adafactor_row_kernel[(M,)](
        grad, row_var, row_fp, M, N, omb2, BLOCK=1024, num_warps=8)
    _adafactor_col_kernel[(N,)](
        grad, col_var, col_fp, M, N, omb2, BLOCK=1024, num_warps=8)
    _adafactor_update_kernel[(1,)](
        param, grad, row_fp, col_fp, M, N, lr, rho_t, eps1, eps2, d,
        weight_decay, BLOCK=1024, num_warps=8)
    return param, row_var, col_var
