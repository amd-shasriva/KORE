"""GENERATED breadth tr_muon_ns5 seed. (Nesterov) momentum + aspect-scaled decoupled update in Triton elementwise kernels; the 5-iter Newton-Schulz orthogonalization runs as fp32 torch matmuls (FUSE them into Triton). Returns (param, momentum_buffer)."""
from __future__ import annotations
import torch, triton, triton.language as tl

_A, _B, _C, _STEPS, _EPS = 3.4445, -4.775, 2.0315, 5, 1e-07


@triton.jit
def _tr_muon_ns5_mom_kernel(g_ptr, buf_ptr, out_ptr, numel, momentum, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    g = tl.load(g_ptr + offs, mask=mask).to(tl.float32)
    buf = tl.load(buf_ptr + offs, mask=mask).to(tl.float32)
    buf = buf + (1.0 - momentum) * (g - buf)
    upd = g + momentum * (buf - g)
    tl.store(buf_ptr + offs, buf.to(tl.float16), mask=mask)
    tl.store(out_ptr + offs, upd, mask=mask)


@triton.jit
def _tr_muon_ns5_upd_kernel(p_ptr, o_ptr, numel, decay, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    p = tl.load(p_ptr + offs, mask=mask).to(tl.float32)
    o = tl.load(o_ptr + offs, mask=mask).to(tl.float32)
    p = p * decay - alpha * o
    tl.store(p_ptr + offs, p.to(tl.float16), mask=mask)


def _ns(x):
    t = x.shape[-2] > x.shape[-1]
    if t:
        x = x.mT
    x = x / x.norm().clamp(min=_EPS)
    for _ in range(_STEPS):
        A = x @ x.mT
        B = _B * A + _C * (A @ A)
        x = _A * x + B @ x
    if t:
        x = x.mT
    return x


def tr_muon_ns5(param, grad, momentum_buffer, lr, weight_decay, momentum, eps, ns_a, ns_b, ns_c, ns_steps):
    numel = param.numel()
    geff = torch.empty_like(param, dtype=torch.float32)
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _tr_muon_ns5_mom_kernel[grid](grad, momentum_buffer, geff, numel, momentum, BLOCK=BLOCK, num_warps=4)
    o = _ns(geff.view_as(param)).contiguous()
    scale = max(1.0, param.shape[-2] / param.shape[-1]) ** 0.5
    _tr_muon_ns5_upd_kernel[grid](param, o, numel, 1.0 - lr * weight_decay, lr * scale, BLOCK=BLOCK, num_warps=4)
    return param, momentum_buffer
