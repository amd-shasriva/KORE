"""GENERATED breadth fused_muon seed (bf16). One Muon step on a 2D param, UPDATING
param + momentum_buffer IN PLACE. Triton elementwise kernels do the (nesterov)
momentum accumulation and the final scaled update; the 5-iter Newton-Schulz
orthogonalization (the quintic X = a*X + (b*A + c*A@A)@X) runs as torch matmuls in
fp32 - FUSING those matmuls into Triton is the optimization target."""
from __future__ import annotations
import torch, triton, triton.language as tl

_NS_A, _NS_B, _NS_C = 3.4445, -4.775, 2.0315
_NS_STEPS = 5


@triton.jit
def _muon_momentum_kernel(g_ptr, buf_ptr, geff_ptr, numel, momentum, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    g = tl.load(g_ptr + offs, mask=mask).to(tl.float32)
    buf = tl.load(buf_ptr + offs, mask=mask).to(tl.float32)
    buf = momentum * buf + (1.0 - momentum) * g
    geff = (1.0 - momentum) * g + momentum * buf
    tl.store(buf_ptr + offs, buf.to(tl.bfloat16), mask=mask)
    tl.store(geff_ptr + offs, geff, mask=mask)


@triton.jit
def _muon_update_kernel(p_ptr, o_ptr, numel, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    p = tl.load(p_ptr + offs, mask=mask).to(tl.float32)
    o = tl.load(o_ptr + offs, mask=mask).to(tl.float32)
    p = p - alpha * o
    tl.store(p_ptr + offs, p.to(tl.bfloat16), mask=mask)


def _newton_schulz5(gm):
    x = gm.float()
    transposed = False
    if x.shape[-2] > x.shape[-1]:
        x = x.mT
        transposed = True
    x = x / (x.norm() + 1e-7)
    for _ in range(_NS_STEPS):
        a = x @ x.mT
        b = _NS_B * a + _NS_C * (a @ a)
        x = _NS_A * x + b @ x
    if transposed:
        x = x.mT
    return x


def fused_muon(param, grad, momentum_buffer, lr, momentum):
    M, N = param.shape
    numel = param.numel()
    geff = torch.empty((M, N), device=param.device, dtype=torch.float32)
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _muon_momentum_kernel[grid](grad, momentum_buffer, geff, numel, momentum,
                                BLOCK=BLOCK, num_warps=4)
    o = _newton_schulz5(geff).contiguous()
    scale = max(1.0, M / N) ** 0.5
    _muon_update_kernel[grid](param, o, numel, lr * scale, BLOCK=BLOCK, num_warps=4)
    return param, momentum_buffer
