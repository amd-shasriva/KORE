"""GENERATED breadth tr_adam_8bit seed. Blockwise int8-quantized Adam states: dequant (torch) -> fused fp32 Adam update in a Triton kernel -> requant (torch). Fusing the de/requant into the Triton kernel is the target."""
from __future__ import annotations
import torch, triton, triton.language as tl

_QB, _QMAX, _QDT = 128, 127.0, torch.int8


def _dequant(q, scale):
    qf = q.reshape(-1)
    s = scale.repeat_interleave(_QB)[:qf.numel()]
    return (qf.float() * s).reshape(q.shape)


def _quant(x):
    xf = x.reshape(-1).float()
    n = xf.numel()
    nb = (n + _QB - 1) // _QB
    pad = nb * _QB - n
    xp = torch.cat([xf, xf.new_zeros(pad)]) if pad else xf
    xb = xp.reshape(nb, _QB)
    amax = xb.abs().amax(1)
    scale = torch.where(amax > 0, amax / _QMAX, torch.ones_like(amax))
    q = (xb / scale[:, None]).round().clamp(-127, 127).to(_QDT).reshape(-1)[:n].reshape(x.shape)
    return q, scale


@triton.jit
def _tr_adam_8bit_kernel(p_ptr, g_ptr, m_ptr, v_ptr, mo_ptr, vo_ptr, numel, lr, weight_decay, b1, b2, eps, step_size, bc2sqrt, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    p = tl.load(p_ptr + offs, mask=mask).to(tl.float32)
    g = tl.load(g_ptr + offs, mask=mask).to(tl.float32)
    m = tl.load(m_ptr + offs, mask=mask).to(tl.float32)
    v = tl.load(v_ptr + offs, mask=mask).to(tl.float32)
    g = g + weight_decay * p
    m = m + (1.0 - b1) * (g - m)
    v = b2 * v + (1.0 - b2) * g * g
    p = p - step_size * m / (tl.sqrt(v) / bc2sqrt + eps)
    tl.store(p_ptr + offs, p.to(tl.float32), mask=mask)
    tl.store(mo_ptr + offs, m, mask=mask)
    tl.store(vo_ptr + offs, v, mask=mask)


def tr_adam_8bit(param, grad, q_exp_avg, s_exp_avg, q_exp_avg_sq, s_exp_avg_sq, lr, b1, b2, eps, weight_decay, step):
    m = _dequant(q_exp_avg, s_exp_avg)
    v = _dequant(q_exp_avg_sq, s_exp_avg_sq)
    mo = torch.empty_like(m); vo = torch.empty_like(v)
    bc1 = 1.0 - b1 ** step
    step_size = lr / bc1
    bc2sqrt = (1.0 - b2 ** step) ** 0.5
    numel = param.numel()
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _tr_adam_8bit_kernel[grid](param, grad, m, v, mo, vo, numel, lr, weight_decay, b1, b2, eps, step_size, bc2sqrt, BLOCK=BLOCK, num_warps=4)
    qm, sm = _quant(mo); qv, sv = _quant(vo)
    return param, qm, sm, qv, sv
