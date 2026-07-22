"""GENERATED breadth tr_ema_update seed (bf16). Fused EMA (Polyak) weight update."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _tr_ema_update_kernel(ema_ptr, param_ptr, decay, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    ema = tl.load(ema_ptr + offs, mask=mask).to(tl.float32)
    param = tl.load(param_ptr + offs, mask=mask).to(tl.float32)
    ema = decay * ema + (1.0 - decay) * param
    tl.store(ema_ptr + offs, ema.to(tl.bfloat16), mask=mask)


def tr_ema_update(ema, param, decay):
    numel = ema.numel()
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _tr_ema_update_kernel[grid](ema, param, decay, numel, BLOCK=BLOCK, num_warps=4)
    return ema
