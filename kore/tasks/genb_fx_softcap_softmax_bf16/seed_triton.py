from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _fx_softcap_softmax_kernel(x_ptr, y_ptr, Ncol, cap, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < Ncol
    x = tl.load(x_ptr + row * Ncol + offs, mask=mask, other=0.0).to(tl.float32)
    s = cap * tl.math.tanh(x / cap)
    s = tl.where(mask, s, -1e30)
    mx = tl.max(s, axis=0)
    e = tl.exp(s - mx)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    tl.store(y_ptr + row * Ncol + offs, (e / denom).to(tl.bfloat16), mask=mask)


def fx_softcap_softmax(scores, cap: float = 50.0):
    R, Ncol = scores.shape
    y = torch.empty_like(scores)
    _fx_softcap_softmax_kernel[(R,)](scores, y, Ncol, cap, BLOCK=triton.next_power_of_2(Ncol), num_warps=8)
    return y
