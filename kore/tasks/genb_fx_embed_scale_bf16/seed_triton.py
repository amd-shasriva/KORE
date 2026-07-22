from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _fx_embed_scale_kernel(ids_ptr, w_ptr, y_ptr, D, scale, BLOCK: tl.constexpr):
    m = tl.program_id(0)
    idx = tl.load(ids_ptr + m)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    w = tl.load(w_ptr + idx * D + offs, mask=mask, other=0.0).to(tl.float32)
    v = w * scale
    tl.store(y_ptr + m * D + offs, v.to(tl.bfloat16), mask=mask)


def fx_embed_scale(ids, weight):
    M = ids.shape[0]
    D = weight.shape[1]
    scale = float(D) ** 0.5
    y = torch.empty((M, D), device=weight.device, dtype=weight.dtype)
    _fx_embed_scale_kernel[(M,)](ids, weight, y, D, scale, BLOCK=triton.next_power_of_2(D), num_warps=4)
    return y
