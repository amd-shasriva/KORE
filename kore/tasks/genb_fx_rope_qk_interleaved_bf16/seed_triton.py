from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _fx_rope_qk_interleaved_kernel(x_ptr, cos_ptr, sin_ptr, y_ptr, S, H, D, HALF, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    pos = (row // H) % S
    base = row * D
    cb = pos * HALF
    offs = tl.arange(0, BLOCK)
    mask = offs < HALF
    xe = tl.load(x_ptr + base + 2 * offs, mask=mask, other=0.0).to(tl.float32)
    xo = tl.load(x_ptr + base + 2 * offs + 1, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(cos_ptr + cb + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(sin_ptr + cb + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + base + 2 * offs, (xe * c - xo * s).to(tl.bfloat16), mask=mask)
    tl.store(y_ptr + base + 2 * offs + 1, (xe * s + xo * c).to(tl.bfloat16), mask=mask)


def fx_rope_qk_interleaved(q, k, cos, sin):
    B, S, H, D = q.shape
    HALF = D // 2
    qc, kc = q.contiguous(), k.contiguous()
    qn, kn = torch.empty_like(qc), torch.empty_like(kc)
    BLK = triton.next_power_of_2(HALF)
    grid = (B * S * H,)
    _fx_rope_qk_interleaved_kernel[grid](qc, cos, sin, qn, S, H, D, HALF, BLOCK=BLK, num_warps=4)
    _fx_rope_qk_interleaved_kernel[grid](kc, cos, sin, kn, S, H, D, HALF, BLOCK=BLK, num_warps=4)
    return qn, kn
