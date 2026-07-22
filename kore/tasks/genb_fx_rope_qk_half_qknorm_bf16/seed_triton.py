from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _fx_rope_qk_half_qknorm_kernel(x_ptr, w_ptr, cos_ptr, sin_ptr, y_ptr, S, H, D, HALF, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    pos = (row // H) % S
    base = row * D
    cb = pos * D
    offs = tl.arange(0, BLOCK)
    mask = offs < HALF
    x1 = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + base + HALF + offs, mask=mask, other=0.0).to(tl.float32)
    ss = tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0)
    rstd = 1.0 / tl.sqrt(ss / D + eps)
    w1 = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    w2 = tl.load(w_ptr + HALF + offs, mask=mask, other=0.0).to(tl.float32)
    n1 = x1 * rstd * w1
    n2 = x2 * rstd * w2
    c = tl.load(cos_ptr + cb + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(sin_ptr + cb + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + base + offs, (n1 * c - n2 * s).to(tl.bfloat16), mask=mask)
    tl.store(y_ptr + base + HALF + offs, (n2 * c + n1 * s).to(tl.bfloat16), mask=mask)


def fx_rope_qk_half_qknorm(q, k, wq, wk, cos, sin, eps: float = 1e-06):
    B, S, H, D = q.shape
    HALF = D // 2
    qc, kc = q.contiguous(), k.contiguous()
    qn, kn = torch.empty_like(qc), torch.empty_like(kc)
    BLK = triton.next_power_of_2(HALF)
    grid = (B * S * H,)
    _fx_rope_qk_half_qknorm_kernel[grid](qc, wq, cos, sin, qn, S, H, D, HALF, eps, BLOCK=BLK, num_warps=4)
    _fx_rope_qk_half_qknorm_kernel[grid](kc, wk, cos, sin, kn, S, H, D, HALF, eps, BLOCK=BLK, num_warps=4)
    return qn, kn
