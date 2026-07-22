"""GENERATED breadth smp_spec_residual seed (fp16). corrected residual distribution normalize(relu(p_target - q_draft)). Naive but correct; the
data-dependent selection runs host-side in torch (the policy fuses it)."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _resid_kernel(p_ptr, q_ptr, o_ptr, sp, sq, so, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    offs = col * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    p = tl.load(p_ptr + row * sp + offs, mask=mask, other=0.0).to(tl.float32)
    q = tl.load(q_ptr + row * sq + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(o_ptr + row * so + offs, tl.maximum(p - q, 0.0), mask=mask)


def smp_spec_residual(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    M, N = q.shape
    resid = torch.empty((M, N), device=q.device, dtype=torch.float32)
    BLOCK_N = 1024
    grid = (M, triton.cdiv(N, BLOCK_N))
    _resid_kernel[grid](p, q, resid, p.stride(0), q.stride(0), resid.stride(0), N, BLOCK_N=BLOCK_N, num_warps=4)
    denom = resid.sum(-1, keepdim=True).clamp_min(1e-20)
    return (resid / denom).to(q.dtype)
