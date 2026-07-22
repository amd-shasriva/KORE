"""GENERATED breadth smp_spec_accept seed (bf16). rejection-sampling accept: u <= min(1, p_target[d]/q_draft[d]). Naive but correct; the
data-dependent selection runs host-side in torch (the policy fuses it)."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _acc_kernel(a_ptr, u_ptr, o_ptr, M, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    o = tl.where(u <= a, 1.0, 0.0)
    tl.store(o_ptr + offs, o.to(tl.bfloat16), mask=mask)


def smp_spec_accept(q: torch.Tensor, p: torch.Tensor, d: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    M, V = q.shape
    di = d.long().view(-1, 1)
    ratio = p.float().gather(-1, di).squeeze(-1) / q.float().gather(-1, di).squeeze(-1)
    accept = torch.clamp(ratio, max=1.0).contiguous()
    uu = u.float().contiguous()
    o = torch.empty((M,), device=q.device, dtype=torch.bfloat16)
    BLOCK = 256
    _acc_kernel[(triton.cdiv(M, BLOCK),)](accept, uu, o, M, BLOCK=BLOCK, num_warps=4)
    return o
