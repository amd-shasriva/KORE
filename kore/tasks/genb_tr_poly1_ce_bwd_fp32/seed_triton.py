"""GENERATED breadth tr_poly1_ce_bwd seed. Naive: fp32 forward + autograd input-gradient, then a Triton elementwise pass materializes the gradient (the FUSED loss+backward Triton kernel is the optimization target)."""
from __future__ import annotations
import torch, triton, triton.language as tl
import torch.nn.functional as F


@triton.jit
def _tr_poly1_ce_bwd_copy_kernel(src_ptr, dst_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    v = tl.load(src_ptr + offs, mask=mask).to(tl.float32)
    tl.store(dst_ptr + offs, v.to(tl.float32), mask=mask)


def tr_poly1_ce_bwd(logits, targets):
    x = logits.float().detach().requires_grad_(True)
    pt = torch.softmax(x, -1).gather(1, targets.long()[:, None]).squeeze(1); loss = (F.cross_entropy(x, targets.long(), reduction='none') + 1.0 * (1.0 - pt)).mean()
    (g,) = torch.autograd.grad(loss, x)
    grad = torch.empty_like(logits)
    numel = grad.numel()
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _tr_poly1_ce_bwd_copy_kernel[grid](g.contiguous(), grad, numel, BLOCK=BLOCK, num_warps=4)
    return loss.detach().to(logits.dtype), grad
