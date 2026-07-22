"""GENERATED breadth tr_js_distill_bwd seed. Naive: fp32 forward + autograd input-gradient, then a Triton elementwise pass materializes the gradient (the FUSED loss+backward Triton kernel is the optimization target)."""
from __future__ import annotations
import torch, triton, triton.language as tl
import torch.nn.functional as F


@triton.jit
def _tr_js_distill_bwd_copy_kernel(src_ptr, dst_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    v = tl.load(src_ptr + offs, mask=mask).to(tl.float32)
    tl.store(dst_ptr + offs, v.to(tl.float32), mask=mask)


def tr_js_distill_bwd(student, teacher):
    x = student.float().detach().requires_grad_(True)
    ps = torch.softmax(x, -1); pt = torch.softmax(teacher.float(), -1); mid = 0.5 * (ps + pt); loss = (0.5 * (ps * (torch.log(ps) - torch.log(mid))).sum(1) + 0.5 * (pt * (torch.log(pt) - torch.log(mid))).sum(1)).mean()
    (g,) = torch.autograd.grad(loss, x)
    grad = torch.empty_like(student)
    numel = grad.numel()
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _tr_js_distill_bwd_copy_kernel[grid](g.contiguous(), grad, numel, BLOCK=BLOCK, num_warps=4)
    return loss.detach().to(student.dtype), grad
