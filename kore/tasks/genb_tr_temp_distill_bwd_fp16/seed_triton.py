"""GENERATED breadth tr_temp_distill_bwd seed. Naive: fp32 forward + autograd input-gradient, then a Triton elementwise pass materializes the gradient (the FUSED loss+backward Triton kernel is the optimization target)."""
from __future__ import annotations
import torch, triton, triton.language as tl
import torch.nn.functional as F


@triton.jit
def _tr_temp_distill_bwd_copy_kernel(src_ptr, dst_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    v = tl.load(src_ptr + offs, mask=mask).to(tl.float32)
    tl.store(dst_ptr + offs, v.to(tl.float16), mask=mask)


def tr_temp_distill_bwd(student, teacher):
    x = student.float().detach().requires_grad_(True)
    T = 2.0; ptT = torch.softmax(teacher.float() / T, -1); loss = (T * T) * (ptT * (torch.log(ptT) - F.log_softmax(x / T, -1))).sum(1).mean()
    (g,) = torch.autograd.grad(loss, x)
    grad = torch.empty_like(student)
    numel = grad.numel()
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _tr_temp_distill_bwd_copy_kernel[grid](g.contiguous(), grad, numel, BLOCK=BLOCK, num_warps=4)
    return loss.detach().to(student.dtype), grad
