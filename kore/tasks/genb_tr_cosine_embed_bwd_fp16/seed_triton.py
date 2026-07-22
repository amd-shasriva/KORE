"""GENERATED breadth tr_cosine_embed_bwd seed. Naive: fp32 forward + autograd input-gradient, then a Triton elementwise pass materializes the gradient (the FUSED loss+backward Triton kernel is the optimization target)."""
from __future__ import annotations
import torch, triton, triton.language as tl
import torch.nn.functional as F


@triton.jit
def _tr_cosine_embed_bwd_copy_kernel(src_ptr, dst_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    v = tl.load(src_ptr + offs, mask=mask).to(tl.float32)
    tl.store(dst_ptr + offs, v.to(tl.float16), mask=mask)


def tr_cosine_embed_bwd(x1, x2):
    x = x1.float().detach().requires_grad_(True)
    loss = F.cosine_embedding_loss(x, x2.float(), torch.ones(x.shape[0], device=x.device))
    (g,) = torch.autograd.grad(loss, x)
    grad = torch.empty_like(x1)
    numel = grad.numel()
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _tr_cosine_embed_bwd_copy_kernel[grid](g.contiguous(), grad, numel, BLOCK=BLOCK, num_warps=4)
    return loss.detach().to(x1.dtype), grad
