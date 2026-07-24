"""GENERATED breadth tr_cosine_embed_bwd seed. One Triton program per row
computes the cosine loss and its analytic gradient; a Triton reduction returns
the mean. No framework loss or autograd delegation."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _cosine_rows_kernel(a_ptr, b_ptr, rows_ptr, grad_ptr, M, N,
                        BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * N
    aa = 0.0
    bb = 0.0
    ab = 0.0
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        aa += tl.sum(a * a, axis=0)
        bb += tl.sum(b * b, axis=0)
        ab += tl.sum(a * b, axis=0)
    n1 = tl.sqrt(aa)
    n2 = tl.sqrt(bb)
    cosv = ab / (n1 * n2)
    tl.store(rows_ptr + row, 1.0 - cosv)
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        grad = -(b / (n1 * n2) - cosv * a / aa) / M
        tl.store(grad_ptr + base + offs, grad.to(tl.float16), mask=mask)


@triton.jit
def _mean_rows_kernel(rows_ptr, loss_ptr, M, BLOCK: tl.constexpr):
    acc = 0.0
    for start in range(0, M, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < M
        acc += tl.sum(tl.load(rows_ptr + offs, mask=mask, other=0.0), axis=0)
    tl.store(loss_ptr, (acc / M).to(loss_ptr.dtype.element_ty))


def tr_cosine_embed_bwd(x1, x2):
    a = x1.contiguous()
    b = x2.contiguous()
    M, N = a.shape
    rows = torch.empty((M,), device=a.device, dtype=torch.float32)
    grad = torch.empty_like(a)
    loss = torch.empty((), device=a.device, dtype=a.dtype)
    _cosine_rows_kernel[(M,)](a, b, rows, grad, M, N, BLOCK=1024, num_warps=8)
    _mean_rows_kernel[(1,)](rows, loss, M, BLOCK=1024, num_warps=8)
    return loss, grad
