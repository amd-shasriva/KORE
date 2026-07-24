"""GENERATED breadth tr_huber_bwd seed. Triton computes both the
elementwise loss contribution and analytic gradient; a Triton reduction returns
the mean loss. No framework loss or autograd delegation."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _elem_loss_kernel(x_ptr, y_ptr, part_ptr, grad_ptr, numel,
                      KIND: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    d = x - y
    ad = tl.abs(d)
    if KIND == 0:
        loss = tl.maximum(x, 0.0) - x * y + tl.log(1.0 + tl.exp(-tl.abs(x)))
        grad = (tl.sigmoid(x) - y) / numel
    elif KIND == 1:
        loss = tl.where(ad <= 1.0, 0.5 * d * d,
                        1.0 * (ad - 0.5 * 1.0))
        grad = tl.where(ad <= 1.0, d,
                        1.0 * tl.where(d >= 0.0, 1.0, -1.0)) / numel
    else:
        loss = tl.where(ad < 0.5,
                        0.5 * d * d / 0.5,
                        ad - 0.5 * 0.5)
        grad = tl.where(ad < 0.5, d / 0.5,
                        tl.where(d >= 0.0, 1.0, -1.0)) / numel
    tl.store(part_ptr + pid, tl.sum(tl.where(mask, loss, 0.0), axis=0))
    tl.store(grad_ptr + offs, grad.to(tl.float16), mask=mask)


@triton.jit
def _finish_loss_kernel(part_ptr, loss_ptr, n_parts, numel, BLOCK: tl.constexpr):
    acc = 0.0
    for start in range(0, n_parts, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < n_parts
        acc += tl.sum(tl.load(part_ptr + offs, mask=mask, other=0.0), axis=0)
    tl.store(loss_ptr, (acc / numel).to(loss_ptr.dtype.element_ty))


def tr_huber_bwd(inp, target):
    x = inp.contiguous()
    y = target.contiguous()
    numel = x.numel()
    BLOCK = 1024
    n_parts = triton.cdiv(numel, BLOCK)
    parts = torch.empty((n_parts,), device=x.device, dtype=torch.float32)
    grad = torch.empty_like(x)
    loss = torch.empty((), device=x.device, dtype=x.dtype)
    _elem_loss_kernel[(n_parts,)](
        x, y, parts, grad, numel, KIND=1, BLOCK=BLOCK, num_warps=4)
    _finish_loss_kernel[(1,)](
        parts, loss, n_parts, numel, BLOCK=1024, num_warps=8)
    return loss, grad
