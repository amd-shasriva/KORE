"""GENERATED breadth red_bce_with_logits seed (bf16). logits[M,N], targets[M,N]
-> per-row mean binary-cross-entropy-with-logits, elementwise
max(x,0) - x*z + log1p(exp(-|x|)) (the numerically stable form). tl.bfloat16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_bce_with_logits_kernel(x_ptr, z_ptr, o_ptr, sx, sz, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    acc = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)
        z = tl.load(z_ptr + row * sz + offs, mask=mask, other=0.0).to(tl.float32)
        elem = tl.maximum(x, 0.0) - x * z + tl.log(1.0 + tl.exp(-tl.abs(x)))
        acc += tl.sum(tl.where(mask, elem, 0.0), axis=0)
    tl.store(o_ptr + row, (acc / N).to(tl.bfloat16))


def red_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    M, N = logits.shape
    o = torch.empty((M,), device=logits.device, dtype=logits.dtype)
    BLOCK_N = 1024
    _red_bce_with_logits_kernel[(M,)](logits, targets, o, logits.stride(0), targets.stride(0), N,
                                      BLOCK_N=BLOCK_N, num_warps=8)
    return o
