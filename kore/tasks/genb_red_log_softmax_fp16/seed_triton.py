"""GENERATED breadth red_log_softmax seed (fp16). x[M,N] -> per-row softmax family
over the last dim. Numerically-stable TWO-pass row kernel: pass 1 an online
(flash-style) running max + rescaled exp-sum in fp32 (no overflow for large
logits); pass 2 reloads x and writes the normalized output. INV_T folds in the
temperature. tl.float16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_log_softmax_kernel(x_ptr, o_ptr, sx, so, N, INV_T, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    m = -float("inf")
    s = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=-float("inf")).to(tl.float32) * INV_T
        blk = tl.max(x, axis=0)
        new_m = tl.maximum(m, blk)
        s = s * tl.exp(m - new_m) + tl.sum(tl.where(mask, tl.exp(x - new_m), 0.0), axis=0)
        m = new_m
    logs = tl.log(s)
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32) * INV_T
        z = x - m
        out = z - logs
        tl.store(o_ptr + row * so + offs, out.to(tl.float16), mask=mask)


def red_log_softmax(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty_like(x)
    BLOCK_N = 1024 if N > 1024 else triton.next_power_of_2(N)
    _red_log_softmax_kernel[(M,)](x, o, x.stride(0), o.stride(0), N, 1.0,
                       BLOCK_N=BLOCK_N, num_warps=8)
    return o
