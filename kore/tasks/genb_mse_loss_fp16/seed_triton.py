"""GENERATED breadth mse_loss seed (fp16). input[M,N], target[M,N] -> mean((a-b)^2).
Per-row fp32 sum of squared error, then sum over rows / (M*N). Matches F.mse_loss."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _mse_loss_kernel(a_ptr, b_ptr, out_ptr, sm, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    acc = 0.0
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(a_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        d = a - b
        acc += tl.sum(d * d, axis=0)
    tl.store(out_ptr + row, acc)


def mse_loss(inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    M, N = inp.shape
    rows = torch.empty((M,), device=inp.device, dtype=torch.float32)
    _mse_loss_kernel[(M,)](inp, target, rows, inp.stride(0), N, BLOCK=1024, num_warps=8)
    return (rows.sum() / (M * N)).to(inp.dtype)
