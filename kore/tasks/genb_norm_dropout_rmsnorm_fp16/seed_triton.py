"""GENERATED breadth norm_dropout_rmsnorm seed (fp16) - RMSNorm then inverted dropout with a
supplied deterministic mask: y = rmsnorm(x)*w * mask * 1.1111111111111112."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _norm_dropout_rmsnorm_kernel(x_ptr, w_ptr, msk_ptr, y_ptr, sm, N, eps, inv_keep, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * sm + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    d = tl.load(msk_ptr + row * sm + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * sm + offs, (x * rstd * w * d * inv_keep).to(tl.float16), mask=mask)


def norm_dropout_rmsnorm(x: torch.Tensor, weight: torch.Tensor, mask: torch.Tensor, eps: float = 1e-06) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    _norm_dropout_rmsnorm_kernel[(M,)](x, weight, mask, y, x.stride(0), N, eps, 1.1111111111111112,
                       BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return y
