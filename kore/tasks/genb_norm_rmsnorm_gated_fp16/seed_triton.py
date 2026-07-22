"""GENERATED breadth norm_rmsnorm_gated seed (fp16) - gated RMSNorm: rmsnorm(x)*w * silu(gate)."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _norm_rmsnorm_gated_kernel(x_ptr, w_ptr, g_ptr, y_ptr, sm, sg, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * sm + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + row * sg + offs, mask=mask, other=0.0).to(tl.float32)
    out = (x * rstd * w) * (g * tl.sigmoid(g))
    tl.store(y_ptr + row * sm + offs, out.to(tl.float16), mask=mask)


def norm_rmsnorm_gated(x: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor, eps: float = 1e-06) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    _norm_rmsnorm_gated_kernel[(M,)](x, weight, gate, y, x.stride(0), gate.stride(0), N, eps,
                       BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return y
