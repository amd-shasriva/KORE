"""GENERATED vendor-baselined LayerNorm seed (fp16) vs aiter.layer_norm.
One program/row: fp32 mean+var, affine, tl.float16 store. Regenerate via
kore/tasks/generate_vendor_ops.py."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _layernorm_kernel(x_ptr, w_ptr, b_ptr, y_ptr, sm, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * sm + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * sm + offs, (xc * rstd * w + b).to(tl.float16), mask=mask)


def layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
              eps: float = 1e-6) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    _layernorm_kernel[(M,)](x, weight, bias, y, x.stride(0), N, eps,
                            BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return y
