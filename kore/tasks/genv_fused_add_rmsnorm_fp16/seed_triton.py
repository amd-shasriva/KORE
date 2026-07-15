"""GENERATED vendor-baselined fused add-RMSNorm seed (fp16) vs aiter.fused_add_rms_norm_cu.
added = x + residual (the new residual); y = RMSNorm(added) * weight. One program
per row, fp32 accumulate, tl.float16 store. Returns (y, added) - the candidate writes
NEW tensors (the vendor baseline is in-place). Regenerate via generate_vendor_ops.py."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _fused_add_rmsnorm_kernel(x_ptr, res_ptr, w_ptr, y_ptr, added_ptr, sm, N, eps,
                              BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(res_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    added = x + r
    tl.store(added_ptr + base + offs, added.to(tl.float16), mask=mask)
    var = tl.sum(added * added, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + base + offs, (added * rstd * w).to(tl.float16), mask=mask)


def fused_add_rmsnorm(x, residual, weight, eps: float = 1e-6):
    M, N = x.shape
    y = torch.empty_like(x)
    added = torch.empty_like(x)
    _fused_add_rmsnorm_kernel[(M,)](x, residual, weight, y, added, x.stride(0), N, eps,
                                    BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return y, added
