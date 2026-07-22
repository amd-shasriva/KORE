"""GENERATED breadth norm_qk_rmsnorm seed (bf16) - per-head RMSNorm on q, k over the
head-dim D (one program per (b,s,h) row). Returns (q_normed, k_normed)."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _norm_qk_rmsnorm_kernel(x_ptr, w_ptr, y_ptr, D, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * D
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + base + offs, (x * rstd * w).to(tl.bfloat16), mask=mask)


def norm_qk_rmsnorm(q: torch.Tensor, k: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, eps: float = 1e-06):
    D = q.shape[-1]
    qc, kc = q.contiguous(), k.contiguous()
    rows = qc.numel() // D
    qn = torch.empty_like(qc)
    kn = torch.empty_like(kc)
    B = triton.next_power_of_2(D)
    _norm_qk_rmsnorm_kernel[(rows,)](qc, wq, qn, D, eps, BLOCK=B, num_warps=4)
    _norm_qk_rmsnorm_kernel[(rows,)](kc, wk, kn, D, eps, BLOCK=B, num_warps=4)
    return qn, kn
