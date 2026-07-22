"""GENERATED breadth norm_qk_layernorm seed (fp16) - per-head LayerNorm on q, k over the
head-dim D. Returns (q_normed, k_normed)."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _norm_qk_layernorm_kernel(x_ptr, w_ptr, b_ptr, y_ptr, D, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * D
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / D
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + base + offs, (xc * rstd * w + b).to(tl.float16), mask=mask)


def norm_qk_layernorm(q: torch.Tensor, k: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor,
         bq: torch.Tensor, bk: torch.Tensor, eps: float = 1e-06):
    D = q.shape[-1]
    qc, kc = q.contiguous(), k.contiguous()
    rows = qc.numel() // D
    qn = torch.empty_like(qc)
    kn = torch.empty_like(kc)
    B = triton.next_power_of_2(D)
    _norm_qk_layernorm_kernel[(rows,)](qc, wq, bq, qn, D, eps, BLOCK=B, num_warps=4)
    _norm_qk_layernorm_kernel[(rows,)](kc, wk, bk, kn, D, eps, BLOCK=B, num_warps=4)
    return qn, kn
