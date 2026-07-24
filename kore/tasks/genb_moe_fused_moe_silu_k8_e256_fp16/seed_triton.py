"""GENERATED breadth MoE seed: moe_fused_moe_silu_k8_e256 (fp16).

end-to-end fused top-k MoE MLP (route ids/weights given). Naive, COMPILING, CORRECT starting point: host-side routing/permute
selection (torch) with a Triton kernel for the dominant primitive. The policy is
expected to fuse the routing + grouped GEMM + activation + combine into one kernel.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl



@triton.jit
def _mm_nt_kernel(a_ptr, b_ptr, c_ptr, Mr, N, K,
                  sam, sak, sbn, sbk, scm, scn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offm = pid_m * BM + tl.arange(0, BM)
    offn = pid_n * BN + tl.arange(0, BN)
    offk = tl.arange(0, BK)
    a_ptrs = a_ptr + offm[:, None] * sam + offk[None, :] * sak
    b_ptrs = b_ptr + offn[:, None] * sbn + offk[None, :] * sbk
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        km = offk[None, :] < (K - k0)
        a = tl.load(a_ptrs, mask=(offm[:, None] < Mr) & km, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=(offn[:, None] < N) & km, other=0.0).to(tl.float32)
        acc += tl.dot(a, tl.trans(b))
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    cmask = (offm[:, None] < Mr) & (offn[None, :] < N)
    tl.store(c_ptr + offm[:, None] * scm + offn[None, :] * scn,
             acc.to(c_ptr.dtype.element_ty), mask=cmask)


def _mm_nt(a, b):
    """a [m,K], b [N,K] -> a @ b.T (fp32 accumulate); out dtype = a.dtype."""
    m, K = a.shape
    N = b.shape[0]
    c = torch.empty((m, N), device=a.device, dtype=a.dtype)
    BM, BN, BK = 64, 64, 32
    grid = (triton.cdiv(m, BM), triton.cdiv(N, BN))
    _mm_nt_kernel[grid](a, b, c, m, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1), BM=BM, BN=BN, BK=BK)
    return c


def _grouped_mm(x, w, expert_ids):
    """Per-expert grouped GEMM: out[m] = x[m] @ w[expert_ids[m]].T (naive: one GEMM
    launch per non-empty expert -- the bar a fused variable-M grouped kernel beats)."""
    M, K = x.shape
    E, N, _ = w.shape
    out = torch.zeros((M, N), device=x.device, dtype=x.dtype)
    eids = expert_ids.to(torch.long)
    for e in range(E):
        idx = (eids == e).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        ye = _mm_nt(x.index_select(0, idx).contiguous(), w[e].contiguous())
        out.index_copy_(0, idx, ye)
    return out


@triton.jit
def _gated_act_kernel(gu_ptr, out_ptr, I, sgm, sgi, som, soi,
                      GELU: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < I
    gate = tl.load(gu_ptr + row * sgm + offs * sgi,
                   mask=mask, other=0.0).to(tl.float32)
    up = tl.load(gu_ptr + row * sgm + (I + offs) * sgi,
                 mask=mask, other=0.0).to(tl.float32)
    if GELU:
        z = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
        act = 0.5 * gate * (1.0 + (2.0 * tl.sigmoid(2.0 * z) - 1.0))
    else:
        act = gate * tl.sigmoid(gate)
    tl.store(out_ptr + row * som + offs * soi,
             (act * up).to(out_ptr.dtype.element_ty), mask=mask)


def _swiglu(gu, gelu):
    """Triton gated activation on [M, 2I] -> [M, I]."""
    M = gu.shape[0]
    I = gu.shape[1] // 2
    out = torch.empty((M, I), device=gu.device, dtype=gu.dtype)
    BLOCK = 256
    _gated_act_kernel[(M, triton.cdiv(I, BLOCK))](
        gu, out, I, gu.stride(0), gu.stride(1), out.stride(0), out.stride(1),
        GELU=gelu, BLOCK=BLOCK)
    return out


def _fused_run(hidden, w1, w2, tw, ti):
    """Naive top-k fused MoE MLP: per non-empty expert gather its tokens, run the
    gate/up GEMM -> gated activation -> down GEMM, then weighted-combine over top-k."""
    M, D = hidden.shape
    E = w1.shape[0]
    ids = ti.to(torch.long)
    twf = tw.float()
    out = torch.zeros((M, D), device=hidden.device, dtype=torch.float32)
    for e in range(E):
        mask = ids == e
        tok = mask.any(dim=1)
        if not bool(tok.any()):
            continue
        idx = tok.nonzero(as_tuple=True)[0]
        gu = _mm_nt(hidden.index_select(0, idx).contiguous(), w1[e].contiguous())
        h = _swiglu(gu, False).to(hidden.dtype)
        ye = _mm_nt(h, w2[e].contiguous()).float()
        we = (twf * mask.float()).sum(dim=1)[idx]
        out.index_add_(0, idx, ye * we[:, None])
    return out.to(hidden.dtype)

def moe_fused_moe_silu_k8_e256(hidden, w1, w2, tw, ti):
    return _fused_run(hidden, w1, w2, tw, ti)
