"""GENERATED breadth tr_muon_ns5 seed. (Nesterov) momentum + aspect-scaled decoupled update plus 5-iter Newton-Schulz orthogonalization implemented with naive Triton GEMM/normalization/axpby kernels. Returns (param, momentum_buffer)."""
from __future__ import annotations
import torch, triton, triton.language as tl

_A, _B, _C, _STEPS, _EPS = 3.4445, -4.775, 2.0315, 5, 1e-07




@triton.jit
def _seed_mm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                    sam, sak, sbk, sbn, scm, scn,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_m = pid_m * BM + tl.arange(0, BM)
    off_n = pid_n * BN + tl.arange(0, BN)
    off_k = tl.arange(0, BK)
    a_ptrs = a_ptr + off_m[:, None] * sam + off_k[None, :] * sak
    b_ptrs = b_ptr + off_k[:, None] * sbk + off_n[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kmask = off_k < (K - k0)
        av = tl.load(a_ptrs, mask=(off_m[:, None] < M) & kmask[None, :],
                     other=0.0).to(tl.float32)
        bv = tl.load(b_ptrs, mask=kmask[:, None] & (off_n[None, :] < N),
                     other=0.0).to(tl.float32)
        acc += tl.dot(av, bv)
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    tl.store(c_ptr + off_m[:, None] * scm + off_n[None, :] * scn, acc,
             mask=(off_m[:, None] < M) & (off_n[None, :] < N))


def _seed_mm(a, b, trans_a=False, trans_b=False):
    M = a.shape[1] if trans_a else a.shape[0]
    K = a.shape[0] if trans_a else a.shape[1]
    N = b.shape[0] if trans_b else b.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    sam = a.stride(1) if trans_a else a.stride(0)
    sak = a.stride(0) if trans_a else a.stride(1)
    sbk = b.stride(1) if trans_b else b.stride(0)
    sbn = b.stride(0) if trans_b else b.stride(1)
    BM, BN, BK = 64, 64, 32
    _seed_mm_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        a, b, c, M, N, K, sam, sak, sbk, sbn, c.stride(0), c.stride(1),
        BM=BM, BN=BN, BK=BK, num_warps=4)
    return c


@triton.jit
def _seed_axpby_kernel(a_ptr, b_ptr, out_ptr, numel, ca, cb,
                       BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    av = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    bv = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, ca * av + cb * bv, mask=mask)


def _seed_axpby(a, b, ca, cb):
    out = torch.empty_like(a, dtype=torch.float32)
    numel = a.numel()
    BLOCK = 1024
    _seed_axpby_kernel[(triton.cdiv(numel, BLOCK),)](
        a, b, out, numel, ca, cb, BLOCK=BLOCK, num_warps=4)
    return out


@triton.jit
def _seed_normalize_kernel(x_ptr, out_ptr, norm_ptr, numel, eps,
                           BLOCK: tl.constexpr):
    acc = 0.0
    for start in range(0, numel, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < numel
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x * x, axis=0)
    denom = tl.maximum(tl.sqrt(acc), eps)
    tl.store(norm_ptr, denom)
    for start in range(0, numel, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < numel
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, x / denom, mask=mask)


def _seed_normalize(x, eps):
    x = x.contiguous()
    out = torch.empty_like(x, dtype=torch.float32)
    norm = torch.empty((), device=x.device, dtype=torch.float32)
    _seed_normalize_kernel[(1,)](
        x, out, norm, x.numel(), eps, BLOCK=1024, num_warps=8)
    return out


@triton.jit
def _tr_muon_ns5_mom_kernel(g_ptr, buf_ptr, out_ptr, numel, momentum, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    g = tl.load(g_ptr + offs, mask=mask).to(tl.float32)
    buf = tl.load(buf_ptr + offs, mask=mask).to(tl.float32)
    buf = buf + (1.0 - momentum) * (g - buf)
    upd = g + momentum * (buf - g)
    tl.store(buf_ptr + offs, buf.to(tl.bfloat16), mask=mask)
    tl.store(out_ptr + offs, upd, mask=mask)


@triton.jit
def _tr_muon_ns5_upd_kernel(p_ptr, o_ptr, numel, decay, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    p = tl.load(p_ptr + offs, mask=mask).to(tl.float32)
    o = tl.load(o_ptr + offs, mask=mask).to(tl.float32)
    p = p * decay - alpha * o
    tl.store(p_ptr + offs, p.to(tl.bfloat16), mask=mask)


def _ns(x):
    t = x.shape[-2] > x.shape[-1]
    if t:
        x = x.mT.contiguous()
    x = _seed_normalize(x, _EPS)
    for _ in range(_STEPS):
        A = _seed_mm(x, x, trans_b=True)
        A2 = _seed_mm(A, A)
        B = _seed_axpby(A, A2, _B, _C)
        BX = _seed_mm(B, x)
        x = _seed_axpby(x, BX, _A, 1.0)
    if t:
        x = x.mT.contiguous()
    return x


def tr_muon_ns5(param, grad, momentum_buffer, lr, weight_decay, momentum, eps, ns_a, ns_b, ns_c, ns_steps):
    numel = param.numel()
    geff = torch.empty_like(param, dtype=torch.float32)
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _tr_muon_ns5_mom_kernel[grid](grad, momentum_buffer, geff, numel, momentum, BLOCK=BLOCK, num_warps=4)
    o = _ns(geff.view_as(param)).contiguous()
    scale = max(1.0, param.shape[-2] / param.shape[-1]) ** 0.5
    _tr_muon_ns5_upd_kernel[grid](param, o, numel, 1.0 - lr * weight_decay, lr * scale, BLOCK=BLOCK, num_warps=4)
    return param, momentum_buffer
