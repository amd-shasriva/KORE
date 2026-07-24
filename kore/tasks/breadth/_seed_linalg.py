"""Self-contained Triton source snippets shared by breadth seed generators.

These strings are embedded into generated ``seed_triton.py`` files. They provide
honest starter linear algebra without dispatching through torch/vendor matmul.
"""

TRITON_LINALG_BLOCK = r'''

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
'''
