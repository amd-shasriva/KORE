from __future__ import annotations
import torch, triton, triton.language as tl

@triton.jit
def _fx_swiglu_mlp_mm(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptrs = a_ptr + (rm[:, None] * sam + rk[None, :] * sak)
    b_ptrs = b_ptr + (rk[:, None] * sbk + rn[None, :] * sbn)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] + k0 < K), other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=(rk[:, None] + k0 < K) & (rn[None, :] < N), other=0.0).to(tl.float32)
        acc += tl.dot(a, b)
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    c_ptrs = c_ptr + (rm[:, None] * scm + rn[None, :] * scn)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))

@triton.jit
def _fx_swiglu_mlp_gate(g_ptr, u_ptr, h_ptr, Ntot, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < Ntot
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(h_ptr + offs, ((g * tl.sigmoid(g)) * u).to(tl.bfloat16), mask=mask)


def fx_swiglu_mlp(x, wg, wu, wd):
    M, K = x.shape
    N = wg.shape[1]
    BM, BN, BK = 64, 64, 32
    g = torch.empty((M, N), device=x.device, dtype=x.dtype)
    u = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid1 = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _fx_swiglu_mlp_mm[grid1](x, wg, g, M, N, K, x.stride(0), x.stride(1), wg.stride(0), wg.stride(1), g.stride(0), g.stride(1), BM=BM, BN=BN, BK=BK)
    _fx_swiglu_mlp_mm[grid1](x, wu, u, M, N, K, x.stride(0), x.stride(1), wu.stride(0), wu.stride(1), u.stride(0), u.stride(1), BM=BM, BN=BN, BK=BK)
    h = torch.empty((M, N), device=x.device, dtype=x.dtype)
    ntot = M * N
    _fx_swiglu_mlp_gate[(triton.cdiv(ntot, 1024),)](g, u, h, ntot, BLOCK=1024)
    out = torch.empty((M, K), device=x.device, dtype=x.dtype)
    grid2 = (triton.cdiv(M, BM), triton.cdiv(K, BN))
    _fx_swiglu_mlp_mm[grid2](h, wd, out, M, K, N, h.stride(0), h.stride(1), wd.stride(0), wd.stride(1), out.stride(0), out.stride(1), BM=BM, BN=BN, BK=BK)
    return out
