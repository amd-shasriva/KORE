from __future__ import annotations
import torch, triton, triton.language as tl

@triton.jit
def _fx_rmsnorm_linear_mm(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
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
    tl.store(c_ptrs, acc.to(tl.float16), mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _fx_rmsnorm_linear_norm(x_ptr, w_ptr, y_ptr, sm, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * sm + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * sm + offs, (x * rstd * w).to(tl.float16), mask=mask)


def fx_rmsnorm_linear(x, weight, W, eps: float = 1e-06):
    M, K = x.shape
    N = W.shape[1]
    normed = torch.empty_like(x)
    _fx_rmsnorm_linear_norm[(M,)](x, weight, normed, x.stride(0), K, eps, BLOCK_N=triton.next_power_of_2(K), num_warps=8)
    BM, BN, BK = 64, 64, 32
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _fx_rmsnorm_linear_mm[grid](normed, W, out, M, N, K, normed.stride(0), normed.stride(1), W.stride(0), W.stride(1), out.stride(0), out.stride(1), BM=BM, BN=BN, BK=BK)
    return out
