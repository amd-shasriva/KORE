from __future__ import annotations
import torch, triton, triton.language as tl

@triton.jit
def _fx_qkv_proj_split_mm(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
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


def fx_qkv_proj_split(x, weight):
    M, K = x.shape
    N3 = weight.shape[1]
    BM, BN, BK = 64, 64, 32
    c = torch.empty((M, N3), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(M, BM), triton.cdiv(N3, BN))
    _fx_qkv_proj_split_mm[grid](x, weight, c, M, N3, K, x.stride(0), x.stride(1), weight.stride(0), weight.stride(1), c.stride(0), c.stride(1), BM=BM, BN=BN, BK=BK)
    N = N3 // 3
    return c[:, 0:N].contiguous(), c[:, N:2 * N].contiguous(), c[:, 2 * N:3 * N].contiguous()
