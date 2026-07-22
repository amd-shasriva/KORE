"""GENERATED breadth sddmm seed (bf16). C = mask ⊙ (A @ B), mask over [M,N].
Dense GEMM (fp32 accumulate) with the sampling mask applied in the epilogue before
the tl.bfloat16 store. Correct + compiling; a real SDDMM would skip masked-out tiles."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _sddmm_kernel(a_ptr, b_ptr, m_ptr, c_ptr, M, N, K,
                  sam, sak, sbk, sbn, smm, smn, scm, scn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * sam + offs_k[None, :] * sak
    b_ptrs = b_ptr + offs_k[:, None] * sbk + offs_n[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        krem = K - k * BK
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < krem), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < krem) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    cmask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    mm = tl.load(m_ptr + offs_m[:, None] * smm + offs_n[None, :] * smn,
                 mask=cmask, other=0.0).to(tl.float32)
    acc = acc * mm
    c_ptrs = c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=cmask)


def sddmm(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BM, BN, BK = 64, 64, 32
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _sddmm_kernel[grid](a, b, mask, c, M, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                        mask.stride(0), mask.stride(1), c.stride(0), c.stride(1),
                        BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=2)
    return c
