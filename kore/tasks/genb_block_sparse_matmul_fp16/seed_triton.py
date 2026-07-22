"""GENERATED breadth block-sparse GEMM seed (fp16). Correct partial-fusion starting point:
the (data-dependent) sparsity selection is done host-side in torch, the dense GEMM
runs in this Triton kernel (fp32 accumulate, tl.float16 store). 2D tiling + K-mask
(ROCm/CDNA-safe). The teacher is expected to fuse the masking into the kernel."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                 sam, sak, sbk, sbn, scm, scn,
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
    c_ptrs = c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn
    cmask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=cmask)


def _gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BM, BN, BK = 64, 64, 32
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _gemm_kernel[grid](a, b, c, M, N, K,
                       a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                       c.stride(0), c.stride(1),
                       BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=2)
    return c


def block_sparse_matmul(x: torch.Tensor, w: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    K, N = w.shape
    Kb, Nb = mask.shape
    bk, bn = K // Kb, N // Nb
    wm = (w.reshape(Kb, bk, Nb, bn) * mask.reshape(Kb, 1, Nb, 1).to(w.dtype)).reshape(K, N).contiguous()
    return _gemm(x, wm)
