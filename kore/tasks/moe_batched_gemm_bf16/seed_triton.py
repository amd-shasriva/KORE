"""Seed Triton bf16 batched expert GEMM ``C[e] = A[e] @ B[e]^T`` for gfx950.

Exposes ``batched_gemm(a, b)`` with a ``[E,m,K]``, b ``[E,N,K]`` -> c ``[E,m,N]``
bf16, fp32 accumulate. One program per (expert, m-tile, n-tile); standard tiled
matmul over K. A correct, simple seed the KORE policy optimizes against the
AITER batched_gemm_bf16 bar.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _bgemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  sab, sam, sak, sbb, sbn, sbk, scb, scm, scn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    per_batch = num_m * num_n
    batch = pid // per_batch
    rem = pid % per_batch
    pid_m = rem // num_n
    pid_n = rem % num_n
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + batch * sab + (offs_m[:, None] * sam + offs_k[None, :] * sak)
    b_ptrs = b_ptr + batch * sbb + (offs_n[None, :] * sbn + offs_k[:, None] * sbk)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        kmask = offs_k < K - k * BK
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & kmask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=(offs_n[None, :] < N) & kmask[:, None], other=0.0)
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    c_ptrs = c_ptr + batch * scb + offs_m[:, None] * scm + offs_n[None, :] * scn
    cmask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=cmask)


def batched_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    E, M, K = a.shape
    N = b.shape[1]
    c = torch.empty((E, M, N), device=a.device, dtype=torch.bfloat16)
    BM, BN, BK = 64, 64, 64
    grid = (E * triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _bgemm_kernel[grid](a, b, c, M, N, K,
                        a.stride(0), a.stride(1), a.stride(2),
                        b.stride(0), b.stride(1), b.stride(2),
                        c.stride(0), c.stride(1), c.stride(2),
                        BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=2)
    return c
