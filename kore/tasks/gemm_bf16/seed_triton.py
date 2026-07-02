"""Seed Triton dense bf16 GEMM for gfx942 (MI325X, CDNA3).

Exposes ``gemm(a, b) -> c`` computing ``C = A @ B`` in bf16:
    A: [M, K] bf16
    B: [K, N] bf16
    C: [M, N] bf16   (fp32 accumulate)

Standard grouped/super-blocked tiling with fp32 accumulation and K-loop masking
(so non-multiple-of-BLOCK_K tails like K=4095 are handled correctly). A correct,
reasonably-tuned baseline the KORE policy learns to edit/optimize against the
hipBLASLt vendor serving bar.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_rem, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_rem, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c = acc.to(tl.bfloat16)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "incompatible dims"
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    if M <= 16:
        # tiny-M decode / GEMV: skinny tile to avoid wasting rows.
        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 16, 128, 64, 1
        num_warps, num_stages = 4, 2
    else:
        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 128, 128, 32, 8
        num_warps, num_stages = 4, 2
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _gemm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
        num_warps=num_warps, num_stages=num_stages,
    )
    return c
