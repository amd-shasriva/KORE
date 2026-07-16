"""Seed GEMM BACKWARD (dgrad + wgrad) for gfx950 (CDNA4).

Exposes ``gemm_backward(x, w, dy) -> (dx, dw)`` for the linear layer Y = X @ W^T
with X[M,K], W[N,K], dY[M,N] (all bf16):
    dx = dY @ W        [M,K]   (dgrad, contracts over N)
    dw = dY^T @ X      [N,K]   (wgrad, contracts over M tokens)

Both are plain GEMMs, so one generic tiled matmul kernel (fp32 accumulation on the
CDNA4 matrix cores, L2-friendly program grouping) is called twice with the right
strides -- for dw the dY^T is expressed purely by swapping dY's row/col strides, no
physical transpose. A correct baseline the KORE policy optimizes: tune BLOCK/GROUP,
fuse the two GEMMs, exploit that both read dY.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _mm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    sam, sak, sbk, sbn, scm, scn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """C[M,N] = A[M,K] @ B[K,N], fp32 accumulate, bf16 store."""
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
    a_ptrs = a_ptr + (offs_am[:, None] * sam + offs_k[None, :] * sak)
    b_ptrs = b_ptr + (offs_k[:, None] * sbk + offs_bn[None, :] * sbn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        kmask = offs_k[None, :] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=kmask, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * sak
        b_ptrs += BLOCK_K * sbk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + scm * offs_cm[:, None] + scn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


def _launch_mm(a, b, c, M, N, K, sam, sak, sbk, sbn):
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 64, 64, 32, 8
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _mm_kernel[grid](
        a, b, c, M, N, K,
        sam, sak, sbk, sbn, c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
        num_warps=4, num_stages=2,
    )


def gemm_backward(x: torch.Tensor, w: torch.Tensor, dy: torch.Tensor):
    M, K = x.shape
    N = w.shape[0]
    dx = torch.empty((M, K), device=x.device, dtype=x.dtype)
    dw = torch.empty((N, K), device=x.device, dtype=x.dtype)

    # dx[M,K] = dY[M,N] @ W[N,K]   (contract over N)
    _launch_mm(dy, w, dx, M, K, N,
               dy.stride(0), dy.stride(1), w.stride(0), w.stride(1))
    # dw[N,K] = dY^T[N,M] @ X[M,K] (contract over M); dY^T via swapped strides.
    _launch_mm(dy, x, dw, N, K, M,
               dy.stride(1), dy.stride(0), x.stride(0), x.stride(1))
    return dx, dw
