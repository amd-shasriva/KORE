"""Seed fp8 a8w8 GEMM with a FUSED bias + fp8 requant epilogue for gfx950 / CDNA4.

Exposes ``gemm(xq, wq, x_scale, w_scale, bias, out_scale) -> y_fp8`` computing
``y_fp8 = requant((XQ*x_scale) @ (WQ*w_scale)^T + bias, out_scale)``:

    XQ: [M,K] fp8, WQ: [N,K] fp8, x_scale: [M,1] (per-token), w_scale: [1,N] (per-channel),
    bias: [N] fp32, out_scale: fp32 scalar (static output requant scale),
    y_fp8: [M,N] fp8 e4m3fn.

fp8 operands are up-converted to fp32 and accumulated in fp32; the scales + bias are
applied on the fp32 accumulator, then the fused epilogue divides by out_scale, clamps to
+/- FP8_MAX, and stores the result directly as fp8 (no bf16 HBM round trip). A correct
starter the KORE policy optimizes against the unfused vendor-GEMM-plus-requant bar.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_requant_kernel(
    a_ptr, b_ptr, c_ptr, xs_ptr, ws_ptr, bias_ptr, out_scale, fmax, M, N, K,
    stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn,
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
    b_ptrs = b_ptr + (offs_bn[None, :] * stride_bn + offs_k[:, None] * stride_bk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        kmask = offs_k[None, :] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=kmask, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    xs = tl.load(xs_ptr + offs_cm, mask=offs_cm < M, other=0.0).to(tl.float32)
    ws = tl.load(ws_ptr + offs_cn, mask=offs_cn < N, other=0.0).to(tl.float32)
    bs = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0).to(tl.float32)
    acc = acc * xs[:, None] * ws[None, :] + bs[None, :]
    # Fused fp8 requant epilogue: divide by the static output scale, clamp, store as fp8.
    acc = acc / out_scale
    acc = tl.minimum(tl.maximum(acc, -fmax), fmax)

    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


def gemm(xq: torch.Tensor, wq: torch.Tensor, x_scale: torch.Tensor,
         w_scale: torch.Tensor, bias: torch.Tensor, out_scale: torch.Tensor) -> torch.Tensor:
    M, K = xq.shape
    N, K2 = wq.shape
    assert K == K2, "incompatible dims"
    c = torch.empty((M, N), device=xq.device, dtype=xq.dtype)   # fp8 output
    xs = x_scale.reshape(-1).contiguous()   # [M]
    ws = w_scale.reshape(-1).contiguous()   # [N]
    bs = bias.reshape(-1).contiguous()      # [N]
    fmax = float(torch.finfo(xq.dtype).max)
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 64, 128, 64, 8
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _gemm_requant_kernel[grid](
        xq, wq, c, xs, ws, bs, float(out_scale), fmax, M, N, K,
        xq.stride(0), xq.stride(1), wq.stride(0), wq.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
        num_warps=4, num_stages=2,
    )
    return c
