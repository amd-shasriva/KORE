"""GENERATED vendor-baselined a8w8 GEMM seed (int8) vs aiter.gemm_a8w8.
Y = (XQ*x_scale) @ (WQ*w_scale)^T, bf16 out. 8-bit (fp8/int8) operands up-converted
in-register to fp32, fp32 accumulate, scales on the accumulator. Dtype-agnostic:
the load->fp32 path handles both fp8 e4m3fnuz and int8. Regenerate via generate_vendor_ops.py."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _gemm_a8w8_kernel(a_ptr, b_ptr, c_ptr, xs_ptr, ws_ptr, M, N, K,
                      stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                      GROUP_M: tl.constexpr):
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
    acc = acc * xs[:, None] * ws[None, :]
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


def gemm_a8w8(xq: torch.Tensor, wq: torch.Tensor,
              x_scale: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    M, K = xq.shape
    N, _ = wq.shape
    c = torch.empty((M, N), device=xq.device, dtype=torch.bfloat16)
    xs = x_scale.reshape(-1).contiguous()
    ws = w_scale.reshape(-1).contiguous()
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 64, 128, 64, 8
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _gemm_a8w8_kernel[grid](xq, wq, c, xs, ws, M, N, K,
                            xq.stride(0), xq.stride(1), wq.stride(0), wq.stride(1),
                            c.stride(0), c.stride(1),
                            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                            GROUP_M=GROUP_M, num_warps=4, num_stages=2)
    return c
