"""GENERATED vendor-baselined block-scaled fp8 GEMM seed (fp8) vs aiter.gemm_a8w8_blockscale.
DeepSeek-V3 blockscale: XQ[M,K] fp8 with x_scale[M,K//128] (1x128), WQ[N,K] fp8 with
w_scale[N//128,K//128] (128x128) -> Y = X_deq @ W_deq^T bf16. Per-128-K-block dequant
applied on the fp32 accumulator. Regenerate via generate_vendor_ops.py."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _bs_kernel(a_ptr, b_ptr, c_ptr, xs_ptr, ws_ptr, M, N, K, KB,
               sam, sak, sbn, sbk, scm, scn, sxm, sxk, swn, swk,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    num_n = N // BN                       # BN == 128 == weight n-block
    pid_m = pid // num_n
    pid_n = pid % num_n
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kb in range(0, KB):
        koff = kb * BK + offs_k
        a = tl.load(a_ptr + offs_m[:, None] * sam + koff[None, :] * sak,
                    mask=offs_m[:, None] < M, other=0.0).to(tl.float32)   # [BM,BK]
        b = tl.load(b_ptr + offs_n[None, :] * sbn + koff[:, None] * sbk,
                    mask=offs_n[None, :] < N, other=0.0).to(tl.float32)   # [BK,BN]
        p = tl.dot(a, b)                                                  # [BM,BN]
        xs = tl.load(xs_ptr + offs_m * sxm + kb * sxk,
                     mask=offs_m < M, other=0.0).to(tl.float32)           # [BM]
        ws = tl.load(ws_ptr + pid_n * swn + kb * swk).to(tl.float32)      # scalar
        acc += p * xs[:, None] * ws
    c_ptrs = c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn
    tl.store(c_ptrs, acc.to(tl.bfloat16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def gemm_a8w8_blockscale(xq, wq, x_scale, w_scale) -> torch.Tensor:
    M, K = xq.shape
    N = wq.shape[0]
    c = torch.empty((M, N), device=xq.device, dtype=torch.bfloat16)
    BM, BN, BK = 64, 128, 128
    grid = (triton.cdiv(M, BM) * (N // BN),)
    _bs_kernel[grid](xq, wq, c, x_scale, w_scale, M, N, K, K // BK,
                     xq.stride(0), xq.stride(1), wq.stride(0), wq.stride(1),
                     c.stride(0), c.stride(1), x_scale.stride(0), x_scale.stride(1),
                     w_scale.stride(0), w_scale.stride(1),
                     BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=2)
    return c
