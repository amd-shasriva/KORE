"""GENERATED breadth fused_linear_cross_entropy seed (bf16).
x[M,H], W[V,H], targets[M] -> mean CE of logits = x @ W^T. Naive TWO-pass seed:
a tiled fp32 GEMM materializes logits[M,V], then a streaming logsumexp computes the
row CE. The Liger-style FUSION (never materialize [M,V]) is the optimization target."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _flce_gemm_kernel(a_ptr, b_ptr, c_ptr, M, V, H,
                      sam, sah, sbv, sbh, scm, scv,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + (offs_m[:, None] * sam + offs_k[None, :] * sah)
    b_ptrs = b_ptr + (offs_n[None, :] * sbv + offs_k[:, None] * sbh)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(H, BK)):
        kmask = offs_k < H - k * BK
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & kmask[None, :], other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=(offs_n[None, :] < V) & kmask[:, None], other=0.0).to(tl.float32)
        acc += tl.dot(a, b)
        a_ptrs += BK * sah
        b_ptrs += BK * sbh
    c_ptrs = c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scv
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < V))


@triton.jit
def _flce_ce_kernel(logits_ptr, tgt_ptr, loss_ptr, sm, V, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    m = -float("inf")
    s = 0.0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(logits_ptr + base + offs, mask=mask, other=-float("inf")).to(tl.float32)
        blk = tl.max(x, axis=0)
        new_m = tl.maximum(m, blk)
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m), axis=0)
        m = new_m
    lse = m + tl.log(s)
    tgt = tl.load(tgt_ptr + row)
    xt = tl.load(logits_ptr + base + tgt).to(tl.float32)
    tl.store(loss_ptr + row, lse - xt)


def fused_linear_cross_entropy(x: torch.Tensor, weight: torch.Tensor,
                               targets: torch.Tensor) -> torch.Tensor:
    M, H = x.shape
    V = weight.shape[0]
    logits = torch.empty((M, V), device=x.device, dtype=torch.float32)
    BM, BN, BK = 64, 64, 32
    grid = (triton.cdiv(M, BM), triton.cdiv(V, BN))
    _flce_gemm_kernel[grid](x, weight, logits, M, V, H,
                            x.stride(0), x.stride(1), weight.stride(0), weight.stride(1),
                            logits.stride(0), logits.stride(1),
                            BM=BM, BN=BN, BK=BK, num_warps=4)
    loss = torch.empty((M,), device=x.device, dtype=torch.float32)
    _flce_ce_kernel[(M,)](logits, targets, loss, logits.stride(0), V, BLOCK=1024, num_warps=8)
    return loss.mean().to(x.dtype)
