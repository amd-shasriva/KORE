"""GENERATED breadth cv_winograd_filter_transform_f2x2 seed (bf16). Naive Winograd F(2x2,3x3) FILTER transform U = G.g.Gt (exact rational transform, unrolled per 3x3 filter) vs the batched-matmul oracle. tl.bfloat16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _cv_winograd_filter_transform_f2x2_kernel(g_ptr, y_ptr, NROW, s0, s1, s2, s3, o0, o1, o2, o3):
    pid = tl.program_id(0)
    j = pid % NROW
    i = pid // NROW
    base = i * s0 + j * s1
    obase = i * o0 + j * o1
    g00 = tl.load(g_ptr + base + 0 * s2 + 0 * s3).to(tl.float32)
    g01 = tl.load(g_ptr + base + 0 * s2 + 1 * s3).to(tl.float32)
    g02 = tl.load(g_ptr + base + 0 * s2 + 2 * s3).to(tl.float32)
    g10 = tl.load(g_ptr + base + 1 * s2 + 0 * s3).to(tl.float32)
    g11 = tl.load(g_ptr + base + 1 * s2 + 1 * s3).to(tl.float32)
    g12 = tl.load(g_ptr + base + 1 * s2 + 2 * s3).to(tl.float32)
    g20 = tl.load(g_ptr + base + 2 * s2 + 0 * s3).to(tl.float32)
    g21 = tl.load(g_ptr + base + 2 * s2 + 1 * s3).to(tl.float32)
    g22 = tl.load(g_ptr + base + 2 * s2 + 2 * s3).to(tl.float32)
    u00 = g00
    tl.store(y_ptr + obase + 0 * o2 + 0 * o3, (u00).to(tl.bfloat16))
    u01 = 0.5 * g00 + 0.5 * g01 + 0.5 * g02
    tl.store(y_ptr + obase + 0 * o2 + 1 * o3, (u01).to(tl.bfloat16))
    u02 = 0.5 * g00 - 0.5 * g01 + 0.5 * g02
    tl.store(y_ptr + obase + 0 * o2 + 2 * o3, (u02).to(tl.bfloat16))
    u03 = g02
    tl.store(y_ptr + obase + 0 * o2 + 3 * o3, (u03).to(tl.bfloat16))
    u10 = 0.5 * g00 + 0.5 * g10 + 0.5 * g20
    tl.store(y_ptr + obase + 1 * o2 + 0 * o3, (u10).to(tl.bfloat16))
    u11 = 0.25 * g00 + 0.25 * g01 + 0.25 * g02 + 0.25 * g10 + 0.25 * g11 + 0.25 * g12 + 0.25 * g20 + 0.25 * g21 + 0.25 * g22
    tl.store(y_ptr + obase + 1 * o2 + 1 * o3, (u11).to(tl.bfloat16))
    u12 = 0.25 * g00 - 0.25 * g01 + 0.25 * g02 + 0.25 * g10 - 0.25 * g11 + 0.25 * g12 + 0.25 * g20 - 0.25 * g21 + 0.25 * g22
    tl.store(y_ptr + obase + 1 * o2 + 2 * o3, (u12).to(tl.bfloat16))
    u13 = 0.5 * g02 + 0.5 * g12 + 0.5 * g22
    tl.store(y_ptr + obase + 1 * o2 + 3 * o3, (u13).to(tl.bfloat16))
    u20 = 0.5 * g00 - 0.5 * g10 + 0.5 * g20
    tl.store(y_ptr + obase + 2 * o2 + 0 * o3, (u20).to(tl.bfloat16))
    u21 = 0.25 * g00 + 0.25 * g01 + 0.25 * g02 - 0.25 * g10 - 0.25 * g11 - 0.25 * g12 + 0.25 * g20 + 0.25 * g21 + 0.25 * g22
    tl.store(y_ptr + obase + 2 * o2 + 1 * o3, (u21).to(tl.bfloat16))
    u22 = 0.25 * g00 - 0.25 * g01 + 0.25 * g02 - 0.25 * g10 + 0.25 * g11 - 0.25 * g12 + 0.25 * g20 - 0.25 * g21 + 0.25 * g22
    tl.store(y_ptr + obase + 2 * o2 + 2 * o3, (u22).to(tl.bfloat16))
    u23 = 0.5 * g02 - 0.5 * g12 + 0.5 * g22
    tl.store(y_ptr + obase + 2 * o2 + 3 * o3, (u23).to(tl.bfloat16))
    u30 = g20
    tl.store(y_ptr + obase + 3 * o2 + 0 * o3, (u30).to(tl.bfloat16))
    u31 = 0.5 * g20 + 0.5 * g21 + 0.5 * g22
    tl.store(y_ptr + obase + 3 * o2 + 1 * o3, (u31).to(tl.bfloat16))
    u32 = 0.5 * g20 - 0.5 * g21 + 0.5 * g22
    tl.store(y_ptr + obase + 3 * o2 + 2 * o3, (u32).to(tl.bfloat16))
    u33 = g22
    tl.store(y_ptr + obase + 3 * o2 + 3 * o3, (u33).to(tl.bfloat16))


def cv_winograd_filter_transform_f2x2(g: torch.Tensor) -> torch.Tensor:
    A, B, IH, IW = g.shape
    y = torch.empty((A, B, 4, 4), device=g.device, dtype=g.dtype)
    grid = (A * B,)
    _cv_winograd_filter_transform_f2x2_kernel[grid](g, y, B,
                       g.stride(0), g.stride(1), g.stride(2), g.stride(3),
                       y.stride(0), y.stride(1), y.stride(2), y.stride(3), num_warps=1)
    return y
