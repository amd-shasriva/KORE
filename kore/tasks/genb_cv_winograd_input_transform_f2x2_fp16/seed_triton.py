"""GENERATED breadth cv_winograd_input_transform_f2x2 seed (fp16). Naive Winograd F(2x2,3x3) INPUT transform V = Bt.d.B (exact integer transform, unrolled per 4x4 tile) vs the batched-matmul oracle. tl.float16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _cv_winograd_input_transform_f2x2_kernel(d_ptr, y_ptr, NROW, s0, s1, s2, s3, o0, o1, o2, o3):
    pid = tl.program_id(0)
    j = pid % NROW
    i = pid // NROW
    base = i * s0 + j * s1
    obase = i * o0 + j * o1
    d00 = tl.load(d_ptr + base + 0 * s2 + 0 * s3).to(tl.float32)
    d01 = tl.load(d_ptr + base + 0 * s2 + 1 * s3).to(tl.float32)
    d02 = tl.load(d_ptr + base + 0 * s2 + 2 * s3).to(tl.float32)
    d03 = tl.load(d_ptr + base + 0 * s2 + 3 * s3).to(tl.float32)
    d10 = tl.load(d_ptr + base + 1 * s2 + 0 * s3).to(tl.float32)
    d11 = tl.load(d_ptr + base + 1 * s2 + 1 * s3).to(tl.float32)
    d12 = tl.load(d_ptr + base + 1 * s2 + 2 * s3).to(tl.float32)
    d13 = tl.load(d_ptr + base + 1 * s2 + 3 * s3).to(tl.float32)
    d20 = tl.load(d_ptr + base + 2 * s2 + 0 * s3).to(tl.float32)
    d21 = tl.load(d_ptr + base + 2 * s2 + 1 * s3).to(tl.float32)
    d22 = tl.load(d_ptr + base + 2 * s2 + 2 * s3).to(tl.float32)
    d23 = tl.load(d_ptr + base + 2 * s2 + 3 * s3).to(tl.float32)
    d30 = tl.load(d_ptr + base + 3 * s2 + 0 * s3).to(tl.float32)
    d31 = tl.load(d_ptr + base + 3 * s2 + 1 * s3).to(tl.float32)
    d32 = tl.load(d_ptr + base + 3 * s2 + 2 * s3).to(tl.float32)
    d33 = tl.load(d_ptr + base + 3 * s2 + 3 * s3).to(tl.float32)
    u00 = d00 - d02 - d20 + d22
    tl.store(y_ptr + obase + 0 * o2 + 0 * o3, (u00).to(tl.float16))
    u01 = d01 + d02 - d21 - d22
    tl.store(y_ptr + obase + 0 * o2 + 1 * o3, (u01).to(tl.float16))
    u02 = -d01 + d02 + d21 - d22
    tl.store(y_ptr + obase + 0 * o2 + 2 * o3, (u02).to(tl.float16))
    u03 = d01 - d03 - d21 + d23
    tl.store(y_ptr + obase + 0 * o2 + 3 * o3, (u03).to(tl.float16))
    u10 = d10 - d12 + d20 - d22
    tl.store(y_ptr + obase + 1 * o2 + 0 * o3, (u10).to(tl.float16))
    u11 = d11 + d12 + d21 + d22
    tl.store(y_ptr + obase + 1 * o2 + 1 * o3, (u11).to(tl.float16))
    u12 = -d11 + d12 - d21 + d22
    tl.store(y_ptr + obase + 1 * o2 + 2 * o3, (u12).to(tl.float16))
    u13 = d11 - d13 + d21 - d23
    tl.store(y_ptr + obase + 1 * o2 + 3 * o3, (u13).to(tl.float16))
    u20 = -d10 + d12 + d20 - d22
    tl.store(y_ptr + obase + 2 * o2 + 0 * o3, (u20).to(tl.float16))
    u21 = -d11 - d12 + d21 + d22
    tl.store(y_ptr + obase + 2 * o2 + 1 * o3, (u21).to(tl.float16))
    u22 = d11 - d12 - d21 + d22
    tl.store(y_ptr + obase + 2 * o2 + 2 * o3, (u22).to(tl.float16))
    u23 = -d11 + d13 + d21 - d23
    tl.store(y_ptr + obase + 2 * o2 + 3 * o3, (u23).to(tl.float16))
    u30 = d10 - d12 - d30 + d32
    tl.store(y_ptr + obase + 3 * o2 + 0 * o3, (u30).to(tl.float16))
    u31 = d11 + d12 - d31 - d32
    tl.store(y_ptr + obase + 3 * o2 + 1 * o3, (u31).to(tl.float16))
    u32 = -d11 + d12 + d31 - d32
    tl.store(y_ptr + obase + 3 * o2 + 2 * o3, (u32).to(tl.float16))
    u33 = d11 - d13 - d31 + d33
    tl.store(y_ptr + obase + 3 * o2 + 3 * o3, (u33).to(tl.float16))


def cv_winograd_input_transform_f2x2(d: torch.Tensor) -> torch.Tensor:
    A, B, IH, IW = d.shape
    y = torch.empty((A, B, 4, 4), device=d.device, dtype=d.dtype)
    grid = (A * B,)
    _cv_winograd_input_transform_f2x2_kernel[grid](d, y, B,
                       d.stride(0), d.stride(1), d.stride(2), d.stride(3),
                       y.stride(0), y.stride(1), y.stride(2), y.stride(3), num_warps=1)
    return y
