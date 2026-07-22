"""GENERATED breadth ssm_gilr seed (fp16). GILR gated impulse linear recurrent over
the last dim: h_t = f_t h_{t-1} + i_t z_t, f=sigmoid(f_l), i=sigmoid(i_l). One
program per flattened row; sequential fp32 scan. tl.float16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _ssm_gilr_kernel(f_ptr, i_ptr, z_ptr, h_ptr, L, srow):
    row = tl.program_id(0)
    base = row * srow
    h = 0.0
    for t in range(0, L):
        f = tl.sigmoid(tl.load(f_ptr + base + t).to(tl.float32))
        ii = tl.sigmoid(tl.load(i_ptr + base + t).to(tl.float32))
        z = tl.load(z_ptr + base + t).to(tl.float32)
        h = f * h + ii * z
        tl.store(h_ptr + base + t, h.to(tl.float16))


def ssm_gilr(f_l, i_l, z):
    L = f_l.shape[-1]
    ff = f_l.contiguous().reshape(-1, L)
    ii = i_l.contiguous().reshape(-1, L)
    zz = z.contiguous().reshape(-1, L)
    h = torch.empty_like(ff)
    _ssm_gilr_kernel[(ff.shape[0],)](ff, ii, zz, h, L, ff.stride(0), num_warps=1)
    return h.reshape(z.shape)
