"""GENERATED breadth ssm_hgrn seed (fp16). HGRN gated linear RNN over the last dim:
h_t = f_t h_{t-1} + (1-f_t) g_t, f=sigmoid(f_l). One program per flattened row;
sequential fp32 scan (the policy builds the parallel prefix scan). tl.float16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _ssm_hgrn_kernel(f_ptr, g_ptr, h_ptr, L, srow):
    row = tl.program_id(0)
    base = row * srow
    h = 0.0
    for i in range(0, L):
        f = tl.sigmoid(tl.load(f_ptr + base + i).to(tl.float32))
        g = tl.load(g_ptr + base + i).to(tl.float32)
        h = f * h + (1.0 - f) * g
        tl.store(h_ptr + base + i, h.to(tl.float16))


def ssm_hgrn(f_l, g):
    L = f_l.shape[-1]
    ff = f_l.contiguous().reshape(-1, L)
    gg = g.contiguous().reshape(-1, L)
    h = torch.empty_like(ff)
    _ssm_hgrn_kernel[(ff.shape[0],)](ff, gg, h, L, ff.stride(0), num_warps=1)
    return h.reshape(g.shape)
