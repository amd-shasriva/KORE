from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _fx_rope_kvcache_half_kernel(k_ptr, cos_ptr, sin_ptr, cache_ptr, S, H, D, HALF, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    pos = (row // H) % S
    base = row * D
    cb = pos * D
    offs = tl.arange(0, BLOCK)
    mask = offs < HALF
    x1 = tl.load(k_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(k_ptr + base + HALF + offs, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(cos_ptr + cb + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(sin_ptr + cb + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(cache_ptr + base + offs, (x1 * c - x2 * s).to(tl.float16), mask=mask)
    tl.store(cache_ptr + base + HALF + offs, (x2 * c + x1 * s).to(tl.float16), mask=mask)


def fx_rope_kvcache_half(k, cos, sin, cache):
    B, S, H, D = k.shape
    HALF = D // 2
    kc = k.contiguous()
    _fx_rope_kvcache_half_kernel[(B * S * H,)](kc, cos, sin, cache, S, H, D, HALF,
                               BLOCK=triton.next_power_of_2(HALF), num_warps=4)
    return cache
