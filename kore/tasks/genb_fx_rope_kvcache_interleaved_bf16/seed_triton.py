from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _fx_rope_kvcache_interleaved_kernel(k_ptr, cos_ptr, sin_ptr, cache_ptr, S, H, D, HALF, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    pos = (row // H) % S
    base = row * D
    cb = pos * HALF
    offs = tl.arange(0, BLOCK)
    mask = offs < HALF
    xe = tl.load(k_ptr + base + 2 * offs, mask=mask, other=0.0).to(tl.float32)
    xo = tl.load(k_ptr + base + 2 * offs + 1, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(cos_ptr + cb + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(sin_ptr + cb + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(cache_ptr + base + 2 * offs, (xe * c - xo * s).to(tl.bfloat16), mask=mask)
    tl.store(cache_ptr + base + 2 * offs + 1, (xe * s + xo * c).to(tl.bfloat16), mask=mask)


def fx_rope_kvcache_interleaved(k, cos, sin, cache):
    B, S, H, D = k.shape
    HALF = D // 2
    kc = k.contiguous()
    _fx_rope_kvcache_interleaved_kernel[(B * S * H,)](kc, cos, sin, cache, S, H, D, HALF,
                               BLOCK=triton.next_power_of_2(HALF), num_warps=4)
    return cache
