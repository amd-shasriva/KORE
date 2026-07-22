"""GENERATED breadth nucleus (top-p) seed (fp16). logits[M,N] -> renormalized
top-p probabilities. Pass 1 fp32 softmax; then a selection loop keeps tokens in
descending prob order while the EXCLUSIVE cumulative mass <= p (arithmetic masks,
no data-dependent branch), then renormalizes the kept set. O(N^2)/row - a correct
partial seed the teacher is expected to replace with a real top-p. tl.float16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _topp_mask_kernel(x_ptr, o_ptr, sx, so, N, P, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * sx + offs, mask=mask, other=-float("inf")).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.where(mask, tl.exp(x - m), 0.0)
    probs = e / tl.sum(e, axis=0)
    work = tl.where(mask, probs, -1.0)
    keep = tl.zeros([BLOCK_N], dtype=tl.float32)
    cum = 0.0
    for i in range(0, N):
        v = tl.max(work, axis=0)
        j = tl.argmax(work, axis=0)
        tk = tl.where(cum <= P, 1.0, 0.0)
        is_j = tl.where(offs == j, 1.0, 0.0)
        keep = keep + is_j * tk
        cum = cum + v * tk
        work = tl.where(offs == j, -1.0, work)
    kp = probs * keep
    denom = tl.sum(kp, axis=0)
    tl.store(o_ptr + row * so + offs, (kp / denom).to(tl.float16), mask=mask)


def topp_mask(x: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty_like(x)
    _topp_mask_kernel[(M,)](x, o, x.stride(0), o.stride(0), N, float(p),
                            BLOCK_N=triton.next_power_of_2(N), num_warps=4)
    return o
