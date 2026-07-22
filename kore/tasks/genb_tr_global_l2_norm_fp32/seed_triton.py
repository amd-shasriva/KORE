"""GENERATED breadth tr_global_l2_norm seed (fp32). Multi-tensor global L2 norm."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _tr_global_l2_norm_kernel(g_ptr, part_ptr, sm, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    acc = 0.0
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < N
        x = tl.load(g_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
        acc += tl.sum(x * x, axis=0)
    tl.store(part_ptr + row, acc)


def tr_global_l2_norm(blob):
    G, N = blob.shape
    part = torch.empty((G,), device=blob.device, dtype=torch.float32)
    _tr_global_l2_norm_kernel[(G,)](blob, part, blob.stride(0), N, BLOCK=1024, num_warps=8)
    return torch.sqrt(part.sum()).to(blob.dtype)
