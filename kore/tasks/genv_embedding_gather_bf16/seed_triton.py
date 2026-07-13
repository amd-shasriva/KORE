"""GENERATED embedding-gather seed (bf16). weight[V,Dim], ids[T] -> out[T,Dim].
One program per token; copies weight[ids[t]] into out[t] in BLOCK-wide chunks. tl.bfloat16."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _embed_kernel(w_ptr, id_ptr, o_ptr, Dim, sw, so, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    idx = tl.load(id_ptr + t)
    for start in range(0, Dim, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < Dim
        x = tl.load(w_ptr + idx * sw + offs, mask=mask, other=0.0)
        tl.store(o_ptr + t * so + offs, x, mask=mask)


def embedding_gather(weight: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    T = ids.shape[0]
    V, Dim = weight.shape
    o = torch.empty((T, Dim), device=weight.device, dtype=weight.dtype)
    _embed_kernel[(T,)](weight, ids, o, Dim, weight.stride(0), o.stride(0),
                        BLOCK=1024, num_warps=4)
    return o
