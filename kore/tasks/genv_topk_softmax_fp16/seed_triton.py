"""GENERATED vendor-baselined MoE router seed (fp16) vs aiter.topk_softmax.
gate[M,E] -> fp32 softmax over experts -> top-k (masked argmax) -> renorm; returned
as a DENSE [M,E] weight tensor (order-independent grading; the vendor baseline is
scattered to dense the same way). Regenerate via generate_vendor_ops.py."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _topk_softmax_kernel(gate_ptr, w_ptr, id_ptr, sg_m, sw_m, sid_m, E, topk,
                         EMAX: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, EMAX)
    mask = offs < E
    g = tl.load(gate_ptr + row * sg_m + offs, mask=mask, other=-float("inf")).to(tl.float32)
    m = tl.max(g, axis=0)
    ex = tl.where(mask, tl.exp(g - m), 0.0)
    probs = tl.where(mask, ex / tl.sum(ex, axis=0), -1.0)
    pw = probs
    wsum = 0.0
    for _ in range(0, topk):
        wsum += tl.max(pw, axis=0)
        pw = tl.where(offs == tl.argmax(pw, axis=0), -1.0, pw)
    pw = probs
    for k in range(0, topk):
        bv = tl.max(pw, axis=0)
        bi = tl.argmax(pw, axis=0)
        tl.store(id_ptr + row * sid_m + k, bi.to(tl.int32))
        tl.store(w_ptr + row * sw_m + k, bv / wsum)
        pw = tl.where(offs == bi, -1.0, pw)


def topk_softmax(gate: torch.Tensor, topk: int) -> torch.Tensor:
    M, E = gate.shape
    w = torch.empty((M, topk), device=gate.device, dtype=torch.float32)
    ids = torch.empty((M, topk), device=gate.device, dtype=torch.int32)
    _topk_softmax_kernel[(M,)](gate, w, ids, gate.stride(0), w.stride(0), ids.stride(0),
                              E, topk, EMAX=triton.next_power_of_2(E), num_warps=4)
    dense = torch.zeros((M, E), device=gate.device, dtype=torch.float32)
    dense.scatter_(1, ids.long(), w)
    return dense
