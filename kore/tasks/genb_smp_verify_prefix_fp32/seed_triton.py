"""GENERATED breadth smp_verify_prefix seed (fp32). verify-and-accept the longest matching draft/target prefix per row. Naive but correct; the
data-dependent selection runs host-side in torch (the policy fuses it)."""
from __future__ import annotations
import torch, triton, triton.language as tl


def smp_verify_prefix(draft: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    eq = (draft == target).to(torch.int64)
    return eq.cumprod(dim=1).sum(dim=1).to(torch.int64)
