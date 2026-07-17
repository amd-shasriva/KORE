"""KORE search: value-guided test-time search over kernel transformations.

Public API:

    from kore.search import search, AlphaKernelConfig, Edit, ProposeContext

``search(root_source, task, env, policy, value_model, budget) -> dict`` runs
AlphaKernel (P1 test-time search) against a verified environment used as a perfect
simulator; see :mod:`kore.search.alphakernel`. :mod:`kore.search.bandit` provides
the Budget + Successive-Halving measurement allocator.
"""

from kore.search.alphakernel import (
    AlphaKernelConfig,
    Edit,
    Node,
    ProposeContext,
    ProposePolicy,
    ValueModel,
    canonicalize_source,
    fingerprint,
    io_signature,
    roofline_speedup_ceiling,
    search,
)
from kore.search.bandit import (
    Budget,
    CallbackArm,
    MeasureStats,
    successive_halving,
)

__all__ = [
    "search",
    "AlphaKernelConfig",
    "Edit",
    "Node",
    "ProposeContext",
    "ProposePolicy",
    "ValueModel",
    "canonicalize_source",
    "fingerprint",
    "io_signature",
    "roofline_speedup_ceiling",
    "Budget",
    "CallbackArm",
    "MeasureStats",
    "successive_halving",
]
