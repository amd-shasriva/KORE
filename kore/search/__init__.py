"""KORE search: value-guided test-time search over kernel transformations.

Public API:

    from kore.search import search, AlphaKernelConfig, Edit, ProposeContext

``search(root_source, task, env, policy, value_model, budget) -> dict`` runs
AlphaKernel (P1 test-time search) against a verified environment used as a perfect
simulator; see :mod:`kore.search.alphakernel`. :mod:`kore.search.bandit` provides
the Budget + Successive-Halving measurement allocator.

    from kore.search import evolve, EvolveAgentConfig, HarnessProposer

``evolve(task, proposer, env, cfg) -> EvolveAgentResult`` runs the evolve-agent
loop: a population of executable candidates, an archive niched by optimisation
STRATEGY rather than by speedup, model-proposed mutations through
:class:`~kore.agent.harness.AgentHarness`, stabilised measurement, and
best-of-history selection. See :mod:`kore.search.evolve_agent`. AlphaKernel and
the evolve-agent are complementary: AlphaKernel searches one DAG of edits under a
value model, the evolve-agent keeps several designs alive under a live proposer.
"""

from kore.search.alphakernel import (
    AlphaKernelConfig,
    Edit,
    Node,
    ProposeContext,
    ProposePolicy,
    RooflineCeiling,
    ValueModel,
    canonicalize_source,
    fingerprint,
    io_signature,
    make_roofline_ub_fn,
    roofline_speedup_ceiling,
    search,
)
from kore.search.bandit import (
    Budget,
    CallbackArm,
    MeasureStats,
    successive_halving,
)
from kore.search.evolve_agent import (
    Archive,
    BestAcrossSteps,
    Candidate,
    EvolveAgentConfig,
    EvolveAgentResult,
    HarnessProposer,
    StableEvaluator,
    StrategySignature,
    evolve,
    strategy_signature,
)

__all__ = [
    "search",
    "AlphaKernelConfig",
    "Edit",
    "Node",
    "ProposeContext",
    "ProposePolicy",
    "ValueModel",
    "RooflineCeiling",
    "canonicalize_source",
    "fingerprint",
    "io_signature",
    "make_roofline_ub_fn",
    "roofline_speedup_ceiling",
    "Budget",
    "CallbackArm",
    "MeasureStats",
    "successive_halving",
    "evolve",
    "Archive",
    "BestAcrossSteps",
    "Candidate",
    "EvolveAgentConfig",
    "EvolveAgentResult",
    "HarnessProposer",
    "StableEvaluator",
    "StrategySignature",
    "strategy_signature",
]
