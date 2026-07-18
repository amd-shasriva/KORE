"""KORE verification-in-the-loop correctness oracle (verified reward half).

A multi-pronged equivalence verdict that is far stronger and harder to hack than a
single sampled-SNR gate: it combines many reseeded random trials (tight per-element
rel-err bound + SNR), a deterministic adversarial battery, deterministic metamorphic
identities, and a determinism check. See :mod:`kore.verify.equivalence` for the full
"provable vs statistical" characterisation and the honest false-accept bound.

Self-contained and CPU-testable: the decision logic (:func:`equivalence_verdict`) is a
pure function over arrays; ``torch`` is imported lazily. NOT wired into the reward yet.

Public API
----------
    verify_equivalence      end-to-end oracle (runs kernel + applies the verdict)
    equivalence_verdict     PURE decision logic over per-prong output arrays
    VerificationResult      structured verdict (verified / confidence / prongs / bounds)
    ProngSamples/ProngResult, PairComparison, compare_pair
    Tolerance / tolerance_for
    false_accept_probability
    adversarial_inputs / adversarial_patterns / dtype_extremes
    metamorphic_relations / MetamorphicRelation

Coevolutionary adversarial generation (ADDITIVE, OFF BY DEFAULT; nothing above invokes
it - a caller opts in explicitly):
    coevolve_tests          minimal-criterion coevolution that evolves test-cases to
                            BREAK currently-passing candidates (injectable, pure CPU)
    TestCase / generate_cases / mutate_case / crossover_cases   evolvable case genomes
    random_search           undirected baseline (the honest control)
    fold_breaking_cases     fold discovered breaks into a strengthened deterministic
                            battery -> verify_equivalence(..., adversarial_inputs_fn=...)
    CoevolutionResult / RoundStats / CaseOutcome / RandomSearchResult / FoldResult
"""

from __future__ import annotations

from kore.verify.equivalence import (
    PairComparison,
    ProngResult,
    ProngSamples,
    Tolerance,
    VerificationResult,
    compare_pair,
    equivalence_verdict,
    false_accept_probability,
    tolerance_for,
    verify_equivalence,
)
from kore.verify.adversarial import (
    CaseOutcome,
    CoevolutionResult,
    FoldResult,
    RandomSearchResult,
    RoundStats,
    TestCase,
    adversarial_inputs,
    adversarial_patterns,
    coevolve_tests,
    crossover_cases,
    dtype_extremes,
    dtype_max,
    fold_breaking_cases,
    generate_cases,
    list_families,
    make_strengthened_inputs,
    mutate_case,
    random_search,
)
from kore.verify.metamorphic import MetamorphicRelation, metamorphic_relations

__all__ = [
    "verify_equivalence",
    "equivalence_verdict",
    "VerificationResult",
    "ProngSamples",
    "ProngResult",
    "PairComparison",
    "compare_pair",
    "Tolerance",
    "tolerance_for",
    "false_accept_probability",
    "adversarial_inputs",
    "adversarial_patterns",
    "dtype_extremes",
    "dtype_max",
    "metamorphic_relations",
    "MetamorphicRelation",
    # coevolutionary adversarial test-case generation (additive, off by default)
    "TestCase",
    "list_families",
    "generate_cases",
    "mutate_case",
    "crossover_cases",
    "coevolve_tests",
    "CoevolutionResult",
    "RoundStats",
    "CaseOutcome",
    "random_search",
    "RandomSearchResult",
    "fold_breaking_cases",
    "FoldResult",
    "make_strengthened_inputs",
]
