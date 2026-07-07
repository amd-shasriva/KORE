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
    adversarial_inputs,
    adversarial_patterns,
    dtype_extremes,
    dtype_max,
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
]
