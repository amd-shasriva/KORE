"""CPU-only tests for the verification-in-the-loop correctness oracle.

Covers (a) the PURE decision logic on synthetic candidate/reference arrays, (b) the
adversarial + metamorphic generators, and (c) the end-to-end oracle on numpy kernels:
a known-wrong kernel (right on random inputs, wrong on an adversarial input) is
REJECTED, a genuinely-correct fp32-accumulate reordering is ACCEPTED, and the
metamorphic + determinism prongs each reject their respective cheats. No GPU / Triton.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from kore.verify import (
    MetamorphicRelation,
    PairComparison,
    ProngSamples,
    Tolerance,
    VerificationResult,
    adversarial_inputs,
    adversarial_patterns,
    compare_pair,
    dtype_extremes,
    equivalence_verdict,
    false_accept_probability,
    metamorphic_relations,
    tolerance_for,
    verify_equivalence,
)

FP32 = Tolerance()  # tight fp32 defaults


# =========================================================================== #
# compare_pair (per-pair primitive)
# =========================================================================== #
def test_compare_pair_identical_passes():
    a = np.linspace(-1, 1, 100).reshape(10, 10)
    cmp = compare_pair(a, a.copy(), FP32)
    assert cmp.ok and cmp.worst_rel_err == 0.0 and math.isinf(cmp.snr_db)


def test_compare_pair_tiny_fp_noise_passes():
    e = np.random.default_rng(0).standard_normal((32, 32))
    a = e + 1e-7 * np.random.default_rng(1).standard_normal((32, 32))
    cmp = compare_pair(a, e, FP32)
    assert cmp.ok


def test_compare_pair_localized_defect_fails_on_rel_err():
    e = np.ones((16, 16))
    a = e.copy()
    a[3, 4] = 2.0  # one very wrong element; norm-SNR would nearly average it away
    cmp = compare_pair(a, e, FP32)
    assert not cmp.ok and cmp.worst_rel_err > FP32.rtol
    assert "rel-err" in cmp.reason


def test_compare_pair_nonfinite_mismatch_fails():
    e = np.array([1.0, 2.0, 3.0])
    a = np.array([1.0, np.nan, 3.0])  # nan where oracle is finite
    cmp = compare_pair(a, e, FP32)
    assert not cmp.ok and math.isinf(cmp.worst_rel_err)
    assert "non-finite" in cmp.reason


def test_compare_pair_matching_inf_and_nan_passes():
    e = np.array([np.inf, -np.inf, np.nan, 1.0])
    a = np.array([np.inf, -np.inf, np.nan, 1.0])
    assert compare_pair(a, e, FP32).ok


def test_compare_pair_shape_mismatch_fails():
    assert not compare_pair(np.zeros((2, 3)), np.zeros((3, 2)), FP32).ok


# =========================================================================== #
# equivalence_verdict (THE pure decision logic)
# =========================================================================== #
def _pairs(n, fn_actual, fn_expected, shape=(8, 8), seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        x = rng.standard_normal(shape)
        out.append((fn_actual(x), fn_expected(x)))
    return out


def test_verdict_accepts_all_prongs_passing():
    ident = lambda x: x
    prongs = [
        ProngSamples("random", "random", _pairs(20, ident, ident)),
        ProngSamples("adversarial", "adversarial", _pairs(5, ident, ident)),
        ProngSamples("metamorphic", "metamorphic", _pairs(4, ident, ident)),
        ProngSamples("determinism", "determinism", _pairs(2, ident, ident)),
    ]
    res = equivalence_verdict(prongs, FP32)
    assert isinstance(res, VerificationResult)
    assert res.verified is True
    assert set(res.passed_prongs()) == {"random", "adversarial", "metamorphic", "determinism"}
    # confidence == 1 - statistical false-accept bound; positive and consistent.
    assert 0.0 < res.confidence <= 1.0
    assert abs(res.confidence - (1.0 - res.false_accept_bound)) < 1e-12
    assert res.n_random_trials == 20
    res.summary()  # must not raise


def test_verdict_rejects_when_one_required_prong_fails():
    ident = lambda x: x
    wrong = lambda x: x + 1.0
    prongs = [
        ProngSamples("random", "random", _pairs(20, ident, ident)),
        # adversarial prong is systematically wrong -> reject
        ProngSamples("adversarial", "adversarial", _pairs(5, wrong, ident)),
    ]
    res = equivalence_verdict(prongs, FP32)
    assert res.verified is False
    assert res.confidence == 0.0
    assert "adversarial" in res.failed_prongs()
    assert res.prong("random").passed is True


def test_verdict_empty_is_not_verified():
    res = equivalence_verdict([], FP32)
    assert res.verified is False and "no prongs" in res.detail


def test_verdict_determinism_prong_uses_tight_tolerance():
    # a drift that is FINE for the random prong (within rtol) must FAIL determinism,
    # which compares a kernel against its OWN prior run and demands near-exactness.
    base = np.random.default_rng(3).standard_normal((16, 16))
    drift = base * (1.0 + 2e-3)  # 2e-3 relative: passes fp32 rtol(3e-3), fails det rtol(1e-5)
    rand = ProngSamples("random", "random", [(drift, base)])
    det = ProngSamples("determinism", "determinism", [(drift, base)])
    assert equivalence_verdict([rand], FP32).prong("random").passed is True
    assert equivalence_verdict([det], FP32).prong("determinism").passed is False


def test_verdict_non_required_prong_does_not_block():
    ident = lambda x: x
    wrong = lambda x: x + 1.0
    prongs = [
        ProngSamples("random", "random", _pairs(10, ident, ident)),
        ProngSamples("optional", "adversarial", _pairs(3, wrong, ident), required=False),
    ]
    res = equivalence_verdict(prongs, FP32)
    assert res.verified is True
    assert res.prong("optional").passed is False


# =========================================================================== #
# false-accept bound (statistical characterisation)
# =========================================================================== #
def test_false_accept_probability_shrinks_with_more_elements():
    p = 1e-3
    assert false_accept_probability(p, 0) == 1.0
    lo = false_accept_probability(p, 100)
    hi = false_accept_probability(p, 100_000)
    assert 0.0 < hi < lo < 1.0
    # closed form check: (1-p)^m
    assert abs(false_accept_probability(p, 100) - (1 - p) ** 100) < 1e-12


def test_false_accept_probability_edges():
    assert false_accept_probability(0.0, 10) == 1.0     # zero-measure defect: undetectable
    assert false_accept_probability(1.0, 10) == 0.0     # everywhere-wrong: always caught


# =========================================================================== #
# adversarial battery
# =========================================================================== #
def test_adversarial_patterns_cover_hard_regimes():
    pats = dict(adversarial_patterns((8, 8), "fp32"))
    for name in ("zeros", "ones", "large_pos", "large_neg", "denormal",
                 "signed_ramp", "sign_alternating", "sparse_spikes",
                 "activation_knots", "mixed_magnitude", "all_equal_const"):
        assert name in pats, f"missing adversarial regime: {name}"
    assert np.all(pats["zeros"] == 0.0)
    assert pats["signed_ramp"].min() < 0 < pats["signed_ramp"].max()  # crosses zero


def test_adversarial_inputs_arity_and_injection():
    cases = list(adversarial_inputs((8, 8), "fp32", arity=2, op_class="elementwise"))
    names = [n for n, _ in cases]
    assert any(n.startswith("all::") for n in names)
    assert any(n.startswith("op0::") for n in names)  # single-operand injection
    assert any(n.startswith("op1::") for n in names)
    for _, inputs in cases:
        assert len(inputs) == 2
        for t in inputs:
            assert t.shape == (8, 8)


def test_dtype_extremes_respect_dtype_range():
    big16, _, _ = dtype_extremes("fp16")
    big32, _, _ = dtype_extremes("fp32")
    assert big16 < 65504.0        # inside fp16 range
    assert big32 > big16          # fp32 can go far larger


# =========================================================================== #
# metamorphic relations
# =========================================================================== #
def test_metamorphic_elementwise_relations_hold_for_true_elementwise():
    # silu is genuinely elementwise -> every relation's lhs must equal rhs
    silu = lambda x: x * (1.0 / (1.0 + np.exp(-x)))
    inputs = (np.random.default_rng(5).standard_normal((16, 16)),)
    for rel in metamorphic_relations("elementwise"):
        lhs, rhs = rel.apply(silu, inputs)
        assert compare_pair(lhs, rhs, FP32,
                            rtol=FP32.metamorphic_rtol,
                            snr_db_min=FP32.metamorphic_snr_db_min).ok, rel.name


def test_metamorphic_locality_rejects_nonlocal_kernel():
    # a "pointwise" kernel that secretly subtracts the GLOBAL mean violates locality
    nonlocal_fn = lambda x: x - x.mean()
    inputs = (np.random.default_rng(6).standard_normal((16, 16)),)
    rel = {r.name: r for r in metamorphic_relations("elementwise")}["elem_locality"]
    lhs, rhs = rel.apply(nonlocal_fn, inputs)
    assert not compare_pair(lhs, rhs, FP32,
                            rtol=FP32.metamorphic_rtol,
                            snr_db_min=FP32.metamorphic_snr_db_min).ok


def test_metamorphic_reduction_relations_hold_for_row_sum():
    row_sum = lambda x: x.sum(axis=1)
    inputs = (np.random.default_rng(7).standard_normal((16, 16)),)
    rels = metamorphic_relations("reduction")
    assert {r.name for r in rels} == {
        "reduce_col_perm_invariance", "reduce_row_perm_equivariance",
        "reduce_row_locality"}
    for rel in rels:
        lhs, rhs = rel.apply(row_sum, inputs)
        assert compare_pair(lhs, rhs, FP32,
                            rtol=FP32.metamorphic_rtol,
                            snr_db_min=FP32.metamorphic_snr_db_min).ok, rel.name


def test_metamorphic_generic_has_no_relations():
    assert metamorphic_relations("generic") == []


# =========================================================================== #
# END-TO-END oracle on numpy kernels (the headline behaviours)
# =========================================================================== #
def _elemwise_gen(shape, dtype, seed, device):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape).astype(np.float32),)


def _reduce_gen(shape, dtype, seed, device):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape).astype(np.float32),)


def test_correct_elementwise_kernel_is_accepted():
    ref = lambda x: (x.astype(np.float64) * (1.0 / (1.0 + np.exp(-x.astype(np.float64)))))
    cand = lambda x: (x * (1.0 / (1.0 + np.exp(-x)))).astype(np.float32)  # fp32 silu
    res = verify_equivalence(cand, ref, _elemwise_gen, dtype="fp32",
                             shape=(64, 128), op_class="elementwise",
                             n_random=24, n_determinism=3)
    assert res.verified is True, res.summary()
    assert res.failed_prongs() == []
    assert res.confidence > 0.999
    assert res.false_accept_bound < 1e-6


def test_correct_fp32_accumulate_reordering_is_accepted():
    # reference: naive fp32 row sum in NATURAL order (the "obvious" implementation).
    # Computing at the candidate's own precision keeps the overflow/saturation regimes
    # (inf_adjacent) fair - both saturate identically.
    def ref(x):
        acc = np.zeros(x.shape[0], dtype=np.float32)
        for j in range(x.shape[1]):
            acc = acc + x[:, j]
        return acc
    # candidate: fp32 accumulate in REVERSED order (a legitimate reassociation a real
    # Triton reduction kernel does) -> not bit-exact vs naive, but within tolerance.
    def cand(x):
        acc = np.zeros(x.shape[0], dtype=np.float32)
        for j in range(x.shape[1] - 1, -1, -1):
            acc = acc + x[:, j]
        return acc
    res = verify_equivalence(cand, ref, _reduce_gen, dtype="fp32",
                             shape=(64, 128), op_class="reduction",
                             n_random=24, n_determinism=3)
    assert res.verified is True, res.summary()
    assert set(res.passed_prongs()) >= {"random", "adversarial", "metamorphic", "determinism"}


def test_known_wrong_kernel_lucky_on_random_is_rejected_by_adversarial():
    # square with a bug EXACTLY at x==0 (returns 1 instead of 0). Random N(0,1) draws
    # never hit exactly 0.0, so it passes the random prong -> a classic LUCKY PASS.
    ref = lambda x: (x.astype(np.float64) ** 2)

    def cand(x):
        y = (x.astype(np.float64) ** 2).astype(np.float32)
        y[x == 0.0] = 1.0  # the planted defect on a measure-zero random slice
        return y

    res = verify_equivalence(cand, ref, _elemwise_gen, dtype="fp32",
                             shape=(64, 128), op_class="elementwise",
                             n_random=24, n_determinism=3)
    assert res.verified is False, res.summary()
    # random got (un)lucky and passed; the DETERMINISTIC adversarial 'zeros' caught it.
    assert res.prong("random").passed is True
    assert res.prong("adversarial").passed is False
    assert "adversarial" in res.failed_prongs()


def test_known_wrong_on_large_magnitude_is_rejected_by_adversarial():
    # correct on random (|x|~O(1)) but clips inputs to [-10,10] before squaring, so it
    # is wrong for the adversarial large-magnitude regime.
    ref = lambda x: (x.astype(np.float64) ** 2)
    cand = lambda x: (np.clip(x.astype(np.float64), -10.0, 10.0) ** 2).astype(np.float32)
    res = verify_equivalence(cand, ref, _elemwise_gen, dtype="fp32",
                             shape=(64, 128), op_class="elementwise",
                             n_random=24, n_determinism=3)
    assert res.verified is False, res.summary()
    assert res.prong("random").passed is True
    assert res.prong("adversarial").passed is False


def test_nondeterministic_kernel_is_rejected_by_determinism_prong():
    ref = lambda x: (x.astype(np.float64) * 2.0)
    _state = {"n": 0}

    def cand(x):
        _state["n"] += 1
        noise = 0.05 * np.random.default_rng(_state["n"]).standard_normal(x.shape)
        return (x * 2.0 + noise.astype(np.float32)).astype(np.float32)

    res = verify_equivalence(cand, ref, _elemwise_gen, dtype="fp32",
                             shape=(64, 128), op_class="elementwise",
                             n_random=8, n_determinism=3)
    assert res.verified is False, res.summary()
    assert res.prong("determinism").passed is False


def test_crashing_kernel_is_rejected():
    ref = lambda x: (x.astype(np.float64) + 1.0)

    def cand(x):
        raise RuntimeError("kernel launch failed")

    res = verify_equivalence(cand, ref, _elemwise_gen, dtype="fp32",
                             shape=(32, 32), op_class="elementwise",
                             n_random=4, n_determinism=2)
    assert res.verified is False
    assert res.prong("random").passed is False


def test_tolerance_for_relaxes_low_precision():
    fp32 = tolerance_for("fp32")
    bf16 = tolerance_for("bf16")
    assert bf16.rtol > fp32.rtol and bf16.snr_db_min < fp32.snr_db_min


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
