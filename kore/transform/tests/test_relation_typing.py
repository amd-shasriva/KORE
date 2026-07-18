"""Honest ε-typing invariant + relation-lattice consistency (audit fix).

CPU-only, pure-string. These tests pin the corrected relation/ε typing so the
calculus's core promise - **``exact`` == bit-preserving** - holds honestly:

  * ``fp32_accumulator`` and ``set_num_warps`` were mislabeled ``exact`` even
    though they perturb output bits; they are now ``approx`` (fp32_accumulator is
    precision-IMPROVING with ε≈0; set_num_warps can reassociate a cross-warp
    reduction).
  * ``split_k`` (already ``approx``) now zero-initializes its atomic-add output,
    so its ``≈_ε`` contract is no longer a "≈_garbage" gap.
  * the relation lattice composes consistently (``exact ⊔ approx = approx``) and
    a trajectory carries the WEAKEST (max) step ε while the additive meter tracks
    the sum.

None of this is a proof: correctness is still enforced downstream by the env SNR
oracle. These tests assert the *labels are honest*, not that the kernels are
verified.
"""

from __future__ import annotations

from kore.transform import (
    APPROX,
    EXACT,
    LIBRARY,
    RELATION_APPROX,
    RELATION_EXACT,
    ErrorBudget,
    apply_sequence,
    compose_eps,
    compose_relation,
    get,
)
from kore.transform.library import _EPS_FP32_ACC, _EPS_SET_NUM_WARPS
from kore.transform.tests.test_transform import BF16_ACC, GEMM

# Names whose rewrite can (or does) change output bits - they MUST NOT be exact.
_BIT_CHANGING = {
    "set_num_warps", "fp32_accumulator", "retile_block", "split_k",
    "downcast_dtype", "reassociate_reduction", "fast_math_recip",
}
# The honest bit-preserving allowlist (the only transforms allowed to be exact).
_BIT_PRESERVING = {
    "set_num_stages", "set_waves_per_eu", "swizzle_group_m",
    "vectorize_loads", "add_mask_boundary", "reorder_loads",
}


def _budget(total=10.0):
    return ErrorBudget(total=total)


# --------------------------------------------------------------------------- #
# The two audit relabels
# --------------------------------------------------------------------------- #
def test_fp32_accumulator_is_now_approx_with_tiny_eps():
    t = get("fp32_accumulator")
    assert t.is_approx() and not t.is_exact()
    assert t.relation == RELATION_APPROX
    # ε≈0 (precision-improving) but strictly > 0 (not bit-identical to the original)
    assert t.epsilon() == _EPS_FP32_ACC
    assert 0.0 < _EPS_FP32_ACC < 1e-3
    # FUNCTIONAL behavior preserved: still forces the acc to fp32
    out = t.apply(BF16_ACC)
    assert out is not None and "dtype=tl.float32" in out


def test_set_num_warps_is_now_approx():
    t = get("set_num_warps")
    assert t.is_approx() and not t.is_exact()
    assert t.relation == RELATION_APPROX
    assert t.epsilon(value=8) == _EPS_SET_NUM_WARPS
    assert 0.0 < _EPS_SET_NUM_WARPS < 0.02
    # FUNCTIONAL behavior preserved: still edits num_warps
    out = t.apply(GEMM, value=8)
    assert out is not None and "num_warps=8" in out


def test_split_k_is_approx_and_zero_inits_its_atomic_output():
    t = get("split_k")
    assert t.is_approx() and not t.is_exact()
    out = t.apply(GEMM, value=2)
    assert out is not None
    # atomic accumulation is now guarded by a zero-initialized destination
    assert "tl.atomic_add(" in out
    assert "c = torch.zeros(" in out
    assert "torch.empty(" not in out  # the (only) empty output alloc was zeroed


def test_relabels_moved_out_of_exact_into_approx():
    exact_names = {t.name for t in EXACT}
    approx_names = {t.name for t in APPROX}
    for name in ("set_num_warps", "fp32_accumulator"):
        assert name not in exact_names
        assert name in approx_names
    # library partition is exactly exact ⊔ approx, disjoint and complete
    assert exact_names.isdisjoint(approx_names)
    assert exact_names | approx_names == {t.name for t in LIBRARY}


# --------------------------------------------------------------------------- #
# The honest "exact == bit-preserving" invariant
# --------------------------------------------------------------------------- #
def test_exact_set_is_exactly_the_bit_preserving_allowlist():
    exact_names = {t.name for t in EXACT}
    assert exact_names == _BIT_PRESERVING
    # no bit-changing transform sneaks in as exact
    assert exact_names.isdisjoint(_BIT_CHANGING)


def test_every_exact_transform_costs_zero_epsilon_for_all_params():
    for t in EXACT:
        assert t.relation == RELATION_EXACT
        assert t.epsilon() == 0.0
        for params in t.candidate_params(GEMM):
            assert t.epsilon(**params) == 0.0  # exact short-circuits to 0


def test_every_approx_transform_has_nonnegative_eps():
    assert len(APPROX) >= 5
    for t in APPROX:
        assert t.relation == RELATION_APPROX
        assert t.default_eps >= 0.0
        for params in t.candidate_params(GEMM):
            assert t.epsilon(**params) >= 0.0


# --------------------------------------------------------------------------- #
# Relation lattice + ε composition (exact ⊔ approx = approx; carried ε = max)
# --------------------------------------------------------------------------- #
def test_relation_lattice_join_is_consistent():
    assert compose_relation(RELATION_EXACT, RELATION_EXACT) == RELATION_EXACT
    assert compose_relation(RELATION_EXACT, RELATION_APPROX) == RELATION_APPROX
    assert compose_relation(RELATION_APPROX, RELATION_EXACT) == RELATION_APPROX
    assert compose_relation(RELATION_APPROX, RELATION_APPROX) == RELATION_APPROX
    # carried ε is the WEAKEST (max) link, never the sum
    assert compose_eps(0.005, 0.06) == 0.06
    assert compose_eps(_EPS_FP32_ACC, 0.03) == 0.03
    assert compose_eps(0.0, 0.0) == 0.0


def test_exact_then_approx_composes_to_approx_via_apply_sequence():
    # set_num_stages (exact) ⊔ set_num_warps (approx) == approx
    budget = _budget()
    new, applied, rejected, state = apply_sequence(
        GEMM,
        [("set_num_stages", {"value": 4}), ("set_num_warps", {"value": 8})],
        budget)
    assert [a["name"] for a in applied] == ["set_num_stages", "set_num_warps"]
    assert not rejected
    assert state["relation"] == RELATION_APPROX          # exact ⊔ approx = approx
    # the exact step spent nothing; only the approx num_warps drew ε
    assert abs(state["cumulative_eps"] - _EPS_SET_NUM_WARPS) < 1e-12
    assert abs(state["weakest_eps"] - _EPS_SET_NUM_WARPS) < 1e-12
    assert state["n_approx"] == 1 and state["n_steps"] == 2


def test_carried_eps_is_the_weakest_max_while_meter_is_the_sum():
    # fp32_accumulator (ε≈0) then downcast_dtype fp16 (ε=0.03) on the bf16-acc kernel
    budget = _budget()
    new, applied, rejected, state = apply_sequence(
        BF16_ACC,
        [("fp32_accumulator", {}), ("downcast_dtype", {"to": "fp16"})],
        budget)
    assert [a["name"] for a in applied] == ["fp32_accumulator", "downcast_dtype"]
    assert state["relation"] == RELATION_APPROX
    # carried contract = weakest (max) step ε = 0.03 ...
    assert abs(state["weakest_eps"] - 0.03) < 1e-9
    # ... while the additive meter spent the SUM (tiny + 0.03)
    assert abs(state["cumulative_eps"] - (_EPS_FP32_ACC + 0.03)) < 1e-9
    assert state["cumulative_eps"] > state["weakest_eps"]


def test_tiny_eps_approx_is_budget_gated_not_free():
    # fp32_accumulator is approx (a numeric contract), so on a ZERO budget it is
    # inadmissible even though its ε is ~0 - "exact" would have been free.
    zero = ErrorBudget(total=0.0)
    assert not zero.admissible(get("fp32_accumulator"))
    _, applied, rejected, _ = apply_sequence(BF16_ACC, [("fp32_accumulator", {})], zero)
    assert not applied
    assert rejected and rejected[0]["reason"] == "budget_exhausted"
    # with any live budget it applies and spends its tiny ε
    live = ErrorBudget(total=_EPS_FP32_ACC * 10)
    _, applied, rejected, state = apply_sequence(
        BF16_ACC, [("fp32_accumulator", {})], live)
    assert applied and not rejected
    assert abs(state["spent"] - _EPS_FP32_ACC) < 1e-12


def test_num_warps_now_pruned_from_action_space_once_budget_exhausted():
    from kore.transform import admissible_actions

    live = admissible_actions(GEMM, ErrorBudget(total=0.1))
    assert "set_num_warps" in {a.name for a in live}       # affordable while live
    spent = ErrorBudget(total=0.1)
    spent.spend(spent.total)                                # exhaust the budget
    after = admissible_actions(GEMM, spent)
    names_after = {a.name for a in after}
    # now-approx num_warps must vanish; genuinely-exact knobs remain
    assert "set_num_warps" not in names_after
    assert "set_num_stages" in names_after
    assert "swizzle_group_m" in names_after
