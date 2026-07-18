"""CPU-only tests for the self-extending transform library (``discover.py``).

Pure-string, no GPU/torch/triton. Verifies the frontier "library extends itself"
mechanism is SAFE and honest:

  * every proposal is well-typed (conservatively ``approx``, floored ε, discovered
    namespace, unique names) and NEVER raises;
  * a proposal is a **no-op** (``None``) when its precondition doesn't match, and
    applies cleanly when it does;
  * discovery is **OFF by default** - the curated ``LIBRARY`` is never mutated and
    the default action space contains no proposals unless a caller opts in via the
    ``library=`` seam;
  * the registry-merge helper appends + de-dupes without mutating its inputs;
  * ε-budget accounting composes correctly for discovered approx moves.

These are PROPOSALS, not proofs - correctness is still SNR-gated downstream. The
tests assert the *typing/plumbing is safe*, never that a rewrite is verified.
"""

from __future__ import annotations

from kore.transform import (
    LIBRARY,
    RELATION_APPROX,
    RELATION_EXACT,
    ErrorBudget,
    admissible_actions,
    apply_sequence,
)
from kore.transform.discover import (
    DISCOVERED_PREFIX,
    describe_proposals,
    discover_transforms,
    extend_library,
    is_discovered,
    merge_transforms,
    propose_fusions,
    propose_knob_sweeps,
    propose_vectorize_widths,
)
from kore.transform.discover import _DISCOVER_EPS_FLOOR, _FUSION_EPS
from kore.transform.tests.test_transform import BF16_ACC, ELEMENTWISE, GEMM


def _by_name(transforms):
    return {t.name: t for t in transforms}


def _all_proposals():
    """Every proposal from the curated library, source-free (nothing pruned)."""
    return discover_transforms(LIBRARY)


# --------------------------------------------------------------------------- #
# OFF by default: the curated library is never touched
# --------------------------------------------------------------------------- #
def test_discovery_is_off_by_default_library_unchanged():
    before = list(LIBRARY)
    props = _all_proposals()
    # importing/using discover does not mutate the curated library ...
    assert list(LIBRARY) == before
    assert len(LIBRARY) == 13
    assert not any(is_discovered(t) for t in LIBRARY)
    # ... and every proposal is a brand-new object outside the library
    lib_ids = {id(t) for t in LIBRARY}
    assert all(id(t) not in lib_ids for t in props)


def test_default_action_space_has_no_proposals():
    # No library= override -> the default (global LIBRARY) action space, unchanged.
    actions = admissible_actions(GEMM, ErrorBudget.for_op("gemm", "bf16"))
    assert actions
    assert not any(is_discovered(a) or a.name.startswith(DISCOVERED_PREFIX)
                   for a in actions)


# --------------------------------------------------------------------------- #
# Every proposal is well-typed
# --------------------------------------------------------------------------- #
def test_all_proposals_are_well_typed():
    props = _all_proposals()
    assert props, "expected proposals from the curated library"
    names = [t.name for t in props]
    assert len(names) == len(set(names)), "proposal names must be unique"
    for t in props:
        assert is_discovered(t) and t.name.startswith(DISCOVERED_PREFIX)
        assert t.relation == RELATION_APPROX and t.is_approx()  # conservative
        assert t.knob and t.summary
        assert t.default_eps >= _DISCOVER_EPS_FLOOR  # never under-estimate drift


def test_proposals_never_raise_on_degenerate_source():
    props = _all_proposals()
    for t in props:
        for src in ("", "   ", "not a kernel", GEMM[:37], "a =", "x = y ="):
            out = t.apply(src)
            assert out is None or isinstance(out, str)
            assert isinstance(t.side_conditions(src), list)


def test_discovery_is_deterministic():
    a = [t.name for t in _all_proposals()]
    b = [t.name for t in _all_proposals()]
    assert a == b


# --------------------------------------------------------------------------- #
# Knob-sweep proposals: apply when applicable, no-op when not
# --------------------------------------------------------------------------- #
def test_knob_sweep_applies_when_applicable():
    props = _by_name(_all_proposals())
    # num_warps sweep to a value NOT in the base grid (1/2/16)
    out = props["disc:set_num_warps[value=16]"].apply(GEMM)
    assert out is not None and "num_warps=16" in out
    # a block re-tile sweep
    out = props["disc:retile_block[block_m=256]"].apply(GEMM)
    assert out is not None and "BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 256, 128, 64, 8" in out
    # a num_stages sweep
    out = props["disc:set_num_stages[value=5]"].apply(GEMM)
    assert out is not None and "num_stages=5" in out


def test_knob_sweep_is_noop_when_inapplicable():
    props = _by_name(_all_proposals())
    # split_k needs a cdiv K-loop; ELEMENTWISE has none -> no-op (None), never raises
    assert props["disc:split_k[value=8]"].apply(ELEMENTWISE) is None
    # a BLOCK re-tile has no BLOCK_* knob to edit in ELEMENTWISE -> no-op
    assert props["disc:retile_block[block_k=128]"].apply(ELEMENTWISE) is None


def test_sweep_reuses_base_side_conditions_on_the_actual_source():
    props = _by_name(_all_proposals())
    # the downcast sweep inherits _cond_downcast, which rejects a bf16 accumulator
    viol = props["disc:downcast_dtype[to=fp16]"].side_conditions(BF16_ACC)
    assert viol and any("fp32" in v for v in viol)
    # ... and is admissible (no violations) on the fp32-acc GEMM
    assert not props["disc:downcast_dtype[to=fp16]"].side_conditions(GEMM)


def test_conservative_eps_inherits_base_cost_and_floors_it():
    props = _by_name(_all_proposals())
    # a BLOCK_K re-tile inherits the base's larger reduction ε (0.06), not the floor
    assert abs(props["disc:retile_block[block_k=128]"].epsilon() - 0.06) < 1e-9
    # a num_stages sweep (exact base) is floored to the conservative discover ε
    assert props["disc:set_num_stages[value=5]"].epsilon() == _DISCOVER_EPS_FLOOR
    # split_k sweep inherits its (large) ways-dependent ε
    assert props["disc:split_k[value=16]"].epsilon() >= 0.2


# --------------------------------------------------------------------------- #
# Source-relevance filter
# --------------------------------------------------------------------------- #
def test_source_filter_prunes_irrelevant_proposals():
    on_gemm = {t.name for t in discover_transforms(LIBRARY, source=GEMM)}
    on_elt = {t.name for t in discover_transforms(LIBRARY, source=ELEMENTWISE)}
    # a K-split / block re-tile is relevant to the GEMM, not the elementwise kernel
    assert "disc:split_k[value=8]" in on_gemm
    assert "disc:split_k[value=8]" not in on_elt
    assert any(n.startswith("disc:retile_block") for n in on_gemm)
    assert not any(n.startswith("disc:retile_block") for n in on_elt)
    # num_warps is present in both (both kernels launch with num_warps=4)
    assert "disc:set_num_warps[value=16]" in on_gemm
    assert "disc:set_num_warps[value=16]" in on_elt


def test_source_free_discovery_keeps_all_proposals():
    # Without a source, proposals are returned regardless of applicability (they
    # remain no-op-safe on non-matching kernels).
    assert len(_all_proposals()) > len(discover_transforms(LIBRARY, source=ELEMENTWISE))


# --------------------------------------------------------------------------- #
# Registry-merge helper
# --------------------------------------------------------------------------- #
def test_merge_appends_and_preserves_base_order():
    props = _all_proposals()
    merged = merge_transforms(LIBRARY, props)
    assert merged[:len(LIBRARY)] == list(LIBRARY)      # base first, order intact
    assert len(merged) == len(LIBRARY) + len(props)
    assert {t.name for t in props} <= {t.name for t in merged}


def test_merge_dedupes_base_wins_unless_override():
    # A proposal that collides with a base name is dropped (base wins) ...
    base0 = LIBRARY[0]
    from kore.transform.calculus import Transformation
    clash = Transformation(name=base0.name, relation=RELATION_APPROX,
                           knob="x", summary="clash", apply_fn=lambda s, **_: None)
    merged = merge_transforms(LIBRARY, [clash])
    assert len(merged) == len(LIBRARY)
    assert merged[0] is base0                          # original kept
    # ... unless override=True, which replaces in place (position preserved)
    merged2 = merge_transforms(LIBRARY, [clash], override=True)
    assert len(merged2) == len(LIBRARY)
    assert merged2[0] is clash


def test_merge_does_not_mutate_inputs():
    base = list(LIBRARY)
    props = _all_proposals()
    n_base, n_props = len(base), len(props)
    _ = merge_transforms(base, props)
    assert len(base) == n_base and len(props) == n_props
    assert list(LIBRARY) == base


# --------------------------------------------------------------------------- #
# extend_library convenience + end-to-end opt-in through the calculus
# --------------------------------------------------------------------------- #
def test_extend_library_defaults_to_curated_library():
    ext = extend_library()
    assert ext[:len(LIBRARY)] == list(LIBRARY)
    assert len(ext) > len(LIBRARY)


def test_discovered_actions_are_opt_in_and_applicable_end_to_end():
    ext = extend_library(source=GEMM)
    budget = ErrorBudget.for_op("gemm", "bf16")
    actions = admissible_actions(GEMM, budget, library=ext)
    disc = [a for a in actions if a.name.startswith(DISCOVERED_PREFIX)]
    assert disc, "expected discovered actions once the caller opts in"
    assert all(a.relation == RELATION_APPROX for a in disc)
    # every discovered action really applies through the SAME merged library
    for a in disc[:8]:
        new, applied, rejected, _ = apply_sequence(
            GEMM, [a.as_step()], ErrorBudget.for_op("gemm", "bf16"), library=ext)
        assert applied and not rejected and new and new != GEMM


def test_discovered_proposal_is_noop_safe_through_apply_sequence():
    # A discovered transform applied to a kernel it does not match is REJECTED as
    # inapplicable (source + budget untouched) - never an out-of-contract rewrite.
    ext = extend_library()  # no source filter -> includes GEMM-only proposals
    budget = ErrorBudget(total=0.5)
    new, applied, rejected, state = apply_sequence(
        ELEMENTWISE, [("disc:split_k[value=8]", {})], budget, library=ext)
    assert new == ELEMENTWISE and not applied
    assert rejected and rejected[0]["reason"] == "inapplicable"
    assert state["spent"] == 0.0


# --------------------------------------------------------------------------- #
# ε-budget composition for discovered moves
# --------------------------------------------------------------------------- #
def test_budget_composes_exact_base_with_discovered_approx():
    ext = extend_library()
    budget = ErrorBudget(total=0.5)
    new, applied, rejected, state = apply_sequence(
        GEMM,
        [("set_num_stages", {"value": 4}),              # exact base -> free
         ("disc:retile_block[block_k=128]", {})],       # discovered approx ε=0.06
        budget, library=ext)
    assert [a["name"] for a in applied] == [
        "set_num_stages", "disc:retile_block[block_k=128]"]
    assert not rejected
    assert state["relation"] == RELATION_APPROX          # exact ⊔ approx = approx
    assert abs(state["weakest_eps"] - 0.06) < 1e-9       # carried = max step ε
    assert abs(state["cumulative_eps"] - 0.06) < 1e-9    # exact spent nothing


def test_budget_prunes_expensive_discovered_but_keeps_cheap():
    ext = extend_library(source=GEMM)
    tiny = ErrorBudget(total=_DISCOVER_EPS_FLOOR)         # 0.01
    names = {a.name for a in admissible_actions(GEMM, tiny, library=ext)}
    # a floor-ε (0.01) num_stages sweep is affordable ...
    assert "disc:set_num_stages[value=5]" in names
    # ... but a 0.02+ block re-tile and 0.06 K-retile are pruned at 0.01
    assert "disc:retile_block[block_m=128]" not in names
    assert "disc:retile_block[block_k=128]" not in names
    assert "disc:split_k[value=16]" not in names


# --------------------------------------------------------------------------- #
# Fusion proposal
# --------------------------------------------------------------------------- #
def test_fusion_fires_on_adjacent_elementwise_and_eliminates_temp():
    fuse = propose_fusions()[0]
    frag = "    a = x + 1.0\n    b = a * 2.0\n    tl.store(p, b, mask=m)\n"
    out = fuse.apply(frag)
    assert out is not None
    assert "b = (x + 1.0) * 2.0" in out
    assert "a = x + 1.0" not in out                       # the temp was inlined away


def test_fusion_is_noop_when_temp_is_read_later():
    fuse = propose_fusions()[0]
    # `a` is read on the next line AND a later line -> unsafe to fold -> no-op
    frag = "    a = x + 1.0\n    b = a * 2.0\n    c = a + 3.0\n    tl.store(p, c)\n"
    assert fuse.apply(frag) is None


def test_fusion_never_inlines_a_load_or_reduction():
    fuse = propose_fusions()[0]
    assert fuse.apply("    a = tl.load(p)\n    b = a + 1.0\n") is None
    assert fuse.apply("    a = tl.sum(v)\n    b = a + 1.0\n") is None
    # and a plain non-fusable source is a no-op
    assert fuse.apply("x = 1\n") is None


# --------------------------------------------------------------------------- #
# Vectorization-width proposal
# --------------------------------------------------------------------------- #
def test_vectorize_width_annotates_an_unannotated_arange():
    props = _by_name(propose_vectorize_widths((8,)))
    t = props["disc:vectorize_width[8]"]
    src = "    offs = tl.arange(0, BLOCK)\n"
    out = t.apply(src)
    assert out is not None
    assert "tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK), 8), 8)" in out
    # no-op when there is no arange to annotate, and never raises
    assert t.apply("x = 1\n") is None


# --------------------------------------------------------------------------- #
# Introspection helpers
# --------------------------------------------------------------------------- #
def test_describe_proposals_and_is_discovered():
    merged = merge_transforms(LIBRARY, _all_proposals())
    rows = describe_proposals(merged)
    assert len(rows) == len(_all_proposals())
    for row in rows:
        assert set(row) == {"name", "relation", "knob", "summary"}
        assert row["name"].startswith(DISCOVERED_PREFIX)
        assert row["relation"] == RELATION_APPROX
    # is_discovered cleanly separates curated from proposed
    assert not any(is_discovered(t) for t in LIBRARY)


def test_strategy_entry_points_are_disjoint_and_typed():
    sweeps = propose_knob_sweeps(LIBRARY)
    fusions = propose_fusions()
    widths = propose_vectorize_widths()
    assert sweeps and fusions and widths
    all_names = [t.name for t in (sweeps + fusions + widths)]
    assert len(all_names) == len(set(all_names))          # no cross-strategy dupes
    for t in sweeps + fusions + widths:
        assert t.is_approx() and t.name.startswith(DISCOVERED_PREFIX)
