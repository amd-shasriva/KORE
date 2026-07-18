"""CPU-only tests for the COEVOLUTIONARY adversarial test-case generator.

The fixed adversarial battery (``adversarial_patterns`` / ``adversarial_inputs``) can
only reject a kernel wrong on a regime someone thought to write down. This suite
exercises the *minimal-criterion coevolution* layer that GROWS the battery: a population
of parametric test-case genomes that mutate to BREAK kernels which currently pass, so
the correctness bar escalates automatically and the discovered breaks are folded back
into the deterministic oracle (the FMSP "find-then-patch" loop).

Everything is pure CPU / numpy (no GPU, no Triton, no network). We use deliberately
buggy candidates that are wrong ONLY on a thin slice NOT covered by the fixed battery:

  * ``near_tie`` argmax: correct everywhere except when a row's top-two values are
    within a tiny gap ``<= delta`` (then it returns the wrong index -> a HUGE error).
    Random ``randn`` inputs have gaps of O(1); the fixed battery's exact-tie / ramp /
    knot patterns never produce a *near* tie -> both miss it.
  * ``kink_neighborhood`` sin: correct except within ``delta`` of a NON-enumerated
    location (1.5 / 2.5 / 4.0) -> a ~1.0 error the ramp steps over and randn misses.

The proven claims (asserted below): coevolution FINDS these breaks while equal-budget
random sampling does not; difficulty ESCALATES round over round; and folding the
discovered case in makes ``verify_equivalence`` REJECT the buggy kernel it previously
accepted - WITHOUT falsely rejecting a correct kernel, and with the new hook OFF by
default byte-identical to the shipped behaviour.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from kore.verify.adversarial import (
    CoevolutionResult,
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
from kore.verify.equivalence import (
    ProngSamples,
    equivalence_verdict,
    tolerance_for,
    verify_equivalence,
)

# --------------------------------------------------------------------------- #
# Config + buggy/reference fixtures (pure numpy CPU)
# --------------------------------------------------------------------------- #
DT = "fp64"                 # fp64 so tiny gaps/bands are exactly representable on CPU
SHAPE = (12, 24)
NEAR_TIE_DELTA = 1e-9       # the buggy argmax is wrong only for 0 < gap <= this
KINK_DELTA = 1e-9
KINK_TARGETS = (1.5, 2.5, 4.0)      # NON-stock kink locations (not in the fixed battery)


def ref_argmax(x):
    """Reference: index of the row max (first occurrence on ties), as float."""
    x = np.asarray(x, dtype=np.float64)
    return np.argmax(x, axis=1).astype(np.float64)


def make_near_tie_argmax_bug(delta=NEAR_TIE_DELTA):
    """A kernel identical to ``ref_argmax`` EXCEPT it returns the wrong (near-tie
    partner) index when a row's top-two values are within ``delta`` - a thin-slice
    defect that produces a large index error but is invisible to random/fixed inputs."""

    def buggy(x):
        x = np.asarray(x, dtype=np.float64)
        m, n = x.shape
        out = np.empty(m, dtype=np.float64)
        for i in range(m):
            row = x[i]
            order = np.argsort(row, kind="stable")
            top, second = order[-1], order[-2]
            gap = row[top] - row[second]
            out[i] = float(second) if (0.0 < gap <= delta) else float(np.argmax(row))
        return out

    return buggy


def ref_sin(x):
    return np.sin(np.asarray(x, dtype=np.float64))


def make_kink_sin_bug(delta=KINK_DELTA, targets=KINK_TARGETS):
    """``sin`` that is off by ~1.0 only within ``delta`` of a non-enumerated location."""

    def buggy(x):
        x = np.asarray(x, dtype=np.float64)
        out = np.sin(x)
        bad = np.zeros(x.shape, dtype=bool)
        for t in targets:
            bad |= np.abs(x - t) <= delta
        return out - bad.astype(np.float64)

    return buggy


def make_input_gen(shape=SHAPE):
    def input_gen(_shape, _dtype, seed, _device):
        rng = np.random.default_rng(seed)
        return (rng.standard_normal(shape).astype(np.float64),)

    return input_gen


NEAR_TIE_BUG = make_near_tie_argmax_bug()


@pytest.fixture(scope="module")
def near_tie_run() -> CoevolutionResult:
    """One shared near-tie coevolution run (deterministic; reused by several tests)."""
    return coevolve_tests(ref_argmax, NEAR_TIE_BUG, shape=SHAPE, dtype=DT,
                          families=["near_tie"], seed=0, rounds=30, population_size=48)


# --------------------------------------------------------------------------- #
# 1. Generator / genome: pure-data, deterministic, evolvable
# --------------------------------------------------------------------------- #
def test_family_registry_lists_expected_regimes():
    fams = set(list_families())
    assert {"constant", "kink_neighborhood", "denormal_sweep", "extreme_magnitude",
            "near_tie", "sparse_spike"} <= fams


def test_near_tie_genome_builds_controlled_gap():
    tc = TestCase("near_tie", {"log_gap": -9.0, "base": 1.0})
    (arr,) = tc.build(SHAPE, DT)
    assert arr.shape == SHAPE and arr.dtype == np.float64
    # column 0 is the max, last column its near-tie partner exactly ~1e-9 below it
    gap = float(arr[0, 0] - arr[0, -1])
    assert abs(gap - 1e-9) < 1e-12
    assert int(np.argmax(arr[0])) == 0            # the max is unique at column 0


def test_build_is_deterministic_and_dtype_cast():
    tc = TestCase("sparse_spike", {"log_density": -1.0, "log_mag": 1.5, "sign": 1.0},
                  benign_seed=7)
    a1 = tc.build(SHAPE, DT)[0]
    a2 = tc.build(SHAPE, DT)[0]
    np.testing.assert_array_equal(a1, a2)         # same genome -> identical arrays
    # fp32 cast produces a float32 numpy array on CPU (no torch needed)
    f32 = TestCase("constant", {"value": 0.5}).build(SHAPE, "fp32")[0]
    assert f32.dtype == np.float32


def test_multi_operand_places_pattern_at_op_index():
    tc = TestCase("constant", {"value": 3.0}, arity=3, op_index=1)
    ops = tc.build((4, 5), DT)
    assert len(ops) == 3 and all(o.shape == (4, 5) for o in ops)
    assert np.allclose(ops[1], 3.0)               # the pattern is in slot 1 ...
    assert not np.allclose(ops[0], 3.0)           # ... others are a benign draw
    assert not np.allclose(ops[2], 3.0)


def test_generate_cases_is_deterministic_and_respects_families():
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    a = generate_cases(20, rng1, families=["near_tie", "kink_neighborhood"], dtype=DT)
    b = generate_cases(20, rng2, families=["near_tie", "kink_neighborhood"], dtype=DT)
    assert [c.signature() for c in a] == [c.signature() for c in b]
    assert all(c.family in {"near_tie", "kink_neighborhood"} for c in a)


def test_unknown_family_rejected():
    with pytest.raises(KeyError):
        generate_cases(1, np.random.default_rng(0), families=["nope"])


def test_difficulty_is_grounded_and_monotone():
    easy = TestCase("near_tie", {"log_gap": -2.0, "base": 1.0})
    hard = TestCase("near_tie", {"log_gap": -12.0, "base": 1.0})
    assert hard.difficulty(DT) > easy.difficulty(DT)
    assert hard.difficulty(DT) <= 14.0 + 1e-9     # bounded (no runaway)


def test_mutation_respects_clamp_and_explores_beyond_prior():
    # The sampling prior floors gaps at 1e-4 (log_gap >= -4); the mutation OPERATOR must
    # be able to explore below that. It is a random walk, so reliable *tightening* comes
    # from SELECTION (see test_difficulty_escalates_round_over_round) - here we only
    # assert the operator's clamp invariant and that it reaches beyond the prior floor.
    rng = np.random.default_rng(0)
    cur = TestCase("near_tie", {"log_gap": -2.0, "base": 1.0})
    logs = []
    for _ in range(150):
        cur = mutate_case(cur, rng, scale=1.0, dtype=DT, families=["near_tie"])
        assert -14.0 - 1e-9 <= cur.params["log_gap"] <= 1e-9      # clamp invariant
        logs.append(cur.params["log_gap"])
    assert min(logs) < -4.0                                        # explores below prior


def test_crossover_within_family_blends_genes():
    a = TestCase("near_tie", {"log_gap": -3.0, "base": 1.0})
    b = TestCase("near_tie", {"log_gap": -9.0, "base": 1.0})
    child = crossover_cases(a, b, np.random.default_rng(1), dtype=DT)
    assert child.family == "near_tie"
    assert child.params["log_gap"] in (-3.0, -9.0)


def test_layout_perturbations_build_without_error():
    tc = TestCase("near_tie", {"log_gap": -9.0, "base": 1.0}, perturbation="reverse_cols")
    (arr,) = tc.build((4, 6), DT)
    assert arr.shape == (4, 6)                     # value-preserving layout variant
    (tr,) = TestCase("constant", {"value": 1.0}, perturbation="transpose").build((4, 6), DT)
    assert tr.shape == (6, 4)


# --------------------------------------------------------------------------- #
# 2. THE CRUX: coevolution finds a break that random sampling misses
# --------------------------------------------------------------------------- #
def test_coevolution_finds_near_tie_break(near_tie_run):
    res = near_tie_run
    assert res.broke_any
    assert res.n_candidates_broken == 1
    best = res.best_case()
    assert best is not None and best.family == "near_tie"
    # the discovered breaking case really is inside the buggy slice (gap <= delta)
    assert 10.0 ** best.params["log_gap"] <= NEAR_TIE_DELTA


def test_random_sampling_misses_near_tie_at_equal_budget():
    budget = 48 * 30                              # same #evaluations as the coevolve run
    natural = random_search(ref_argmax, NEAR_TIE_BUG, shape=SHAPE, dtype=DT, seed=0,
                            n_samples=budget, mode="natural")
    family = random_search(ref_argmax, NEAR_TIE_BUG, shape=SHAPE, dtype=DT, seed=0,
                           n_samples=budget, mode="family", families=["near_tie"])
    # randn inputs (what the shipped SNR gate samples) and undirected family draws
    # both live above the thin defect slice -> neither ever triggers it.
    assert natural.n_breaking == 0
    assert family.n_breaking == 0


def test_difficulty_escalates_round_over_round(near_tie_run):
    res = near_tie_run
    trend = res.difficulty_trend()
    assert len(trend) == 30
    assert trend[-1] > trend[0] + 2.0             # elite hardness climbs substantially
    assert res.rounds[-1].max_difficulty >= res.rounds[0].max_difficulty


def test_break_requires_escalation_not_initial_luck(near_tie_run):
    # the break is absent from the initial random population (round 0) and only appears
    # after several rounds of escalation -> it is NOT something random sampling finds.
    first_break = next((r.round for r in near_tie_run.rounds if r.n_breaking > 0), None)
    assert first_break is not None and first_break >= 1
    assert near_tie_run.rounds[0].n_breaking == 0


# --------------------------------------------------------------------------- #
# 3. FMSP find-then-patch: folding the break makes the oracle reject the kernel
# --------------------------------------------------------------------------- #
def test_stock_oracle_lucky_passes_the_buggy_kernel():
    # WITHOUT the discovered regime, the buggy argmax clears the full oracle (random +
    # fixed adversarial + determinism) - the loophole the coevolution exists to close.
    res = verify_equivalence(NEAR_TIE_BUG, ref_argmax, make_input_gen(), dtype=DT,
                             shape=SHAPE, op_class="generic", metamorphic=False,
                             device="cpu", n_random=24)
    assert res.verified is True


def test_folded_oracle_rejects_the_buggy_kernel(near_tie_run):
    fold = fold_breaking_cases(near_tie_run.breaking_cases, dtype=DT, max_cases=8)
    assert fold.n_folded >= 1
    strong = verify_equivalence(NEAR_TIE_BUG, ref_argmax, make_input_gen(), dtype=DT,
                                shape=SHAPE, op_class="generic", metamorphic=False,
                                device="cpu", n_random=24,
                                adversarial_inputs_fn=fold.adversarial_inputs_fn())
    assert strong.verified is False
    adv = strong.prong("adversarial")
    assert adv is not None and adv.passed is False
    assert "folded::" in adv.detail                # a folded regime is the one that fired


def test_folding_does_not_falsely_reject_a_correct_kernel(near_tie_run):
    fold = fold_breaking_cases(near_tie_run.breaking_cases, dtype=DT, max_cases=8)
    good = verify_equivalence(ref_argmax, ref_argmax, make_input_gen(), dtype=DT,
                              shape=SHAPE, op_class="generic", metamorphic=False,
                              device="cpu", n_random=24,
                              adversarial_inputs_fn=fold.adversarial_inputs_fn())
    assert good.verified is True                   # the strengthened battery is sound


def test_fold_rejection_is_visible_in_pure_verdict(near_tie_run):
    # The decision logic is a PURE function over arrays: build the folded adversarial
    # (candidate, reference) pairs by hand and confirm equivalence_verdict rejects.
    fold = fold_breaking_cases(near_tie_run.breaking_cases, dtype=DT, max_cases=4)
    gen = fold.adversarial_inputs_fn()
    pairs, labels = [], []
    for name, inputs in gen(SHAPE, DT, arity=1, op_class="generic", device="cpu"):
        pairs.append((NEAR_TIE_BUG(*inputs), ref_argmax(*inputs)))
        labels.append(name)
    verdict = equivalence_verdict(
        [ProngSamples("adversarial", "adversarial", pairs, labels=labels)],
        tolerance_for(DT))
    assert verdict.verified is False


# --------------------------------------------------------------------------- #
# 4. ADDITIVE SAFETY: the new hook is OFF by default and byte-identical
# --------------------------------------------------------------------------- #
def test_verify_equivalence_hook_off_by_default_is_identical():
    kw = dict(dtype=DT, shape=SHAPE, op_class="generic", metamorphic=False,
              device="cpu", n_random=12)
    default = verify_equivalence(ref_argmax, ref_argmax, make_input_gen(), **kw)
    explicit_none = verify_equivalence(ref_argmax, ref_argmax, make_input_gen(),
                                       adversarial_inputs_fn=None, **kw)
    explicit_stock = verify_equivalence(ref_argmax, ref_argmax, make_input_gen(),
                                        adversarial_inputs_fn=adversarial_inputs, **kw)
    assert default.summary() == explicit_none.summary() == explicit_stock.summary()
    assert default.verified is True


def test_existing_fixed_battery_is_unchanged():
    # Guard against accidental regression of the pre-existing enumerated API.
    names = [n for n, _ in adversarial_patterns((8, 8), "fp32")]
    for expected in ("zeros", "denormal", "activation_knots", "inf_adjacent_pos",
                     "sparse_spikes", "signed_ramp", "mixed_magnitude"):
        assert expected in names
    cases = [n for n, _ in adversarial_inputs((8, 8), "fp32", arity=1)]
    assert "all::zeros" in cases and "all::activation_knots" in cases
    assert dtype_max("fp16") == float(np.finfo(np.float16).max)
    assert dtype_extremes("fp32")[0] == 1.0e18


def test_strengthened_generator_prepends_fixed_battery():
    tc = TestCase("near_tie", {"log_gap": -9.0, "base": 1.0})
    gen = make_strengthened_inputs([tc], include_base=True)
    emitted = [name for name, _ in gen(SHAPE, DT, arity=1, op_class="generic")]
    assert any(n.startswith("all::") for n in emitted)      # fixed battery kept ...
    assert any(n.startswith("folded::") for n in emitted)   # ... plus the folded case


# --------------------------------------------------------------------------- #
# 5. GENERALITY: a different family / defect (activation-kink neighbourhood)
# --------------------------------------------------------------------------- #
def test_coevolution_finds_kink_neighbourhood_break():
    bug = make_kink_sin_bug()
    res = coevolve_tests(ref_sin, bug, shape=(8, 16), dtype=DT,
                         families=["kink_neighborhood"], seed=1, rounds=40,
                         population_size=64)
    assert res.broke_any
    best = res.best_case()
    assert best is not None and best.family == "kink_neighborhood"
    assert abs(best.params["loc"] - min(KINK_TARGETS, key=lambda t: abs(t - best.params["loc"]))) < 1e-9

    budget = 64 * 40
    natural = random_search(ref_sin, bug, shape=(8, 16), dtype=DT, seed=1,
                            n_samples=budget, mode="natural")
    assert natural.n_breaking == 0                 # randn never lands in the thin band

    fold = fold_breaking_cases(res.breaking_cases, dtype=DT, max_cases=8)
    strong = verify_equivalence(bug, ref_sin, make_input_gen((8, 16)), dtype=DT,
                                shape=(8, 16), op_class="elementwise", device="cpu",
                                n_random=24, adversarial_inputs_fn=fold.adversarial_inputs_fn())
    assert strong.verified is False


# --------------------------------------------------------------------------- #
# 6. Multi-candidate escalation + injectability + fail-safe
# --------------------------------------------------------------------------- #
def test_escalation_across_candidate_set():
    # two candidates: one broken by an easy (shallow) gap, one only by a very tight gap.
    easy = make_near_tie_argmax_bug(delta=1e-3)
    hard = make_near_tie_argmax_bug(delta=1e-11)
    res = coevolve_tests(ref_argmax, [easy, hard], shape=(8, 16), dtype=DT,
                         families=["near_tie"], seed=3, rounds=45, population_size=48)
    assert res.n_candidates_broken == 2
    # there is a phase where only the easy candidate is broken -> the bar escalated to
    # reach the harder one (open-ended: keep breaking the still-passing candidate).
    assert any(r.n_candidates_broken == 1 for r in res.rounds)


def test_injected_runner_is_used_and_crash_is_a_break():
    calls = {"n": 0}

    def counting_runner(fn, inputs):
        calls["n"] += 1
        try:
            return fn(*inputs)
        except Exception as exc:      # noqa: BLE001
            return exc

    def crash_on_extremes(x):
        x = np.asarray(x, dtype=np.float64)
        if np.any(np.abs(x) > 1e17):
            raise FloatingPointError("boom")
        return x * 2.0

    res = coevolve_tests(lambda x: np.asarray(x, dtype=np.float64) * 2.0,
                         crash_on_extremes, shape=(8, 8), dtype=DT,
                         families=["extreme_magnitude"], seed=0, rounds=20,
                         population_size=32, run_candidate=counting_runner)
    assert calls["n"] > 0                          # the injected runner was used ...
    assert res.broke_any                           # ... and a raised candidate = a break


def test_reference_that_raises_fails_minimal_criterion_safely():
    def flaky_ref(x):
        raise RuntimeError("reference undefined here")

    # A reference that always raises makes every case inadmissible: no crash, no breaks.
    res = coevolve_tests(flaky_ref, lambda x: np.asarray(x) * 1.0, shape=(4, 8),
                         dtype=DT, families=["constant"], seed=0, rounds=5,
                         population_size=16)
    assert res.broke_any is False
    assert all(r.n_valid == 0 for r in res.rounds)


def test_determinism_same_seed_same_archive():
    a = coevolve_tests(ref_argmax, NEAR_TIE_BUG, shape=(8, 16), dtype=DT,
                       families=["near_tie"], seed=5, rounds=15, population_size=32)
    b = coevolve_tests(ref_argmax, NEAR_TIE_BUG, shape=(8, 16), dtype=DT,
                       families=["near_tie"], seed=5, rounds=15, population_size=32)
    assert [c.signature() for c in a.breaking_cases] == [c.signature() for c in b.breaking_cases]
    assert a.difficulty_trend() == b.difficulty_trend()


def test_pure_cpu_no_torch_import():
    # Importing the module and running a full coevolution must not pull in torch (proves
    # the search is pure CPU and never touches the GPU/env itself).
    code = (
        "import sys, numpy as np\n"
        "from kore.verify.adversarial import coevolve_tests\n"
        "ref = lambda x: np.asarray(x, dtype=np.float64) * 2.0\n"
        "coevolve_tests(ref, ref, shape=(4, 8), dtype='fp64', families=['near_tie'],"
        " seed=0, rounds=3, population_size=8)\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
        "print('NO_TORCH_OK')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "NO_TORCH_OK" in out.stdout


def test_result_summary_and_describe_are_safe(near_tie_run):
    assert isinstance(near_tie_run.summary(), str) and near_tie_run.summary()
    best = near_tie_run.best_case()
    assert isinstance(best.describe(), str) and "near_tie" in best.describe()
