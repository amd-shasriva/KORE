"""CPU-only tests for the safe, throttled GRPO-loop adversarial hook.

:mod:`kore.verify.adversarial_hook` is the one-call bridge the orchestrator invokes
every N steps on a batch of candidate kernels. It runs a BOUNDED coevolution round
(:mod:`kore.verify.adversarial`), screens the discovered breaks for monotone-safety, and
accumulates a per-``(op, dtype)`` strengthened ``adversarial_inputs_fn`` in a
process-global registry that ``verify_equivalence(adversarial_inputs_fn=...)`` consumes.

The proven claims asserted below:
  * gating: OFF by default, driven by ``KORE_ADVERSARIAL_COEVOLVE`` (disabled => no-op,
    and :func:`get_adversarial_inputs_fn` returns ``None`` == byte-identical stock).
  * throttling: at most one real round per ``budget.every`` calls/steps.
  * effectiveness (heuristic): given a buggy candidate wrong only on a NON-enumerated
    thin slice + a correct reference, the hook discovers the break within the small
    default budget and accumulates it.
  * FMSP find-then-patch: the accumulated fn makes ``verify_equivalence`` REJECT the
    buggy kernel while still ACCEPTING a correct kernel (exact, and within-tolerance).
  * monotone-safety: a non-reproducible reference case is screened out (never folded),
    and folding never rejects a within-tolerance kernel.
  * fail-safe: an internal error / a raising injected runner degrades to a no-op that
    leaves the registry unchanged; the hook never raises.
  * bounded: cost is capped by ``rounds x population``, and the wall-clock deadline
    winds the search down (no real runner calls once the deadline is hit).

Everything is pure numpy CPU (no GPU/Triton/torch/network), mirroring the buggy/reference
fixtures from ``test_adversarial_coevolution.py``.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from kore.verify.adversarial import TestCase
from kore.verify.adversarial_hook import (
    AdversarialHook,
    AdversarialRegistry,
    HookBudget,
    HookReport,
    default_hook,
    default_registry,
    enabled_from_env,
    get_adversarial_inputs_fn,
    maybe_coevolve,
    registry_key,
    registry_stats,
    reset_registry,
)
from kore.verify.equivalence import tolerance_for, verify_equivalence

# --------------------------------------------------------------------------- #
# Fixtures (pure numpy CPU) - a defect wrong ONLY on a thin, non-enumerated slice
# --------------------------------------------------------------------------- #
DT = "fp64"                     # fp64: tiny gaps exactly representable on CPU
SHAPE = (12, 24)
BUG_DELTA = 1e-2                # near-tie gap the buggy argmax mishandles (fixed battery
                                # and randn both miss it; a small bounded round finds it)


def ref_argmax(x):
    """Reference: index of the row max (first occurrence on ties), as float."""
    x = np.asarray(x, dtype=np.float64)
    return np.argmax(x, axis=1).astype(np.float64)


def make_near_tie_bug(delta=BUG_DELTA):
    """argmax that returns the WRONG (near-tie partner) index when a row's top-two
    values are within ``delta`` - a thin-slice defect invisible to random/fixed inputs."""

    def buggy(x):
        x = np.asarray(x, dtype=np.float64)
        m, _ = x.shape
        out = np.empty(m, dtype=np.float64)
        for i in range(m):
            row = x[i]
            order = np.argsort(row, kind="stable")
            top, second = order[-1], order[-2]
            gap = row[top] - row[second]
            out[i] = float(second) if (0.0 < gap <= delta) else float(np.argmax(row))
        return out

    return buggy


def make_input_gen(shape=SHAPE):
    def input_gen(_shape, _dtype, seed, _device):
        rng = np.random.default_rng(seed)
        return (rng.standard_normal(shape).astype(np.float64),)

    return input_gen


BUG = make_near_tie_bug()
VERIFY_KW = dict(dtype=DT, shape=SHAPE, op_class="generic", metamorphic=False,
                 device="cpu", n_random=16)


def _fresh_hook(**budget_kw):
    """An enabled hook over an isolated registry (no cross-test contamination)."""
    reg = AdversarialRegistry()
    return AdversarialHook(budget=HookBudget(**budget_kw), registry=reg, enabled=True), reg


def _accumulate(hook, reg, *, reference_fn=ref_argmax, candidate_fns=(BUG,), **run_kw):
    return hook.run(op="argmax", dtype=DT, reference_fn=reference_fn,
                    candidate_fns=list(candidate_fns), shape=SHAPE, op_class="generic",
                    arity=1, force=True, **run_kw)


# --------------------------------------------------------------------------- #
# 1. Gating: OFF by default, driven by KORE_ADVERSARIAL_COEVOLVE
# --------------------------------------------------------------------------- #
def test_enabled_from_env_off_by_default(monkeypatch):
    monkeypatch.delenv("KORE_ADVERSARIAL_COEVOLVE", raising=False)
    assert enabled_from_env() is False


@pytest.mark.parametrize("val,expect", [
    ("1", True), ("true", True), ("TRUE", True), ("yes", True), ("on", True),
    ("0", False), ("", False), ("no", False), ("off", False),
])
def test_enabled_from_env_reads_lever(monkeypatch, val, expect):
    monkeypatch.setenv("KORE_ADVERSARIAL_COEVOLVE", val)
    assert enabled_from_env() is expect


def test_disabled_hook_is_a_noop():
    reg = AdversarialRegistry()
    hook = AdversarialHook(registry=reg, enabled=False)
    rep = _accumulate(hook, reg)
    assert rep.ran is False and rep.reason == "disabled"
    assert reg.get("argmax", DT) == []                       # oracle untouched
    # disabled consumption is byte-identical to stock (None == fixed battery)
    assert get_adversarial_inputs_fn("argmax", DT, enabled=False, registry=reg) is None


def test_disabled_via_env_even_with_accumulated_cases(monkeypatch):
    # Accumulate some cases into a registry, then prove the env-gate hides them.
    reg = AdversarialRegistry()
    reg.add("argmax", DT, [TestCase("near_tie", {"log_gap": -9.0, "base": 1.0})])
    monkeypatch.setenv("KORE_ADVERSARIAL_COEVOLVE", "0")
    assert get_adversarial_inputs_fn("argmax", DT, registry=reg) is None
    monkeypatch.setenv("KORE_ADVERSARIAL_COEVOLVE", "1")
    assert callable(get_adversarial_inputs_fn("argmax", DT, registry=reg))


# --------------------------------------------------------------------------- #
# 2. Throttling: one real round per budget.every calls/steps
# --------------------------------------------------------------------------- #
def test_step_based_throttle_runs_only_on_multiples():
    hook, reg = _fresh_hook(every=10)
    # non-multiples are throttled no-ops; multiples run.
    for step in (1, 5, 9, 11, 19):
        rep = hook.run(op="argmax", dtype=DT, reference_fn=ref_argmax,
                       candidate_fns=[BUG], shape=SHAPE, op_class="generic",
                       arity=1, step=step)
        assert rep.ran is False and rep.reason == "throttled"
    rep = hook.run(op="argmax", dtype=DT, reference_fn=ref_argmax, candidate_fns=[BUG],
                   shape=SHAPE, op_class="generic", arity=1, step=20)
    assert rep.ran is True and rep.broke_any is True


def test_call_counter_throttle_without_step():
    hook, reg = _fresh_hook(every=3)
    reasons = []
    for _ in range(6):
        rep = hook.run(op="argmax", dtype=DT, reference_fn=ref_argmax,
                       candidate_fns=[BUG], shape=SHAPE, op_class="generic", arity=1)
        reasons.append(rep.ran)
    # every=3 => the 3rd and 6th calls run, the rest are throttled.
    assert reasons == [False, False, True, False, False, True]


def test_should_run_is_advisory_and_gated():
    hook, _ = _fresh_hook(every=5)
    assert hook.should_run(step=5) is True
    assert hook.should_run(step=4) is False
    disabled = AdversarialHook(enabled=False)
    assert disabled.should_run(step=5) is False
    assert disabled.should_run(force=True) is False          # disabled beats force


# --------------------------------------------------------------------------- #
# 3. Effectiveness: discover the thin-slice break within the small default budget
# --------------------------------------------------------------------------- #
def test_hook_discovers_break_and_accumulates_default_budget():
    hook, reg = _fresh_hook()                                # DEFAULT budget (rounds=6,pop=24)
    rep = _accumulate(hook, reg)
    assert rep.ran is True and rep.broke_any is True
    assert rep.n_added >= 1 and rep.battery_size == rep.n_added
    assert rep.n_evaluations <= HookBudget().rounds * HookBudget().population_size
    assert reg.get("argmax", DT)                             # battery grew
    assert callable(get_adversarial_inputs_fn("argmax", DT, enabled=True, registry=reg))


def test_accumulation_is_additive_across_invocations():
    hook, reg = _fresh_hook()
    r1 = _accumulate(hook, reg)
    size1 = r1.battery_size
    r2 = _accumulate(hook, reg)                              # same defect again
    # dedup by signature => battery never shrinks; only grows or stays put.
    assert r2.battery_size >= size1


# --------------------------------------------------------------------------- #
# 4. FMSP find-then-patch: accumulated fn REJECTS buggy, ACCEPTS correct
# --------------------------------------------------------------------------- #
def test_stock_adversarial_prong_misses_but_folded_prong_rejects_with_certainty():
    # Isolate the PROVABLE (deterministic) adversarial prong with n_random=0: this is
    # the sound half of the oracle and does not depend on random-draw luck. The fixed
    # battery has no near-tie regime, so it MISSES the defect; folding the discovered
    # near-tie case in makes the deterministic prong reject it with certainty. (With a
    # slice this wide the *random* prong may also catch it - a thinner slice needs deep
    # escalation, honestly out of scope for one bounded round; the DETERMINISTIC fold is
    # what the hook makes certain.)
    hook, reg = _fresh_hook()
    _accumulate(hook, reg)
    kw = dict(dtype=DT, shape=SHAPE, op_class="generic", metamorphic=False,
              device="cpu", n_random=0)
    stock = verify_equivalence(BUG, ref_argmax, make_input_gen(), **kw)
    assert stock.verified is True                            # stock deterministic prongs miss it
    assert stock.prong("adversarial").passed is True
    # WITH the folded regime, the adversarial prong fires on a folded case and rejects.
    adv = get_adversarial_inputs_fn("argmax", DT, enabled=True, registry=reg)
    strong = verify_equivalence(BUG, ref_argmax, make_input_gen(),
                                adversarial_inputs_fn=adv, **kw)
    assert strong.verified is False
    prong = strong.prong("adversarial")
    assert prong is not None and prong.passed is False
    assert "folded::" in prong.detail


def test_folded_oracle_accepts_correct_kernel_exact_and_within_tol():
    hook, reg = _fresh_hook()
    _accumulate(hook, reg)
    adv = get_adversarial_inputs_fn("argmax", DT, enabled=True, registry=reg)
    # exact reference-vs-reference: the strengthened battery is sound.
    good = verify_equivalence(ref_argmax, ref_argmax, make_input_gen(),
                              adversarial_inputs_fn=adv, **VERIFY_KW)
    assert good.verified is True
    # a genuinely within-tolerance kernel (<= rtol, >= snr floor) is still accepted -
    # folding ADDS inputs, it never tightens the bound (monotone-safety in action).
    tol = tolerance_for(DT)
    assert 1e-3 <= tol.rtol
    noisy = lambda x: ref_argmax(x) * (1.0 + 1e-3)
    noisy_res = verify_equivalence(noisy, ref_argmax, make_input_gen(),
                                   adversarial_inputs_fn=adv, **VERIFY_KW)
    assert noisy_res.verified is True


# --------------------------------------------------------------------------- #
# 5. Monotone-safety screen: a non-reproducible reference is never folded
# --------------------------------------------------------------------------- #
def test_nondeterministic_reference_is_screened_out():
    def flaky_ref(x):
        x = np.asarray(x, dtype=np.float64)
        base = np.argmax(x, axis=1).astype(np.float64)
        return base + np.random.default_rng().standard_normal(x.shape[0])   # unseeded!

    hook, reg = _fresh_hook()
    rep = _accumulate(hook, reg, reference_fn=flaky_ref)
    # coevolution may "find" breaks against a noisy reference, but the safety screen
    # drops every one (reference not reproducible) => nothing is added to the oracle.
    assert rep.n_added == 0 and rep.battery_size == 0
    assert rep.n_screened_out == rep.n_folded
    assert get_adversarial_inputs_fn("argmax", DT, enabled=True, registry=reg) is None


# --------------------------------------------------------------------------- #
# 6. Fail-safe: any error => no-op that leaves the registry (oracle) unchanged
# --------------------------------------------------------------------------- #
def test_fail_safe_on_internal_error_leaves_registry_unchanged():
    # an invalid family makes coevolve_tests raise internally; the hook must swallow it.
    hook, reg = _fresh_hook(families=("does_not_exist",))
    rep = _accumulate(hook, reg)
    assert rep.ran is False and rep.reason == "error" and rep.error
    assert reg.get("argmax", DT) == []                       # untouched


def test_fail_safe_on_raising_injected_runner():
    def boom(fn, inputs):
        raise RuntimeError("gpu exploded")

    hook, reg = _fresh_hook()
    rep = _accumulate(hook, reg, run_reference=boom)
    assert rep.ran is False and rep.reason == "error"
    assert "gpu exploded" in (rep.error or "")
    assert reg.get("argmax", DT) == []


def test_no_candidates_or_no_reference_is_a_noop():
    hook, reg = _fresh_hook()
    assert hook.run(op="argmax", dtype=DT, reference_fn=ref_argmax, candidate_fns=[],
                    force=True).reason == "no-candidates"
    assert hook.run(op="argmax", dtype=DT, reference_fn=None, candidate_fns=[BUG],
                    force=True).reason == "no-reference"
    assert reg.get("argmax", DT) == []


# --------------------------------------------------------------------------- #
# 7. Bounded cost: capped by rounds x population, wound down by the deadline
# --------------------------------------------------------------------------- #
def test_bounded_cost_capped_by_rounds_and_population():
    calls = {"cand": 0, "ref": 0}

    def rc(fn, inputs):
        calls["cand"] += 1
        try:
            return fn(*inputs)
        except Exception as exc:      # noqa: BLE001
            return exc

    def rr(fn, inputs):
        calls["ref"] += 1
        return fn(*inputs)

    b = HookBudget(rounds=4, population_size=12, max_candidates=2, max_seconds=60.0)
    hook = AdversarialHook(budget=b, registry=AdversarialRegistry(), enabled=True)
    hook.run(op="argmax", dtype=DT, reference_fn=ref_argmax,
             candidate_fns=[BUG, make_near_tie_bug(1e-3)], shape=SHAPE,
             op_class="generic", arity=1, run_candidate=rc, run_reference=rr, force=True)
    genome_evals = b.rounds * b.population_size
    assert calls["ref"] <= genome_evals + 2 * b.fold_max_cases + 4   # + bounded screen
    assert calls["cand"] <= genome_evals * b.max_candidates


def test_deadline_zero_winds_down_to_no_real_work():
    calls = {"cand": 0, "ref": 0}

    def rc(fn, inputs):
        calls["cand"] += 1
        return fn(*inputs)

    def rr(fn, inputs):
        calls["ref"] += 1
        return fn(*inputs)

    hook = AdversarialHook(budget=HookBudget(max_seconds=0.0),
                           registry=AdversarialRegistry(), enabled=True)
    rep = hook.run(op="argmax", dtype=DT, reference_fn=ref_argmax, candidate_fns=[BUG],
                   shape=SHAPE, op_class="generic", arity=1,
                   run_candidate=rc, run_reference=rr, force=True)
    # immediate wind-down: NO real runner calls happen, and no (spurious) break is found.
    assert calls["cand"] == 0 and calls["ref"] == 0
    assert rep.ran is True and rep.broke_any is False and rep.n_added == 0


# --------------------------------------------------------------------------- #
# 8. Registry: keyed by (op, dtype), snapshot/stats/reset
# --------------------------------------------------------------------------- #
def test_registry_is_keyed_by_op_and_dtype():
    reg = AdversarialRegistry()
    reg.add("argmax", DT, [TestCase("near_tie", {"log_gap": -9.0, "base": 1.0})])
    assert reg.get("argmax", DT)                             # present ...
    assert reg.get("argmax", "fp32") == []                   # ... different dtype: empty
    assert reg.get("sin", DT) == []                          # ... different op: empty
    assert reg.inputs_fn("sin", DT) is None
    assert registry_key("ArgMax", "FP64") == ("argmax", "fp64")   # normalised


def test_registry_add_dedups_and_caps():
    reg = AdversarialRegistry()
    tc = TestCase("near_tie", {"log_gap": -9.0, "base": 1.0})
    assert reg.add("op", DT, [tc]) == 1
    assert reg.add("op", DT, [tc]) == 0                      # dedup by signature
    many = [TestCase("near_tie", {"log_gap": -float(i), "base": 1.0}) for i in range(1, 20)]
    reg.add("op", DT, many, max_battery=5)
    assert len(reg.get("op", DT)) <= 5                       # capped


def test_registry_stats_and_reset():
    reg = AdversarialRegistry()
    reg.add("argmax", DT, [TestCase("near_tie", {"log_gap": -9.0, "base": 1.0})])
    assert reg.stats().get("argmax::fp64") == 1
    reg.clear()
    assert reg.stats() == {}


# --------------------------------------------------------------------------- #
# 9. Process-global one-call path + observability
# --------------------------------------------------------------------------- #
def test_maybe_coevolve_uses_process_global_registry(monkeypatch):
    monkeypatch.setenv("KORE_ADVERSARIAL_COEVOLVE", "1")
    reset_registry()
    try:
        rep = maybe_coevolve(op="argmax", dtype=DT, reference_fn=ref_argmax,
                             candidate_fns=[BUG], shape=SHAPE, op_class="generic",
                             arity=1, force=True)
        assert rep.ran is True and rep.n_added >= 1
        # the env-side consumption reads it back from the same global registry.
        adv = get_adversarial_inputs_fn("argmax", DT)
        assert callable(adv)
        assert registry_stats().get("argmax::fp64", 0) >= 1
    finally:
        reset_registry()


def test_default_hook_is_a_singleton():
    assert default_hook() is default_hook()
    assert default_hook().registry is default_registry()


def test_hook_report_summary_is_a_safe_string():
    hook, reg = _fresh_hook()
    rep = _accumulate(hook, reg)
    assert isinstance(rep, HookReport)
    assert isinstance(rep.summary(), str) and rep.summary()
    assert "adv-hook" in rep.summary()


# --------------------------------------------------------------------------- #
# 10. Purity: importing + running the hook never pulls in torch / touches a GPU
# --------------------------------------------------------------------------- #
def test_pure_cpu_no_torch_import():
    code = (
        "import sys, numpy as np\n"
        "from kore.verify.adversarial_hook import AdversarialHook, AdversarialRegistry, HookBudget\n"
        "ref = lambda x: np.asarray(x, dtype=np.float64) * 2.0\n"
        "h = AdversarialHook(budget=HookBudget(rounds=2, population_size=8),"
        " registry=AdversarialRegistry(), enabled=True)\n"
        "h.run(op='dbl', dtype='fp64', reference_fn=ref, candidate_fns=[ref],"
        " shape=(4, 8), op_class='elementwise', arity=1, force=True)\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
        "print('NO_TORCH_OK')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "NO_TORCH_OK" in out.stdout
