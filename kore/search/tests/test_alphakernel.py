"""CPU-only tests for AlphaKernel value-guided test-time search.

No GPU, no torch/vllm. Mirroring ``tests/test_agent.py``: a ``FakeEnv`` returns
scripted ``Observation``s keyed by markers in the kernel source, a ``FakePolicy``
yields scripted edits per parent, and a ``FakeValueModel`` scores sources. The
verifier is used as a perfect simulator.

Covers the AlphaKernel contract:
  * fingerprint / canonicalization (semantic transposition key)
  * Budget hard cap + MeasureStats LCB + Successive-Halving (bandit)
  * MAX-backup returns the best CORRECT leaf (even when it is deep)
  * roofline admissible branch-and-bound skips dominated subtrees
  * transposition table dedups equivalent kernels (no double measure; DAG links)
  * LCB selection prefers low-variance-fast over high-variance-noisy kernels
  * anytime incumbent is monotone non-decreasing
  * global verification budget is respected
  * an incorrect leaf is a LOW-value but REPAIRABLE node (not dead)
"""

from __future__ import annotations

from kore.reward.reward import Observation
from kore.search import (
    Budget,
    CallbackArm,
    Edit,
    MeasureStats,
    canonicalize_source,
    fingerprint,
    io_signature,
    search,
    successive_halving,
)


# --------------------------------------------------------------------------- #
# Fakes (scripted; deterministic)
# --------------------------------------------------------------------------- #
class FakeTask:
    task_id = "fake_gemm_bf16"
    operation = "gemm"
    dtype = "bf16"
    gpu_target = "gfx950"
    snr_threshold = 25.0
    shapes = []


def _obs_compile_fail():
    return Observation(compiled=False, dtype="bf16", error_text="SyntaxError: bad")


def _obs_incorrect():
    return Observation(compiled=True, dtype="bf16", validation_passed=False,
                       snr_by_shape={"primary": 5.0}, snr_db=5.0,
                       error_text="worst SNR 5.0 < 25.0")


def _obs_correct_gate():
    """Correctness-only verdict (full_validation=False): no timing."""
    return Observation(compiled=True, dtype="bf16", validation_passed=True,
                       snr_by_shape={"primary": 40.0}, snr_db=40.0)


def _obs_correct_bench(speedup: float):
    """A timed verdict encoding ``speedup`` = baseline_ms / wall_ms."""
    return Observation(compiled=True, dtype="bf16", validation_passed=True,
                       snr_by_shape={"primary": 40.0}, snr_db=40.0,
                       wall_by_shape={"primary": 1.0},
                       baseline_by_shape={"primary": float(speedup)},
                       wall_ms=1.0, baseline_ms=float(speedup))


class FakeEnv:
    """Scripted env. A marker in the source selects the verdict; ``speedups`` maps a
    marker to a sequence of speedup samples (cycled) returned on timed
    (``full_validation=True``) calls, so repeated measures can carry variance.
    """

    def __init__(self, speedups=None):
        self.calls: list[tuple] = []
        self.speedups = speedups or {}
        self._idx: dict = {}

    def step(self, source, full_validation=True, multi_shape=True):
        self.calls.append((source, full_validation, multi_shape))
        if "__BAD__" in source:
            return _obs_compile_fail()
        if "__WRONG__" in source:
            return _obs_incorrect()
        if not full_validation:
            return _obs_correct_gate()
        return _obs_correct_bench(self._next_speedup(source))

    def _next_speedup(self, source: str) -> float:
        for marker, seq in self.speedups.items():
            if marker in source:
                i = self._idx.get(marker, 0)
                self._idx[marker] = i + 1
                return float(seq[i % len(seq)])
        return 1.0

    # assertion helpers
    def any_call(self, needle: str) -> bool:
        return any(needle in s for (s, _fv, _ms) in self.calls)

    def bench_count(self, needle: str) -> int:
        return sum(1 for (s, fv, _ms) in self.calls if fv and needle in s)


class FakePolicy:
    """Returns scripted edits for the FIRST marker found in the node's source."""

    def __init__(self, rules: list[tuple]):
        self.rules = rules            # [(marker, [Edit, ...]), ...]
        self.seen: list[str] = []

    def propose(self, state):
        self.seen.append(state.source)
        for marker, edits in self.rules:
            if marker in state.source:
                return list(edits)
        return []


class FakeValueModel:
    """Scores sources by a marker->score map (default 0.0). Sets PUCT priors."""

    def __init__(self, scores: dict):
        self.scores = scores

    def score(self, sources, task):
        out = []
        for s in sources:
            v = 0.0
            for marker, sc in self.scores.items():
                if marker in s:
                    v = sc
                    break
            out.append(float(v))
        return out


def _find(root, needle: str):
    seen: set = set()
    stack = [root]
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        if needle in n.source:
            return n
        stack.extend(n.children)
    return None


# --------------------------------------------------------------------------- #
# 1. Fingerprint / canonicalization (transposition key)
# --------------------------------------------------------------------------- #
def test_canonicalize_strips_comments_and_whitespace():
    a = "def k(x):\n    return x    # comment A\n"
    b = "def k(x):\n        return x   # a totally different comment\n\n"
    assert canonicalize_source(a) == canonicalize_source(b) == "def k(x):\nreturn x"


def test_fingerprint_dedups_equivalent_and_splits_changed():
    a = "kernel body __X__  # note A"
    b = "kernel body __X__      # a different note entirely"
    c = "kernel body __X__ + 1  # note A"     # a real change
    assert fingerprint(a) == fingerprint(b)     # cosmetic diff -> same fingerprint
    assert fingerprint(a) != fingerprint(c)     # semantic diff -> distinct


def test_io_signature_extraction():
    assert io_signature("def foo(a, b: int, c=3):\n  pass") == "foo(a,b,c)"
    assert io_signature("def f(x):\n  pass\ndef g(y, z):\n  pass") == "f(x)|g(y,z)"
    assert io_signature("no defs here") == ""


# --------------------------------------------------------------------------- #
# 2. Bandit: Budget, LCB stats, Successive Halving
# --------------------------------------------------------------------------- #
def test_budget_is_a_hard_cap():
    b = Budget(3)
    assert b.spend(2) and b.used == 2
    assert not b.spend(2)          # would exceed -> nothing consumed
    assert b.used == 2
    assert b.spend(1) and b.used == 3
    assert not b.can_afford(1) and b.remaining == 0


def test_measure_stats_lcb_penalizes_variance():
    stable = MeasureStats(z=1.0)
    for x in [2.0, 2.0, 2.0, 2.0]:
        stable.add(x)
    assert stable.mean == 2.0 and stable.var == 0.0 and stable.lcb == 2.0

    noisy = MeasureStats(z=1.0)
    for x in [3.5, 0.5]:
        noisy.add(x)
    assert abs(noisy.mean - 2.0) < 1e-9      # SAME mean as the stable arm
    assert noisy.lcb < 2.0                    # ...but the LCB is pulled down
    assert noisy.n == 2


def test_successive_halving_invests_in_low_variance_survivor():
    stable_seq = [2.0] * 12
    noisy_seq = [3.5, 0.5] * 6
    si = {"i": 0}
    ni = {"i": 0}

    def stable():
        v = stable_seq[si["i"]]
        si["i"] += 1
        return v

    def noisy():
        v = noisy_seq[ni["i"]]
        ni["i"] += 1
        return v

    a = CallbackArm("stable", stable, MeasureStats(z=1.0))
    b = CallbackArm("noisy", noisy, MeasureStats(z=1.0))
    budget = Budget(50)
    ranked = successive_halving([a, b], budget, eta=2, min_measures=2, max_measures=6)

    assert ranked[0].key == "stable"          # LCB ranks the stable arm first
    assert a.n > b.n                           # survivor got extra measurements
    assert b.n == 2                            # eliminated after the first rung
    assert budget.used <= 50


def test_successive_halving_stops_at_budget():
    seq = [1.0] * 100
    i = {"i": 0}

    def s():
        v = seq[i["i"]]
        i["i"] += 1
        return v

    arms = [CallbackArm(k, s, MeasureStats()) for k in range(4)]
    budget = Budget(5)
    successive_halving(arms, budget, eta=2, min_measures=2, max_measures=8)
    assert budget.used <= 5
    assert sum(a.n for a in arms) == budget.used


def test_successive_halving_skips_roofline_dominated_arm():
    seq = [2.0] * 20
    i = {"i": 0}

    def s():
        v = seq[i["i"]]
        i["i"] += 1
        return v

    live = CallbackArm("live", s, MeasureStats(), ceiling=5.0)
    dom = CallbackArm("dominated", s, MeasureStats(), ceiling=1.0)
    budget = Budget(50)
    # incumbent LCB 3.0 dominates the ceiling-1.0 arm: it must never be measured.
    successive_halving([live, dom], budget, min_measures=2, max_measures=6,
                       incumbent_lcb=3.0)
    assert dom.n == 0
    assert live.n >= 2


# --------------------------------------------------------------------------- #
# 3. MAX-backup returns the best CORRECT leaf (even when deep)
# --------------------------------------------------------------------------- #
def _deep_tree():
    env = FakeEnv(speedups={"__ROOT__": [1.0], "__GOOD__": [1.5],
                            "__BEST__": [3.0], "__MID__": [1.2]})
    policy = FakePolicy([
        ("__ROOT__", [Edit("cand __GOOD__", "widen"),
                      Edit("cand __WRONG__", "bad-numerics")]),
        ("__GOOD__", [Edit("cand __BEST__", "pipeline"),
                      Edit("cand __MID__", "retune")]),
    ])
    return env, policy


def test_max_backup_returns_best_correct_leaf():
    env, policy = _deep_tree()
    res = search("root __ROOT__", FakeTask(), env, policy, value_model=None, budget=128)

    assert "__BEST__" in res["best_source"]
    assert abs(res["best_speedup_lcb"] - 3.0) < 1e-9

    root = res["root"]
    best = res["best_node"]
    assert best.correct and best.children == []          # it is a leaf
    # MAX-backup: the root's best-descendant value == the best leaf's node value
    # (correctness base + pessimistic speedup), NOT an average over the tree.
    assert abs(root.best_descendant_reward - best.self_value) < 1e-9
    assert abs(root.best_descendant_reward - (1.0 + 3.0)) < 1e-9
    # the slow/mid/wrong siblings did not win
    assert "__MID__" not in res["best_source"]
    assert res["tree_stats"]["n_correct"] == 4           # root, GOOD, BEST, MID


# --------------------------------------------------------------------------- #
# 4. Roofline admissible branch-and-bound skips dominated subtrees
# --------------------------------------------------------------------------- #
def test_roofline_pruning_skips_dominated_subtree():
    env = FakeEnv(speedups={"__ROOT__": [1.0], "__GOOD__": [3.0],
                            "__GBEST__": [4.0], "__DOM__": [1.0],
                            "__DOMCHILD__": [9.0]})
    policy = FakePolicy([
        ("__ROOT__", [Edit("cand __GOOD__", "good"), Edit("cand __DOM__", "dominated")]),
        ("__GOOD__", [Edit("cand __GBEST__", "best")]),
        # This would be a huge win, but its parent's roofline ceiling is too low,
        # so the whole subtree must be pruned and NEVER measured.
        ("__DOM__", [Edit("cand __DOMCHILD__", "trap")]),
    ])

    def roofline_ub_fn(source, task):
        return 1.2 if "__DOM" in source else 5.0

    res = search("root __ROOT__", FakeTask(), env, policy, value_model=None,
                 budget=256, roofline_ub_fn=roofline_ub_fn)

    dom = _find(res["root"], "__DOM__")
    assert dom is not None and dom.pruned and dom.status == "pruned"
    # the dominated subtree was never expanded => its child was never generated/measured
    assert not env.any_call("__DOMCHILD__")
    assert res["tree_stats"]["n_pruned"] >= 1
    # the admissible branch still found its best leaf
    assert "__GBEST__" in res["best_source"]
    assert abs(res["best_speedup_lcb"] - 4.0) < 1e-9


# --------------------------------------------------------------------------- #
# 5. Transposition table dedups equivalent kernels (no double measure; DAG)
# --------------------------------------------------------------------------- #
def test_transposition_dedups_same_parent_no_double_measure():
    env = FakeEnv(speedups={"__ROOT__": [1.0], "__DUP__": [2.0]})
    policy = FakePolicy([
        ("__ROOT__", [Edit("cand __DUP__  # variant A", "a"),
                      Edit("cand __DUP__\n# variant B differs only in comment", "b")]),
    ])
    res = search("root __ROOT__", FakeTask(), env, policy, value_model=None, budget=64)

    root = res["root"]
    assert len(root.children) == 1                     # both edits collapsed to 1 node
    assert not env.any_call("variant B")               # the duplicate was never stepped
    assert res["tree_stats"]["n_transpositions"] >= 1
    dup = root.children[0]
    assert dup.correct and dup.n_measures > 0


def test_transposition_links_across_parents_and_inherits_value():
    env = FakeEnv(speedups={"__ROOT__": [1.0], "__P1__": [1.6],
                            "__P2__": [1.5], "__SHARED__": [2.0]})
    policy = FakePolicy([
        ("__ROOT__", [Edit("cand __P1__", "p1"), Edit("cand __P2__", "p2")]),
        ("__P1__", [Edit("cand __SHARED__ # via p1", "s1")]),
        ("__P2__", [Edit("cand __SHARED__ # via p2", "s2")]),
    ])
    res = search("root __ROOT__", FakeTask(), env, policy, value_model=None, budget=256)

    shared = _find(res["root"], "__SHARED__")
    assert shared is not None
    assert len(shared.parents) == 2                    # reached from BOTH p1 and p2 (DAG)
    # measured exactly once (via p1); the p2 path linked and inherited the value
    assert not env.any_call("via p2")
    assert shared.n_measures == env.bench_count("__SHARED__")
    assert res["tree_stats"]["n_transpositions"] >= 1


# --------------------------------------------------------------------------- #
# 6. LCB selection prefers low-variance-fast over high-variance-noisy
# --------------------------------------------------------------------------- #
def test_lcb_prefers_low_variance_over_noisy_same_mean():
    env = FakeEnv(speedups={
        "__ROOT__": [1.0],
        "__STABLE__": [2.0, 2.0, 2.0, 2.0],       # mean 2.0, zero variance
        "__NOISY__": [3.5, 0.5, 3.5, 0.5],        # mean 2.0, high variance
    })
    policy = FakePolicy([
        ("__ROOT__", [Edit("cand __STABLE__", "stable"),
                      Edit("cand __NOISY__", "noisy")]),
    ])
    res = search("root __ROOT__", FakeTask(), env, policy, value_model=None, budget=128)

    stable = _find(res["root"], "__STABLE__")
    noisy = _find(res["root"], "__NOISY__")
    assert abs(stable.speedup_mean - noisy.speedup_mean) < 1e-9   # same mean
    assert stable.var == 0.0 and noisy.var > 0.0
    assert stable.speedup_lcb > noisy.speedup_lcb                 # LCB separates them
    assert "__STABLE__" in res["best_source"]                    # ...and search commits to it
    assert abs(res["best_speedup_lcb"] - 2.0) < 1e-9


# --------------------------------------------------------------------------- #
# 7. Anytime incumbent is monotone non-decreasing
# --------------------------------------------------------------------------- #
def test_anytime_incumbent_is_monotone():
    env = FakeEnv(speedups={"__ROOT__": [1.0], "__S1__": [1.5],
                            "__S2__": [2.5], "__S3__": [3.5]})
    policy = FakePolicy([
        ("__ROOT__", [Edit("cand __S1__", "s1")]),
        ("__S1__", [Edit("cand __S2__", "s2")]),
        ("__S2__", [Edit("cand __S3__", "s3")]),
    ])
    res = search("root __ROOT__", FakeTask(), env, policy, value_model=None, budget=256)

    trace = [t for t in res["tree_stats"]["incumbent_trace"] if t is not None]
    assert trace, "expected at least one incumbent"
    assert all(b >= a - 1e-12 for a, b in zip(trace, trace[1:]))   # never regresses
    assert trace[0] == 1.0 and abs(trace[-1] - 3.5) < 1e-9


def test_deep_tree_incumbent_trace_never_regresses():
    env, policy = _deep_tree()
    res = search("root __ROOT__", FakeTask(), env, policy, value_model=None, budget=256)
    trace = [t for t in res["tree_stats"]["incumbent_trace"] if t is not None]
    assert all(b >= a - 1e-12 for a, b in zip(trace, trace[1:]))


# --------------------------------------------------------------------------- #
# 8. Global verification budget is respected
# --------------------------------------------------------------------------- #
def test_budget_is_respected_across_the_search():
    env = FakeEnv(speedups={"__ROOT__": [1.5]})
    # a policy that would expand forever if allowed
    policy = FakePolicy([
        ("__ROOT__", [Edit(f"cand child{i} fast", f"e{i}") for i in range(6)]),
        ("child", [Edit("cand grandchild fast", "g")] * 6),
    ])
    budget = Budget(7)
    res = search("root __ROOT__", FakeTask(), env, policy, value_model=None, budget=budget)

    assert budget.used <= 7
    assert res["tree_stats"]["env_calls"] == budget.used
    assert len(env.calls) == res["tree_stats"]["env_calls"] <= 7
    # anytime: the root was correct, so a best is still reported despite the tiny budget
    assert res["best_source"] is not None


# --------------------------------------------------------------------------- #
# 9. Incorrect leaf is LOW-value but REPAIRABLE (not dead)
# --------------------------------------------------------------------------- #
def test_incorrect_leaf_is_repairable_not_dead():
    env = FakeEnv(speedups={"__ROOT__": [1.0], "__FIXED__": [2.0]})
    policy = FakePolicy([
        ("__ROOT__", [Edit("cand __WRONG__", "buggy")]),
        # the incorrect node is expanded (repaired) into a correct, fast child
        ("__WRONG__", [Edit("cand __FIXED__", "repair")]),
    ])
    res = search("root __ROOT__", FakeTask(), env, policy, value_model=None, budget=128)

    wrong = _find(res["root"], "__WRONG__")
    fixed = _find(res["root"], "__FIXED__")
    assert wrong is not None and wrong.status == "incorrect" and not wrong.pruned
    assert wrong.expanded                              # it was selectable / repairable
    assert wrong.self_value < 1.0                      # low value...
    assert fixed is not None and fixed.self_value >= 1.0   # ...but the repair is high value
    assert "__FIXED__" in res["best_source"]
    assert abs(res["best_speedup_lcb"] - 2.0) < 1e-9


# --------------------------------------------------------------------------- #
# 10. No correct kernel anywhere -> honest empty incumbent
# --------------------------------------------------------------------------- #
def test_no_correct_kernel_returns_none_incumbent():
    env = FakeEnv()
    policy = FakePolicy([])          # root is a compile failure; nothing to expand
    res = search("root __BAD__", FakeTask(), env, policy, value_model=None, budget=32)

    assert res["best_source"] is None
    assert res["best_speedup_lcb"] is None
    assert res["tree_stats"]["n_correct"] == 0
    assert all(t is None for t in res["tree_stats"]["incumbent_trace"])


# --------------------------------------------------------------------------- #
# 11. Value model sets priors (exploration steering) without breaking outcomes
# --------------------------------------------------------------------------- #
def test_value_model_priors_are_used():
    env = FakeEnv(speedups={"__ROOT__": [1.0], "__A__": [2.0], "__B__": [2.0]})
    policy = FakePolicy([
        ("__ROOT__", [Edit("cand __A__", "a"), Edit("cand __B__", "b")]),
    ])
    vm = FakeValueModel({"__A__": 5.0, "__B__": 0.0})
    res = search("root __ROOT__", FakeTask(), env, policy, value_model=vm, budget=128)

    a = _find(res["root"], "__A__")
    b = _find(res["root"], "__B__")
    # softmax(5,0) -> A gets the dominant prior mass
    assert a.prior > b.prior
    assert a.value_prior == 5.0 and b.value_prior == 0.0
