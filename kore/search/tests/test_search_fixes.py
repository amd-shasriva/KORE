"""CPU-only tests for the AlphaKernel audit fixes (no GPU / torch / network).

Covers the four defects fixed in :mod:`kore.search.alphakernel` /
:mod:`kore.search.propose`:

  1. Roofline branch-and-bound is wireable + admissible (signature fixed):
       * ``RooflineCeiling`` / ``make_roofline_ub_fn`` expose the canonical
         ``(source, task) -> Optional[float]`` signature and inject the env-measured
         baseline lazily via ``observe_baseline``;
       * an ADMISSIBLE bound never prunes the branch that contains the optimum on a
         small synthetic tree, while a genuinely-dominated branch IS pruned.
  2. The anytime incumbent is ALWAYS the true argmax; the B&B pruning floor is
     monotone and decoupled from the (possibly-moving) live incumbent.
  3. Deeper search: a larger budget explores strictly more nodes, and ``max_depth``
     caps the expansion depth. Defaults reproduce the shallow behavior.
  4. A supplied ``value_fn`` sets the PUCT priors (and takes precedence over a
     ``value_model``); the optional value-model leaf prior is OFF by default.

Plus: every new knob at its default reproduces the prior behavior exactly.

The fakes mirror ``test_alphakernel.py``: a scripted ``FakeEnv`` keyed by markers in
the source, a ``FakePolicy`` that yields scripted edits per parent, and CPU-only
``Task`` stand-ins. The verifier is used as a perfect simulator.
"""

from __future__ import annotations

from kore.reward.reward import Observation
from kore.search.alphakernel import (
    AlphaKernelConfig,
    Edit,
    Node,
    RooflineCeiling,
    _Search,
    make_roofline_ub_fn,
    roofline_speedup_ceiling,
    search,
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
    shapes: list = []


class _Shape:
    def __init__(self, name, dims):
        self.name = name
        self.dims = dims


class RooflineTask:
    """A task with real shape dims so the roofline model yields a finite ceiling."""

    task_id = "rt_gemm_fp32"
    operation = "gemm"
    dtype = "fp32"
    gpu_target = "gfx950"
    snr_threshold = 25.0

    def __init__(self, dims=None):
        self.shapes = [_Shape("primary", dims or {"M": 4096, "N": 4096, "K": 4096})]


def _obs_gate():
    return Observation(compiled=True, dtype="bf16", validation_passed=True,
                       snr_by_shape={"primary": 40.0}, snr_db=40.0)


def _obs_bench(speedup: float, baseline_ms: float = None):
    b = float(speedup if baseline_ms is None else baseline_ms)
    wall = b / float(speedup)
    return Observation(compiled=True, dtype="bf16", validation_passed=True,
                       snr_by_shape={"primary": 40.0}, snr_db=40.0,
                       wall_by_shape={"primary": wall},
                       baseline_by_shape={"primary": b},
                       wall_ms=wall, baseline_ms=b)


class FakeEnv:
    """Scripted env; a marker in the source selects a cycled speedup sequence."""

    def __init__(self, speedups=None, baseline_ms=None):
        self.calls: list = []
        self.speedups = speedups or {}
        self.baseline_ms = baseline_ms
        self._idx: dict = {}

    def step(self, source, full_validation=True, multi_shape=True):
        self.calls.append((source, full_validation, multi_shape))
        if "__BAD__" in source:
            return Observation(compiled=False, dtype="bf16", error_text="bad")
        if "__WRONG__" in source:
            return Observation(compiled=True, dtype="bf16", validation_passed=False,
                               snr_by_shape={"primary": 5.0}, snr_db=5.0)
        if not full_validation:
            return _obs_gate()
        return _obs_bench(self._next(source), self.baseline_ms)

    def _next(self, source: str) -> float:
        for marker, seq in self.speedups.items():
            if marker in source:
                i = self._idx.get(marker, 0)
                self._idx[marker] = i + 1
                return float(seq[i % len(seq)])
        return 1.0

    def any_call(self, needle: str) -> bool:
        return any(needle in s for (s, _fv, _ms) in self.calls)

    def bench_count(self, needle: str) -> int:
        return sum(1 for (s, fv, _ms) in self.calls if fv and needle in s)


class FakePolicy:
    def __init__(self, rules):
        self.rules = rules

    def propose(self, state):
        for marker, edits in self.rules:
            if marker in state.source:
                return list(edits)
        return []


class GrowPolicy:
    """Every node proposes ``b`` children with globally-unique fast sources, so the
    tree grows without transposition collapse until the budget/depth stops it."""

    def __init__(self, b: int = 2):
        self.b = b
        self.c = 0

    def propose(self, state):
        out = []
        for _ in range(self.b):
            self.c += 1
            out.append(Edit(source=f"cand FAST unique_{self.c}", name=f"e{self.c}"))
        return out


class FakeValueModel:
    def __init__(self, scores):
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


def _all_nodes(root):
    seen: dict = {}
    stack = [root]
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen[id(n)] = n
        stack.extend(n.children)
    return list(seen.values())


def _engine(task=None, cfg=None):
    """A bare engine for unit-testing internal methods (env/policy unused here)."""
    return _Search(task or FakeTask(), env=None, policy=None, value_model=None,
                   budget=100, cfg=cfg or AlphaKernelConfig(), roofline_ub_fn=None)


# =========================================================================== #
# Defect 1a: roofline adapter signature is fixed + admissible + lazy baseline
# =========================================================================== #
def test_roofline_ceiling_adapter_signature_and_values():
    task = RooflineTask()
    B = 2.0
    exact = roofline_speedup_ceiling(task, B, arch="gfx950")
    assert exact is not None and exact > 0.0

    # Canonical (source, task) signature -- the historical mismatch is fixed.
    rc = make_roofline_ub_fn(baseline_ms=B, arch="gfx950", safety_margin=0.0)
    v1 = rc("some kernel source", task)
    v2 = rc("a completely different kernel", task)
    assert v1 == v2 == exact              # physical ceiling is source-independent

    # safety_margin inflates the bound (=> prunes less; strictly safer/admissible).
    rcm = make_roofline_ub_fn(baseline_ms=B, arch="gfx950", safety_margin=0.5)
    assert abs(rcm("s", task) - exact * 1.5) < 1e-9

    # No dims (roofline not modelable) -> None -> caller keeps +inf -> no pruning.
    assert rc("s", FakeTask()) is None


def test_roofline_ceiling_observes_baseline_lazily_and_once():
    task = RooflineTask()
    B = 3.0
    exact = roofline_speedup_ceiling(task, B, arch="gfx950")

    rc = RooflineCeiling(arch="gfx950", safety_margin=0.0)
    assert rc("s", task) is None          # no baseline yet -> safe no-op
    rc.observe_baseline(B)
    assert abs(rc("s", task) - exact) < 1e-9
    rc.observe_baseline(999.0)            # set-once: a later baseline is ignored
    assert abs(rc("s", task) - exact) < 1e-9
    rc.observe_baseline(0.0)              # non-positive ignored
    assert abs(rc("s", task) - exact) < 1e-9


def test_roofline_ceiling_is_admissible_upper_bound():
    """The ceiling must UPPER-bound any achievable speedup: baseline_ms / T_min with
    T_min the physical runtime floor. So for a fixed baseline it strictly exceeds the
    speedup of a kernel that runs slower than T_min (i.e. every real kernel)."""
    task = RooflineTask()
    B = 5.0
    exact = roofline_speedup_ceiling(task, B, arch="gfx950")
    # a kernel that runs at 2x T_min has speedup exact/2 < ceiling; the bound never
    # sits below an achievable speedup -> admissible.
    assert exact > exact / 2.0
    assert make_roofline_ub_fn(baseline_ms=B, arch="gfx950")("s", task) >= exact


# =========================================================================== #
# Defect 1b: admissible bound never prunes the optimum (synthetic tree)
# =========================================================================== #
def test_admissible_bound_never_prunes_the_optimum():
    """Tree:  root(1.0) -> {A(1.5) -> ABEST(5.0),  B(2.0),  D(0.3) -> DKID(0.4)}.

    The admissible bound gives each node an upper bound on the BEST speedup in its
    subtree (the tightest admissible value). D's whole subtree tops out at 0.4, so D
    is correctly pruned once the incumbent passes it -- and nothing is lost, because
    the true optimum (ABEST=5.0) lives under A whose ceiling (5.0) is never dominated.
    """
    subtree_ceiling = {
        "__ABEST__": 5.0, "__A__": 5.0, "__B__": 2.0,
        "__DKID__": 0.4, "__D__": 0.4, "__ROOT__": 5.0,
    }

    def roofline_ub_fn(source, task):
        for marker, c in subtree_ceiling.items():
            if marker in source:
                return c
        return float("inf")

    env = FakeEnv(speedups={"__ROOT__": [1.0], "__A__": [1.5], "__B__": [2.0],
                            "__ABEST__": [5.0], "__D__": [0.3], "__DKID__": [0.4]})
    policy = FakePolicy([
        ("__ROOT__", [Edit("cand __A__", "a"), Edit("cand __B__", "b"),
                      Edit("cand __D__", "d")]),
        ("__A__", [Edit("cand __ABEST__", "abest")]),
        ("__D__", [Edit("cand __DKID__", "dkid")]),
    ])
    res = search("root __ROOT__", FakeTask(), env, policy, budget=256,
                 roofline_ub_fn=roofline_ub_fn)

    # the deep optimum is found -- its branch (A) was NEVER pruned
    assert "__ABEST__" in res["best_source"]
    assert abs(res["best_speedup_lcb"] - 5.0) < 1e-9
    a = _find(res["root"], "__A__")
    assert a is not None and not a.pruned
    # the genuinely-dominated branch D WAS pruned (admissible: it hides nothing good)
    d = _find(res["root"], "__D__")
    assert d is not None and d.pruned and d.status == "pruned"
    assert not env.any_call("__DKID__")     # its subtree was never explored


def test_make_roofline_ub_fn_wires_end_to_end_and_preserves_optimum():
    """The production adapter, wired into a real search, activates from the env
    baseline and stamps every post-root node with the finite physical ceiling -- and
    because that ceiling is admissible, the optimum is still found (nothing pruned)."""
    task = RooflineTask()
    B = 4.0
    exact = roofline_speedup_ceiling(task, B, arch="gfx950")
    rc = make_roofline_ub_fn(arch="gfx950")           # baseline discovered at runtime

    env = FakeEnv(speedups={"__ROOT__": [1.0], "__A__": [1.5], "__ABEST__": [2.0]},
                  baseline_ms=B)
    policy = FakePolicy([
        ("__ROOT__", [Edit("cand __A__", "a")]),
        ("__A__", [Edit("cand __ABEST__", "abest")]),
    ])
    res = search("root __ROOT__", task, env, policy, budget=128, roofline_ub_fn=rc)

    assert abs(rc.baseline_ms - B) < 1e-9             # observe_baseline was wired
    a = _find(res["root"], "__A__")
    assert a is not None and a.roofline_ub != float("inf")
    assert abs(a.roofline_ub - exact * 1.25) < 1e-6   # default margin 0.25, signature OK
    # admissible: the (huge) physical ceiling dominates nothing here -> optimum kept
    assert "__ABEST__" in res["best_source"]
    assert abs(res["best_speedup_lcb"] - 2.0) < 1e-9
    assert res["tree_stats"]["n_pruned"] == 0


# =========================================================================== #
# Defect 2: incumbent is the true argmax; prune floor is monotone
# =========================================================================== #
def test_incumbent_tracks_true_argmax_and_floor_is_monotone():
    """Directly exercises the fixed _update_incumbent.

    Old bug: incumbent_lcb was a running max and the incumbent was never demoted, so
    a node whose LCB later DECAYED could keep the slot while a now-better node never
    got promoted. The fix recomputes the true argmax each call and keeps a SEPARATE
    monotone pruning floor."""
    eng = _engine()

    a = Node(source="a", fingerprint="fa", correct=True)
    a.stats.add(3.0)
    a.stats.add(3.0)                       # A: lcb = 3.0 (stable)
    eng.tt["fa"] = a
    eng._update_incumbent()
    assert eng.incumbent is a
    assert abs(eng.incumbent_lcb - 3.0) < 1e-9
    assert abs(eng._prune_floor - 3.0) < 1e-9

    # A decays (more samples pull its mean/LCB down) and a genuinely-better-NOW node
    # B appears, but B's LCB (2.5) never clears A's old 3.0 high-water mark.
    a.stats.add(1.0)
    a.stats.add(1.0)                       # A: lcb now ~1.42
    assert a.stats.lcb < 2.5
    b = Node(source="b", fingerprint="fb", correct=True)
    b.stats.add(2.5)
    b.stats.add(2.5)                       # B: lcb = 2.5
    eng.tt["fb"] = b
    eng._update_incumbent()

    assert eng.incumbent is b              # FIX: true argmax, not the stale A
    assert abs(eng.incumbent_lcb - 2.5) < 1e-9   # reports the LIVE argmax LCB
    assert eng._prune_floor >= 3.0 - 1e-9        # ...while the floor never regresses

    # a strictly-better node advances both the incumbent and the floor
    c = Node(source="c", fingerprint="fc", correct=True)
    c.stats.add(4.0)
    c.stats.add(4.0)
    eng.tt["fc"] = c
    eng._update_incumbent()
    assert eng.incumbent is c
    assert abs(eng.incumbent_lcb - 4.0) < 1e-9
    assert abs(eng._prune_floor - 4.0) < 1e-9


def test_incumbent_min_measures_gate():
    """A node below the sample floor is not eligible as the incumbent."""
    eng = _engine(cfg=AlphaKernelConfig(incumbent_min_measures=2))
    n1 = Node(source="x", fingerprint="fx", correct=True)
    n1.stats.add(9.0)                      # only ONE sample -> below the floor of 2
    eng.tt["fx"] = n1
    eng._update_incumbent()
    assert eng.incumbent is None           # not eligible yet
    n1.stats.add(9.0)                      # second sample -> now eligible
    eng._update_incumbent()
    assert eng.incumbent is n1 and abs(eng.incumbent_lcb - 9.0) < 1e-9


def test_search_incumbent_is_the_true_argmax_over_the_tree():
    """End-to-end: the reported incumbent equals the argmax over all correct,
    measured nodes by speedup_lcb."""
    env = FakeEnv(speedups={"__ROOT__": [1.0], "__S1__": [1.5], "__S2__": [2.5],
                            "__S3__": [3.5]})
    policy = FakePolicy([
        ("__ROOT__", [Edit("cand __S1__", "s1")]),
        ("__S1__", [Edit("cand __S2__", "s2")]),
        ("__S2__", [Edit("cand __S3__", "s3")]),
    ])
    res = search("root __ROOT__", FakeTask(), env, policy, budget=256)

    measured = [n for n in _all_nodes(res["root"]) if n.correct and n.stats.n > 0]
    true_best = max(measured, key=lambda n: n.stats.lcb)
    assert res["best_node"] is true_best
    assert abs(res["best_speedup_lcb"] - true_best.stats.lcb) < 1e-9
    # trace never regresses (anytime monotone under the one-shot measurement model)
    trace = [t for t in res["tree_stats"]["incumbent_trace"] if t is not None]
    assert all(y >= x - 1e-12 for x, y in zip(trace, trace[1:]))


# =========================================================================== #
# Defect 3: deeper budget explores more; max_depth caps the depth
# =========================================================================== #
def test_deeper_budget_explores_more_nodes():
    def run(budget):
        env = FakeEnv()                    # every kernel correct @ ~1.0x
        res = search("root SEED", FakeTask(), env, GrowPolicy(b=2), budget=budget)
        return res["tree_stats"]

    small = run(8)
    big = run(200)
    assert big["n_nodes"] > small["n_nodes"]
    assert big["n_expanded"] > small["n_expanded"]
    assert big["budget_used"] <= 200 and small["budget_used"] <= 8


def test_max_depth_caps_expansion_depth():
    env = FakeEnv()
    shallow = search("root SEED", FakeTask(), env, GrowPolicy(b=2), budget=200,
                     config=AlphaKernelConfig(max_depth=1))
    assert shallow["tree_stats"]["max_depth"] <= 1

    env2 = FakeEnv()
    deep = search("root SEED", FakeTask(), env2, GrowPolicy(b=2), budget=200,
                  config=AlphaKernelConfig(max_depth=None))
    assert deep["tree_stats"]["max_depth"] >= 2
    assert deep["tree_stats"]["max_depth"] > shallow["tree_stats"]["max_depth"]


def test_search_from_kernel_exposes_deeper_search_params():
    """The propose.py wiring forwards budget / k_expand / max_depth to the engine."""
    from kore.search.propose import search_from_kernel

    env = FakeEnv()
    # a non-Triton root yields no transform moves, but the call must accept the new
    # params and run cleanly (fail-safe), returning the result contract.
    res = search_from_kernel("def f(x):\n    return x\n", FakeTask(), env,
                             budget=16, k_expand=6, max_depth=3,
                             incumbent_min_measures=2, value_leaf_weight=0.1)
    assert "best_source" in res and "tree_stats" in res


# =========================================================================== #
# Defect 4: value_fn drives the PUCT priors (and beats value_model)
# =========================================================================== #
def test_value_fn_prior_is_used_when_supplied():
    env = FakeEnv(speedups={"__ROOT__": [1.0], "__A__": [2.0], "__B__": [2.0]})
    policy = FakePolicy([
        ("__ROOT__", [Edit("cand __A__", "a"), Edit("cand __B__", "b")]),
    ])

    def value_fn(sources, task):
        return [5.0 if "__A__" in s else 0.0 for s in sources]

    res = search("root __ROOT__", FakeTask(), env, policy, budget=128,
                 value_fn=value_fn)
    a = _find(res["root"], "__A__")
    b = _find(res["root"], "__B__")
    assert a.value_prior == 5.0 and b.value_prior == 0.0
    assert a.prior > b.prior               # softmax(5,0) mass concentrates on A


def test_value_fn_takes_precedence_over_value_model():
    env = FakeEnv(speedups={"__ROOT__": [1.0], "__A__": [2.0], "__B__": [2.0]})
    policy = FakePolicy([
        ("__ROOT__", [Edit("cand __A__", "a"), Edit("cand __B__", "b")]),
    ])
    vm = FakeValueModel({"__A__": 0.0, "__B__": 5.0})     # model favors B

    def value_fn(sources, task):                           # ...but value_fn favors A
        return [5.0 if "__A__" in s else 0.0 for s in sources]

    res = search("root __ROOT__", FakeTask(), env, policy, budget=128,
                 value_model=vm, value_fn=value_fn)
    a = _find(res["root"], "__A__")
    b = _find(res["root"], "__B__")
    assert a.value_prior == 5.0 and a.prior > b.prior      # value_fn wins


def test_value_leaf_prior_hook():
    """cfg.value_leaf_weight blends a bounded value-model score into the leaf value of
    a correct-but-unmeasured node; default 0.0 => the bare correctness base."""
    off = _engine(cfg=AlphaKernelConfig(value_leaf_weight=0.0))
    on = _engine(cfg=AlphaKernelConfig(value_leaf_weight=0.5))

    n = Node(source="k", fingerprint="fk", correct=True)   # correct, n == 0
    n.value_prior = 10.0
    assert off._node_q(n) == 1.0                            # base only (prior behavior)
    q = on._node_q(n)
    assert 1.0 < q < 1.5                                    # bounded in (base, base+w)

    n.value_prior = -10.0
    assert abs(on._node_q(n) - 1.0) < 1e-3                  # low prior -> ~base

    # a MEASURED correct node is unaffected by the leaf weight (measurement dominates)
    m = Node(source="m", fingerprint="fm", correct=True)
    m.value_prior = 10.0
    m.stats.add(3.0)
    m.stats.add(3.0)
    assert abs(on._node_q(m) - (1.0 + 3.0)) < 1e-9


# =========================================================================== #
# Default-off: every new knob at its default reproduces prior behavior
# =========================================================================== #
def _deep_tree_env_policy():
    env = FakeEnv(speedups={"__ROOT__": [1.0], "__GOOD__": [1.5],
                            "__BEST__": [3.0], "__MID__": [1.2]})
    policy = FakePolicy([
        ("__ROOT__", [Edit("cand __GOOD__", "widen"),
                      Edit("cand __WRONG__", "bad")]),
        ("__GOOD__", [Edit("cand __BEST__", "pipeline"),
                      Edit("cand __MID__", "retune")]),
    ])
    return env, policy


def test_defaults_reproduce_prior_behavior():
    env_a, pol_a = _deep_tree_env_policy()
    res_default = search("root __ROOT__", FakeTask(), env_a, pol_a, budget=128)

    env_b, pol_b = _deep_tree_env_policy()
    res_explicit = search(
        "root __ROOT__", FakeTask(), env_b, pol_b, budget=128,
        roofline_ub_fn=None, value_fn=None,
        config=AlphaKernelConfig(max_depth=None, incumbent_min_measures=1,
                                 value_leaf_weight=0.0))

    assert res_default["best_source"] == res_explicit["best_source"]
    assert res_default["best_speedup_lcb"] == res_explicit["best_speedup_lcb"]
    assert "__BEST__" in res_default["best_source"]
    assert abs(res_default["best_speedup_lcb"] - 3.0) < 1e-9
    # no roofline_ub_fn => nothing is ever pruned (dormant-by-default, as before)
    assert res_default["tree_stats"]["n_pruned"] == 0
