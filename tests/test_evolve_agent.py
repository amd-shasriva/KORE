"""CPU-only tests for the evolve-agent loop (``kore.search.evolve_agent``).

These are deliberately weighted toward FAILURE modes rather than the happy path,
because every one of them has already cost this project or its sources something
real:

* a claimed **1541x** kernel arrived in third-party data - a decoy that never
  ran, where genuine fused wins on this hardware are 1-10x;
* Dr. Kernel (arXiv 2602.05885) measured a "successful" optimisation covering
  **0.014%** of CUDA time where proper fusion covers **86.15%**;
* Kernel-Smith (arXiv 2603.28342 §3.3) report that timing noise makes an
  evolutionary search "preserve suboptimal kernels or eliminate genuinely
  promising ones, and such mistakes compound across generations";
* and the archive this repository already had collapses to a single cell on a
  single task, which :func:`test_speedup_binned_archive_collapses_where_strategy_archive_does_not`
  measures rather than asserts by assumption.

Everything runs against scripted fakes: no GPU, no model, no network.
"""

from __future__ import annotations

import json
import random

import pytest

from kore.agent.harness import AgentHarness, WinsKB
from kore.data.evolve import MapElitesArchive
from kore.data.step_centric import MIN_GAIN
from kore.data.teacher import StubTeacher
from kore.reward.reward import Observation
from kore.search.alphakernel import fingerprint
from kore.search.bandit import Budget, MeasureStats
from kore.search.evolve_agent import (
    Archive,
    BestAcrossSteps,
    Candidate,
    EvolveAgentConfig,
    HarnessProposer,
    Proposal,
    ScaledStep,
    StableEvaluator,
    StrategySignature,
    Trial,
    candidate_from_trial,
    evolve,
    kernel_time_coverage,
    launch_count,
    strategy_signature,
    torch_op_residue,
    trim_outliers,
)


# --------------------------------------------------------------------------- #
# Fixtures: kernels that differ in the ways the descriptor is supposed to see
# --------------------------------------------------------------------------- #
def kernel(
    name: str = "k",
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 64,
    warps: int = 8,
    stages: int = 3,
    mode: str = "mma",
    torch_ops: int = 0,
    comment: str = "",
    tweak: str = "",
) -> str:
    """A syntactically plausible Triton kernel with controllable strategy knobs."""
    if mode == "mma":
        body = ("    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)\n"
                "    for kk in range(0, K, BLOCK_K):\n"
                "        acc += tl.dot(tl.load(a_ptr), tl.load(b_ptr))\n"
                "    tl.store(c_ptr, acc)")
    elif mode == "reduce":
        body = ("    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)\n"
                "    for kk in range(0, K, BLOCK_K):\n"
                "        acc += tl.load(a_ptr) * tl.load(b_ptr)\n"
                "    tl.store(c_ptr, acc)")
    elif mode == "atomic":
        body = ("    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)\n"
                "    for kk in range(0, K, BLOCK_K):\n"
                "        acc += tl.dot(tl.load(a_ptr), tl.load(b_ptr))\n"
                "    tl.atomic_add(c_ptr, acc)")
    else:  # elementwise
        body = ("    x = tl.load(a_ptr)\n"
                "    tl.store(c_ptr, x * 2.0)")
    eager = "".join(
        f"    a = torch.{op}(a)\n"
        for op in ("relu", "softmax", "sigmoid", "tanh", "log", "sqrt",
                   "exp", "cos")[:torch_ops]
    )
    # A comment is erased by ``canonicalize_source`` (so the fingerprint dedups
    # it); ``tweak`` is a real statement, so it produces a genuinely different
    # kernel that still makes every strategy decision the same way.
    head = f"# {comment}\n" if comment else ""
    body_tweak = f"    {tweak}\n" if tweak else ""
    return (
        f"{head}import torch\n"
        "import triton\n"
        "import triton.language as tl\n\n"
        "@triton.jit\n"
        f"def _{name}(a_ptr, b_ptr, c_ptr, M, N, K,\n"
        "        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):\n"
        f"{body}\n\n"
        "def entry(a, b):\n"
        f"    BLOCK_M, BLOCK_N, BLOCK_K = {block_m}, {block_n}, {block_k}\n"
        f"{body_tweak}{eager}"
        "    c = torch.empty_like(a)\n"
        f"    _{name}[(1,)](a, b, c, 1, 1, 1, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,\n"
        f"        BLOCK_K=BLOCK_K, num_warps={warps}, num_stages={stages})\n"
        "    return c\n"
    )


#: Dr. Kernel's lazy-optimisation shape: one trivial sub-op in Triton, the whole
#: bottleneck left in eager torch.
LAZY_KERNEL = kernel("lazy", mode="elementwise", torch_ops=5,
                     block_m=0, block_n=0, block_k=0, warps=0, stages=0)
#: The same task, properly fused: nothing left in eager.
FUSED_KERNEL = kernel("fused", mode="mma", torch_ops=0)


class FakeTask:
    task_id = "fake_gemm_bf16"
    operation = "gemm"
    dtype = "bf16"
    gpu_target = "gfx950"
    seed_source = FUSED_KERNEL


def observation(speedup=None, correct=True, compiled=True, hack=None,
                error=None) -> Observation:
    """A verifier Observation with the fields ``compute_reward`` actually reads."""
    if not compiled:
        return Observation(compiled=False, dtype="bf16",
                           error_text=error or "SyntaxError")
    baseline = 2.0
    wall = baseline / float(speedup) if speedup else None
    return Observation(
        compiled=True, dtype="bf16", validation_passed=bool(correct),
        snr_db=40.0 if correct else 5.0,
        snr_by_shape={"primary": 40.0 if correct else 5.0},
        wall_ms=wall, baseline_ms=baseline if wall else None,
        wall_by_shape={"primary": wall} if wall else {},
        baseline_by_shape={"primary": baseline} if wall else {},
        flagged_hack=bool(hack), hack_reason=hack,
        error_text=error,
    )


class ScriptedEnv:
    """Env keyed by a ``#K:<name>`` marker in the source.

    ``plan`` maps a marker to a list of per-call outcomes; the last entry repeats
    once exhausted, so a kernel can be scripted to be lucky on its first
    measurement and mediocre afterwards. That is the exact shape of the
    noise-driven-collapse failure mode.
    """

    def __init__(self, plan: dict, default=("ok", 1.0)):
        self.plan = plan
        self.default = default
        self.calls: list[str] = []

    def _marker(self, source: str) -> str:
        for line in (source or "").splitlines():
            if line.strip().startswith("#K:"):
                return line.strip()[3:].strip()
        return "?"

    def step(self, source, full_validation=True, multi_shape=True):
        marker = self._marker(source)
        self.calls.append(marker)
        outcomes = self.plan.get(marker)
        if outcomes is None:
            outcome = self.default
        else:
            index = min(sum(1 for c in self.calls if c == marker) - 1,
                        len(outcomes) - 1)
            outcome = outcomes[index]
        kind, value = outcome
        if kind == "ok":
            return observation(speedup=value)
        if kind == "wrong":
            return observation(correct=False)
        if kind == "hack":
            return observation(speedup=value, hack="kernel never launched")
        return observation(compiled=False)


def marked(source: str, marker: str) -> str:
    return f"#K:{marker}\n{source}"


def trial_for(source: str, samples, correct=True, hack=None,
              exceeds=False) -> Trial:
    """A Trial built directly, for archive tests that need no env."""
    values = [float(s) for s in samples]
    return Trial(source=source, fingerprint=fingerprint(source), compiled=True,
                 correct=correct, samples=values,
                 trimmed=trim_outliers(values), hack_reason=hack,
                 exceeds_credible=exceeds)


def candidate(source: str, samples, generation=1, parent=None, lineage=None,
              correct=True, hack=None, exceeds=False) -> Candidate:
    return candidate_from_trial(
        trial_for(source, samples, correct=correct, hack=hack, exceeds=exceeds),
        generation, parent, lineage=lineage)


# --------------------------------------------------------------------------- #
# 1. The diversity metric
# --------------------------------------------------------------------------- #
def test_cosmetic_edits_do_not_open_a_new_niche():
    """Niche explosion is the mirror failure of collapse.

    If every candidate gets its own niche, nothing ever competes and the archive
    degenerates into "keep everything" - diversity preservation becomes a no-op
    with no selection pressure. A descriptor earns its keep only if it collides
    for kernels that are the same design.
    """
    base = kernel("k")
    same = kernel("k", comment="retuned for gfx950, see notes")
    assert strategy_signature(base) == strategy_signature(same)
    assert strategy_signature(base).distance(strategy_signature(same)) == 0


@pytest.mark.parametrize("variant,axis", [
    (dict(block_m=32, block_n=32), "tiling"),
    (dict(warps=4), "warps"),
    (dict(stages=1), "stages"),
    (dict(mode="reduce"), "compute"),
    (dict(mode="atomic"), "compute"),
    (dict(torch_ops=4), "fusion"),
])
def test_a_different_optimisation_decision_is_a_different_niche(variant, axis):
    """Each axis has to move when the decision it names moves, or it is decoration."""
    base = strategy_signature(kernel("k"))
    other = strategy_signature(kernel("k", **variant))
    assert base != other, axis
    assert getattr(base, axis) != getattr(other, axis), axis


def test_split_k_is_not_folded_into_plain_mma():
    """A split-K GEMM (tl.dot + atomics) is a distinct schedule family.

    Ranking atomics above matrix-core use in the compute precedence is what keeps
    the archive from silently dropping whichever of the two was measured second.
    """
    plain = strategy_signature(kernel("k", mode="mma"))
    split = strategy_signature(kernel("k", mode="atomic"))
    assert plain.compute != split.compute


def test_distance_is_a_metric():
    """Non-negativity, identity, symmetry and the triangle inequality.

    The archive uses this distance for crowding, eviction and farthest-point
    exemplar selection; if it is not a metric those three disagree with each
    other in ways no single test would catch.
    """
    signatures = [
        strategy_signature(kernel("k", **kw)) for kw in (
            {}, dict(warps=4), dict(mode="reduce"), dict(torch_ops=6),
            dict(block_m=32, block_n=32, stages=1), dict(mode="atomic"),
        )
    ]
    for a in signatures:
        assert a.distance(a) == 0
        for b in signatures:
            assert a.distance(b) >= 0
            assert a.distance(b) == b.distance(a)
            assert (a.distance(b) == 0) == (a == b)
            for c in signatures:
                assert a.distance(c) <= a.distance(b) + b.distance(c)


def test_the_lazy_kernel_is_far_from_the_fused_one():
    """Dr. Kernel's 0.014%-vs-86.15% pair must not share a niche.

    If it did, the archive would let a trivial rewrite evict a properly fused
    kernel of similar measured speedup and the search would lose the only member
    with real headroom.
    """
    lazy, fused = strategy_signature(LAZY_KERNEL), strategy_signature(FUSED_KERNEL)
    assert lazy != fused
    assert lazy.distance(fused) >= 4
    assert torch_op_residue(LAZY_KERNEL) >= 5
    assert torch_op_residue(FUSED_KERNEL) == 0


def test_allocation_helpers_are_not_counted_as_unfused_computation():
    """``torch.empty_like`` is how a correct Triton kernel gets its output.

    Counting plumbing as residue would flag every honest kernel as lazy, which is
    a false positive on the one signal meant to catch a real one.
    """
    assert torch_op_residue("import torch\nc = torch.empty_like(a)\n") == 0
    assert torch_op_residue("import torch\nc = torch.matmul(a, b)\n") == 1


def test_launch_count_sees_the_grid_call():
    assert launch_count(FUSED_KERNEL) == 1
    assert launch_count("def entry(a):\n    return a\n") == 0


def test_kernel_time_coverage_returns_nothing_without_a_profiler():
    """The honest-nothing rule: no measurement means no number.

    Dr. Kernel's PR is a share of measured CUDA time. Without profiler output
    there is no honest value for it, and substituting the static residue proxy
    here is precisely how a plausible-looking figure ends up quoted as measured.
    """
    assert kernel_time_coverage(None) is None
    assert kernel_time_coverage({}) is None
    assert kernel_time_coverage({"kernel_ms": 1.0}) is None
    assert kernel_time_coverage({"kernel_ms": 1.0, "total_ms": 0.0}) is None
    assert kernel_time_coverage({"kernel_ms": 0.0014, "total_ms": 10.0}) == pytest.approx(0.00014)
    assert kernel_time_coverage({"kernel_ms": 8.615, "total_ms": 10.0}) == pytest.approx(0.8615)


# --------------------------------------------------------------------------- #
# 2. Archive: diversity vs top-k by speedup
# --------------------------------------------------------------------------- #
def _converged_population():
    """Six strategically distinct kernels that have all reached 3x or better.

    This is what a search looks like once it is working: the interesting
    variation is in HOW the kernels are fast, not in whether they are.
    """
    designs = [
        (kernel("a", mode="mma", warps=8, stages=3), 3.1),
        (kernel("b", mode="mma", warps=4, stages=2), 3.4),
        (kernel("c", mode="reduce", warps=8, stages=3), 3.2),
        (kernel("d", mode="atomic", warps=8, stages=3), 3.6),
        (kernel("e", mode="mma", block_m=32, block_n=32), 3.3),
        (kernel("f", mode="mma", torch_ops=3), 3.0),
    ]
    return designs


def test_speedup_binned_archive_collapses_where_strategy_archive_does_not():
    """The measurement behind this whole module.

    ``kore.data.evolve.MapElitesArchive`` keys cells on
    ``(op_family, speedup_bin, correct)``. Within ONE task op_family is constant,
    so once the population converges every member lands in the same top speedup
    bin and the archive holds exactly one kernel - top-k by speedup with k=1.
    The strategy-keyed archive keeps every distinct design.
    """
    population = _converged_population()

    binned = MapElitesArchive()
    for source, speedup in population:
        binned.insert(source, True, speedup, 40.0, "gemm")

    strategic = Archive(capacity=32)
    for source, speedup in population:
        strategic.add(candidate(source, [speedup] * 3))

    assert binned.coverage() == 1, (
        "the speedup-binned archive was expected to collapse to one cell on a "
        f"converged single-task population, got {binned.coverage()}")
    assert strategic.coverage() == len(population)
    assert strategic.mean_pairwise_distance() > 0.0

    # The collapsed archive keeps only the fastest; the strategic one keeps the
    # slower designs that are still the only member of their region.
    kept = {c.fingerprint for c in strategic.elites()}
    assert fingerprint(population[-1][0]) in kept       # slowest, unique design
    assert len(binned.elites()) == 1


def test_a_slower_but_structurally_unique_kernel_survives():
    """The property top-k cannot have: being the only one of its kind is value."""
    archive = Archive(capacity=32)
    archive.add(candidate(kernel("fast", mode="mma"), [4.0] * 3))
    archive.add(candidate(kernel("odd", mode="atomic", warps=4), [1.4] * 3))
    survivors = {c.fingerprint for c in archive.elites()}
    assert len(survivors) == 2
    assert archive.champion().speedup_mean == pytest.approx(4.0)


def test_exemplars_buy_coverage_after_the_champion():
    """Prompt context is not the leaderboard.

    Kernel-Smith prompt from "top-performing AND diverse regions"; showing the
    model four re-tunings of one design teaches it that the design is settled.
    Only the first exemplar slot is spent on rank.
    """
    archive = Archive(capacity=32)
    # Three near-identical strong designs and one weaker, structurally distant one.
    archive.add(candidate(kernel("a", warps=8, stages=3), [4.0] * 3))
    archive.add(candidate(kernel("b", warps=8, stages=2), [3.9] * 3))
    archive.add(candidate(kernel("c", warps=4, stages=3), [3.8] * 3))
    outlier = kernel("d", mode="elementwise", torch_ops=6, block_m=32, block_n=32)
    archive.add(candidate(outlier, [1.1] * 3))

    top_by_speed = [c.fingerprint for c in archive.best(3)]
    chosen = archive.exemplars(3)
    assert chosen[0].fingerprint == archive.champion().fingerprint
    assert fingerprint(outlier) in {c.fingerprint for c in chosen}
    assert [c.fingerprint for c in chosen] != top_by_speed

    spread = min(chosen[0].signature.distance(c.signature) for c in chosen[1:])
    assert spread > 0


def test_eviction_drops_from_the_crowded_region_and_never_the_champion():
    """Evicting the globally worst member re-derives top-k as soon as it fills."""
    archive = Archive(capacity=4)
    crowd = [kernel("x", warps=w, stages=s)
             for w, s in ((8, 3), (8, 2), (4, 3), (4, 2))]
    for i, source in enumerate(crowd):
        archive.add(candidate(source, [2.0 + 0.1 * i] * 3))
    lonely = kernel("y", mode="elementwise", torch_ops=7, block_m=0, block_n=0,
                    block_k=0, warps=0, stages=0)
    archive.add(candidate(lonely, [1.0] * 3))

    assert len(archive) == 4
    kept = {c.fingerprint for c in archive.members()}
    assert fingerprint(lonely) in kept, "the sparse region was evicted"
    champion_kernel = crowd[-1]
    assert fingerprint(champion_kernel) in kept, "the champion was evicted"


def test_capacity_is_respected_under_a_long_stream():
    archive = Archive(capacity=5)
    rng = random.Random(0)
    for i in range(60):
        source = kernel(f"k{i}", warps=rng.choice([2, 4, 8, 16]),
                        stages=rng.choice([1, 2, 3, 4]),
                        mode=rng.choice(["mma", "reduce", "atomic", "elementwise"]),
                        torch_ops=rng.choice([0, 1, 3, 6]))
        archive.add(candidate(source, [1.0 + rng.random()] * 3))
        assert len(archive) <= 5


# --------------------------------------------------------------------------- #
# 3. Failure mode: reward hacking
# --------------------------------------------------------------------------- #
def test_a_1541x_candidate_is_recorded_but_never_becomes_an_elite():
    """The exact number that arrived in third-party data.

    An archive is a persistence mechanism: a hacked kernel admitted once becomes
    a parent and a few-shot exemplar for every later generation. So the guard has
    to be at admission, and it has to bar all four roles - elite, champion,
    exemplar, parent - not merely rank the kernel low.
    """
    archive = Archive(capacity=8)
    honest = kernel("honest", mode="mma")
    decoy = kernel("decoy", mode="elementwise", torch_ops=0)
    archive.add(candidate(honest, [2.0] * 3))
    decision = archive.add(candidate(decoy, [1541.0] * 3, exceeds=True))

    assert decision.verdict == "implausible"
    assert "1541" in decision.detail or "credible" in decision.detail
    assert fingerprint(decoy) not in {c.fingerprint for c in archive.elites()}
    assert archive.champion().fingerprint == fingerprint(honest)
    assert fingerprint(decoy) not in {c.fingerprint for c in archive.exemplars(8)}

    rng = random.Random(0)
    drawn = {archive.sample_parent(rng).fingerprint for _ in range(200)}
    assert fingerprint(decoy) not in drawn

    # The measurement itself is preserved, unaltered, for audit.
    rejected = [c for c in archive.rejected if c.fingerprint == fingerprint(decoy)]
    assert len(rejected) == 1
    assert rejected[0].speedup_mean == pytest.approx(1541.0)


def test_env_flagged_hack_is_reported_as_a_hack_not_as_incorrect():
    """The reason recorded has to be the earliest one that applies.

    ``compute_reward`` returns ``correct=False`` for a flagged hack, so an
    admission order that checked correctness first would file every hack as an
    ordinary wrong answer and the run log would show no hacks at all.
    """
    archive = Archive()
    decision = archive.add(candidate(kernel("h"), [], correct=False,
                                     hack="kernel never launched"))
    assert decision.verdict == "hack"
    assert archive.verdict_counts()["hack"] == 1
    assert archive.verdict_counts()["incorrect"] == 0


def test_credibility_ceiling_comes_from_the_repository_config():
    """One ceiling, set in one place: ``cfg.excessive_speedup_flag``."""
    from kore.config import CONFIG
    from kore.data.schemas import credible_speedup_max

    assert Archive().credible_max == credible_speedup_max(CONFIG)
    assert Archive().credible_max == pytest.approx(10.0)


def test_evaluator_marks_an_implausible_measurement_without_altering_it():
    env = ScriptedEnv({"decoy": [("ok", 1541.0)]})
    evaluator = StableEvaluator(env, FakeTask(), Budget(10))
    trial = evaluator.screen(marked(kernel("decoy"), "decoy"))
    assert trial.correct is True                 # the oracle really did pass
    assert trial.exceeds_credible is True        # but it is not admissible
    assert trial.speedup_mean == pytest.approx(1541.0)
    assert trial.verified is False


# --------------------------------------------------------------------------- #
# 4. Failure mode: lazy optimisation
# --------------------------------------------------------------------------- #
def test_a_within_noise_same_niche_revision_does_not_take_the_slot():
    """Lazy optimisation, seen from the population.

    A revision that is correct, in the same strategy niche as its parent, and
    within timing noise of it has changed nothing. Letting it take the slot
    churns the archive, resets stall counters, and lets a run report progress it
    did not make.
    """
    archive = Archive(capacity=8)
    parent_src = kernel("p", mode="mma")
    parent = candidate(parent_src, [2.00] * 4)
    archive.add(parent)

    nudge = candidate(kernel("p", mode="mma", tweak="stride_a = a.stride(0)"),
                      [2.02] * 4)
    assert nudge.fingerprint != parent.fingerprint      # a real, distinct kernel
    assert nudge.signature == parent.signature          # ... with no new strategy
    decision = archive.add(nudge)
    assert decision.verdict == "dominated"
    assert archive.champion().fingerprint == parent.fingerprint

    real = candidate(kernel("p", mode="mma", tweak="stride_b = b.stride(1)"),
                     [2.40] * 4)
    assert archive.add(real).verdict == "improved"


def test_the_gain_floor_is_the_step_centric_constant():
    """The step filter that BUILDS training data and the archive that GENERATES
    it must use one number, or the loop admits steps the trainer then discards.
    """
    assert Archive().min_gain == MIN_GAIN
    assert MIN_GAIN == pytest.approx(0.05)


def test_a_lazy_rewrite_does_not_displace_a_fused_kernel_of_similar_speed():
    """The concrete Dr. Kernel case, at similar measured speedup.

    Both kernels time about the same, so speedup alone cannot separate them. The
    strategy descriptor puts them in different niches, so the fused kernel - the
    one with the remaining headroom - is never evicted by the trivial rewrite.
    """
    archive = Archive(capacity=8)
    archive.add(candidate(FUSED_KERNEL, [1.30] * 3))
    archive.add(candidate(LAZY_KERNEL, [1.32] * 3))
    kept = {c.fingerprint for c in archive.elites()}
    assert fingerprint(FUSED_KERNEL) in kept
    assert fingerprint(LAZY_KERNEL) in kept
    assert archive.coverage() == 2


# --------------------------------------------------------------------------- #
# 5. Failure mode: population collapse
# --------------------------------------------------------------------------- #
def test_collapse_is_stagnation_not_a_young_run():
    """A run that is still opening or beating niches is not collapsed.

    The fail-safe direction matters: before a full window of admissions exists
    there is no evidence either way, and forcing exploration on absent evidence
    would throw away the productive early generations.
    """
    archive = Archive(capacity=16)
    archive.add(candidate(kernel("r"), [1.0] * 3, lineage="L1"))
    assert archive.recent_progress() == pytest.approx(1.0)   # no evidence yet
    assert not archive.collapsed()

    for i in range(8):
        archive.add(candidate(kernel(f"c{i}", warps=[2, 4, 8, 16][i % 4],
                                     stages=[1, 2, 3, 4][i % 4]),
                              [1.0 + i * 0.3] * 3, lineage="L1"))
    assert archive.recent_novelty() > 0.0
    assert archive.recent_progress() > 0.0
    assert not archive.collapsed()

    # Now nothing lands: repeated dominated insertions in occupied niches.
    for i in range(archive.novelty_window + 2):
        archive.add(candidate(kernel("c0", warps=2, stages=1,
                                     tweak=f"pad_{i} = {i}"),
                              [0.4] * 3, lineage="L1"))
    assert archive.recent_progress() == 0.0
    assert archive.collapsed()

    # An improvement inside an occupied niche is progress even at zero novelty,
    # so it clears the trip without opening any new territory.
    for i in range(archive.novelty_window):
        archive.add(candidate(kernel("c0", warps=2, stages=1,
                                     tweak=f"real_{i} = {i}"),
                              [5.0 + i] * 3, lineage="L1"))
    assert archive.recent_novelty() == 0.0
    assert archive.recent_progress() > 0.0
    assert not archive.collapsed()


def test_lineage_concentration_sees_what_coverage_cannot():
    """Descendants of one early commitment can occupy many niches.

    Coverage would call that a healthy run. It is single-trajectory anchoring
    with extra steps, which is the failure the archive exists to break.
    """
    archive = Archive(capacity=16)
    for i in range(6):
        archive.add(candidate(kernel(f"k{i}", warps=[2, 4, 8, 16][i % 4],
                                     mode=["mma", "reduce", "atomic"][i % 3]),
                              [2.0] * 3, lineage="one-branch"))
    assert archive.coverage() >= 5
    assert archive.lineage_concentration() == pytest.approx(1.0)


def test_forced_exploration_stops_drawing_the_incumbent():
    """The collapse trip has to change behaviour, not just log a metric."""
    archive = Archive(capacity=16, elite_band=1)
    champ = kernel("champ", mode="mma")
    archive.add(candidate(champ, [9.0] * 4))
    for i in range(5):
        archive.add(candidate(kernel(f"o{i}", warps=[2, 4, 16][i % 3],
                                     mode=["reduce", "atomic", "elementwise"][i % 3],
                                     torch_ops=i),
                              [1.0] * 4))

    rng = random.Random(7)
    greedy = [archive.sample_parent(rng, p_elite=1.0).fingerprint
              for _ in range(50)]
    assert set(greedy) == {fingerprint(champ)}

    rng = random.Random(7)
    explored = [archive.sample_parent(rng, p_elite=1.0, force_explore=True).fingerprint
                for _ in range(50)]
    assert len(set(explored)) > 1


def test_parent_sampling_is_not_argmax():
    archive = Archive(capacity=16, elite_band=3)
    for i in range(6):
        archive.add(candidate(kernel(f"k{i}", warps=[2, 4, 8, 16][i % 4],
                                     mode=["mma", "reduce", "atomic"][i % 3]),
                              [1.0 + i] * 3))
    rng = random.Random(1)
    drawn = {archive.sample_parent(rng).fingerprint for _ in range(200)}
    assert len(drawn) >= 3


def test_archive_is_deterministic_for_a_seed():
    def run(seed):
        archive = Archive(capacity=8)
        rng = random.Random(seed)
        for i in range(10):
            archive.add(candidate(kernel(f"k{i}", warps=[2, 4, 8][i % 3]),
                                  [1.0 + 0.2 * i] * 3))
        return [archive.sample_parent(rng).fingerprint for _ in range(20)]

    assert run(3) == run(3)


# --------------------------------------------------------------------------- #
# 6. Failure mode: noise-driven collapse (stable evaluation)
# --------------------------------------------------------------------------- #
def test_one_lucky_sample_would_win_but_the_lcb_refuses():
    """Kernel-Smith §3.3, made concrete.

    ``noisy`` measures 3.0x once and ~0.5x afterwards; ``stable`` measures 1.5x
    every time. On a single screening sample the noisy kernel wins by 2x and
    becomes the elite, the parent and the exemplar for the rest of the run. The
    test asserts BOTH halves: that single-sample ranking really would pick the
    wrong one, and that repeated measurement plus a pessimistic LCB does not.
    """
    plan = {
        "noisy": [("ok", 3.0), ("ok", 0.5), ("ok", 0.4), ("ok", 0.5), ("ok", 0.45)],
        "stable": [("ok", 1.5)],
    }
    env = ScriptedEnv(plan)
    evaluator = StableEvaluator(env, FakeTask(), Budget(40), min_for_trim=99)
    noisy = marked(kernel("noisy", mode="mma"), "noisy")
    stable = marked(kernel("stable", mode="reduce"), "stable")

    screens = [evaluator.screen(noisy), evaluator.screen(stable)]
    # The counterfactual: one look each, and the wrong kernel is ahead.
    assert screens[0].speedup_mean > screens[1].speedup_mean

    evaluator.stabilize(screens, min_measures=4)
    archive = Archive(capacity=8)
    for trial in screens:
        archive.add(candidate_from_trial(trial, 1, None))

    champion = archive.champion()
    assert champion.fingerprint == fingerprint(stable), (
        "a single lucky timing was allowed to install the wrong elite; "
        f"noisy mean={screens[0].speedup_mean}, stable mean={screens[1].speedup_mean}")
    assert champion.speedup_lcb == pytest.approx(1.5)


def test_lcb_penalises_variance_at_equal_means():
    """The ordering property the archive relies on, in isolation."""
    steady, jittery = MeasureStats(), MeasureStats()
    for value in (2.0, 2.0, 2.0, 2.0):
        steady.add(value)
    for value in (0.5, 3.5, 0.5, 3.5):
        jittery.add(value)
    assert steady.mean == pytest.approx(jittery.mean)
    assert steady.lcb > jittery.lcb


def test_a_kernel_that_verifies_once_and_not_again_is_dropped():
    """Non-determinism is a correctness failure, not a slow kernel.

    Averaging the passing run with the failing one would let a kernel that passes
    the oracle by luck accumulate archive tenure.
    """
    env = ScriptedEnv({"flaky": [("ok", 2.0), ("wrong", None)]})
    evaluator = StableEvaluator(env, FakeTask(), Budget(20))
    trial = evaluator.screen(marked(kernel("flaky"), "flaky"))
    assert trial.correct is True
    evaluator.stabilize([trial], min_measures=2)
    assert trial.correct is False
    archive = Archive()
    assert archive.add(candidate_from_trial(trial, 1, None)).verdict == "incorrect"


def test_trim_refuses_to_trim_below_five_samples():
    """Dropping the extremes of three samples leaves one and a variance of zero.

    A confidently wrong LCB is worse than a wide honest one.
    """
    assert trim_outliers([1.0, 5.0, 2.0]) == [1.0, 2.0, 5.0]
    assert trim_outliers([1.0, 5.0, 2.0, 3.0]) == [1.0, 2.0, 3.0, 5.0]
    assert trim_outliers([1.0, 5.0, 2.0, 3.0, 4.0]) == [2.0, 3.0, 4.0]
    assert trim_outliers([]) == []


def test_the_verifier_budget_is_a_hard_cap():
    """Anytime means anytime: the cap is a fact, not an intention."""
    env = ScriptedEnv({}, default=("ok", 1.2))
    budget = Budget(5)
    evaluator = StableEvaluator(env, FakeTask(), budget)
    trials = [evaluator.screen(marked(kernel(f"k{i}"), f"k{i}")) for i in range(10)]
    evaluator.stabilize(trials, min_measures=4)
    assert budget.used <= 5
    assert evaluator.env_calls <= 5
    assert len(env.calls) <= 5


def test_a_budget_stop_is_not_recorded_as_a_wrong_kernel():
    """"We did not measure it" and "it was wrong" are different facts.

    Filing a budget stop under the incorrect count puts a verdict about a kernel
    into the run log that no oracle ever produced - the same defect as a
    fabricated timing, wearing a different hat. It also silently inflates the
    failure rate of whatever the run was truncated in the middle of.
    """
    env = ScriptedEnv({}, default=("ok", 1.5))
    evaluator = StableEvaluator(env, FakeTask(), Budget(1))
    first = evaluator.screen(marked(kernel("a"), "a"))
    stopped = evaluator.screen(marked(kernel("b"), "b"))

    assert first.correct is True and first.budget_exhausted is False
    assert stopped.budget_exhausted is True
    assert stopped.verified is False
    assert len(env.calls) == 1, "a stopped evaluation must not touch the verifier"

    archive = Archive()
    archive.add(candidate_from_trial(first, 1, None))
    decision = archive.add(candidate_from_trial(stopped, 1, None))
    assert decision.verdict == "unmeasured"
    assert archive.verdict_counts()["incorrect"] == 0
    assert archive.rejected == []            # not a guard rejection
    assert len(archive.unmeasured) == 1
    assert archive.summary()["unmeasured"] == 1
    assert decision.candidate.admissible is False


def test_a_crashing_verifier_is_a_datum_not_an_exception():
    class Exploding:
        def step(self, *a, **kw):
            raise RuntimeError("hipErrorIllegalAddress")

    evaluator = StableEvaluator(Exploding(), FakeTask(), Budget(4))
    trial = evaluator.screen(kernel("boom"))
    assert trial.compiled is False and trial.correct is False
    assert "hipErrorIllegalAddress" in (trial.error_text or "")


# --------------------------------------------------------------------------- #
# 7. Sequential test-time scaling
# --------------------------------------------------------------------------- #
def _step(value, verified=True, generation=1, turn=0):
    return ScaledStep(generation=generation, turn=turn,
                      fingerprint=f"fp{value}", speedup=value, verified=verified)


def test_best_of_history_curve_is_monotone_and_equals_the_running_max():
    scaling = BestAcrossSteps()
    values = [1.0, 2.5, 1.2, 3.1, 0.4, 2.9]
    for value in values:
        scaling.offer(_step(value))
    assert scaling.curve == sorted(scaling.curve)
    assert scaling.best_speedup == pytest.approx(max(values))
    running = []
    best = 0.0
    for value in values:
        best = max(best, value)
        running.append(best)
    assert scaling.curve == pytest.approx(running)


def test_best_of_history_beats_reporting_the_last_step():
    """Dr. Kernel's Table 1 in miniature: 31.6 last-turn vs 47.8 best-of-history.

    The search already paid for the intermediate kernels; reporting the final one
    throws the result away.
    """
    scaling = BestAcrossSteps()
    trajectory = [1.0, 2.2, 3.4, 1.1]     # the last step regresses
    for i, value in enumerate(trajectory):
        scaling.offer(_step(value, generation=i))
    assert scaling.best_speedup == pytest.approx(3.4)
    assert scaling.best.generation == 2
    assert trajectory[-1] < scaling.best_speedup


def test_an_unverified_step_can_never_lift_the_curve():
    """Anything that failed, was flagged, or was never measured is not a result."""
    scaling = BestAcrossSteps()
    scaling.offer(_step(2.0))
    for bad in (_step(99.0, verified=False),
                ScaledStep(1, 0, "fp", None, True),
                ScaledStep(1, 0, "fp", float("inf"), True),
                ScaledStep(1, 0, "fp", -1.0, True)):
        assert scaling.offer(bad) is False
    assert scaling.best_speedup == pytest.approx(2.0)
    assert scaling.rejected == 4


# --------------------------------------------------------------------------- #
# 8. The loop end to end
# --------------------------------------------------------------------------- #
class ScriptedProposer:
    """Returns a fixed list of sources per generation; records what it was given."""

    def __init__(self, script):
        self.script = script
        self.seen_parents: list = []
        self.seen_exemplars: list = []

    def propose(self, task, env, parent, exemplars, generation):
        self.seen_parents.append(parent.fingerprint if parent else None)
        self.seen_exemplars.append([c.fingerprint for c in exemplars])
        sources = self.script.get(generation, [])
        return [Proposal(source=s, turn=i) for i, s in enumerate(sources)]


def test_evolve_end_to_end_admits_only_stabilised_measurements():
    script = {
        1: [marked(kernel("a", mode="mma"), "a"),
            marked(kernel("b", mode="reduce"), "b")],
        2: [marked(kernel("c", mode="atomic"), "c"),
            marked(kernel("d", mode="elementwise", warps=2), "d"),
            marked(kernel("e", mode="elementwise", warps=16), "e")],
    }
    plan = {
        "seed": [("ok", 1.0)],
        "a": [("ok", 2.0)],
        "b": [("ok", 1.4)],
        "c": [("ok", 2.6)],
        "d": [("ok", 1541.0)],          # the decoy: verified, but not credible
        "e": [("hack", 8.0)],           # flagged by the env's own scanner
    }
    env = ScriptedEnv(plan, default=("ok", 1.0))
    task = FakeTask()
    task.seed_source = marked(kernel("seed", mode="mma", warps=4), "seed")
    result = evolve(task, ScriptedProposer(script), env,
                    EvolveAgentConfig(generations=2, max_env_calls=120, seed=0))

    assert result.best is not None
    assert result.best.speedup_mean == pytest.approx(2.6)
    assert result.stats["rejected_implausible"] == 1
    assert result.stats["rejected_hacks"] == 1
    assert result.scaling.best_speedup == pytest.approx(2.6)
    # Neither disqualified kernel reaches the reported result by any route.
    for source in (script[2][1], script[2][2]):
        bad = fingerprint(source)
        assert bad not in {c.fingerprint for c in result.archive.elites()}
        assert bad not in {s.fingerprint for s in result.scaling.steps}
    assert json.loads(json.dumps(result.to_dict()))["archive"]["coverage"] >= 3


def test_evolve_never_exceeds_its_verifier_budget():
    script = {g: [marked(kernel(f"g{g}i{i}", warps=[2, 4, 8][i % 3]), f"g{g}i{i}")
                  for i in range(4)] for g in range(1, 9)}
    env = ScriptedEnv({}, default=("ok", 1.3))
    result = evolve(FakeTask(), ScriptedProposer(script), env,
                    EvolveAgentConfig(generations=8, max_env_calls=25, seed=0))
    assert result.env_calls <= 25
    assert result.stats["budget_used"] <= 25
    assert len(result.generations) <= 8


def test_evolve_is_deterministic_for_a_seed():
    script = {g: [marked(kernel(f"g{g}i{i}", warps=[2, 4, 8][i % 3],
                                mode=["mma", "reduce"][i % 2]), f"g{g}i{i}")
                  for i in range(3)] for g in range(1, 5)}

    def run():
        env = ScriptedEnv({}, default=("ok", 1.7))
        return evolve(FakeTask(), ScriptedProposer(script), env,
                      EvolveAgentConfig(generations=4, max_env_calls=200, seed=11))

    a, b = run(), run()
    assert [c.fingerprint for c in a.archive.elites()] == \
           [c.fingerprint for c in b.archive.elites()]
    assert a.scaling.curve == b.scaling.curve
    assert a.env_calls == b.env_calls


def test_a_lineage_is_a_branch_not_descent_from_the_seed():
    """Every candidate descends from the seed, so that cannot be the lineage.

    Inheriting through the seed pegs ``lineage_concentration`` at 1.0 for every
    run and silently reduces the collapse trip to its novelty half - a guard that
    reports a number it can never fail.
    """
    seed = candidate_from_trial(trial_for(kernel("seed"), [1.0] * 3), 0, None)
    first = [candidate_from_trial(trial_for(kernel(f"b{i}", warps=[2, 4][i]),
                                            [2.0] * 3), 1, seed)
             for i in range(2)]
    assert first[0].lineage != first[1].lineage, "gen-1 branches were merged"
    assert first[0].lineage != seed.lineage

    child = candidate_from_trial(trial_for(kernel("c", warps=2, stages=1),
                                           [2.5] * 3), 2, first[0])
    assert child.lineage == first[0].lineage, "a descendant left its branch"

    archive = Archive(capacity=8)
    for cand in (seed, *first, child):
        archive.add(cand)
    assert archive.lineage_concentration() < 1.0


def test_the_collapse_trip_actually_fires_in_a_run():
    """Archive-level unit tests do not prove the loop ever reaches the branch.

    A proposer that only ever re-tunes one design inside one lineage is exactly
    the anchored single trajectory this module exists to break, so a run against
    it has to end up forcing exploration.
    """
    # Every proposal is a distinct source in the SAME strategy niche.
    script = {g: [marked(kernel("same", tweak=f"pad_{g}_{i} = {i}"), f"s{g}{i}")
                  for i in range(2)] for g in range(1, 12)}
    env = ScriptedEnv({}, default=("ok", 1.2))
    task = FakeTask()
    task.seed_source = marked(kernel("seed", mode="reduce"), "seed")
    result = evolve(task, ScriptedProposer(script), env,
                    EvolveAgentConfig(generations=11, max_env_calls=400, seed=0))

    assert any(record.forced_explore for record in result.generations), (
        "the run never detected collapse; "
        f"lineage={result.archive.lineage_concentration()} "
        f"novelty={result.archive.recent_novelty()} "
        f"coverage={result.archive.coverage()}")


def test_evolve_feeds_the_archive_back_into_the_proposer():
    """Parent and exemplars must come from the archive, or it is decoration."""
    script = {g: [marked(kernel(f"g{g}", mode=["mma", "reduce", "atomic"][g % 3]),
                         f"g{g}")] for g in range(1, 5)}
    env = ScriptedEnv({}, default=("ok", 1.5))
    task = FakeTask()
    task.seed_source = marked(kernel("seed"), "seed")
    proposer = ScriptedProposer(script)
    result = evolve(task, proposer, env,
                    EvolveAgentConfig(generations=4, max_env_calls=200, seed=2))
    assert proposer.seen_parents[0] is not None
    assert any(proposer.seen_exemplars[-1])
    known = {c.fingerprint for c in result.archive.members()}
    assert set(proposer.seen_exemplars[-1]) <= known


# --------------------------------------------------------------------------- #
# 9. The production proposer: the harness as the mutation operator
# --------------------------------------------------------------------------- #
def _tool_call(name, arguments):
    return f'<tool_call>\n{json.dumps({"name": name, "arguments": arguments})}\n</tool_call>'


def scripted_model(script):
    state = {"i": 0}

    def fn(_messages):
        i = state["i"]
        state["i"] = i + 1
        return script[i] if i < len(script) else "no further changes."

    return StubTeacher(fn=fn)


def test_harness_proposer_returns_every_turns_kernel():
    """Best-of-history needs the intermediate kernels, not just the episode best."""
    first = marked(kernel("t1", mode="mma"), "t1")
    second = marked(kernel("t2", mode="reduce"), "t2")
    model = scripted_model([
        _tool_call("bench", {"kernel_src": first}),
        _tool_call("bench", {"kernel_src": second}),
    ])
    env = ScriptedEnv({"t1": [("ok", 1.4)], "t2": [("ok", 2.2)]})
    proposals = HarnessProposer(model, max_turns=2).propose(
        FakeTask(), env, None, [], 1)
    sources = [p.source for p in proposals]
    assert first in sources and second in sources
    assert [p.turn for p in proposals] == sorted(p.turn for p in proposals)


def test_archive_exemplars_reach_the_model_prompt():
    """The archive's diverse pool has to arrive as context or it changes nothing."""
    archive = Archive(capacity=8)
    fast = kernel("fast", mode="mma")
    odd = kernel("odd", mode="atomic", warps=2, torch_ops=3)
    archive.add(candidate(fast, [3.0] * 3))
    archive.add(candidate(odd, [1.2] * 3))

    proposer = HarnessProposer(scripted_model([]), max_turns=1)
    kb = proposer._kb(FakeTask(), archive.exemplars(2))
    assert isinstance(kb, WinsKB)
    retrieved = kb.retrieve("gemm", "bf16", k=2)
    assert {entry["final_source"] for entry in retrieved} == {fast, odd}

    harness = AgentHarness(FakeTask(), scripted_model([]),
                           ScriptedEnv({}), max_turns=1, seed_src=fast,
                           kb=kb, kb_top_k=2)
    prompt = harness.run().messages[1]["content"]
    assert odd in prompt, "the diverse exemplar never reached the prompt"


def test_evolve_with_the_real_harness_proposer():
    """Whole stack on fakes: archive -> prompt -> harness -> env -> archive."""
    better = marked(kernel("better", mode="reduce", warps=4), "better")
    model = scripted_model([_tool_call("bench", {"kernel_src": better})] * 8)
    env = ScriptedEnv({"seed": [("ok", 1.0)], "better": [("ok", 2.5)]})
    task = FakeTask()
    task.seed_source = marked(kernel("seed", mode="mma"), "seed")

    result = evolve(task, HarnessProposer(model, max_turns=2), env,
                    EvolveAgentConfig(generations=2, max_env_calls=200, seed=0))
    assert result.best is not None
    assert result.best.speedup_mean == pytest.approx(2.5)
    assert result.scaling.best_speedup == pytest.approx(2.5)
    assert result.archive.coverage() >= 2

    # The harness runs its own tool loop, so its verifier traffic is counted
    # rather than gated. It has to be counted SEPARATELY: reporting one env_calls
    # number would imply the cap governs traffic it cannot refuse.
    assert result.stats["proposer_env_calls"] > 0
    assert result.stats["measurement_env_calls"] > 0
    assert result.stats["env_calls"] == (result.stats["proposer_env_calls"]
                                         + result.stats["measurement_env_calls"])
    assert result.env_calls == len(env.calls)
    assert result.stats["budget_used"] >= result.stats["proposer_env_calls"]


def test_a_proposers_uncounted_traffic_cannot_hide_from_the_budget():
    """A generation that spends the whole cap inside the harness must stop the run.

    Without charging the proposer's calls, an agent that benches heavily inside
    its own loop would run every configured generation regardless of the cap and
    the reported budget would be fiction.
    """
    busy = [marked(kernel(f"b{i}", warps=[2, 4, 8][i % 3]), f"b{i}") for i in range(6)]
    model = scripted_model([_tool_call("bench", {"kernel_src": s}) for s in busy] * 4)
    env = ScriptedEnv({}, default=("ok", 1.1))
    task = FakeTask()
    task.seed_source = marked(kernel("seed"), "seed")

    result = evolve(task, HarnessProposer(model, max_turns=6), env,
                    EvolveAgentConfig(generations=6, max_env_calls=12,
                                      generation_reserve=6, seed=0))
    assert len(result.generations) < 6, "the cap did not stop the run"
    assert result.stats["budget_used"] == result.stats["budget_total"]


def test_a_correct_kernel_with_no_timing_is_unmeasured_not_incorrect():
    """A missing timing sample is a measurement failure, not a wrong kernel.

    ``StableEvaluator``'s sampler used to condemn a trial on
    ``not result.correct or result.speedup is None``, which folds two unrelated
    outcomes together. The first is a real finding -- a kernel that verified once
    and not again is unstable. The second is the node being busy.

    It matters because contention is the normal case in production: every rank
    benches at once, and on gfx950 a run whose cv_pct rose past ~10% had all
    eight of its proposals recorded as 'incorrect' while the env had reported
    compiled=True correct=True for every one of them. The archive then refuses
    them forever (``admissible`` requires ``correct``), so the search discards
    good designs because of noise and the run log blames the model.

    The scripted env below returns correct-but-untimed on every call, which is
    exactly that situation.
    """
    class NoTimingEnv:
        def __init__(self):
            self.calls = 0

        def step(self, source, full_validation=True, multi_shape=True):
            self.calls += 1
            return observation(speedup=None, correct=True)

    env = NoTimingEnv()
    evaluator = StableEvaluator(env, FakeTask(), Budget(8))
    trial = evaluator.screen(marked(kernel("a"), "a"))

    # Screening saw a correct kernel and must not have relabelled it.
    assert trial.correct is True, (
        "a correct kernel that produced no timing was marked incorrect; that is "
        "the conflation this test exists to prevent")
    assert trial.error_text != "unstable correctness"

    # In the archive it must not be counted as a wrong answer. It is also not an
    # elite -- with no samples ``stats.n`` is 0, so ``admissible`` is False and it
    # cannot become a champion, parent or exemplar on the strength of a
    # measurement that never happened. Held out of the elite pool, not condemned.
    archive = Archive()
    cand = candidate_from_trial(trial, 1, None)
    archive.add(cand)
    assert archive.verdict_counts()["incorrect"] == 0
    assert cand.admissible is False
    assert archive.champion() is None
