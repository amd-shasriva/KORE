"""Evolve-agent: population-based test-time kernel optimisation over ``KoreEnv``.

This is the search half of the Kernel-Smith recipe (arXiv 2603.28342, "A Unified
Recipe for Evolutionary Kernel Optimization"). Their claim is specific and it is
the one this project is betting on: under a *unified* evolutionary protocol -
same agent scaffold for every model - a 235B specialist reaches 3.70 average
speedup on KernelBench Triton against Claude-4.6-opus at 3.33, and their 30B MACA
model beats DeepSeek-V3.2-think and Qwen3-235B. The scaffold is the multiplier;
our product model is 30B-class, so the scaffold is where the leverage is.

The failure mode this module exists to prevent is stated plainly in their §1: a
multi-turn refinement loop "can anchor later proposals to early decisions and
limit exploration diversity". :class:`kore.agent.harness.AgentHarness` is exactly
such a loop - one trajectory, one lineage, a reseed only after a stall. It is a
good LOCAL IMPROVER. This module makes it the mutation operator of a search that
keeps more than one lineage alive, so a bad early commitment costs one branch
rather than the run.

Four things have to be true for that to work, and each maps to a section below.

1. **The archive must preserve diversity, not top-k by speedup.**
   Kernel-Smith prompt the model with "archived candidates sampled from both
   top-performing and diverse regions of the search space" (§3.2), citing
   MAP-Elites. This repository already has a MAP-Elites archive
   (``kore.data.evolve.MapElitesArchive``) whose cell key is
   ``(op_family, speedup_bin, correct)`` - but *within a single task* op_family
   is constant, so the key degenerates to ``(speedup_bin, correct)``: about six
   cells, converging to one as the search improves. That archive is top-k by
   speedup wearing a MAP-Elites hat, which is why this module defines its own
   descriptor over OPTIMISATION STRATEGY (:class:`StrategySignature`) instead of
   over performance. ``tests/test_evolve_agent.py`` measures both on the same
   candidate stream; the speedup-binned archive collapses to a single occupied
   cell and this one does not.

2. **Evaluation must be stable, or the evolutionary dynamics are noise.**
   Kernel-Smith §3.3: profiling noise makes the search "preserve suboptimal
   kernels or eliminate genuinely promising ones, and such mistakes compound
   across generations". A single cold-cache timing on a shared MI355X moves
   several percent, so one lucky sample can install a mediocre kernel as the
   elite of its niche and every later generation inherits it. Admission is
   therefore by pessimistic LCB over REPEATED measurements
   (:class:`StableEvaluator`, reusing ``kore.search.bandit``), never by a single
   screening number.

3. **Test-time compute must be spent on the best-of-history, not the last turn.**
   Dr. Kernel (arXiv 2602.05885, ICML 2026) get KernelBench L2 Fast@1.2 from
   31.6 to **47.8** purely by selecting the best turn across history instead of
   the final one, at a fixed model. :class:`BestAcrossSteps` is that selector,
   and it refuses unverified steps so the reported best is always a measured
   kernel (§ "Sequential test-time scaling").

4. **The guards have to hold under an archive, which is a persistence mechanism.**
   A hacked kernel that reaches the archive is not one bad sample: it becomes a
   parent and a few-shot exemplar, so it is re-proposed for the rest of the run.
   Three guards, each tied to an incident this project or its sources already
   hit, are enforced at ADMISSION (see :meth:`Archive.add`):

   * *reward hacking* - third-party data handed us a claimed **1541x** kernel
     that never ran; real fused wins on this hardware are 1-10x. Anything the
     env flags, or anything above ``credible_speedup_max`` (10.0), is recorded
     with its measured value and permanently barred from elite/parent/exemplar
     status. Kernel-Smith §3.3 and Dr. Kernel §2 both report the same class.
   * *lazy optimisation* - Dr. Kernel's example is a kernel covering 0.014% of
     CUDA time where proper fusion covers 86.15%. The population-level symptom
     is a revision that is correct, in the SAME strategy niche as its parent,
     and within timing noise of it. Such a revision may not displace its parent
     (``MIN_GAIN``, imported from :mod:`kore.data.step_centric` so the two stages
     cannot drift apart) and does not count as progress.
   * *population collapse* - measured, not assumed:
     :meth:`Archive.lineage_concentration` and :meth:`Archive.recent_novelty`.
     When both trip, parent selection is forced off the incumbent lineage.

Everything here is pure CPU orchestration: the only GPU contact is through the
injected ``env`` and the injected proposer, so the whole loop runs against a
scripted fake in the unit tests. Nothing in this module may relax a correctness
oracle or a timing gate - correctness and speedup come from
``kore.reward.compute_reward`` on a ``KoreEnv`` ``Observation``, unmodified, and
a measurement that cannot be made honestly yields ``None`` rather than a
plausible number.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from kore.config import CONFIG
from kore.data.schemas import credible_speedup_max
from kore.data.step_centric import MIN_GAIN
from kore.obs import get_logger
from kore.reward.reward import Observation, compute_reward
from kore.search.alphakernel import canonicalize_source, fingerprint
from kore.search.bandit import Budget, CallbackArm, MeasureStats, successive_halving
from kore.value.features import extract_schedule_features

log = get_logger("search.evolve_agent")


# --------------------------------------------------------------------------- #
# 1. The diversity metric
#
# The archive's descriptor is the single load-bearing design choice in this
# module, so the argument for it is written out rather than implied.
#
# What the descriptor has to buy us: a population collapses when every surviving
# member is a re-tuning of one design, because then every mutation lands in a
# neighbourhood the search has already exhausted. The descriptor's job is to say
# "these two kernels are the same DESIGN" so that they compete for one slot, and
# "these two are different designs" so that both survive.
#
# Three candidate metrics, and why this one:
#
#   * AST / structural edit distance. Rejected. It answers the wrong question in
#     both directions. Two kernels that differ by a hundred AST nodes of
#     index arithmetic but share tile shape, warp count and memory strategy are
#     the same design - mutating either explores the same region - yet edit
#     distance calls them far apart. Conversely a kernel that swaps `tl.dot` for
#     a manual FMA loop moves to a different roofline, a different occupancy
#     regime and a different failure mode while barely moving in AST distance.
#     It is also inherently pairwise, so it forces either O(n^2) novelty scoring
#     or a tuned threshold; a discrete descriptor gives niching for free.
#
#   * Operator mix (which torch/tl ops appear). Rejected as the PRIMARY axis, but
#     kept as one: for a fixed task the operator set is largely fixed by the
#     reference, so on its own it barely varies. Its one genuinely informative
#     component - how much computation is left in eager torch - is kept as the
#     `fusion` axis precisely because that is Dr. Kernel's lazy-optimisation axis.
#
#   * Optimisation-strategy signature. Chosen. The axes are the decisions a
#     performance engineer actually makes and that a Triton autotuner actually
#     sweeps: how much is fused into the kernel, the tile shape, the warp count,
#     the pipeline depth, and which compute path the inner loop takes. These are
#     the coordinates the search is searching over, so distance in this space is
#     distance in the space that matters. They are extractable statically and
#     deterministically from source (`kore.value.features.extract_schedule_features`,
#     already the value model's featurizer - the archive and the ranker therefore
#     agree on what a schedule IS), and they are integers, so niching is O(1) and
#     coverage is a number you can report.
#
# Cost, stated honestly: this is a syntactic descriptor. A kernel that computes
# something completely different with identical schedule knobs lands in the same
# niche and the two compete on measured fitness. That is the correct trade for an
# archive whose members are all correct implementations of ONE task - correctness
# is already established by the oracle before anything is inserted, so the only
# remaining question is which schedule region a candidate occupies.
# --------------------------------------------------------------------------- #

#: Compute-path classes for the inner loop, in precedence order (highest first).
#: Atomics outrank matrix-core use deliberately: a split-K GEMM (``tl.dot`` plus
#: ``tl.atomic_add``) is a different schedule family from a non-split-K GEMM even
#: though both issue MFMA, and keeping them in one niche would let the archive
#: silently drop whichever was measured second.
COMPUTE_NONE = 0          # no Triton kernel in the source at all
COMPUTE_ELEMENTWISE = 1   # kernel with no reduction, no dot, no atomics
COMPUTE_REDUCTION = 2     # reduction/contraction loop, scalar math
COMPUTE_MMA = 3           # tl.dot / MFMA matrix-core path
COMPUTE_ATOMIC = 4        # atomics: split-K, scatter, cross-program accumulation

_JIT = re.compile(r"@\s*triton\s*\.\s*jit\b")
#: ``name[grid](...)`` - a Triton launch. Deliberately narrow: a subscript
#: immediately followed by a call is not a construct honest host code uses for
#: anything else in these kernels.
_LAUNCH = re.compile(r"\b\w+\s*\[[^\]\n]*\]\s*\(")
_TORCH_CALL = re.compile(r"\b(?:torch|F|nn\.functional|torch\.nn\.functional)\.(\w+)\s*\(")

#: ``torch.<name>`` calls that are plumbing rather than computation. An honest
#: Triton kernel has to allocate its output and read shapes/strides, so counting
#: those as "computation left in eager" would flag every correct kernel as lazy.
#: Anything NOT on this list is arithmetic the candidate declined to fuse.
_TORCH_PLUMBING = frozenset({
    "empty", "empty_like", "empty_strided", "zeros", "zeros_like", "ones",
    "ones_like", "full", "full_like", "as_tensor", "tensor", "from_numpy",
    "device", "cuda", "contiguous", "view", "reshape", "permute", "transpose",
    "stride", "size", "numel", "dim", "shape", "to", "type", "float", "half",
    "bfloat16", "int", "long", "is_contiguous", "cdiv", "next_power_of_2",
    "get_default_dtype", "set_grad_enabled", "no_grad", "inference_mode",
})


@dataclass(frozen=True)
class StrategySignature:
    """A kernel's optimisation strategy as five integer coordinates.

    ``fusion``   0-3  how much computation stays in eager torch (0 = none left).
    ``tiling``   0-4  log-bucketed tile area; 0 when no tile constants exist.
    ``warps``    0-4  declared ``num_warps`` bucket; 0 when undeclared.
    ``stages``   0-3  declared ``num_stages`` bucket; 0 when undeclared.
    ``compute``  0-4  one of the ``COMPUTE_*`` classes.

    L1 distance over these coordinates (:meth:`distance`) is the archive's
    diversity metric. It is a genuine metric on the coordinate space, it is
    integer-valued so crowding is countable, and each unit step is a decision a
    kernel author would describe as "a different approach".
    """

    fusion: int
    tiling: int
    warps: int
    stages: int
    compute: int

    def as_tuple(self) -> tuple:
        return (self.fusion, self.tiling, self.warps, self.stages, self.compute)

    def distance(self, other: "StrategySignature") -> int:
        return sum(abs(a - b) for a, b in zip(self.as_tuple(), other.as_tuple()))


def torch_op_residue(source: str) -> int:
    """Count arithmetic ``torch.*`` calls the candidate left outside its kernels.

    This is a STATIC PROXY for Dr. Kernel's Profiling-based Reward - the fraction
    of CUDA time covered by generated kernels (0.014% in their lazy case, 86.15%
    with proper fusion). It is deliberately not called a coverage number: it is
    a count of unfused ops, not a measurement of runtime, and
    :func:`kernel_time_coverage` returns ``None`` rather than fabricate the real
    one when no profiler data exists.
    """
    src = canonicalize_source(source or "")
    return sum(1 for m in _TORCH_CALL.finditer(src)
               if m.group(1) not in _TORCH_PLUMBING)


def kernel_time_coverage(profile: Optional[dict]) -> Optional[float]:
    """Dr. Kernel's PR: share of measured device time inside generated kernels.

    ``profile`` must carry ``kernel_ms`` and ``total_ms`` from a real profiler
    run. Without both this returns ``None`` - the loop then reports no coverage
    at all rather than substituting the static proxy, because a plausible number
    in the slot where a measurement belongs is how the 1541x claim survived.
    """
    if not isinstance(profile, dict):
        return None
    kernel_ms = profile.get("kernel_ms")
    total_ms = profile.get("total_ms")
    for value in (kernel_ms, total_ms):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if not math.isfinite(float(value)) or float(value) < 0.0:
            return None
    if float(total_ms) <= 0.0:
        return None
    return max(0.0, min(1.0, float(kernel_ms) / float(total_ms)))


def _bucket(value: Optional[float], edges: tuple) -> int:
    """1-based bucket index for ``value`` against ascending ``edges``; 0 if None."""
    if value is None:
        return 0
    out = 1
    for edge in edges:
        if value >= edge:
            out += 1
    return out


def strategy_signature(source: str) -> StrategySignature:
    """Extract the strategy signature of a kernel source. Pure and deterministic."""
    src = source or ""
    feats = extract_schedule_features(src)

    residue = torch_op_residue(src)
    fusion = 0 if residue == 0 else (1 if residue <= 2 else (2 if residue <= 5 else 3))

    bm, bn, bk = feats.get("block_m"), feats.get("block_n"), feats.get("block_k")
    if bm is not None and bn is not None:
        tile: Optional[float] = float(bm) * float(bn)
    else:
        first = next((b for b in (bm, bn, bk) if b is not None), None)
        tile = float(first) if first is not None else None
    # 4096 / 16384 / 65536 == 64x64, 128x128, 256x256: the tile sizes an author
    # would name as different designs, not arbitrary cut points.
    tiling = _bucket(tile, (4096.0, 16384.0, 65536.0))

    warps = _bucket(feats.get("num_warps"), (4.0, 8.0, 16.0))
    stages = _bucket(feats.get("num_stages"), (2.0, 3.0))

    has_kernel = bool(_JIT.search(src))
    if not has_kernel:
        compute = COMPUTE_NONE
    elif feats.get("has_atomic"):
        compute = COMPUTE_ATOMIC
    elif feats.get("has_tl_dot") or feats.get("has_mfma"):
        compute = COMPUTE_MMA
    elif feats.get("has_reduction_loop"):
        compute = COMPUTE_REDUCTION
    else:
        compute = COMPUTE_ELEMENTWISE

    return StrategySignature(fusion=fusion, tiling=tiling, warps=warps,
                             stages=stages, compute=compute)


def launch_count(source: str) -> int:
    """Number of Triton launches (``kernel[grid](...)``) in the source."""
    return len(_LAUNCH.findall(canonicalize_source(source or "")))


# --------------------------------------------------------------------------- #
# 2. Population members
# --------------------------------------------------------------------------- #
@dataclass
class Candidate:
    """One executable member of the population, with its measurement evidence.

    ``stats`` accumulates every speedup sample ever taken of this exact kernel
    (deduplicated by :func:`fingerprint`), so re-measuring an incumbent tightens
    its LCB instead of starting a new estimate. ``fitness`` is the pessimistic
    LCB, never the mean: a fast-but-noisy kernel must not evict a slightly
    slower one that reproduces.
    """

    source: str
    fingerprint: str
    signature: StrategySignature
    correct: bool
    stats: MeasureStats = field(default_factory=MeasureStats)
    generation: int = 0
    parent_fingerprint: Optional[str] = None
    lineage: str = ""
    hack_reason: Optional[str] = None
    exceeds_credible: bool = False
    torch_residue: int = 0
    launches: int = 0
    coverage: Optional[float] = None
    error_text: Optional[str] = None
    #: False when the oracle never ran on this source (budget stop). Kept
    #: separate from ``correct`` so "we did not measure it" can never be read
    #: off the run log as "it was wrong".
    measured: bool = True

    @property
    def admissible(self) -> bool:
        """Eligible to be an elite, a parent, or a few-shot exemplar."""
        return (self.measured and self.correct and self.hack_reason is None
                and not self.exceeds_credible and self.stats.n > 0)

    @property
    def speedup_mean(self) -> Optional[float]:
        return self.stats.mean if self.stats.n else None

    @property
    def speedup_lcb(self) -> Optional[float]:
        return self.stats.lcb if self.stats.n else None

    @property
    def fitness(self) -> float:
        """Pessimistic speedup for admissible candidates, ``-inf`` otherwise.

        An inadmissible candidate is not merely ranked last, it is unrankable:
        returning a finite score would let a hacked kernel win an empty niche.
        """
        return self.stats.lcb if self.admissible else float("-inf")

    def summary(self) -> dict:
        return {
            "fingerprint": self.fingerprint,
            "niche": self.signature.as_tuple(),
            "correct": self.correct,
            "n_measures": self.stats.n,
            "speedup_mean": self.speedup_mean,
            "speedup_lcb": self.speedup_lcb,
            "generation": self.generation,
            "lineage": self.lineage,
            "hack_reason": self.hack_reason,
            "exceeds_credible": self.exceeds_credible,
            "torch_residue": self.torch_residue,
            "coverage": self.coverage,
        }


@dataclass
class Admission:
    """The archive's decision about one candidate, and why."""

    verdict: str            # see Archive.VERDICTS
    candidate: Candidate
    incumbent: Optional[Candidate] = None
    evicted: Optional[Candidate] = None
    detail: str = ""


# --------------------------------------------------------------------------- #
# 3. The archive
# --------------------------------------------------------------------------- #
class Archive:
    """MAP-Elites over :class:`StrategySignature`, with a bounded grid.

    One elite per strategy niche, plus a bounded capacity so that niches
    genuinely compete. Two pools are exposed because Kernel-Smith prompt from
    both: :meth:`best` is the top-performing pool and :meth:`exemplars` is the
    diverse one (farthest-point over niche coordinates, seeded with the champion),
    which is the concrete reading of "sampled from both top-performing and
    diverse regions of the search space".

    Inadmissible candidates (env-flagged hacks, implausible ratios, incorrect
    kernels) never enter ``cells``. They are retained in ``rejected`` with their
    measured values so a run stays auditable - the measurement is never altered,
    only its eligibility, which is the same discipline
    ``kore.data.schemas.speedup_credibility`` applies to emitted records.
    """

    VERDICTS = ("new", "improved", "dominated", "duplicate",
                "incorrect", "hack", "implausible", "unmeasured")

    def __init__(
        self,
        capacity: int = 32,
        elite_band: int = 6,
        min_gain: float = MIN_GAIN,
        credible_max: Optional[float] = None,
        novelty_window: int = 8,
    ):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = int(capacity)
        self.elite_band = max(1, int(elite_band))
        self.min_gain = float(min_gain)
        self.credible_max = float(
            credible_max if credible_max is not None else credible_speedup_max(CONFIG))
        self.novelty_window = max(1, int(novelty_window))
        self.cells: dict[tuple, Candidate] = {}
        self.by_fingerprint: dict[str, Candidate] = {}
        self.rejected: list[Candidate] = []
        self.unmeasured: list[Candidate] = []
        self.admissions: list[Admission] = []

    # -- admission ---------------------------------------------------------- #
    def add(self, cand: Candidate) -> Admission:
        """Insert a measured candidate. The three guards are enforced here.

        Order matters: unmeasured before hack before credibility before
        correctness before niche competition, so the reason recorded is the
        earliest one that applies - a hacked kernel is never described as merely
        "incorrect", and a kernel the budget stopped us evaluating is never
        described as wrong.
        """
        if not cand.measured:
            # Deliberately NOT in ``rejected``: that list is the audit trail of
            # kernels the guards refused, and a stopped evaluation is not one.
            self.unmeasured.append(cand)
            return self._record(Admission("unmeasured", cand,
                                          detail=cand.error_text or "never evaluated"))
        if cand.hack_reason:
            return self._reject(cand, "hack", cand.hack_reason)
        if cand.exceeds_credible:
            return self._reject(
                cand, "implausible",
                f"speedup {cand.speedup_mean!r} exceeds credible ceiling "
                f"{self.credible_max}")
        if not cand.correct:
            return self._reject(cand, "incorrect", cand.error_text or "")

        seen = self.by_fingerprint.get(cand.fingerprint)
        if seen is not None:
            # Same kernel, more evidence: fold the samples in so the LCB tightens
            # rather than restarting the estimate from one sample.
            for sample in cand.stats.samples:
                seen.stats.add(sample)
            return self._record(Admission("duplicate", seen, incumbent=seen,
                                          detail="samples merged"))

        key = cand.signature.as_tuple()
        incumbent = self.cells.get(key)
        if incumbent is None:
            self.cells[key] = cand
            self.by_fingerprint[cand.fingerprint] = cand
            evicted = self._evict_if_full(protect=cand)
            return self._record(Admission("new", cand, evicted=evicted))

        # Lazy-optimisation guard: a same-niche revision has to beat its
        # incumbent by more than timing noise to take the slot. MIN_GAIN is
        # imported from kore.data.step_centric so the step filter that builds
        # training data and the archive that generates it cannot drift apart.
        floor = incumbent.fitness * (1.0 + self.min_gain)
        if cand.fitness > floor:
            self.cells[key] = cand
            self.by_fingerprint.pop(incumbent.fingerprint, None)
            self.by_fingerprint[cand.fingerprint] = cand
            evicted = self._evict_if_full(protect=cand)
            return self._record(Admission("improved", cand, incumbent=incumbent,
                                          evicted=evicted))
        return self._record(Admission(
            "dominated", cand, incumbent=incumbent,
            detail=f"fitness {cand.fitness:.4f} <= {floor:.4f} (incumbent "
                   f"{incumbent.fitness:.4f} + {self.min_gain:.0%})"))

    def _reject(self, cand: Candidate, verdict: str, detail: str) -> Admission:
        self.rejected.append(cand)
        return self._record(Admission(verdict, cand, detail=detail))

    def _record(self, admission: Admission) -> Admission:
        self.admissions.append(admission)
        return admission

    def _evict_if_full(self, protect: Candidate) -> Optional[Candidate]:
        """Drop from the DENSEST region when over capacity, never the champion.

        Evicting the globally worst member would quietly re-derive top-k by
        speedup as soon as the archive fills. Evicting the most crowded member
        instead spends capacity on the regions the search has least explored,
        which is the property the archive exists for.
        """
        if len(self.cells) <= self.capacity:
            return None
        champion = self.best(1)
        keep = {protect.fingerprint} | {c.fingerprint for c in champion}
        pool = [c for c in self.cells.values() if c.fingerprint not in keep]
        if not pool:
            return None
        victim = max(pool, key=lambda c: (self._crowding(c), -c.fitness,
                                          c.fingerprint))
        self.cells.pop(victim.signature.as_tuple(), None)
        self.by_fingerprint.pop(victim.fingerprint, None)
        return victim

    def _crowding(self, cand: Candidate, radius: int = 1) -> int:
        """How many other members sit within ``radius`` of this one's niche."""
        return sum(1 for other in self.cells.values()
                   if other.fingerprint != cand.fingerprint
                   and cand.signature.distance(other.signature) <= radius)

    # -- pools -------------------------------------------------------------- #
    def members(self) -> list[Candidate]:
        return list(self.cells.values())

    def elites(self) -> list[Candidate]:
        """Every admissible member, deterministically ordered."""
        return sorted((c for c in self.cells.values() if c.admissible),
                      key=lambda c: (-c.fitness, c.fingerprint))

    def best(self, n: int = 1) -> list[Candidate]:
        return self.elites()[:max(0, int(n))]

    def champion(self) -> Optional[Candidate]:
        top = self.best(1)
        return top[0] if top else None

    def exemplars(self, k: int = 4) -> list[Candidate]:
        """Champion first, then farthest-point over niche coordinates.

        This is the prompt context. Dr. Kernel's context management keeps the
        top-w turns by reward with w=4 and reports it as strictly more reliable
        than appending the whole history as T grows; the default k matches. The
        difference here is that only the FIRST slot is spent on rank - the rest
        buy coverage of schedule space, so the model sees that more than one
        design is viable and is not being taught to converge.
        """
        pool = self.elites()
        k = max(0, int(k))
        if not pool or k == 0:
            return []
        chosen = [pool[0]]
        rest = pool[1:]
        while rest and len(chosen) < k:
            pick = max(rest, key=lambda c: (
                min(c.signature.distance(s.signature) for s in chosen),
                c.fitness, c.fingerprint))
            chosen.append(pick)
            rest.remove(pick)
        return chosen

    # -- parent selection --------------------------------------------------- #
    def sample_parent(self, rng: random.Random, p_elite: float = 0.5,
                      force_explore: bool = False) -> Optional[Candidate]:
        """Draw a parent from the top pool or the sparse pool.

        ``force_explore`` is set by the loop when collapse is detected; it skips
        the top pool entirely so the next generation cannot be another re-tune of
        the incumbent. Without that branch a single dominant kernel is drawn as
        parent with probability ~1 and the population is a population in name
        only.
        """
        pool = self.elites()
        if not pool:
            return None
        if not force_explore and rng.random() < float(p_elite):
            band = pool[:self.elite_band]
            return band[rng.randrange(len(band))]
        weights = [1.0 / (1.0 + self._crowding(c)) for c in pool]
        return rng.choices(pool, weights=weights, k=1)[0]

    # -- collapse diagnostics ----------------------------------------------- #
    def coverage(self) -> int:
        """Occupied strategy niches. The archive's exploration footprint."""
        return len(self.cells)

    def mean_pairwise_distance(self) -> float:
        """Mean L1 niche distance over admissible members; 0.0 below two members.

        This is the number that goes to zero when a population collapses, and it
        is the number ``tests/test_evolve_agent.py`` compares against a
        speedup-binned archive on the same candidate stream.
        """
        pool = self.elites()
        if len(pool) < 2:
            return 0.0
        total, pairs = 0, 0
        for i, a in enumerate(pool):
            for b in pool[i + 1:]:
                total += a.signature.distance(b.signature)
                pairs += 1
        return total / pairs if pairs else 0.0

    def lineage_concentration(self) -> float:
        """Largest share of members descending from one first-generation branch.

        1.0 means every survivor is a descendant of a single early commitment -
        exactly the single-trajectory anchoring the evolutionary loop exists to
        break, and invisible to a coverage count because those descendants can
        still occupy different niches.
        """
        pool = self.elites()
        if not pool:
            return 0.0
        counts: dict[str, int] = {}
        for cand in pool:
            key = cand.lineage or cand.fingerprint
            counts[key] = counts.get(key, 0) + 1
        return max(counts.values()) / len(pool)

    def recent_novelty(self, window: Optional[int] = None) -> float:
        """Share of the last ``window`` admissions that opened a NEW niche.

        Complements lineage concentration: a run can keep exploring lineages
        while re-landing in explored territory, and vice versa. Zero here means
        the proposer is re-tuning, which is when forced exploration pays.
        """
        window = self.novelty_window if window is None else max(1, int(window))
        scored = [a for a in self.admissions
                  if a.verdict in ("new", "improved", "dominated", "duplicate")]
        if not scored:
            return 0.0
        recent = scored[-window:]
        return sum(1 for a in recent if a.verdict == "new") / len(recent)

    def collapsed(self, lineage_threshold: float = 0.85,
                  min_members: int = 3) -> bool:
        """True when the population has stopped being a population.

        Both conditions must hold: enough members to judge, one lineage owning
        nearly all of them, and no new territory in the recent window. Any one of
        those alone is normal early-run behaviour.
        """
        pool = self.elites()
        if len(pool) < max(2, int(min_members)):
            return False
        return (self.lineage_concentration() >= float(lineage_threshold)
                and self.recent_novelty() <= 0.0)

    def verdict_counts(self) -> dict:
        counts = {v: 0 for v in self.VERDICTS}
        for admission in self.admissions:
            counts[admission.verdict] = counts.get(admission.verdict, 0) + 1
        return counts

    def summary(self) -> dict:
        champ = self.champion()
        return {
            "coverage": self.coverage(),
            "members": len(self.cells),
            "rejected": len(self.rejected),
            "unmeasured": len(self.unmeasured),
            "mean_pairwise_distance": round(self.mean_pairwise_distance(), 4),
            "lineage_concentration": round(self.lineage_concentration(), 4),
            "recent_novelty": round(self.recent_novelty(), 4),
            "best_speedup_lcb": champ.speedup_lcb if champ else None,
            "best_speedup_mean": champ.speedup_mean if champ else None,
            "verdicts": self.verdict_counts(),
        }

    def __len__(self) -> int:
        return len(self.cells)


# --------------------------------------------------------------------------- #
# 4. Stable evaluation
# --------------------------------------------------------------------------- #
@dataclass
class Trial:
    """The verified outcome of evaluating one kernel source.

    ``samples`` are speedup measurements, one per ``env.step``. ``speedup_mean``
    is over the TRIMMED samples; the untrimmed list is kept so the trim is
    auditable rather than a number that appeared.
    """

    source: str
    fingerprint: str
    compiled: bool
    correct: bool
    samples: list = field(default_factory=list)
    trimmed: list = field(default_factory=list)
    hack_reason: Optional[str] = None
    exceeds_credible: bool = False
    infra_error: bool = False
    error_text: Optional[str] = None
    env_calls: int = 0
    coverage: Optional[float] = None
    #: True when the verifier-call cap stopped this evaluation before the oracle
    #: ran. A budget stop is NOT a kernel failure, and recording it as one would
    #: put a number that was never measured into the run's incorrect count - the
    #: same defect as a fabricated timing, wearing a different hat.
    budget_exhausted: bool = False

    @property
    def speedup_mean(self) -> Optional[float]:
        return sum(self.trimmed) / len(self.trimmed) if self.trimmed else None

    @property
    def verified(self) -> bool:
        """Measured, correct, and not disqualified by a guard."""
        return bool(self.correct and self.hack_reason is None
                    and not self.exceeds_credible and self.trimmed
                    and not self.budget_exhausted)


def trim_outliers(samples: list, min_for_trim: int = 5) -> list:
    """Drop one high and one low sample once there are enough to spare them.

    Kernel-Smith reduce timing variance with warm-up, repeated measurement and
    outlier removal, reporting fluctuation held under 1% (§3.3). Warm-up and
    cold-cache flushing already live in the KORE drivers; this is the outlier
    step. It refuses to trim below ``min_for_trim`` because dropping the extremes
    of three samples leaves one sample and a variance of zero - a confidently
    wrong LCB is worse than a wide honest one.
    """
    values = sorted(float(s) for s in samples)
    if len(values) < max(3, int(min_for_trim)):
        return values
    return values[1:-1]


class StableEvaluator:
    """Repeated, guarded measurement of candidates through a ``KoreEnv``.

    Two phases, because they have different costs:

    * :meth:`screen` - one ``env.step``: compile, correctness, first timing. This
      is what the proposer's turns are worth; it is never what the archive
      records.
    * :meth:`stabilize` - extra measurements allocated across a generation's
      correct candidates by ``kore.search.bandit.successive_halving``, so budget
      goes to the contenders instead of being spread evenly over kernels one
      look already ruled out.

    Every ``env.step`` is charged to a shared :class:`Budget`, so the whole run
    is anytime and the verifier-call cap is a fact rather than an intention.
    """

    def __init__(
        self,
        env,
        task,
        budget: Budget,
        reward_cfg=CONFIG,
        credible_max: Optional[float] = None,
        min_for_trim: int = 5,
        max_measures: int = 6,
    ):
        self.env = env
        self.task = task
        self.budget = budget
        self.reward_cfg = reward_cfg
        self.credible_max = float(
            credible_max if credible_max is not None else credible_speedup_max(reward_cfg))
        self.min_for_trim = int(min_for_trim)
        self.max_measures = max(1, int(max_measures))
        self.env_calls = 0

    # -- one measurement ---------------------------------------------------- #
    def _step(self, source: str):
        """One verified ``env.step``; never raises, never fabricates a timing."""
        try:
            obs = self.env.step(source, full_validation=True, multi_shape=True)
        except Exception as exc:  # noqa: BLE001 - a crashing verifier is a datum
            obs = Observation(compiled=False,
                              dtype=getattr(self.task, "dtype", "fp32"),
                              error_text=str(exc)[:200])
        self.env_calls += 1
        result = compute_reward(obs, source,
                                dtype=getattr(self.task, "dtype", "fp32"),
                                cfg=self.reward_cfg)
        return obs, result

    def screen(self, source: str) -> Trial:
        """One look: does it compile, is it correct, and roughly how fast."""
        fp = fingerprint(source)
        if not self.budget.spend(1):
            return Trial(source=source, fingerprint=fp, compiled=False,
                         correct=False, error_text="budget exhausted",
                         env_calls=0, budget_exhausted=True)
        obs, result = self._step(source)
        samples: list = []
        speedup = result.speedup
        if result.correct and speedup is not None:
            samples.append(float(speedup))
        exceeds = bool(speedup is not None and float(speedup) > self.credible_max)
        trial = Trial(
            source=source,
            fingerprint=fp,
            compiled=bool(obs.compiled),
            correct=bool(result.correct),
            samples=samples,
            trimmed=trim_outliers(samples, self.min_for_trim),
            hack_reason=(obs.hack_reason or "flagged_hack") if obs.flagged_hack else None,
            exceeds_credible=exceeds,
            infra_error=bool(obs.infra_error),
            error_text=obs.error_text,
            env_calls=1,
        )
        log.debug("evolve_screen", fingerprint=fp, correct=trial.correct,
                  speedup=speedup, hack=trial.hack_reason,
                  exceeds_credible=exceeds)
        return trial

    def stabilize(self, trials: list, min_measures: int = 2) -> list:
        """Spend the remaining generation budget tightening contenders' LCBs.

        Arms are the correct, non-disqualified trials. Successive halving gives
        everyone a cheap second look, then re-invests on the survivors ranked by
        LCB - the same pessimism the archive admits on, so allocation and
        admission cannot disagree about who is winning.
        """
        live = [t for t in trials if t.correct and not t.hack_reason
                and not t.exceeds_credible]
        if not live:
            return trials

        def sampler_for(trial: Trial):
            def sample() -> Optional[float]:
                obs, result = self._step(trial.source)
                trial.env_calls += 1
                if obs.flagged_hack and trial.hack_reason is None:
                    trial.hack_reason = obs.hack_reason or "flagged_hack"
                if not result.correct or result.speedup is None:
                    # A kernel that verified once and not again is unstable, not
                    # fast; record it and stop measuring rather than average it
                    # with the run that passed.
                    trial.correct = False
                    trial.error_text = obs.error_text or "unstable correctness"
                    return None
                value = float(result.speedup)
                if value > self.credible_max:
                    trial.exceeds_credible = True
                trial.samples.append(value)
                trial.trimmed = trim_outliers(trial.samples, self.min_for_trim)
                return value
            return sample

        arms = []
        for trial in live:
            arm = CallbackArm(key=trial.fingerprint, sampler=sampler_for(trial))
            for sample in trial.samples:
                arm.stats.add(sample)
            arms.append(arm)
        successive_halving(arms, self.budget, min_measures=max(2, int(min_measures)),
                           max_measures=self.max_measures, rank_key="lcb")
        for trial in trials:
            trial.trimmed = trim_outliers(trial.samples, self.min_for_trim)
        return trials


def candidate_from_trial(trial: Trial, generation: int, parent: Optional[Candidate],
                         lineage: Optional[str] = None) -> Candidate:
    """Build an archive candidate from a verified trial.

    The candidate's stats are seeded from the TRIMMED samples, so the LCB the
    archive ranks on is the same number the trim produced - there is no second,
    friendlier view of the same measurement anywhere in the loop.
    """
    stats = MeasureStats()
    for sample in trial.trimmed:
        stats.add(sample)
    fp = trial.fingerprint
    if lineage is None:
        lineage = (parent.lineage if parent is not None and parent.lineage else fp)
    return Candidate(
        source=trial.source,
        fingerprint=fp,
        signature=strategy_signature(trial.source),
        correct=bool(trial.correct),
        stats=stats,
        generation=int(generation),
        parent_fingerprint=parent.fingerprint if parent is not None else None,
        lineage=lineage,
        hack_reason=trial.hack_reason,
        exceeds_credible=bool(trial.exceeds_credible),
        torch_residue=torch_op_residue(trial.source),
        launches=launch_count(trial.source),
        coverage=trial.coverage,
        error_text=trial.error_text,
        measured=not trial.budget_exhausted,
    )


# --------------------------------------------------------------------------- #
# 5. Sequential test-time scaling
# --------------------------------------------------------------------------- #
@dataclass
class ScaledStep:
    """One evolution step offered to the best-of-history selector."""

    generation: int
    turn: int
    fingerprint: str
    speedup: Optional[float]
    verified: bool
    source: str = ""


class BestAcrossSteps:
    """Best-of-history selection over every step of the search.

    Dr. Kernel's Table 1 is the argument: at a fixed 14B model on KernelBench
    Level 2, reporting the LAST turn gives Fast@1.2 31.6 while selecting the best
    turn across history gives **47.8**. The search already pays for those
    intermediate kernels; reporting only the final one throws away half the
    result. Kernel-Smith make the same measurement from the other side - their
    best-score-per-generation curve is the upper envelope of every competitor's.

    :meth:`offer` refuses unverified steps outright, so the curve can never be
    lifted by a kernel that failed, was flagged, or was never measured. The curve
    is monotone non-decreasing by construction.
    """

    def __init__(self):
        self.steps: list[ScaledStep] = []
        self.curve: list[float] = []
        self.best: Optional[ScaledStep] = None
        self.rejected: int = 0

    def offer(self, step: ScaledStep) -> bool:
        """Record a step; return True iff it became the new best."""
        if not step.verified or step.speedup is None:
            self.rejected += 1
            return False
        value = float(step.speedup)
        if not math.isfinite(value) or value <= 0.0:
            self.rejected += 1
            return False
        self.steps.append(step)
        improved = self.best is None or value > float(self.best.speedup)
        if improved:
            self.best = step
        self.curve.append(float(self.best.speedup))
        return improved

    @property
    def best_speedup(self) -> Optional[float]:
        return float(self.best.speedup) if self.best is not None else None

    def summary(self) -> dict:
        return {
            "steps_recorded": len(self.steps),
            "steps_rejected": self.rejected,
            "best_speedup": self.best_speedup,
            "best_generation": self.best.generation if self.best else None,
            "best_turn": self.best.turn if self.best else None,
            "curve": list(self.curve),
        }


# --------------------------------------------------------------------------- #
# 6. Proposal: the model as a local improver
# --------------------------------------------------------------------------- #
@dataclass
class Proposal:
    """One kernel the proposer produced, with the turn it came from."""

    source: str
    turn: int
    origin: str = "model"


class HarnessProposer:
    """Mutation operator = one :class:`~kore.agent.harness.AgentHarness` episode.

    The harness is the local improver: it already runs build/test/bench with
    structured execution feedback, splits correctness from optimisation, and
    reflects on failures. What it cannot do alone is keep more than one design
    alive, which is what the archive around it supplies - the parent is drawn
    from the archive and the archive's diverse exemplars are injected as the
    episode's few-shot context through the harness's existing ``WinsKB`` hook.
    Every turn's candidate is returned, not just the episode's best, because the
    intermediate kernels are exactly what best-of-history selection needs.
    """

    def __init__(self, model, max_turns: int = 4, reseed_patience: int = 2):
        self.model = model
        self.max_turns = max(1, int(max_turns))
        self.reseed_patience = max(1, int(reseed_patience))

    def _kb(self, task, exemplars: list):
        from kore.agent.harness import WinsKB
        from kore.data.mutate import infer_family

        op = getattr(task, "operation", "") or getattr(task, "task_id", "")
        family = infer_family(op)
        dtype = str(getattr(task, "dtype", "") or "").lower()
        entries = [{
            "task_id": f"{getattr(task, 'task_id', 'task')}@{c.fingerprint}",
            "family": family,
            "dtype": dtype,
            "speedup": c.speedup_lcb,
            "final_source": c.source,
            "snr_db": None,
        } for c in exemplars if c.source]
        return WinsKB(entries)

    def propose(self, task, env, parent: Optional[Candidate],
                exemplars: list, generation: int) -> list:
        from kore.agent.harness import AgentHarness

        harness = AgentHarness(
            task, self.model, env,
            max_turns=self.max_turns,
            seed_src=parent.source if parent is not None else None,
            reseed_patience=self.reseed_patience,
            kb=self._kb(task, exemplars),
            kb_top_k=len(exemplars),
        )
        episode = harness.run()
        out: list[Proposal] = []
        for turn, source in enumerate(episode.turn_codes or []):
            if source:
                out.append(Proposal(source=source, turn=turn))
        log.debug("evolve_propose", generation=generation,
                  turns=episode.turns_used, proposals=len(out))
        return out


# --------------------------------------------------------------------------- #
# 7. The loop
# --------------------------------------------------------------------------- #
@dataclass
class EvolveAgentConfig:
    """Knobs for :func:`evolve`. Defaults are sized for one MI355X node."""

    generations: int = 8
    #: Hard cap on verifier calls. Enforced exactly by StableEvaluator; the
    #: proposer's own env traffic is metered but charged at generation
    #: granularity, so an in-flight episode may overshoot by at most its turn
    #: count. That bound is stated rather than hidden because pretending to a
    #: precision the wrapper does not have is the same defect as a fabricated
    #: timing.
    max_env_calls: int = 400
    #: Reserve below which no new generation starts (screening + stabilisation
    #: for one generation's proposals).
    generation_reserve: int = 12
    turns_per_generation: int = 4
    exemplars: int = 4              # Dr. Kernel's context-management w
    archive_capacity: int = 32
    elite_band: int = 6
    p_elite: float = 0.5
    min_gain: float = MIN_GAIN
    max_measures: int = 6
    min_for_trim: int = 5
    lineage_threshold: float = 0.85
    seed: int = 0
    reward_cfg: Any = CONFIG


@dataclass
class GenerationRecord:
    """What one generation did, for the run log and for the tests."""

    generation: int
    parent: Optional[str]
    forced_explore: bool
    proposals: int
    screened: int
    admitted: int
    verdicts: dict = field(default_factory=dict)
    best_speedup: Optional[float] = None
    coverage: int = 0
    mean_pairwise_distance: float = 0.0
    lineage_concentration: float = 0.0
    env_calls: int = 0


@dataclass
class EvolveAgentResult:
    """The outcome of a run. ``best`` is the best-of-history, not the last step."""

    task_id: str
    archive: Archive
    scaling: BestAcrossSteps
    generations: list = field(default_factory=list)
    env_calls: int = 0
    stats: dict = field(default_factory=dict)

    @property
    def best(self) -> Optional[Candidate]:
        return self.archive.champion()

    @property
    def best_source(self) -> Optional[str]:
        champ = self.best
        return champ.source if champ is not None else None

    def to_dict(self) -> dict:
        champ = self.best
        return {
            "task_id": self.task_id,
            "best": champ.summary() if champ is not None else None,
            "archive": self.archive.summary(),
            "scaling": self.scaling.summary(),
            "env_calls": self.env_calls,
            "generations": [g.__dict__ for g in self.generations],
            "stats": dict(self.stats),
        }


def evolve(
    task,
    proposer,
    env,
    cfg: Optional[EvolveAgentConfig] = None,
    seed_source: Optional[str] = None,
) -> EvolveAgentResult:
    """Run the evolutionary loop on one task.

    ``proposer`` is anything with
    ``propose(task, env, parent, exemplars, generation) -> [Proposal]`` -
    :class:`HarnessProposer` in production, a scripted stub in the tests.
    ``env`` is anything with ``step(source, full_validation=, multi_shape=)``.

    The generation is: draw a parent from the archive (top pool or sparse pool,
    forced to the sparse pool when collapse is detected), hand it and the diverse
    exemplars to the proposer, SCREEN every turn's kernel once, then spend the
    remaining generation budget STABILISING the correct ones before any of them
    is allowed to become an elite. Screening decides what is worth measuring;
    only stabilised measurements decide what the population becomes.
    """
    cfg = cfg or EvolveAgentConfig()
    rng = random.Random(cfg.seed)
    budget = Budget(cfg.max_env_calls)
    evaluator = StableEvaluator(env, task, budget, reward_cfg=cfg.reward_cfg,
                                min_for_trim=cfg.min_for_trim,
                                max_measures=cfg.max_measures)
    archive = Archive(capacity=cfg.archive_capacity, elite_band=cfg.elite_band,
                      min_gain=cfg.min_gain)
    scaling = BestAcrossSteps()
    task_id = str(getattr(task, "task_id", "task"))
    records: list[GenerationRecord] = []
    seen: set[str] = set()

    def admit(trials: list, generation: int, parent: Optional[Candidate],
              turns: dict) -> int:
        admitted = 0
        for trial in trials:
            cand = candidate_from_trial(trial, generation, parent)
            decision = archive.add(cand)
            if decision.verdict in ("new", "improved"):
                admitted += 1
            scaling.offer(ScaledStep(
                generation=generation,
                turn=turns.get(trial.fingerprint, -1),
                fingerprint=trial.fingerprint,
                speedup=trial.speedup_mean,
                # Verified AND admissible: a step the archive refused must not be
                # able to win best-of-history through the back door.
                verified=bool(trial.verified
                              and decision.verdict not in ("hack", "implausible")),
                source=trial.source,
            ))
        return admitted

    with log.stage("evolve_agent", task=task_id, generations=cfg.generations,
                   budget=cfg.max_env_calls):
        # --- population init: the seed is generation 0 --------------------- #
        root = seed_source if seed_source is not None else _safe_seed(task)
        if root:
            seed_trial = evaluator.screen(root)
            evaluator.stabilize([seed_trial])
            seen.add(seed_trial.fingerprint)
            admit([seed_trial], 0, None, {seed_trial.fingerprint: 0})

        for generation in range(1, int(cfg.generations) + 1):
            if budget.remaining < int(cfg.generation_reserve):
                log.event("evolve_budget_stop", task=task_id, generation=generation,
                          remaining=budget.remaining)
                break

            force = archive.collapsed(cfg.lineage_threshold)
            parent = archive.sample_parent(rng, p_elite=cfg.p_elite,
                                           force_explore=force)
            exemplars = archive.exemplars(cfg.exemplars)
            calls_before = evaluator.env_calls

            proposals = proposer.propose(task, env, parent, exemplars, generation)
            trials: list[Trial] = []
            turns: dict[str, int] = {}
            for proposal in proposals:
                fp = fingerprint(proposal.source)
                if fp in seen:
                    continue          # already measured; re-screening buys nothing
                if budget.remaining <= 0:
                    break
                seen.add(fp)
                trial = evaluator.screen(proposal.source)
                turns[trial.fingerprint] = proposal.turn
                trials.append(trial)

            evaluator.stabilize(trials)
            admitted = admit(trials, generation, parent, turns)

            champ = archive.champion()
            records.append(GenerationRecord(
                generation=generation,
                parent=parent.fingerprint if parent is not None else None,
                forced_explore=bool(force),
                proposals=len(proposals),
                screened=len(trials),
                admitted=admitted,
                verdicts=archive.verdict_counts(),
                best_speedup=champ.speedup_lcb if champ else None,
                coverage=archive.coverage(),
                mean_pairwise_distance=archive.mean_pairwise_distance(),
                lineage_concentration=archive.lineage_concentration(),
                env_calls=evaluator.env_calls - calls_before,
            ))
            log.metric("evolve_generation", task=task_id, generation=generation,
                       admitted=admitted, forced_explore=bool(force),
                       **archive.summary())

    stats = {
        "budget_total": budget.total,
        "budget_used": budget.used,
        "env_calls": evaluator.env_calls,
        "generations_run": len(records),
        "rejected_hacks": sum(1 for c in archive.rejected if c.hack_reason),
        "rejected_implausible": sum(1 for c in archive.rejected
                                    if c.exceeds_credible and not c.hack_reason),
        # Reported separately from the rejections so a truncated run is legible
        # as "we ran out of budget", not as "these kernels were wrong".
        "unmeasured": len(archive.unmeasured),
    }
    result = EvolveAgentResult(task_id=task_id, archive=archive, scaling=scaling,
                               generations=records, env_calls=evaluator.env_calls,
                               stats=stats)
    log.event("evolve_agent_done", task=task_id, **stats,
              best_speedup=scaling.best_speedup, coverage=archive.coverage())
    return result


def _safe_seed(task) -> str:
    """The task's seed kernel, or '' when the task carries none (fake tasks)."""
    try:
        return task.seed_source
    except Exception:  # noqa: BLE001 - scripted tasks have no seed file
        return ""
