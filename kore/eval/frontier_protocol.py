"""Publication-grade model-versus-system frontier claim protocol.

This module is deliberately PURE: it consumes frozen benchmark artifacts and
produces a claim report.  It never calls a model, network, benchmark driver, or
GPU.  Live harnesses must adapt their traces into the schemas below and verify
artifact signatures before setting ``ArtifactSignature.verified``.

The protocol has two non-substitutable tracks:

* ``closed-model-harness`` compares model checkpoints while freezing the prompt,
  tools, verifier, hardware, and budget.
* ``open-system-harness`` compares disclosed end-to-end systems while freezing
  the verifier, hardware, and budget.  System prompts/tools may differ because
  they are part of the system under test, but their fingerprints are mandatory.

Every claim is evaluated over the preregistered task x run rectangle.  Missing
cells count as failures *and* invalidate publication integrity.  The canonical
``fast_p`` point estimate is kept distinct from its hierarchical-bootstrap
lower confidence bound.  Speed-of-light (SoL) timing is an integrity ceiling and
attainment metric only; it is never exposed as a beatable baseline.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import urlparse

from kore.eval.fastp import fastp
from kore.eval.paired_stats import wilcoxon_signed_rank


PROTOCOL_SCHEMA_VERSION = "kore.frontier-claim/v1"
REPORT_SCHEMA_VERSION = "kore.frontier-report/v1"
CANONICAL_BASELINE = "vendor-production"
# Named integrity signal for a paired bootstrap whose resampling distribution has
# zero variance (an arm saturated at 0/1 on every preregistered cell).
BOOTSTRAP_DEGENERATE = "bootstrap_degenerate"
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_MOVING_REVISIONS = {"head", "latest", "main", "master", "tip", "trunk"}
_SOL_TOLERANCE = 1e-9


# ---------------------------------------------------------------------------
# Canonical serialization and immutable fingerprints.
# ---------------------------------------------------------------------------
def _primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float) and not math.isfinite(value):
        # Keep malformed artifacts content-addressable so validation can return a
        # finite-input failure instead of crashing while hashing the evidence.
        return {"__nonfinite__": repr(value)}
    if is_dataclass(value):
        return {field.name: _primitive(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(_primitive(key)): _primitive(val) for key, val in value.items()}
    if isinstance(value, (tuple, list)):
        return [_primitive(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _primitive(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_text(text: str) -> str:
    """Return the lowercase SHA-256 of UTF-8 ``text``."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Fingerprint:
    """Auditable immutable fingerprint with its canonical payload attached."""

    kind: str
    sha256: str
    canonical_json: str

    @classmethod
    def from_payload(cls, kind: str, payload: Any) -> "Fingerprint":
        canonical = _canonical_json(payload)
        return cls(kind=str(kind), sha256=sha256_text(canonical), canonical_json=canonical)

    def to_dict(self) -> dict:
        return _primitive(self)


class HarnessTrack(str, Enum):
    CLOSED_MODEL = "closed-model-harness"
    OPEN_SYSTEM = "open-system-harness"


class ArmKind(str, Enum):
    MODEL = "model"
    SYSTEM = "system"


class Disclosure(str, Enum):
    CLOSED = "closed"
    OPEN = "open"


class BaselineKind(str, Enum):
    VENDOR_PRODUCTION = CANONICAL_BASELINE
    BEST_VENDOR = "best-vendor"
    COMPILER = "compiler"
    EAGER = "eager"


REQUIRED_BASELINES: tuple[BaselineKind, ...] = tuple(BaselineKind)


@dataclass(frozen=True)
class BudgetEnvelope:
    """Preregistered per-task resource ceiling for one harness arm."""

    output_tokens: int
    context_policy: Fingerprint
    tool_calls: int
    correctness_calls: int
    fresh_timed_calls: int
    profiler_calls: int
    gpu_seconds: float
    wall_time_seconds: float
    cost_usd: float

    def fingerprint_payload(self) -> dict:
        return {
            "output_tokens": self.output_tokens,
            "context_policy_sha256": self.context_policy.sha256,
            "tool_calls": self.tool_calls,
            "correctness_calls": self.correctness_calls,
            "fresh_timed_calls": self.fresh_timed_calls,
            "profiler_calls": self.profiler_calls,
            "gpu_seconds": self.gpu_seconds,
            "wall_time_seconds": self.wall_time_seconds,
            "cost_usd": self.cost_usd,
        }

    def fingerprint(self) -> Fingerprint:
        return Fingerprint.from_payload("budget", self.fingerprint_payload())


@dataclass(frozen=True)
class ResourceUsage:
    """Observed per-task usage, checked against :class:`BudgetEnvelope`."""

    output_tokens: int
    tool_calls: int
    correctness_calls: int
    fresh_timed_calls: int
    profiler_calls: int
    gpu_seconds: float
    wall_time_seconds: float
    cost_usd: float


@dataclass(frozen=True)
class EvidenceRef:
    """Content-addressed raw evidence and its immutable upstream revision."""

    raw_trace_sha256: str
    sample_sha256: str
    source_url: str
    source_revision: str


@dataclass(frozen=True)
class ArmSpec:
    """Frozen identity and budget for one side of a track."""

    arm_id: str
    display_name: str
    arm_kind: ArmKind
    disclosure: Disclosure
    artifact_fingerprint: Fingerprint
    checkpoint_fingerprint: Fingerprint
    tool_fingerprint: Fingerprint
    prompt_fingerprint: Fingerprint
    verifier_fingerprint: Fingerprint
    hardware_fingerprint: Fingerprint
    budget_fingerprint: Fingerprint
    budget: BudgetEnvelope
    source_url: str
    source_revision: str


@dataclass(frozen=True)
class DataQuality:
    """Conditions that categorically disqualify a frontier claim."""

    smoke_data: bool = False
    fallback_data: bool = False
    contamination_detected: bool = False


@dataclass(frozen=True)
class TrackSpec:
    track: HarnessTrack
    candidate: ArmSpec
    comparator: ArmSpec
    run_ids: tuple[str, ...]
    started_at_utc: str
    skipped: bool = False
    skip_reason: Optional[str] = None
    quality: DataQuality = DataQuality()


@dataclass(frozen=True)
class TaskManifestEntry:
    task_id: str
    family_id: str
    source_url: str
    source_revision: str
    task_sha256: str


@dataclass(frozen=True)
class ArmObservation:
    public_correct: bool
    hidden_correct: Optional[bool]
    time_ms: Optional[float]
    usage: ResourceUsage
    evidence: EvidenceRef


@dataclass(frozen=True)
class BaselineObservation:
    kind: BaselineKind
    hidden_correct: Optional[bool]
    time_ms: float
    evidence: EvidenceRef


@dataclass(frozen=True)
class PairedTaskSample:
    """One candidate/comparator pair on one preregistered task and run."""

    track: HarnessTrack
    run_id: str
    task_id: str
    family_id: str
    candidate: Optional[ArmObservation]
    comparator: Optional[ArmObservation]
    baselines: tuple[BaselineObservation, ...]
    sol_time_ms: float
    sol_evidence: EvidenceRef


@dataclass(frozen=True)
class ArtifactSignature:
    """Detached signature metadata verified by a trusted ingestion adapter."""

    algorithm: str
    signature: str
    signer_key_sha256: str
    payload_sha256: str
    verified: bool


@dataclass(frozen=True)
class Preregistration:
    profile_name: str
    profile_fingerprint: str
    manifest_fingerprint: str
    registered_at_utc: str


@dataclass(frozen=True)
class FrontierArtifact:
    schema_version: str
    artifact_id: str
    source_url: str
    source_revision: str
    preregistration: Preregistration
    manifest: tuple[TaskManifestEntry, ...]
    tracks: tuple[TrackSpec, ...]
    samples: tuple[PairedTaskSample, ...]
    quality: DataQuality = DataQuality()
    signature: Optional[ArtifactSignature] = None


def artifact_payload_sha256(artifact: FrontierArtifact) -> str:
    """Hash the complete artifact except its detached signature."""
    payload = {
        field.name: _primitive(getattr(artifact, field.name))
        for field in fields(artifact)
        if field.name != "signature"
    }
    return sha256_text(_canonical_json(payload))


def manifest_fingerprint(manifest: Sequence[TaskManifestEntry]) -> str:
    """Hash the ordered, preregistered task manifest."""
    return sha256_text(_canonical_json(list(manifest)))


# ---------------------------------------------------------------------------
# Preregistered claim profiles.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClaimProfile:
    name: str
    description: str
    required_tracks: tuple[HarnessTrack, ...]
    canonical_p: float
    alpha: float
    ci_level: float
    n_boot: int
    noninferiority_margin: float
    superiority_margin: float
    required_superiority_tracks: tuple[HarnessTrack, ...]
    superiority_any_of: tuple[HarnessTrack, ...]
    min_tasks: int
    min_families: int
    min_runs_per_track: int

    @property
    def fingerprint(self) -> str:
        return sha256_text(_canonical_json(self))

    def to_dict(self) -> dict:
        result = _primitive(self)
        result["fingerprint"] = self.fingerprint
        return result


_PROFILE_LIST = (
    ClaimProfile(
        name="development",
        description="Protocol-valid development claim; both tracks must be non-inferior.",
        required_tracks=tuple(HarnessTrack),
        canonical_p=1.0,
        alpha=0.10,
        ci_level=0.90,
        n_boot=2_000,
        noninferiority_margin=0.10,
        superiority_margin=0.0,
        required_superiority_tracks=(),
        superiority_any_of=(),
        min_tasks=4,
        min_families=2,
        min_runs_per_track=2,
    ),
    ClaimProfile(
        name="frontier-competitive",
        description=(
            "Non-inferior on both tracks and superior on at least one preregistered track."
        ),
        required_tracks=tuple(HarnessTrack),
        canonical_p=1.0,
        alpha=0.05,
        ci_level=0.95,
        n_boot=10_000,
        noninferiority_margin=0.02,
        superiority_margin=0.0,
        required_superiority_tracks=(),
        superiority_any_of=tuple(HarnessTrack),
        min_tasks=50,
        min_families=5,
        min_runs_per_track=3,
    ),
    ClaimProfile(
        name="best-in-class-model",
        description=(
            "Superior in the frozen closed-model harness and non-inferior as an open system."
        ),
        required_tracks=tuple(HarnessTrack),
        canonical_p=1.0,
        alpha=0.05,
        ci_level=0.95,
        n_boot=20_000,
        noninferiority_margin=0.01,
        superiority_margin=0.0,
        required_superiority_tracks=(HarnessTrack.CLOSED_MODEL,),
        superiority_any_of=(),
        min_tasks=100,
        min_families=8,
        min_runs_per_track=5,
    ),
    ClaimProfile(
        name="best-in-class-system",
        description=(
            "Superior in the disclosed open-system harness and non-inferior as a model."
        ),
        required_tracks=tuple(HarnessTrack),
        canonical_p=1.0,
        alpha=0.05,
        ci_level=0.95,
        n_boot=20_000,
        noninferiority_margin=0.01,
        superiority_margin=0.0,
        required_superiority_tracks=(HarnessTrack.OPEN_SYSTEM,),
        superiority_any_of=(),
        min_tasks=100,
        min_families=8,
        min_runs_per_track=5,
    ),
)

CLAIM_PROFILES: Mapping[str, ClaimProfile] = MappingProxyType(
    {profile.name: profile for profile in _PROFILE_LIST}
)


# ---------------------------------------------------------------------------
# Hierarchical paired bootstrap and secondary tests.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PairedDatum:
    run_id: str
    family_id: str
    task_id: str
    candidate: float
    comparator: float


@dataclass(frozen=True)
class HierarchicalBootstrapResult:
    n: int
    n_runs: int
    n_families: int
    n_boot: int
    ci_level: float
    candidate_estimate: float
    comparator_estimate: float
    delta_estimate: float
    candidate_ci: tuple[float, float]
    comparator_ci: tuple[float, float]
    delta_ci: tuple[float, float]
    delta_se: float
    p_superiority: float
    p_noninferiority: float
    superiority_margin: float
    noninferiority_margin: float
    # Boundary handling.  A 0/1 arm that succeeds (or fails) on every cell makes
    # every resample identical, so the percentile bootstrap collapses to a
    # zero-width point interval.  The reported intervals above are therefore the
    # widest of the design-aware bootstrap and the boundary-exact interval; the
    # raw components are kept here so the substitution is auditable.
    binary_outcomes: bool = True
    candidate_saturated: bool = False
    comparator_saturated: bool = False
    degenerate: bool = False
    interval_method: str = "cluster-bootstrap"
    p_value_method: str = "centered-cluster-bootstrap"
    candidate_successes: Optional[int] = None
    comparator_successes: Optional[int] = None
    candidate_ci_bootstrap: Optional[tuple[float, float]] = None
    comparator_ci_bootstrap: Optional[tuple[float, float]] = None
    delta_ci_bootstrap: Optional[tuple[float, float]] = None
    candidate_ci_exact: Optional[tuple[float, float]] = None
    comparator_ci_exact: Optional[tuple[float, float]] = None
    delta_ci_exact: Optional[tuple[float, float]] = None

    def to_dict(self) -> dict:
        return _primitive(self)


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a quantile of an empty sequence")
    if q <= 0.0:
        return ordered[0]
    if q >= 1.0:
        return ordered[-1]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _sample_se(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _bootstrap_greater_p(
    boot: Sequence[float],
    estimate: float,
    null_boundary: float,
) -> float:
    """Centered-bootstrap one-sided p-value for H1: effect > null boundary."""
    distance = estimate - null_boundary
    extreme = sum(1 for value in boot if (value - estimate) >= distance)
    return (extreme + 1.0) / (len(boot) + 1.0)


# ---------------------------------------------------------------------------
# Boundary-aware proportion intervals.
#
# fast_p is the mean of 0/1 success indicators, so an arm that succeeds on every
# preregistered cell sits exactly on the boundary of the parameter space.  The
# nonparametric bootstrap is inconsistent there: the empirical distribution is
# degenerate, every resample is all-ones, and the percentile interval collapses
# to (1.0, 1.0) no matter how few cells were measured.  A zero-variance
# resampling distribution also drives the centered-bootstrap p-value to its
# floor, which is how a saturated arm can clear a superiority gate on no
# evidence.  We therefore pair the design-aware bootstrap with an exact
# (Clopper-Pearson) interval, which stays valid on the boundary.
# ---------------------------------------------------------------------------
def _normal_quantile(p: float) -> float:
    """Standard-normal quantile by bisection on ``erf`` (no scipy dependency)."""
    if not 0.0 < p < 1.0:
        raise ValueError("normal quantile requires p in (0, 1)")
    low, high = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < p:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _log_binom_coeff(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _binomial_tail(k: int, n: int, p: float, *, upper: bool) -> float:
    """``P(X >= k)`` (upper) or ``P(X <= k)`` (lower) for ``X ~ Binomial(n, p)``."""
    if upper:
        if k <= 0:
            return 1.0
        if k > n:
            return 0.0
    else:
        if k >= n:
            return 1.0
        if k < 0:
            return 0.0
    if p <= 0.0:
        return 1.0 if (upper and k <= 0) or (not upper and k >= 0) else 0.0
    if p >= 1.0:
        return 1.0 if (upper and k <= n) or (not upper and k >= n) else 0.0
    indices = range(k, n + 1) if upper else range(0, k + 1)
    total = 0.0
    for index in indices:
        total += math.exp(
            _log_binom_coeff(n, index)
            + index * math.log(p)
            + (n - index) * math.log1p(-p)
        )
    return min(1.0, max(0.0, total))


def clopper_pearson_interval(
    successes: int, n: int, ci_level: float = 0.95
) -> tuple[float, float]:
    """Exact (Clopper-Pearson) two-sided interval for a binomial proportion.

    Valid at ``p_hat in {0, 1}``: with ``k = n`` successes the lower bound is
    ``(alpha/2) ** (1/n)``, which shrinks toward 1 only as evidence accumulates.
    Solved by bisection on the exact binomial tail, so no scipy/Beta inverse is
    required and the result is reproducible.
    """
    if n <= 0:
        raise ValueError("Clopper-Pearson requires n > 0")
    if not 0 <= successes <= n:
        raise ValueError("successes must lie in [0, n]")
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must be in (0, 1)")
    tail = (1.0 - ci_level) / 2.0

    def _solve(target: float, *, upper_tail: bool, k: int) -> float:
        low, high = 0.0, 1.0
        for _ in range(80):
            mid = 0.5 * (low + high)
            value = _binomial_tail(k, n, mid, upper=upper_tail)
            # P(X >= k) increases in p; P(X <= k) decreases in p.
            if (value < target) == upper_tail:
                low = mid
            else:
                high = mid
        return 0.5 * (low + high)

    lower = 0.0 if successes == 0 else _solve(tail, upper_tail=True, k=successes)
    upper = 1.0 if successes == n else _solve(tail, upper_tail=False, k=successes)
    return (max(0.0, min(1.0, lower)), max(0.0, min(1.0, upper)))


def wilson_interval(
    successes: int, n: int, ci_level: float = 0.95
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (boundary-safe)."""
    if n <= 0:
        raise ValueError("Wilson interval requires n > 0")
    if not 0 <= successes <= n:
        raise ValueError("successes must lie in [0, n]")
    z = _normal_quantile(1.0 - (1.0 - ci_level) / 2.0)
    p_hat = successes / n
    denominator = 1.0 + z * z / n
    centre = (p_hat + z * z / (2.0 * n)) / denominator
    half = (
        z * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n))
    ) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def newcombe_difference_interval(
    candidate: tuple[float, float],
    candidate_estimate: float,
    comparator: tuple[float, float],
    comparator_estimate: float,
) -> tuple[float, float]:
    """Newcombe "square-and-add" interval for a difference of two proportions.

    Combines the two boundary-aware single-proportion intervals (Newcombe 1998).
    Built from independent-arm intervals, so for POSITIVELY correlated paired
    cells -- which is what a matched task x run rectangle produces -- it is
    conservative: it is never narrower than the correct paired interval.
    """
    delta = candidate_estimate - comparator_estimate
    lower_gap = math.hypot(
        candidate_estimate - candidate[0], comparator[1] - comparator_estimate
    )
    upper_gap = math.hypot(
        candidate[1] - candidate_estimate, comparator_estimate - comparator[0]
    )
    return (max(-1.0, delta - lower_gap), min(1.0, delta + upper_gap))


def _widest(
    first: Optional[tuple[float, float]], second: Optional[tuple[float, float]]
) -> tuple[float, float]:
    """The union of two intervals: never narrower than either component."""
    if first is None:
        return second  # type: ignore[return-value]
    if second is None:
        return first
    return (min(first[0], second[0]), max(first[1], second[1]))


def _binary(values: Sequence[float]) -> bool:
    return all(value in (0.0, 1.0) for value in values)


def _sign_greater_p(deltas: Sequence[float], null_boundary: float) -> float:
    """Exact one-sided paired SIGN-test p-value for H1: mean effect > boundary.

    Used when the resampling distribution is degenerate.  It needs no variance
    estimate -- which is exactly what a zero-variance bootstrap cannot supply --
    and for the saturated case (candidate wins every cell, comparator none) it
    reproduces the exact one-sided McNemar result ``0.5 ** n``.  It ignores
    clustering, so it is paired with the conservative boundary-aware interval:
    a gate must clear BOTH.
    """
    shifted = [float(value) - float(null_boundary) for value in deltas]
    positives = sum(1 for value in shifted if value > 0.0)
    negatives = sum(1 for value in shifted if value < 0.0)
    effective = positives + negatives
    if effective == 0:
        return 1.0
    return _binomial_tail(positives, effective, 0.5, upper=True)


def hierarchical_paired_bootstrap(
    data: Sequence[PairedDatum],
    *,
    n_boot: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 0,
    superiority_margin: float = 0.0,
    noninferiority_margin: float = 0.0,
) -> HierarchicalBootstrapResult:
    """Run a task-within-family-within-run paired cluster bootstrap.

    Runs are sampled with replacement.  Within each selected run, families are
    sampled with replacement; within each selected family, its tasks are sampled
    with replacement.  Candidate/comparator values always travel as a pair.

    For 0/1 success indicators the reported intervals are the UNION of that
    design-aware bootstrap interval and the boundary-exact interval
    (Clopper-Pearson per arm, Newcombe square-and-add for the paired delta), so
    they are never narrower than either component and never zero-width: a
    saturated arm reports the evidence its cell count actually supports instead
    of a false ``(1.0, 1.0)``.  When the delta resampling distribution is fully
    degenerate (``delta_se == 0``) the centered-bootstrap p-values are not
    inferential -- zero variance pins them to their ``1/(n_boot+1)`` floor
    regardless of evidence -- so they are replaced by exact one-sided paired
    sign tests.
    """
    rows = list(data)
    if not rows:
        raise ValueError("hierarchical bootstrap requires at least one paired datum")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if not (0.0 < ci_level < 1.0):
        raise ValueError("ci_level must be in (0, 1)")
    for row in rows:
        if not math.isfinite(row.candidate) or not math.isfinite(row.comparator):
            raise ValueError("hierarchical bootstrap inputs must be finite")

    hierarchy: dict[str, dict[str, list[PairedDatum]]] = {}
    for row in rows:
        hierarchy.setdefault(row.run_id, {}).setdefault(row.family_id, []).append(row)
    run_ids = sorted(hierarchy)
    families = sorted({row.family_id for row in rows})

    candidate_estimate = sum(row.candidate for row in rows) / len(rows)
    comparator_estimate = sum(row.comparator for row in rows) / len(rows)
    delta_estimate = candidate_estimate - comparator_estimate
    rng = random.Random(seed)
    candidate_boot: list[float] = []
    comparator_boot: list[float] = []
    delta_boot: list[float] = []

    for _ in range(n_boot):
        candidate_values: list[float] = []
        comparator_values: list[float] = []
        for _run_draw in run_ids:
            run_id = rng.choice(run_ids)
            run_families = sorted(hierarchy[run_id])
            for _family_draw in run_families:
                family_id = rng.choice(run_families)
                tasks = hierarchy[run_id][family_id]
                for _task_draw in tasks:
                    row = rng.choice(tasks)
                    candidate_values.append(row.candidate)
                    comparator_values.append(row.comparator)
        candidate_mean = sum(candidate_values) / len(candidate_values)
        comparator_mean = sum(comparator_values) / len(comparator_values)
        candidate_boot.append(candidate_mean)
        comparator_boot.append(comparator_mean)
        delta_boot.append(candidate_mean - comparator_mean)

    alpha = 1.0 - ci_level
    bounds = (alpha / 2.0, 1.0 - alpha / 2.0)
    candidate_boot_ci = tuple(float(_quantile(candidate_boot, q)) for q in bounds)
    comparator_boot_ci = tuple(float(_quantile(comparator_boot, q)) for q in bounds)
    delta_boot_ci = tuple(float(_quantile(delta_boot, q)) for q in bounds)
    delta_se = _sample_se(delta_boot)

    candidate_values = [row.candidate for row in rows]
    comparator_values = [row.comparator for row in rows]
    binary = _binary(candidate_values) and _binary(comparator_values)
    candidate_successes = comparator_successes = None
    candidate_exact = comparator_exact = delta_exact = None
    if binary:
        candidate_successes = int(round(sum(candidate_values)))
        comparator_successes = int(round(sum(comparator_values)))
        candidate_exact = clopper_pearson_interval(
            candidate_successes, len(rows), ci_level
        )
        comparator_exact = clopper_pearson_interval(
            comparator_successes, len(rows), ci_level
        )
        delta_exact = newcombe_difference_interval(
            candidate_exact, candidate_estimate, comparator_exact, comparator_estimate
        )

    candidate_ci = _widest(candidate_boot_ci, candidate_exact)
    comparator_ci = _widest(comparator_boot_ci, comparator_exact)
    delta_ci = _widest(delta_boot_ci, delta_exact)

    degenerate = delta_se == 0.0
    p_superiority = _bootstrap_greater_p(
        delta_boot, delta_estimate, float(superiority_margin)
    )
    p_noninferiority = _bootstrap_greater_p(
        delta_boot, delta_estimate, -float(noninferiority_margin)
    )
    p_method = "centered-cluster-bootstrap"
    if degenerate:
        # Zero resampling variance: the centered bootstrap tests nothing (its
        # p-value is pinned to the 1/(n_boot+1) floor regardless of evidence).
        # An exact paired sign test needs no variance estimate.
        deltas = [row.candidate - row.comparator for row in rows]
        p_superiority = _sign_greater_p(deltas, float(superiority_margin))
        p_noninferiority = _sign_greater_p(deltas, -float(noninferiority_margin))
        p_method = "exact-one-sided-sign-test"

    return HierarchicalBootstrapResult(
        n=len(rows),
        n_runs=len(run_ids),
        n_families=len(families),
        n_boot=n_boot,
        ci_level=ci_level,
        candidate_estimate=candidate_estimate,
        comparator_estimate=comparator_estimate,
        delta_estimate=delta_estimate,
        candidate_ci=(float(candidate_ci[0]), float(candidate_ci[1])),
        comparator_ci=(float(comparator_ci[0]), float(comparator_ci[1])),
        delta_ci=(float(delta_ci[0]), float(delta_ci[1])),
        delta_se=delta_se,
        p_superiority=p_superiority,
        p_noninferiority=p_noninferiority,
        superiority_margin=float(superiority_margin),
        noninferiority_margin=float(noninferiority_margin),
        binary_outcomes=binary,
        candidate_saturated=bool(
            binary and candidate_successes in (0, len(rows))
        ),
        comparator_saturated=bool(
            binary and comparator_successes in (0, len(rows))
        ),
        degenerate=degenerate,
        interval_method=(
            "union(cluster-bootstrap, clopper-pearson/newcombe)"
            if binary
            else "cluster-bootstrap"
        ),
        p_value_method=p_method,
        candidate_successes=candidate_successes,
        comparator_successes=comparator_successes,
        candidate_ci_bootstrap=candidate_boot_ci,
        comparator_ci_bootstrap=comparator_boot_ci,
        delta_ci_bootstrap=delta_boot_ci,
        candidate_ci_exact=candidate_exact,
        comparator_ci_exact=comparator_exact,
        delta_ci_exact=delta_exact,
    )


def _binomial_lower_tail(k: int, n: int) -> float:
    return sum(math.comb(n, index) for index in range(k + 1)) / (2.0**n)


def mcnemar_exact(candidate: Sequence[bool], comparator: Sequence[bool]) -> dict:
    """Exact two-sided McNemar test over paired binary outcomes."""
    if len(candidate) != len(comparator):
        raise ValueError("McNemar inputs must have equal length")
    candidate_only = sum(bool(a) and not bool(b) for a, b in zip(candidate, comparator))
    comparator_only = sum(not bool(a) and bool(b) for a, b in zip(candidate, comparator))
    discordant = candidate_only + comparator_only
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = min(
            1.0,
            2.0 * _binomial_lower_tail(min(candidate_only, comparator_only), discordant),
        )
    return {
        "candidate_only": candidate_only,
        "comparator_only": comparator_only,
        "discordant": discordant,
        "p_value": float(p_value),
    }


def paired_permutation_test(
    deltas: Sequence[float],
    *,
    seed: int = 0,
    n_permutations: int = 9_999,
    max_exact: int = 16,
) -> dict:
    """Two-sided paired sign-flip permutation test, exact when tractable."""
    values = [float(value) for value in deltas]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("permutation inputs must be finite")
    if not values:
        return {"statistic": 0.0, "p_value": 1.0, "n_effective": 0, "exact": True}
    nonzero = [value for value in values if value != 0.0]
    if not nonzero:
        return {"statistic": 0.0, "p_value": 1.0, "n_effective": 0, "exact": True}
    observed = abs(sum(values) / len(values))

    def statistic(signs: Sequence[int]) -> float:
        return abs(sum(sign * value for sign, value in zip(signs, nonzero)) / len(values))

    if len(nonzero) <= max_exact:
        total = 1 << len(nonzero)
        extreme = 0
        for mask in range(total):
            signs = [1 if mask & (1 << index) else -1 for index in range(len(nonzero))]
            if statistic(signs) >= observed - 1e-15:
                extreme += 1
        p_value = extreme / total
        exact = True
        draws = total
    else:
        if n_permutations <= 0:
            raise ValueError("n_permutations must be positive for Monte Carlo mode")
        rng = random.Random(seed)
        extreme = 0
        for _ in range(n_permutations):
            signs = [1 if rng.random() < 0.5 else -1 for _value in nonzero]
            if statistic(signs) >= observed - 1e-15:
                extreme += 1
        p_value = (extreme + 1.0) / (n_permutations + 1.0)
        exact = False
        draws = n_permutations
    return {
        "statistic": float(observed),
        "p_value": float(p_value),
        "n_effective": len(nonzero),
        "exact": exact,
        "draws": draws,
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down family-wise error correction."""
    checked: list[tuple[str, float]] = []
    for name, value in p_values.items():
        p_value = float(value)
        if not math.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
            raise ValueError(f"invalid p-value for {name}: {value}")
        checked.append((str(name), p_value))
    ordered = sorted(checked, key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * p_value))
        adjusted[name] = running
    return {name: adjusted[name] for name in p_values}


# ---------------------------------------------------------------------------
# Report schema.
# ---------------------------------------------------------------------------
@dataclass
class FastPInference:
    canonical_fast_p: float
    certified_lower_ci_fast_p: Optional[float]
    ci: Optional[tuple[float, float]]
    ci_level: Optional[float]
    successes: int
    denominator: int


@dataclass
class TrackReport:
    track: HarnessTrack
    denominator: int
    task_count: int
    run_count: int
    canonical_baseline: str
    canonical_p: float
    candidate: FastPInference
    comparator: FastPInference
    delta: float
    delta_ci: Optional[tuple[float, float]]
    delta_se: Optional[float]
    hidden_correctness: dict
    public_correctness: dict
    baseline_fast_p: dict
    sol_attainment: dict
    survivor_analysis: dict
    bootstrap: Optional[dict]
    secondary_tests: dict
    fingerprints: dict
    gates: dict


@dataclass
class FrontierReport:
    schema_version: str
    protocol_schema_version: str
    artifact_id: str
    artifact_payload_sha256: str
    profile: dict
    passed: bool
    errors: list[str]
    warnings: list[str]
    primary_holm: dict
    secondary_holm: dict
    tracks: dict[str, TrackReport]
    evidence_manifest: list[dict]
    artifact_source: dict
    preregistration: dict
    signature: Optional[dict]

    def to_dict(self) -> dict:
        return _primitive(self)


# ---------------------------------------------------------------------------
# Validation.
# ---------------------------------------------------------------------------
def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _valid_hash(value: str) -> bool:
    return bool(_HASH_RE.fullmatch(str(value)))


def _valid_source(url: str, revision: str) -> bool:
    parsed = urlparse(str(url))
    exact_revision = (
        bool(revision)
        and str(revision).lower() not in _MOVING_REVISIONS
        and not any(char.isspace() for char in str(revision))
        and len(str(revision)) >= 7
    )
    return parsed.scheme in {"https", "http"} and bool(parsed.netloc) and exact_revision


def _parse_utc(value: str) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo is not None else None


def _validate_fingerprint(
    fingerprint: Fingerprint,
    *,
    expected_kind: str,
    path: str,
    errors: list[str],
) -> None:
    if fingerprint.kind != expected_kind:
        errors.append(
            f"fingerprint: {path} kind={fingerprint.kind!r}, expected {expected_kind!r}"
        )
    if not _valid_hash(fingerprint.sha256):
        errors.append(f"fingerprint: {path} has invalid SHA-256")
        return
    try:
        parsed = json.loads(fingerprint.canonical_json)
        canonical = _canonical_json(parsed)
    except (TypeError, ValueError, json.JSONDecodeError):
        errors.append(f"fingerprint: {path} canonical payload is invalid JSON")
        return
    if canonical != fingerprint.canonical_json:
        errors.append(f"fingerprint: {path} payload is not canonical JSON")
    if sha256_text(fingerprint.canonical_json) != fingerprint.sha256:
        errors.append(f"fingerprint: {path} digest does not match payload")


def _validate_evidence(evidence: EvidenceRef, path: str, errors: list[str]) -> None:
    if not _valid_hash(evidence.raw_trace_sha256):
        errors.append(f"evidence: {path} has invalid raw_trace_sha256")
    if not _valid_hash(evidence.sample_sha256):
        errors.append(f"evidence: {path} has invalid sample_sha256")
    if not _valid_source(evidence.source_url, evidence.source_revision):
        errors.append(f"evidence: {path} lacks an exact source URL/revision")


_BUDGET_INTEGER_FIELDS = (
    "output_tokens",
    "tool_calls",
    "correctness_calls",
    "fresh_timed_calls",
    "profiler_calls",
)
_BUDGET_FLOAT_FIELDS = ("gpu_seconds", "wall_time_seconds", "cost_usd")


def _validate_budget(budget: BudgetEnvelope, path: str, errors: list[str]) -> None:
    for name in _BUDGET_INTEGER_FIELDS:
        value = getattr(budget, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"finite-input: {path}.{name} must be a non-negative integer")
    for name in _BUDGET_FLOAT_FIELDS:
        value = getattr(budget, name)
        if not _finite_number(value) or value < 0:
            errors.append(f"finite-input: {path}.{name} must be finite and non-negative")
    _validate_fingerprint(
        budget.context_policy,
        expected_kind="context-policy",
        path=f"{path}.context_policy",
        errors=errors,
    )


def _validate_usage(
    usage: ResourceUsage,
    budget: BudgetEnvelope,
    path: str,
    errors: list[str],
) -> None:
    for name in _BUDGET_INTEGER_FIELDS:
        value = getattr(usage, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"finite-input: {path}.{name} must be a non-negative integer")
        elif value > getattr(budget, name):
            errors.append(
                f"budget: {path}.{name}={value} exceeds matched ceiling "
                f"{getattr(budget, name)}"
            )
    for name in _BUDGET_FLOAT_FIELDS:
        value = getattr(usage, name)
        if not _finite_number(value) or value < 0:
            errors.append(f"finite-input: {path}.{name} must be finite and non-negative")
        elif value > getattr(budget, name) + 1e-12:
            errors.append(
                f"budget: {path}.{name}={value} exceeds matched ceiling "
                f"{getattr(budget, name)}"
            )


def _validate_arm(arm: ArmSpec, path: str, errors: list[str]) -> None:
    if not arm.arm_id:
        errors.append(f"schema: {path}.arm_id is empty")
    if not _valid_source(arm.source_url, arm.source_revision):
        errors.append(f"source: {path} lacks an exact source URL/revision")
    expected = {
        "artifact_fingerprint": "artifact",
        "checkpoint_fingerprint": "checkpoint",
        "tool_fingerprint": "tool",
        "prompt_fingerprint": "prompt",
        "verifier_fingerprint": "verifier",
        "hardware_fingerprint": "hardware",
        "budget_fingerprint": "budget",
    }
    for field_name, kind in expected.items():
        _validate_fingerprint(
            getattr(arm, field_name),
            expected_kind=kind,
            path=f"{path}.{field_name}",
            errors=errors,
        )
    _validate_budget(arm.budget, f"{path}.budget", errors)
    expected_budget = arm.budget.fingerprint()
    if (
        arm.budget_fingerprint.sha256 != expected_budget.sha256
        or arm.budget_fingerprint.canonical_json != expected_budget.canonical_json
    ):
        errors.append(f"fingerprint: {path}.budget_fingerprint does not match budget")


def _validate_quality(quality: DataQuality, path: str, errors: list[str]) -> None:
    if quality.smoke_data:
        errors.append(f"quality: {path} uses smoke data")
    if quality.fallback_data:
        errors.append(f"quality: {path} uses fallback data")
    if quality.contamination_detected:
        errors.append(f"quality: {path} reports contamination")


def _validate_track(
    track: TrackSpec,
    preregistered_at: Optional[datetime],
    errors: list[str],
) -> None:
    path = f"track[{track.track.value}]"
    _validate_arm(track.candidate, f"{path}.candidate", errors)
    _validate_arm(track.comparator, f"{path}.comparator", errors)
    _validate_quality(track.quality, path, errors)
    if track.skipped:
        errors.append(
            f"track: {track.track.value} comparator track was skipped"
            + (f" ({track.skip_reason})" if track.skip_reason else "")
        )
    if not track.run_ids or len(set(track.run_ids)) != len(track.run_ids):
        errors.append(f"schema: {path}.run_ids must be non-empty and unique")
    started = _parse_utc(track.started_at_utc)
    if started is None:
        errors.append(f"preregistration: {path}.started_at_utc is not timezone-aware ISO-8601")
    elif preregistered_at is not None and started <= preregistered_at:
        errors.append(f"preregistration: {path} started before registration")

    if track.candidate.budget != track.comparator.budget:
        errors.append(
            f"budget: {track.track.value} candidate/comparator envelopes are not exactly matched"
        )
    if (
        track.candidate.budget_fingerprint.sha256
        != track.comparator.budget_fingerprint.sha256
    ):
        errors.append(
            f"budget: {track.track.value} candidate/comparator budget fingerprints differ"
        )
    if (
        track.candidate.verifier_fingerprint.sha256
        != track.comparator.verifier_fingerprint.sha256
    ):
        errors.append(f"harness: {track.track.value} verifier fingerprints differ")
    if (
        track.candidate.hardware_fingerprint.sha256
        != track.comparator.hardware_fingerprint.sha256
    ):
        errors.append(f"harness: {track.track.value} hardware fingerprints differ")
    if (
        track.candidate.checkpoint_fingerprint.sha256
        == track.comparator.checkpoint_fingerprint.sha256
    ):
        errors.append(f"comparator: {track.track.value} compares an identical checkpoint")

    if track.track is HarnessTrack.CLOSED_MODEL:
        if (
            track.candidate.arm_kind is not ArmKind.MODEL
            or track.comparator.arm_kind is not ArmKind.MODEL
        ):
            errors.append("track: closed-model-harness requires two model arms")
        for name in ("tool_fingerprint", "prompt_fingerprint"):
            if getattr(track.candidate, name).sha256 != getattr(track.comparator, name).sha256:
                errors.append(f"harness: closed-model-harness {name} differs")
    elif track.track is HarnessTrack.OPEN_SYSTEM:
        if (
            track.candidate.arm_kind is not ArmKind.SYSTEM
            or track.comparator.arm_kind is not ArmKind.SYSTEM
        ):
            errors.append("track: open-system-harness requires two system arms")
        if (
            track.candidate.disclosure is not Disclosure.OPEN
            or track.comparator.disclosure is not Disclosure.OPEN
        ):
            errors.append("track: open-system-harness requires fully open arm disclosures")


def _baseline_map(sample: PairedTaskSample) -> dict[BaselineKind, BaselineObservation]:
    return {baseline.kind: baseline for baseline in sample.baselines}


def _validate_observation(
    observation: Optional[ArmObservation],
    budget: BudgetEnvelope,
    sol_time_ms: float,
    path: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    if observation is None:
        errors.append(f"coverage: {path} observation is missing")
        return
    _validate_evidence(observation.evidence, f"{path}.evidence", errors)
    _validate_usage(observation.usage, budget, f"{path}.usage", errors)
    if observation.hidden_correct is None:
        errors.append(f"correctness: {path}.hidden_correct is missing")
    elif observation.public_correct != observation.hidden_correct:
        warnings.append(f"correctness: {path} public and hidden verdicts disagree")
    if observation.time_ms is not None:
        if not _finite_number(observation.time_ms) or observation.time_ms <= 0:
            errors.append(f"finite-input: {path}.time_ms must be finite and positive")
        elif observation.time_ms < sol_time_ms * (1.0 - _SOL_TOLERANCE):
            errors.append(
                f"sol-integrity: {path}.time_ms={observation.time_ms} is below "
                f"SoL={sol_time_ms}"
            )
    elif observation.hidden_correct is True:
        errors.append(f"timing: {path} is hidden-correct but has no time")


def _validate_sample(
    sample: PairedTaskSample,
    track: TrackSpec,
    manifest_entry: TaskManifestEntry,
    path: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    if sample.family_id != manifest_entry.family_id:
        errors.append(
            f"manifest: {path}.family_id={sample.family_id!r} does not match "
            f"{manifest_entry.family_id!r}"
        )
    if not _finite_number(sample.sol_time_ms) or sample.sol_time_ms <= 0:
        errors.append(f"finite-input: {path}.sol_time_ms must be finite and positive")
        sol_time = float("inf")
    else:
        sol_time = float(sample.sol_time_ms)
    _validate_evidence(sample.sol_evidence, f"{path}.sol_evidence", errors)
    _validate_observation(
        sample.candidate,
        track.candidate.budget,
        sol_time,
        f"{path}.candidate",
        errors,
        warnings,
    )
    _validate_observation(
        sample.comparator,
        track.comparator.budget,
        sol_time,
        f"{path}.comparator",
        errors,
        warnings,
    )

    kinds = [baseline.kind for baseline in sample.baselines]
    if len(kinds) != len(set(kinds)):
        errors.append(f"baseline: {path} has duplicate baseline kinds")
    missing = [kind.value for kind in REQUIRED_BASELINES if kind not in kinds]
    if missing:
        errors.append(f"baseline: {path} is missing {', '.join(missing)}")
    baselines = _baseline_map(sample)
    for kind, baseline in baselines.items():
        baseline_path = f"{path}.baseline[{kind.value}]"
        _validate_evidence(baseline.evidence, f"{baseline_path}.evidence", errors)
        if baseline.hidden_correct is not True:
            errors.append(f"correctness: {baseline_path} lacks hidden correctness")
        if not _finite_number(baseline.time_ms) or baseline.time_ms <= 0:
            errors.append(f"finite-input: {baseline_path}.time_ms must be finite and positive")
        elif baseline.time_ms < sol_time * (1.0 - _SOL_TOLERANCE):
            errors.append(
                f"sol-integrity: {baseline_path}.time_ms={baseline.time_ms} is below "
                f"SoL={sol_time}"
            )
    production = baselines.get(BaselineKind.VENDOR_PRODUCTION)
    best_vendor = baselines.get(BaselineKind.BEST_VENDOR)
    if (
        production is not None
        and best_vendor is not None
        and _finite_number(production.time_ms)
        and _finite_number(best_vendor.time_ms)
        and best_vendor.time_ms > production.time_ms * (1.0 + 1e-12)
    ):
        errors.append(f"baseline: {path} best-vendor is slower than vendor-production")


def validate_artifact(
    artifact: FrontierArtifact,
    profile: ClaimProfile,
) -> tuple[list[str], list[str]]:
    """Return publication-integrity errors and non-fatal warnings."""
    errors: list[str] = []
    warnings: list[str] = []
    if artifact.schema_version != PROTOCOL_SCHEMA_VERSION:
        errors.append(
            f"schema: artifact version {artifact.schema_version!r} != "
            f"{PROTOCOL_SCHEMA_VERSION!r}"
        )
    if not artifact.artifact_id:
        errors.append("schema: artifact_id is empty")
    if not _valid_source(artifact.source_url, artifact.source_revision):
        errors.append("source: artifact lacks an exact source URL/revision")
    _validate_quality(artifact.quality, "artifact", errors)

    preregistered = _parse_utc(artifact.preregistration.registered_at_utc)
    if preregistered is None:
        errors.append("preregistration: registered_at_utc is not timezone-aware ISO-8601")
    if artifact.preregistration.profile_name != profile.name:
        errors.append(
            f"preregistration: profile {artifact.preregistration.profile_name!r} does not "
            f"match {profile.name!r}"
        )
    if artifact.preregistration.profile_fingerprint != profile.fingerprint:
        errors.append("preregistration: profile fingerprint does not match immutable schema")
    expected_manifest_fingerprint = manifest_fingerprint(artifact.manifest)
    if artifact.preregistration.manifest_fingerprint != expected_manifest_fingerprint:
        errors.append("preregistration: manifest fingerprint does not match artifact")

    if artifact.signature is None:
        errors.append("signature: artifact is unsigned")
    else:
        signature = artifact.signature
        if not signature.verified:
            errors.append("signature: artifact signature was not verified")
        if not signature.signature.strip():
            errors.append("signature: detached signature is empty")
        if not _valid_hash(signature.signer_key_sha256):
            errors.append("signature: signer key fingerprint is invalid")
        if signature.payload_sha256 != artifact_payload_sha256(artifact):
            errors.append("signature: signed payload hash does not match artifact")

    if not artifact.manifest:
        errors.append("manifest: task manifest is empty")
    task_ids = [entry.task_id for entry in artifact.manifest]
    if len(task_ids) != len(set(task_ids)):
        errors.append("manifest: task IDs are not unique")
    for index, entry in enumerate(artifact.manifest):
        path = f"manifest[{index}]"
        if not entry.task_id or not entry.family_id:
            errors.append(f"manifest: {path} has an empty task/family ID")
        if not _valid_hash(entry.task_sha256):
            errors.append(f"manifest: {path}.task_sha256 is invalid")
        if not _valid_source(entry.source_url, entry.source_revision):
            errors.append(f"manifest: {path} lacks an exact source URL/revision")

    track_map = {track.track: track for track in artifact.tracks}
    if len(track_map) != len(artifact.tracks):
        errors.append("track: duplicate track specifications")
    for required_track in profile.required_tracks:
        if required_track not in track_map:
            errors.append(f"track: required {required_track.value} is missing")
    for track in artifact.tracks:
        _validate_track(track, preregistered, errors)

    manifest_map = {entry.task_id: entry for entry in artifact.manifest}
    sample_map: dict[tuple[HarnessTrack, str, str], PairedTaskSample] = {}
    for index, sample in enumerate(artifact.samples):
        key = (sample.track, sample.run_id, sample.task_id)
        path = (
            f"sample[{sample.track.value}/{sample.run_id}/{sample.task_id}]"
        )
        if key in sample_map:
            errors.append(f"coverage: duplicate {path}")
            continue
        sample_map[key] = sample
        track = track_map.get(sample.track)
        manifest_entry = manifest_map.get(sample.task_id)
        if track is None:
            errors.append(f"track: {path} references an unknown track")
            continue
        if sample.run_id not in track.run_ids:
            errors.append(f"coverage: {path} references an unregistered run")
        if manifest_entry is None:
            errors.append(f"coverage: {path} references an unregistered task")
            continue
        _validate_sample(sample, track, manifest_entry, path, errors, warnings)

    for track_kind in profile.required_tracks:
        track = track_map.get(track_kind)
        if track is None:
            continue
        for run_id in track.run_ids:
            for entry in artifact.manifest:
                key = (track_kind, run_id, entry.task_id)
                if key not in sample_map:
                    errors.append(
                        f"coverage: missing sample[{track_kind.value}/{run_id}/"
                        f"{entry.task_id}]"
                    )

    if len(artifact.manifest) < profile.min_tasks:
        errors.append(
            f"profile: {profile.name} requires >= {profile.min_tasks} tasks, "
            f"got {len(artifact.manifest)}"
        )
    family_count = len({entry.family_id for entry in artifact.manifest})
    if family_count < profile.min_families:
        errors.append(
            f"profile: {profile.name} requires >= {profile.min_families} families, "
            f"got {family_count}"
        )
    for track_kind in profile.required_tracks:
        track = track_map.get(track_kind)
        run_count = len(track.run_ids) if track is not None else 0
        if run_count < profile.min_runs_per_track:
            errors.append(
                f"profile: {profile.name} requires >= {profile.min_runs_per_track} "
                f"runs for {track_kind.value}, got {run_count}"
            )

    return list(dict.fromkeys(errors)), list(dict.fromkeys(warnings))


# ---------------------------------------------------------------------------
# Scoring and gates.
# ---------------------------------------------------------------------------
def _timing_valid(observation: Optional[ArmObservation], sol_time_ms: float) -> bool:
    return bool(
        observation is not None
        and observation.hidden_correct is True
        and _finite_number(observation.time_ms)
        and observation.time_ms > 0
        and _finite_number(sol_time_ms)
        and sol_time_ms > 0
        and observation.time_ms >= sol_time_ms * (1.0 - _SOL_TOLERANCE)
    )


def _baseline_valid(
    baseline: Optional[BaselineObservation],
    sol_time_ms: float,
) -> bool:
    return bool(
        baseline is not None
        and baseline.hidden_correct is True
        and _finite_number(baseline.time_ms)
        and baseline.time_ms > 0
        and _finite_number(sol_time_ms)
        and sol_time_ms > 0
        and baseline.time_ms >= sol_time_ms * (1.0 - _SOL_TOLERANCE)
    )


def _fastp_inputs(
    cells: Sequence[Optional[PairedTaskSample]],
    side: str,
    baseline_kind: BaselineKind,
) -> tuple[list[bool], list[Optional[float]], list[float], list[bool]]:
    correctness: list[bool] = []
    baseline_times: list[Optional[float]] = []
    actual_times: list[float] = []
    successes: list[bool] = []
    for sample in cells:
        observation = getattr(sample, side) if sample is not None else None
        baseline = _baseline_map(sample).get(baseline_kind) if sample is not None else None
        valid = bool(
            sample is not None
            and _timing_valid(observation, sample.sol_time_ms)
            and _baseline_valid(baseline, sample.sol_time_ms)
        )
        baseline_time = float(baseline.time_ms) if valid and baseline is not None else None
        actual_time = float(observation.time_ms) if valid and observation is not None else math.inf
        correctness.append(valid)
        baseline_times.append(baseline_time)
        actual_times.append(actual_time)
        successes.append(
            bool(valid and baseline_time is not None and baseline_time / actual_time > 1.0)
        )
    return correctness, baseline_times, actual_times, successes


def _successes_at_p(
    cells: Sequence[Optional[PairedTaskSample]],
    side: str,
    baseline_kind: BaselineKind,
    p: float,
) -> list[bool]:
    outcomes: list[bool] = []
    for sample in cells:
        observation = getattr(sample, side) if sample is not None else None
        baseline = _baseline_map(sample).get(baseline_kind) if sample is not None else None
        valid = bool(
            sample is not None
            and _timing_valid(observation, sample.sol_time_ms)
            and _baseline_valid(baseline, sample.sol_time_ms)
        )
        outcomes.append(
            bool(
                valid
                and observation is not None
                and baseline is not None
                and baseline.time_ms / observation.time_ms > p
            )
        )
    return outcomes


def _point_fastp(
    cells: Sequence[Optional[PairedTaskSample]],
    side: str,
    baseline_kind: BaselineKind,
    p: float,
) -> float:
    correctness: list[bool] = []
    baseline_times: list[Optional[float]] = []
    actual_times: list[float] = []
    for sample in cells:
        observation = getattr(sample, side) if sample is not None else None
        baseline = _baseline_map(sample).get(baseline_kind) if sample is not None else None
        valid = bool(
            sample is not None
            and _timing_valid(observation, sample.sol_time_ms)
            and _baseline_valid(baseline, sample.sol_time_ms)
        )
        correctness.append(valid)
        baseline_times.append(float(baseline.time_ms) if valid and baseline is not None else None)
        actual_times.append(
            float(observation.time_ms)
            if valid and observation is not None
            else math.inf
        )
    return fastp(correctness, baseline_times, actual_times, len(cells), float(p))


def _correctness_summary(
    cells: Sequence[Optional[PairedTaskSample]],
    attribute: str,
) -> dict:
    result: dict[str, dict] = {}
    for side in ("candidate", "comparator"):
        values = []
        missing = 0
        for sample in cells:
            observation = getattr(sample, side) if sample is not None else None
            value = getattr(observation, attribute) if observation is not None else None
            if value is None:
                missing += 1
                values.append(False)
            else:
                values.append(bool(value))
        result[side] = {
            "rate": sum(values) / len(values) if values else 0.0,
            "correct": sum(values),
            "denominator": len(values),
            "missing": missing,
        }
    return result


def _sol_summary(cells: Sequence[Optional[PairedTaskSample]]) -> dict:
    result: dict[str, dict] = {}
    for side in ("candidate", "comparator"):
        attainments: list[float] = []
        for sample in cells:
            observation = getattr(sample, side) if sample is not None else None
            if sample is not None and _timing_valid(observation, sample.sol_time_ms):
                attainment = sample.sol_time_ms / float(observation.time_ms)
                attainments.append(min(1.0, max(0.0, attainment)))
            else:
                attainments.append(0.0)
        result[side] = {
            "mean_full_denominator": (
                sum(attainments) / len(attainments) if attainments else 0.0
            ),
            "max": max(attainments, default=0.0),
            "denominator": len(attainments),
            "interpretation": "fraction of physical SoL ceiling attained; not a comparator",
        }
    return result


def _survivor_summary(
    cells: Sequence[Optional[PairedTaskSample]],
    track: HarnessTrack,
    warnings: list[str],
) -> dict:
    ratios: list[float] = []
    for sample in cells:
        if sample is None:
            continue
        if (
            _timing_valid(sample.candidate, sample.sol_time_ms)
            and _timing_valid(sample.comparator, sample.sol_time_ms)
            and sample.candidate is not None
            and sample.comparator is not None
        ):
            ratios.append(sample.comparator.time_ms / sample.candidate.time_ms)
    ratio = math.exp(sum(math.log(value) for value in ratios) / len(ratios)) if ratios else None
    excluded = len(cells) - len(ratios)
    warning = (
        f"survivor-bias: {track.value} both-correct speed ratio excludes "
        f"{excluded}/{len(cells)} cells and is descriptive only"
    )
    if excluded:
        warnings.append(warning)
    return {
        "both_correct_cells": len(ratios),
        "full_denominator": len(cells),
        "excluded_cells": excluded,
        "geomean_candidate_over_comparator_speed": ratio,
        "gate_eligible": False,
        "warning": warning if excluded else (
            "Both-correct ratio remains descriptive and is never used for a claim gate."
        ),
    }


def _fingerprint_summary(track: TrackSpec) -> dict:
    names = (
        "artifact_fingerprint",
        "checkpoint_fingerprint",
        "tool_fingerprint",
        "prompt_fingerprint",
        "verifier_fingerprint",
        "hardware_fingerprint",
        "budget_fingerprint",
    )
    return {
        side: {
            name.removesuffix("_fingerprint"): getattr(getattr(track, side), name).sha256
            for name in names
        }
        for side in ("candidate", "comparator")
    }


def _empty_track_report(track: TrackSpec, artifact: FrontierArtifact, p: float) -> TrackReport:
    denominator = len(artifact.manifest) * len(track.run_ids)
    empty = FastPInference(0.0, None, None, None, 0, denominator)
    return TrackReport(
        track=track.track,
        denominator=denominator,
        task_count=len(artifact.manifest),
        run_count=len(track.run_ids),
        canonical_baseline=CANONICAL_BASELINE,
        canonical_p=p,
        candidate=empty,
        comparator=FastPInference(0.0, None, None, None, 0, denominator),
        delta=0.0,
        delta_ci=None,
        delta_se=None,
        hidden_correctness={},
        public_correctness={},
        baseline_fast_p={},
        sol_attainment={},
        survivor_analysis={
            "both_correct_cells": 0,
            "full_denominator": denominator,
            "excluded_cells": denominator,
            "geomean_candidate_over_comparator_speed": None,
            "gate_eligible": False,
            "warning": "Track skipped; survivor analysis unavailable.",
        },
        bootstrap=None,
        secondary_tests={},
        fingerprints=_fingerprint_summary(track),
        gates={},
    )


def _evaluate_track(
    artifact: FrontierArtifact,
    track: TrackSpec,
    profile: ClaimProfile,
    sample_map: Mapping[tuple[HarnessTrack, str, str], PairedTaskSample],
    warnings: list[str],
    seed: int,
) -> TrackReport:
    if track.skipped:
        return _empty_track_report(track, artifact, profile.canonical_p)

    cells: list[Optional[PairedTaskSample]] = []
    data: list[PairedDatum] = []
    candidate_successes: list[bool] = []
    comparator_successes: list[bool] = []
    for run_id in track.run_ids:
        for entry in artifact.manifest:
            sample = sample_map.get((track.track, run_id, entry.task_id))
            cells.append(sample)
            candidate_success = _successes_at_p(
                [sample], "candidate", BaselineKind.VENDOR_PRODUCTION, profile.canonical_p
            )[0]
            comparator_success = _successes_at_p(
                [sample], "comparator", BaselineKind.VENDOR_PRODUCTION, profile.canonical_p
            )[0]
            candidate_successes.append(candidate_success)
            comparator_successes.append(comparator_success)
            data.append(
                PairedDatum(
                    run_id=run_id,
                    family_id=entry.family_id,
                    task_id=entry.task_id,
                    candidate=float(candidate_success),
                    comparator=float(comparator_success),
                )
            )

    bootstrap = hierarchical_paired_bootstrap(
        data,
        n_boot=profile.n_boot,
        ci_level=profile.ci_level,
        seed=seed,
        superiority_margin=profile.superiority_margin,
        noninferiority_margin=profile.noninferiority_margin,
    )
    candidate_point = _point_fastp(
        cells, "candidate", BaselineKind.VENDOR_PRODUCTION, profile.canonical_p
    )
    comparator_point = _point_fastp(
        cells, "comparator", BaselineKind.VENDOR_PRODUCTION, profile.canonical_p
    )
    baseline_fast_p: dict[str, dict] = {}
    for baseline_kind in REQUIRED_BASELINES:
        baseline_fast_p[baseline_kind.value] = {
            "candidate": _point_fastp(
                cells, "candidate", baseline_kind, profile.canonical_p
            ),
            "comparator": _point_fastp(
                cells, "comparator", baseline_kind, profile.canonical_p
            ),
            "p": profile.canonical_p,
            "denominator": len(cells),
        }

    deltas = [
        float(candidate) - float(comparator)
        for candidate, comparator in zip(candidate_successes, comparator_successes)
    ]
    mcnemar = mcnemar_exact(candidate_successes, comparator_successes)
    permutation = paired_permutation_test(deltas, seed=seed)
    wilcoxon = wilcoxon_signed_rank(deltas).to_dict()
    secondary = {
        "mcnemar": mcnemar,
        "paired_permutation": permutation,
        "wilcoxon": wilcoxon,
    }
    return TrackReport(
        track=track.track,
        denominator=len(cells),
        task_count=len(artifact.manifest),
        run_count=len(track.run_ids),
        canonical_baseline=CANONICAL_BASELINE,
        canonical_p=profile.canonical_p,
        candidate=FastPInference(
            canonical_fast_p=candidate_point,
            certified_lower_ci_fast_p=bootstrap.candidate_ci[0],
            ci=bootstrap.candidate_ci,
            ci_level=profile.ci_level,
            successes=sum(candidate_successes),
            denominator=len(cells),
        ),
        comparator=FastPInference(
            canonical_fast_p=comparator_point,
            certified_lower_ci_fast_p=bootstrap.comparator_ci[0],
            ci=bootstrap.comparator_ci,
            ci_level=profile.ci_level,
            successes=sum(comparator_successes),
            denominator=len(cells),
        ),
        delta=candidate_point - comparator_point,
        delta_ci=bootstrap.delta_ci,
        delta_se=bootstrap.delta_se,
        hidden_correctness=_correctness_summary(cells, "hidden_correct"),
        public_correctness=_correctness_summary(cells, "public_correct"),
        baseline_fast_p=baseline_fast_p,
        sol_attainment=_sol_summary(cells),
        survivor_analysis=_survivor_summary(cells, track.track, warnings),
        bootstrap=bootstrap.to_dict(),
        secondary_tests=secondary,
        fingerprints=_fingerprint_summary(track),
        gates={},
    )


def _evidence_manifest(artifact: FrontierArtifact) -> list[dict]:
    rows: list[dict] = []
    for sample in artifact.samples:
        rows.append(
            {
                "track": sample.track.value,
                "run_id": sample.run_id,
                "task_id": sample.task_id,
                "candidate": (
                    _primitive(sample.candidate.evidence)
                    if sample.candidate is not None
                    else None
                ),
                "comparator": (
                    _primitive(sample.comparator.evidence)
                    if sample.comparator is not None
                    else None
                ),
                "baselines": {
                    baseline.kind.value: _primitive(baseline.evidence)
                    for baseline in sample.baselines
                },
                "sol": _primitive(sample.sol_evidence),
            }
        )
    return rows


def evaluate_claim(
    artifact: FrontierArtifact,
    *,
    profile_name: Optional[str] = None,
    seed: int = 0,
) -> FrontierReport:
    """Evaluate a signed, preregistered artifact without external side effects."""
    selected_name = profile_name or artifact.preregistration.profile_name
    if selected_name not in CLAIM_PROFILES:
        raise ValueError(f"unknown claim profile: {selected_name!r}")
    profile = CLAIM_PROFILES[selected_name]
    errors, warnings = validate_artifact(artifact, profile)
    if profile_name is not None and profile_name != artifact.preregistration.profile_name:
        errors.append("preregistration: runtime profile differs from preregistered profile")

    track_specs = {track.track: track for track in artifact.tracks}
    sample_map = {
        (sample.track, sample.run_id, sample.task_id): sample for sample in artifact.samples
    }
    reports: dict[str, TrackReport] = {}
    for offset, track_kind in enumerate(profile.required_tracks):
        track = track_specs.get(track_kind)
        if track is None:
            continue
        reports[track_kind.value] = _evaluate_track(
            artifact,
            track,
            profile,
            sample_map,
            warnings,
            seed + offset * 1_000_003,
        )

    # Missing inferential intervals are categorical failures for every profile.
    for track_kind in profile.required_tracks:
        report = reports.get(track_kind.value)
        if (
            report is None
            or report.candidate.ci is None
            or report.comparator.ci is None
            or report.delta_ci is None
        ):
            errors.append(f"ci: required confidence intervals missing for {track_kind.value}")
            continue
        bootstrap = report.bootstrap or {}
        if not bootstrap.get("degenerate"):
            continue
        if bootstrap.get("binary_outcomes"):
            # Recoverable: the paired resampling distribution collapsed because an
            # arm sits on the 0/1 boundary, so the reported interval and p-values
            # come from the boundary-exact substitution instead of the bootstrap.
            warnings.append(
                f"{BOOTSTRAP_DEGENERATE}: {track_kind.value} paired bootstrap has zero "
                f"variance (delta_se=0); interval and p-values come from "
                f"{bootstrap.get('interval_method')} / {bootstrap.get('p_value_method')}"
            )
        else:
            errors.append(
                f"{BOOTSTRAP_DEGENERATE}: {track_kind.value} paired bootstrap has zero "
                "variance and the outcomes are not 0/1 successes, so no boundary-aware "
                "interval exists; the reported interval is not inferential"
            )

    # Primary hypothesis family: NI on both tracks plus preregistered superiority tests.
    superiority_tracks = set(profile.required_superiority_tracks) | set(
        profile.superiority_any_of
    )
    primary_raw: dict[str, float] = {}
    for track_kind in profile.required_tracks:
        report = reports.get(track_kind.value)
        if report is None or report.bootstrap is None:
            continue
        primary_raw[f"{track_kind.value}:non-inferiority"] = float(
            report.bootstrap["p_noninferiority"]
        )
        if track_kind in superiority_tracks:
            primary_raw[f"{track_kind.value}:superiority"] = float(
                report.bootstrap["p_superiority"]
            )
    primary_holm = holm_adjust(primary_raw)

    ni_pass: dict[HarnessTrack, bool] = {}
    sup_pass: dict[HarnessTrack, bool] = {}
    for track_kind in profile.required_tracks:
        report = reports.get(track_kind.value)
        ni_key = f"{track_kind.value}:non-inferiority"
        ci_lower = report.delta_ci[0] if report is not None and report.delta_ci else None
        ni_ok = bool(
            ci_lower is not None
            and ci_lower > -profile.noninferiority_margin
            and primary_holm.get(ni_key, 1.0) <= profile.alpha
        )
        ni_pass[track_kind] = ni_ok
        sup_key = f"{track_kind.value}:superiority"
        sup_ok = bool(
            ci_lower is not None
            and ci_lower > profile.superiority_margin
            and primary_holm.get(sup_key, 1.0) <= profile.alpha
        )
        sup_pass[track_kind] = sup_ok
        if report is not None:
            bootstrap = report.bootstrap or {}
            report.gates = {
                "interval": {
                    "method": bootstrap.get("interval_method"),
                    "p_value_method": bootstrap.get("p_value_method"),
                    BOOTSTRAP_DEGENERATE: bool(bootstrap.get("degenerate")),
                    "candidate_saturated": bool(bootstrap.get("candidate_saturated")),
                    "comparator_saturated": bool(bootstrap.get("comparator_saturated")),
                },
                "noninferiority": {
                    "passed": ni_ok,
                    "margin": profile.noninferiority_margin,
                    "ci_lower": ci_lower,
                    "holm_p": primary_holm.get(ni_key),
                },
                "superiority": {
                    "required": track_kind in superiority_tracks,
                    "passed": sup_ok if track_kind in superiority_tracks else None,
                    "margin": profile.superiority_margin,
                    "ci_lower": ci_lower,
                    "holm_p": primary_holm.get(sup_key),
                },
            }

    # Secondary tests are explicitly secondary and corrected as one family.
    secondary_raw: dict[str, float] = {}
    for track_name, report in reports.items():
        tests = report.secondary_tests
        if not tests:
            continue
        secondary_raw[f"{track_name}:mcnemar"] = float(tests["mcnemar"]["p_value"])
        secondary_raw[f"{track_name}:paired-permutation"] = float(
            tests["paired_permutation"]["p_value"]
        )
        secondary_raw[f"{track_name}:wilcoxon"] = float(tests["wilcoxon"]["p_value"])
    secondary_holm = holm_adjust(secondary_raw)
    for track_name, report in reports.items():
        for test_name, key_name in (
            ("mcnemar", "mcnemar"),
            ("paired_permutation", "paired-permutation"),
            ("wilcoxon", "wilcoxon"),
        ):
            if test_name in report.secondary_tests:
                report.secondary_tests[test_name]["holm_p_value"] = secondary_holm.get(
                    f"{track_name}:{key_name}"
                )

    superiority_required_ok = all(
        sup_pass.get(track_kind, False)
        for track_kind in profile.required_superiority_tracks
    )
    superiority_any_ok = (
        any(sup_pass.get(track_kind, False) for track_kind in profile.superiority_any_of)
        if profile.superiority_any_of
        else True
    )
    statistical_ok = (
        all(ni_pass.get(track_kind, False) for track_kind in profile.required_tracks)
        and superiority_required_ok
        and superiority_any_ok
    )
    passed = not errors and statistical_ok
    if not statistical_ok:
        warnings.append(
            "gate: preregistered non-inferiority/superiority conditions were not met"
        )

    return FrontierReport(
        schema_version=REPORT_SCHEMA_VERSION,
        protocol_schema_version=PROTOCOL_SCHEMA_VERSION,
        artifact_id=artifact.artifact_id,
        artifact_payload_sha256=artifact_payload_sha256(artifact),
        profile=profile.to_dict(),
        passed=passed,
        errors=list(dict.fromkeys(errors)),
        warnings=list(dict.fromkeys(warnings)),
        primary_holm=primary_holm,
        secondary_holm=secondary_holm,
        tracks=reports,
        evidence_manifest=_evidence_manifest(artifact),
        artifact_source={
            "url": artifact.source_url,
            "revision": artifact.source_revision,
        },
        preregistration=_primitive(artifact.preregistration),
        signature=_primitive(artifact.signature) if artifact.signature is not None else None,
    )


def format_frontier_report(report: FrontierReport) -> str:
    """Compact publication-facing markdown summary."""
    status = "PASS" if report.passed else "FAIL"
    lines = [
        f"# Frontier claim: {status}",
        "",
        f"- profile: {report.profile['name']}",
        f"- artifact: `{report.artifact_id}`",
        f"- payload SHA-256: `{report.artifact_payload_sha256}`",
        "- SoL: physical integrity ceiling/attainment metric, never a comparator",
        "",
    ]
    for track_name, track in report.tracks.items():
        lines.extend(
            [
                f"## {track_name}",
                "",
                f"- full denominator: {track.denominator}",
                (
                    "- candidate canonical fast_p: "
                    f"{track.candidate.canonical_fast_p:.4f}"
                ),
                (
                    "- candidate certified lower-CI fast_p: "
                    f"{track.candidate.certified_lower_ci_fast_p}"
                ),
                (
                    "- comparator canonical fast_p: "
                    f"{track.comparator.canonical_fast_p:.4f}"
                ),
                f"- paired delta: {track.delta:+.4f}; CI={track.delta_ci}",
                "",
            ]
        )
    if report.errors:
        lines.extend(["## Integrity errors", ""])
        lines.extend(f"- {error}" for error in report.errors)
        lines.append("")
    if report.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report.warnings)
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "PROTOCOL_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "BOOTSTRAP_DEGENERATE",
    "CANONICAL_BASELINE",
    "HarnessTrack",
    "ArmKind",
    "Disclosure",
    "BaselineKind",
    "REQUIRED_BASELINES",
    "Fingerprint",
    "BudgetEnvelope",
    "ResourceUsage",
    "EvidenceRef",
    "ArmSpec",
    "DataQuality",
    "TrackSpec",
    "TaskManifestEntry",
    "ArmObservation",
    "BaselineObservation",
    "PairedTaskSample",
    "ArtifactSignature",
    "Preregistration",
    "FrontierArtifact",
    "ClaimProfile",
    "CLAIM_PROFILES",
    "PairedDatum",
    "HierarchicalBootstrapResult",
    "FastPInference",
    "TrackReport",
    "FrontierReport",
    "sha256_text",
    "artifact_payload_sha256",
    "manifest_fingerprint",
    "hierarchical_paired_bootstrap",
    "clopper_pearson_interval",
    "wilson_interval",
    "newcombe_difference_interval",
    "mcnemar_exact",
    "paired_permutation_test",
    "holm_adjust",
    "validate_artifact",
    "evaluate_claim",
    "format_frontier_report",
]
