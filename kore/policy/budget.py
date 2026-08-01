"""Versioned, non-overlapping accounting for GRPO compute budgets.

The ledger deliberately does not derive one counter from another.  A timed
evaluation may also perform correctness work, a replay hit performs neither,
and generated tokens are not optimizer tokens when samples are filtered or
reused for multiple PPO epochs.  Callers must therefore report each physical
event explicitly.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


SCHEMA_VERSION = "BudgetLedgerV1"

_INTEGER_COUNTERS = (
    "generated_tokens",
    "optimizer_tokens",
    "correctness_calls",
    "fresh_timed_calls",
    "replay_hits",
    "groups_attempted",
    "groups_kept",
)
_FLOAT_COUNTERS = (
    "verifier_gpu_seconds",
    "profiler_gpu_seconds",
)
_ALL_COUNTERS = _INTEGER_COUNTERS + _FLOAT_COUNTERS

# The evaluation-path subset, in ledger-counter naming so one observed record
# maps onto the ledger without any translation table.
_EVALUATION_INTEGER_FIELDS = (
    "correctness_calls",
    "fresh_timed_calls",
    "replay_hits",
)
_EVALUATION_FLOAT_FIELDS = (
    "verifier_gpu_seconds",
    "profiler_gpu_seconds",
)
_EVALUATION_FIELDS = _EVALUATION_INTEGER_FIELDS + _EVALUATION_FLOAT_FIELDS


class BudgetError(ValueError):
    """Base class for malformed or inconsistent budget state."""


class BudgetExceededError(BudgetError):
    """Raised before a ledger update would exceed a configured hard limit."""

    def __init__(self, counter: str, attempted: float, limit: float) -> None:
        self.counter = counter
        self.attempted = attempted
        self.limit = limit
        super().__init__(
            f"budget exceeded for {counter}: attempted {attempted}, limit {limit}"
        )


def _finite_nonnegative(name: str, value: Any, *, integer: bool) -> int | float:
    if isinstance(value, bool):
        raise BudgetError(f"{name} must be a non-negative number, not bool")
    if integer:
        if not isinstance(value, int):
            raise BudgetError(f"{name} must be a non-negative integer")
        if value < 0:
            raise BudgetError(f"{name} must be non-negative")
        return value
    if not isinstance(value, (int, float)):
        raise BudgetError(f"{name} must be a non-negative finite number")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise BudgetError(f"{name} must be non-negative and finite")
    return value


@dataclass(frozen=True)
class EvaluationWork:
    """The physical work ONE evaluation performed, as observed.

    Every field is an independent observation of a distinct physical event, so
    the ledger's non-derivation rule survives: a timed evaluation reports its own
    timing call, a correctness-only evaluation reports no timing call, and a
    replay hit reports neither because it ran nothing.  The seconds are measured
    wall time attributed to the stage that spent it, never inferred from a count.
    """

    correctness_calls: int = 0
    fresh_timed_calls: int = 0
    replay_hits: int = 0
    verifier_gpu_seconds: float = 0.0
    profiler_gpu_seconds: float = 0.0

    def __post_init__(self) -> None:
        for name in _EVALUATION_INTEGER_FIELDS:
            object.__setattr__(
                self, name,
                _finite_nonnegative(name, getattr(self, name), integer=True),
            )
        for name in _EVALUATION_FLOAT_FIELDS:
            object.__setattr__(
                self, name,
                _finite_nonnegative(name, getattr(self, name), integer=False),
            )
        calls = self.correctness_calls + self.fresh_timed_calls
        seconds = self.verifier_gpu_seconds + self.profiler_gpu_seconds
        if self.replay_hits and (calls or seconds):
            raise BudgetError(
                "a replay hit performs no correctness, timing or profiling work")
        if seconds > 0.0 and not calls:
            raise BudgetError(
                "measured GPU seconds require at least one correctness or timed call")

    @property
    def is_empty(self) -> bool:
        return not any(getattr(self, name) for name in _EVALUATION_FIELDS)

    def to_dict(self) -> dict[str, int | float]:
        return {name: getattr(self, name) for name in _EVALUATION_FIELDS}

    @classmethod
    def from_observation(
        cls,
        observation: Any,
        *,
        verifier_seconds: float,
        profiler_seconds: float = 0.0,
        replayed: bool = False,
        executed: bool = True,
    ) -> "EvaluationWork":
        """Read one KoreEnv observation as physical work.

        ``replayed`` marks a cache-served evaluation: it consumed no GPU, so it
        is charged as a replay hit and nothing else, whatever its wall time was.
        ``executed`` is the caller's assertion that a candidate-bearing
        subprocess actually ran; pass ``False`` for evaluations rejected before
        launch (static hack scan, source-budget violation).  It defaults to
        ``True`` because over-charging an evaluation fails closed while
        under-charging it hands out free compute.
        """
        if replayed:
            return cls(replay_hits=1)
        if not executed:
            return cls()
        return cls(
            correctness_calls=1,
            fresh_timed_calls=1 if _timing_executed(observation) else 0,
            verifier_gpu_seconds=verifier_seconds,
            profiler_gpu_seconds=profiler_seconds,
        )


def _timing_executed(observation: Any) -> bool:
    """Whether the benchmark subprocesses actually ran for an observation.

    Timings that came back prove it.  So does a requested timing whose admission
    was ``rejected``: that verdict means the bench ran but some shape produced no
    usable measurement, which still spent the GPU.  A driver that was refused as
    performance-ineligible never reached the benchmark and is not counted.
    """
    if getattr(observation, "wall_by_shape", None):
        return True
    return bool(
        getattr(observation, "timing_requested", False)
        and getattr(observation, "infra_error", False)
        and str(getattr(observation, "timing_grade", "")) == "rejected"
    )


@dataclass(frozen=True)
class BudgetLimitsV1:
    """Optional hard limits for every physical ledger dimension.

    ``None`` means that the dimension is observed but not capped.  A zero limit
    is valid and explicitly prohibits that kind of work.
    """

    generated_tokens: Optional[int] = None
    optimizer_tokens: Optional[int] = None
    correctness_calls: Optional[int] = None
    fresh_timed_calls: Optional[int] = None
    replay_hits: Optional[int] = None
    verifier_gpu_seconds: Optional[float] = None
    profiler_gpu_seconds: Optional[float] = None
    groups_attempted: Optional[int] = None
    groups_kept: Optional[int] = None

    def __post_init__(self) -> None:
        for name in _INTEGER_COUNTERS:
            value = getattr(self, name)
            if value is not None:
                _finite_nonnegative(name, value, integer=True)
        for name in _FLOAT_COUNTERS:
            value = getattr(self, name)
            if value is not None:
                _finite_nonnegative(name, value, integer=False)
        if (
            self.groups_attempted is not None
            and self.groups_kept is not None
            and self.groups_kept > self.groups_attempted
        ):
            raise BudgetError("groups_kept limit cannot exceed groups_attempted limit")

    @classmethod
    def from_mapping(cls, raw: Optional[Mapping[str, Any]]) -> "BudgetLimitsV1":
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise BudgetError("budget_limits must be a mapping")
        unknown = sorted(set(raw) - set(_ALL_COUNTERS))
        if unknown:
            raise BudgetError(f"unknown budget limit(s): {', '.join(unknown)}")
        return cls(**dict(raw))

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


class BudgetLedgerV1:
    """Thread-safe, exactly resumable GRPO budget ledger."""

    schema_version = SCHEMA_VERSION

    def __init__(
        self,
        *,
        limits: Optional[BudgetLimitsV1 | Mapping[str, Any]] = None,
        generated_tokens: int = 0,
        optimizer_tokens: int = 0,
        correctness_calls: int = 0,
        fresh_timed_calls: int = 0,
        replay_hits: int = 0,
        verifier_gpu_seconds: float = 0.0,
        profiler_gpu_seconds: float = 0.0,
        groups_attempted: int = 0,
        groups_kept: int = 0,
        feature_invocations: Optional[Mapping[str, int]] = None,
    ) -> None:
        if limits is None:
            self.limits = BudgetLimitsV1()
        elif isinstance(limits, BudgetLimitsV1):
            self.limits = limits
        else:
            self.limits = BudgetLimitsV1.from_mapping(limits)
        self._lock = threading.RLock()
        self.generated_tokens = _finite_nonnegative(
            "generated_tokens", generated_tokens, integer=True
        )
        self.optimizer_tokens = _finite_nonnegative(
            "optimizer_tokens", optimizer_tokens, integer=True
        )
        self.correctness_calls = _finite_nonnegative(
            "correctness_calls", correctness_calls, integer=True
        )
        self.fresh_timed_calls = _finite_nonnegative(
            "fresh_timed_calls", fresh_timed_calls, integer=True
        )
        self.replay_hits = _finite_nonnegative("replay_hits", replay_hits, integer=True)
        self.verifier_gpu_seconds = _finite_nonnegative(
            "verifier_gpu_seconds", verifier_gpu_seconds, integer=False
        )
        self.profiler_gpu_seconds = _finite_nonnegative(
            "profiler_gpu_seconds", profiler_gpu_seconds, integer=False
        )
        self.groups_attempted = _finite_nonnegative(
            "groups_attempted", groups_attempted, integer=True
        )
        self.groups_kept = _finite_nonnegative(
            "groups_kept", groups_kept, integer=True
        )
        if self.groups_kept > self.groups_attempted:
            raise BudgetError("groups_kept cannot exceed groups_attempted")
        self.feature_invocations: dict[str, int] = {}
        if feature_invocations is not None and not isinstance(
            feature_invocations, Mapping
        ):
            raise BudgetError("feature_invocations must be a mapping")
        for feature, count in dict(feature_invocations or {}).items():
            self._validate_feature(feature)
            self.feature_invocations[feature] = _finite_nonnegative(
                f"feature_invocations[{feature!r}]", count, integer=True
            )
        self._assert_limits(self._counter_snapshot())

    @staticmethod
    def _validate_feature(feature: Any) -> str:
        if not isinstance(feature, str) or not feature.strip():
            raise BudgetError("feature invocation name must be a non-empty string")
        return feature

    def _counter_snapshot(self) -> dict[str, int | float]:
        return {name: getattr(self, name) for name in _ALL_COUNTERS}

    def _assert_limits(self, candidate: Mapping[str, int | float]) -> None:
        for name in _ALL_COUNTERS:
            limit = getattr(self.limits, name)
            if limit is not None and candidate[name] > limit:
                raise BudgetExceededError(name, candidate[name], limit)

    def _candidate_totals(
        self, deltas: Mapping[str, int | float]
    ) -> dict[str, int | float]:
        unknown = sorted(set(deltas) - set(_ALL_COUNTERS))
        if unknown:
            raise BudgetError(f"unknown budget counter(s): {', '.join(unknown)}")
        candidate = self._counter_snapshot()
        for name, delta in deltas.items():
            checked = _finite_nonnegative(
                name, delta, integer=name in _INTEGER_COUNTERS
            )
            candidate[name] += checked
            if name in _FLOAT_COUNTERS and not math.isfinite(candidate[name]):
                raise BudgetError(f"{name} total must remain finite")
        if candidate["groups_kept"] > candidate["groups_attempted"]:
            raise BudgetError("groups_kept cannot exceed groups_attempted")
        return candidate

    def _increment(self, **deltas: int | float) -> None:
        with self._lock:
            candidate = self._candidate_totals(deltas)
            self._assert_limits(candidate)
            for name, value in candidate.items():
                setattr(self, name, value)

    def record_generated(self, tokens: int) -> None:
        self._increment(generated_tokens=tokens)

    def record_optimizer(self, tokens: int) -> None:
        self._increment(optimizer_tokens=tokens)

    def record_evaluation(
        self,
        *,
        correctness_calls: int = 0,
        fresh_timed_calls: int = 0,
        replay_hits: int = 0,
        verifier_gpu_seconds: float = 0.0,
        profiler_gpu_seconds: float = 0.0,
    ) -> None:
        """Record explicitly observed evaluation work without inferring overlap."""

        self._increment(
            correctness_calls=correctness_calls,
            fresh_timed_calls=fresh_timed_calls,
            replay_hits=replay_hits,
            verifier_gpu_seconds=verifier_gpu_seconds,
            profiler_gpu_seconds=profiler_gpu_seconds,
        )

    def record_evaluation_work(self, work: EvaluationWork) -> EvaluationWork:
        """Charge one observed evaluation in a single atomic ledger update."""

        if not isinstance(work, EvaluationWork):
            raise BudgetError("evaluation work must be an EvaluationWork")
        self.record_evaluation(**work.to_dict())
        return work

    def check_evaluation(self, work: EvaluationWork) -> None:
        """Raise before compute a hard limit already forbids is actually spent.

        Charging after the fact still binds a limit, but only once the GPU work
        is gone.  Calling this with the work an evaluation is ABOUT to do keeps
        the ledger fail-closed: a ``correctness_calls`` limit of 0 refuses to
        launch the evaluation at all.
        """

        if not isinstance(work, EvaluationWork):
            raise BudgetError("evaluation work must be an EvaluationWork")
        with self._lock:
            self._assert_limits(self._candidate_totals(work.to_dict()))

    def record_groups(self, *, attempted: int = 0, kept: int = 0) -> None:
        self._increment(groups_attempted=attempted, groups_kept=kept)

    def record_feature(self, feature: str, count: int = 1) -> None:
        feature = self._validate_feature(feature)
        count = _finite_nonnegative(
            f"feature_invocations[{feature!r}]", count, integer=True
        )
        with self._lock:
            self.feature_invocations[feature] = (
                self.feature_invocations.get(feature, 0) + count
            )

    def feature_count(self, feature: str) -> int:
        with self._lock:
            return int(self.feature_invocations.get(feature, 0))

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": self.schema_version,
                **self._counter_snapshot(),
                "feature_invocations": dict(sorted(self.feature_invocations.items())),
                "limits": self.limits.to_dict(),
            }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BudgetLedgerV1":
        if not isinstance(raw, Mapping):
            raise BudgetError("budget ledger state must be a mapping")
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise BudgetError(
                f"unsupported budget ledger schema: {raw.get('schema_version')!r}"
            )
        required = set(_ALL_COUNTERS) | {
            "schema_version",
            "feature_invocations",
            "limits",
        }
        missing = sorted(required - set(raw))
        unknown = sorted(set(raw) - required)
        if missing:
            raise BudgetError(f"budget ledger state missing: {', '.join(missing)}")
        if unknown:
            raise BudgetError(f"unknown budget ledger state: {', '.join(unknown)}")
        return cls(
            limits=BudgetLimitsV1.from_mapping(raw["limits"]),
            feature_invocations=raw["feature_invocations"],
            **{name: raw[name] for name in _ALL_COUNTERS},
        )

    @classmethod
    def merge(cls, ledgers: Iterable["BudgetLedgerV1"]) -> "BudgetLedgerV1":
        items = list(ledgers)
        if not items:
            return cls()
        limits = items[0].limits
        if any(item.limits != limits for item in items[1:]):
            raise BudgetError("cannot merge ledgers with different hard limits")
        counters = {
            name: sum(getattr(item, name) for item in items)
            for name in _ALL_COUNTERS
        }
        features: dict[str, int] = {}
        for item in items:
            for name, count in item.feature_invocations.items():
                features[name] = features.get(name, 0) + count
        return cls(
            limits=limits,
            feature_invocations=features,
            **counters,
        )

    def digest(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def write_json(self, path: str | os.PathLike[str]) -> Path:
        """Atomically persist exact resume/accounting state."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            self.to_dict(), sort_keys=True, indent=2, allow_nan=False
        ) + "\n"
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, target)
            try:
                dir_fd = os.open(target.parent, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                # Some filesystems do not support directory fsync.  The file itself
                # is still flushed and atomically replaced.
                pass
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        return target


# --------------------------------------------------------------------------- #
# Evaluation-path producer
#
# The evaluation path is the only place that knows what physically happened to a
# candidate, so it owns the five evaluation counters.  These two helpers are the
# whole interface it needs, and both tolerate ``ledger=None`` so an unbudgeted
# evaluation (a bare KoreEnv, a test, a one-off script) stays a no-op.
# --------------------------------------------------------------------------- #
def check_evaluation_budget(
    ledger: Optional[BudgetLedgerV1], work: EvaluationWork
) -> EvaluationWork:
    """Pre-flight the work an evaluation is about to do; raise if it is barred."""

    if ledger is not None:
        ledger.check_evaluation(work)
    return work


def charge_evaluation_work(
    ledger: Optional[BudgetLedgerV1], work: EvaluationWork
) -> EvaluationWork:
    """Charge the work an evaluation actually did."""

    if ledger is not None:
        ledger.record_evaluation_work(work)
    return work
