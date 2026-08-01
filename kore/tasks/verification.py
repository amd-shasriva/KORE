"""On-hardware seed verification evidence and the training-eligibility policy.

``data/gfx950_task_verification.json`` records what actually happened when every
breadth task's declared seed kernel was executed against its own reference on
real gfx950 silicon: 948 PASS, 100 FAIL_CORRECTNESS, 4 INFRA over 1,052 tasks.
That evidence existed but nothing consumed it, so the registry admitted tasks
whose own seed cannot clear the SNR gate the task itself declares.

Three deliberate properties:

* **Three-valued verdicts.**  A task with no record gets ``UNKNOWN``, never
  ``PASS``.  ``is_pass`` is False for ``UNKNOWN`` and for ``INFRA``, so no report
  can launder an unmeasured task into "hardware verified".
* **INFRA is not a task defect.**  The four ``INFRA`` records are harness OOMs on
  a shared GPU (252 GiB exhausted).  They are reported separately and are never
  counted as a correctness failure.
* **The policy is explicit.**  Nothing here changes the registry's train split.
  A caller opts in by asking for an eligibility decision, so the definition of
  "train task" and the definition of "eligible for selection under this policy"
  stay separate, inspectable states.

``FAIL_CORRECTNESS`` is not one failure mode, so it is banded by the seed's SNR
against the gate the task declares (see :data:`NEAR_GATE_MARGIN_DB`):

* ``broken`` (27) - the seed does not compile on this Triton build or returns a
  non-finite result; the sweep records -999 dB.  No rollout can ever earn reward.
* ``near_gate`` (73) - the seed clears (72) or lands within 5 dB of (1) its
  declared gate and fails only ``torch.allclose``'s elementwise tolerance.  The
  evidence points at a tolerance calibrated for fp32 being applied to bf16/fp16/
  fp8 outputs, not at broken math.
* ``shortfall`` (11) - the seed returns finite values but misses its own gate by
  more than 5 dB (4.7-24.6 dB against 30/40 dB gates).  Too large to be a
  rounding artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFICATION_ARTIFACT = REPO_ROOT / "data" / "gfx950_task_verification.json"

# The sweep ran on gfx950; a verdict is evidence about that architecture only.
VERIFIED_ARCHITECTURE = "gfx950"

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL_CORRECTNESS"
STATUS_INFRA = "INFRA"
# Synthesized here for a task the sweep never executed.  It is a third value,
# not a default of either recorded outcome.
STATUS_UNKNOWN = "UNKNOWN"
RECORDED_STATUSES = frozenset({STATUS_PASS, STATUS_FAIL, STATUS_INFRA})
ALL_STATUSES = frozenset(RECORDED_STATUSES | {STATUS_UNKNOWN})

BAND_PASS = "pass"
BAND_NEAR_GATE = "near_gate"
BAND_SHORTFALL = "shortfall"
BAND_BROKEN = "broken"
BAND_INFRA = "infra"
BAND_UNKNOWN = "unknown"
ALL_BANDS = (
    BAND_PASS,
    BAND_NEAR_GATE,
    BAND_SHORTFALL,
    BAND_BROKEN,
    BAND_INFRA,
    BAND_UNKNOWN,
)

# How far below its declared gate a finite SNR may sit and still be read as a
# gate/tolerance calibration question rather than evidence of wrong math.
NEAR_GATE_MARGIN_DB = 5.0
# The harness writes -999 dB for "no comparable result at all" (uncompilable
# seed, inf/NaN output).  Anything at or below this is structurally broken.
BROKEN_SNR_DB = -900.0


class VerificationError(RuntimeError):
    """The hardware verification artifact is missing, malformed, or ambiguous."""


@dataclass(frozen=True)
class HardwareVerdict:
    """One task's measured outcome on real gfx950 silicon.

    ``status`` is what the sweep recorded (or ``UNKNOWN``); ``band`` refines
    ``FAIL_CORRECTNESS`` into the failure modes that call for different action.
    """

    task_id: str
    status: str
    band: str
    snr_db: Optional[float] = None
    threshold_db: Optional[float] = None
    dtype: str = ""
    operation: str = ""
    shape: str = ""
    seconds: Optional[float] = None
    returncode: Optional[int] = None
    detail: str = ""
    architecture: str = VERIFIED_ARCHITECTURE

    @property
    def is_known(self) -> bool:
        return self.status in RECORDED_STATUSES

    @property
    def is_pass(self) -> bool:
        """True only for a recorded PASS.  Never true for UNKNOWN or INFRA."""
        return self.status == STATUS_PASS

    @property
    def is_correctness_failure(self) -> bool:
        return self.status == STATUS_FAIL

    @property
    def is_infra(self) -> bool:
        """A harness/capacity failure, which is not evidence about the task."""
        return self.status == STATUS_INFRA

    @property
    def margin_db(self) -> Optional[float]:
        """Signed SNR headroom over the declared gate, when both are known."""
        if self.snr_db is None or self.threshold_db is None:
            return None
        if self.snr_db <= BROKEN_SNR_DB:
            return None
        return self.snr_db - self.threshold_db

    @property
    def clears_declared_gate(self) -> bool:
        """True when the measured SNR met the task's own gate.

        72 of the 111 correctness failures are in this state: the SNR gate passed
        and only the elementwise ``allclose`` tolerance rejected the seed.
        """
        margin = self.margin_db
        return margin is not None and margin >= 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "band": self.band,
            "snr_db": self.snr_db,
            "threshold_db": self.threshold_db,
            "margin_db": self.margin_db,
            "clears_declared_gate": self.clears_declared_gate,
            "dtype": self.dtype,
            "operation": self.operation,
            "shape": self.shape,
            "architecture": self.architecture,
        }


def unknown_verdict(task_id: str) -> HardwareVerdict:
    """The verdict for a task the sweep never executed."""

    tid = str(task_id or "").strip()
    if not tid:
        raise VerificationError("cannot build a verdict for an empty task_id")
    return HardwareVerdict(task_id=tid, status=STATUS_UNKNOWN, band=BAND_UNKNOWN)


def _float_or_none(value: Any, *, field: str, task_id: str) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise VerificationError(
            f"verdict {task_id!r}: {field} is not a number: {value!r}"
        ) from exc
    if math.isnan(number):
        return None
    return number


def classify_band(status: str, snr_db: Optional[float], threshold_db: Optional[float],
                  *, task_id: str = "") -> str:
    """Band a recorded status, refusing to guess when the record cannot decide."""

    if status == STATUS_PASS:
        return BAND_PASS
    if status == STATUS_INFRA:
        return BAND_INFRA
    if status == STATUS_UNKNOWN:
        return BAND_UNKNOWN
    if status != STATUS_FAIL:
        raise VerificationError(f"verdict {task_id!r}: unknown status {status!r}")
    if snr_db is None or not math.isfinite(snr_db) or snr_db <= BROKEN_SNR_DB:
        return BAND_BROKEN
    if threshold_db is None:
        # Without the gate the seed was judged against, "near miss" and "wrong
        # math" are indistinguishable; a silent guess would admit either.
        raise VerificationError(
            f"verdict {task_id!r}: correctness failure at {snr_db} dB declares no threshold"
        )
    if snr_db >= threshold_db - NEAR_GATE_MARGIN_DB:
        return BAND_NEAR_GATE
    return BAND_SHORTFALL


def verdict_from_record(record: Mapping[str, Any]) -> HardwareVerdict:
    """Parse one sweep record, raising rather than dropping a malformed verdict.

    Dropping would silently downgrade a recorded failure to ``UNKNOWN``, which
    the default policy admits -- the exact leak this module exists to close.
    """

    if not isinstance(record, Mapping):
        raise VerificationError(f"verification record is not a mapping: {record!r}")
    task_id = str(record.get("task") or record.get("task_id") or "").strip()
    if not task_id:
        raise VerificationError("verification record has no task id")
    status = str(record.get("status") or "").strip()
    if status not in RECORDED_STATUSES:
        raise VerificationError(
            f"verdict {task_id!r}: status {status!r} is not one of {sorted(RECORDED_STATUSES)}"
        )
    snr_db = _float_or_none(record.get("snr_db"), field="snr_db", task_id=task_id)
    threshold_db = _float_or_none(
        record.get("threshold"), field="threshold", task_id=task_id
    )
    returncode = record.get("rc")
    return HardwareVerdict(
        task_id=task_id,
        status=status,
        band=classify_band(status, snr_db, threshold_db, task_id=task_id),
        snr_db=snr_db,
        threshold_db=threshold_db,
        dtype=str(record.get("dtype") or ""),
        operation=str(record.get("operation") or ""),
        shape=str(record.get("shape") or ""),
        seconds=_float_or_none(record.get("seconds"), field="seconds", task_id=task_id),
        returncode=int(returncode) if isinstance(returncode, (int, float)) else None,
        detail=str(record.get("detail") or ""),
    )


@dataclass(frozen=True)
class VerificationReport:
    """Every verdict in one artifact, plus its provenance."""

    path: str
    digest: str
    architecture: str
    verdicts: Mapping[str, HardwareVerdict]
    summary: Mapping[str, Any]

    def status_counts(self) -> dict[str, int]:
        counts = {status: 0 for status in sorted(RECORDED_STATUSES)}
        for verdict in self.verdicts.values():
            counts[verdict.status] += 1
        return counts

    def band_counts(self) -> dict[str, int]:
        # An artifact only holds recorded verdicts, so ``unknown`` never appears.
        counts = {band: 0 for band in ALL_BANDS if band != BAND_UNKNOWN}
        for verdict in self.verdicts.values():
            counts[verdict.band] += 1
        return counts

    def task_ids_in_band(self, band: str) -> tuple[str, ...]:
        if band not in ALL_BANDS:
            raise VerificationError(f"unknown verification band {band!r}")
        return tuple(
            sorted(
                task_id
                for task_id, verdict in self.verdicts.items()
                if verdict.band == band
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact": self.path,
            "artifact_digest": self.digest,
            "architecture": self.architecture,
            "verdicts": len(self.verdicts),
            "recorded_summary": dict(self.summary),
            "status_counts": self.status_counts(),
            "band_counts": self.band_counts(),
            "near_gate_margin_db": NEAR_GATE_MARGIN_DB,
        }


def load_verification(path: Optional[Path | str] = None) -> VerificationReport:
    """Read and validate a verification artifact.

    A missing or unreadable artifact raises: silently returning "no verdicts"
    would make every task ``UNKNOWN`` and disable the eligibility policy.
    """

    artifact = Path(path) if path is not None else VERIFICATION_ARTIFACT
    try:
        raw = artifact.read_bytes()
    except OSError as exc:
        raise VerificationError(
            f"hardware verification artifact is unreadable: {artifact}"
        ) from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(
            f"hardware verification artifact is not valid JSON: {artifact}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise VerificationError(f"verification artifact must be a mapping: {artifact}")
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise VerificationError(
            f"verification artifact has no results list: {artifact}"
        )

    verdicts: dict[str, HardwareVerdict] = {}
    for record in results:
        verdict = verdict_from_record(record)
        previous = verdicts.get(verdict.task_id)
        if previous is not None and previous != verdict:
            raise VerificationError(
                f"verdict {verdict.task_id!r} recorded twice with different outcomes: "
                f"{previous.status}/{previous.band} versus {verdict.status}/{verdict.band}"
            )
        verdicts[verdict.task_id] = verdict

    summary = payload.get("summary")
    return VerificationReport(
        path=str(artifact),
        digest=hashlib.sha256(raw).hexdigest(),
        architecture=VERIFIED_ARCHITECTURE,
        verdicts=MappingProxyType(dict(sorted(verdicts.items()))),
        summary=MappingProxyType(dict(summary) if isinstance(summary, Mapping) else {}),
    )


@lru_cache(maxsize=1)
def report() -> VerificationReport:
    """The committed gfx950 verification artifact."""

    return load_verification()


def verdicts() -> Mapping[str, HardwareVerdict]:
    return report().verdicts


def verdict_for(task_id: str) -> HardwareVerdict:
    """Verdict for ``task_id``, or an explicit ``UNKNOWN`` -- never a PASS."""

    tid = str(task_id or "").strip()
    if not tid:
        raise VerificationError("cannot look up a verdict for an empty task_id")
    found = verdicts().get(tid)
    return found if found is not None else unknown_verdict(tid)


def hardware_pass_ids() -> frozenset[str]:
    """Tasks whose seed is proven correct on hardware.  Excludes UNKNOWN/INFRA."""

    return frozenset(
        task_id for task_id, verdict in verdicts().items() if verdict.is_pass
    )


def hardware_failure_ids() -> frozenset[str]:
    """Tasks whose own seed failed its declared correctness gate on hardware."""

    return frozenset(
        task_id
        for task_id, verdict in verdicts().items()
        if verdict.is_correctness_failure
    )


# --------------------------------------------------------------------------- #
# Eligibility policy
# --------------------------------------------------------------------------- #
EXCLUSION_BROKEN = "hardware_broken_seed"
EXCLUSION_SHORTFALL = "hardware_snr_shortfall"
EXCLUSION_NEAR_GATE = "hardware_near_gate_failure"
EXCLUSION_INFRA = "hardware_infra_failure"
EXCLUSION_UNVERIFIED = "hardware_verdict_missing"


@dataclass(frozen=True)
class EligibilityPolicy:
    """Which hardware evidence disqualifies a task from TRAINING selection.

    Every switch is off unless named, so a policy prints as exactly what it
    enforces.  ``require_verdict`` is the strict form: only a recorded PASS is
    admitted, which is meaningful for a "fully hardware-proven scope" campaign
    but is not the default (see :data:`DEFAULT_ELIGIBILITY_POLICY`).
    """

    name: str = "unnamed"
    exclude_broken: bool = False
    exclude_shortfall: bool = False
    exclude_near_gate: bool = False
    exclude_infra: bool = False
    require_verdict: bool = False

    def exclusion_reason(self, verdict: HardwareVerdict) -> Optional[str]:
        """The reason this policy rejects ``verdict``, or ``None`` to admit it."""

        if not isinstance(verdict, HardwareVerdict):
            raise VerificationError(f"expected a HardwareVerdict, got {verdict!r}")
        if verdict.status == STATUS_UNKNOWN:
            return EXCLUSION_UNVERIFIED if self.require_verdict else None
        if verdict.is_infra:
            # A harness OOM says nothing about the task, so this is off by
            # default; a strict campaign may still want only measured tasks.
            if self.exclude_infra or self.require_verdict:
                return EXCLUSION_INFRA
            return None
        if verdict.is_pass:
            return None
        if verdict.band == BAND_BROKEN:
            return EXCLUSION_BROKEN if (self.exclude_broken or self.require_verdict) else None
        if verdict.band == BAND_SHORTFALL:
            return (
                EXCLUSION_SHORTFALL
                if (self.exclude_shortfall or self.require_verdict)
                else None
            )
        if verdict.band == BAND_NEAR_GATE:
            return (
                EXCLUSION_NEAR_GATE
                if (self.exclude_near_gate or self.require_verdict)
                else None
            )
        raise VerificationError(
            f"verdict {verdict.task_id!r}: unhandled band {verdict.band!r}"
        )

    def admits(self, verdict: HardwareVerdict) -> bool:
        return self.exclusion_reason(verdict) is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "exclude_broken": self.exclude_broken,
            "exclude_shortfall": self.exclude_shortfall,
            "exclude_near_gate": self.exclude_near_gate,
            "exclude_infra": self.exclude_infra,
            "require_verdict": self.require_verdict,
            "near_gate_margin_db": NEAR_GATE_MARGIN_DB,
        }


# Default: exclude only the two bands where the hardware evidence contradicts the
# task's own declared contract, and treat everything else as admissible.
#
# * ``broken`` (27) is excluded: an uncompilable or inf-producing seed makes the
#   task's correctness gate unreachable, so every rollout scores zero.
# * ``shortfall`` (11) is excluded: missing your own gate by 5-25 dB is not a
#   rounding artifact.  Several are topk/sampling ops where elementwise SNR is
#   the wrong metric, but until that metric is recalibrated the recorded evidence
#   says the reference and the seed disagree.
# * ``near_gate`` (73) is admitted: 72 of them cleared the declared SNR gate and
#   failed only ``allclose``'s elementwise tolerance (1 LSB of bf16 in
#   attention-backward, one fp8 quantization step in fused norm+quant).  Dropping
#   them would delete real coverage over a threshold artifact.
# * ``INFRA`` (4) is admitted: those runs OOM'd the shared GPU at 252 GiB.
# * ``UNKNOWN`` (280 train tasks) is admitted but never counted as verified: the
#   sweep ran with prefix ``genb_``, so ``gen_``/``genv_``/hand-authored tasks
#   were never executed.  Absence of a run is not evidence of a defect.
DEFAULT_ELIGIBILITY_POLICY = EligibilityPolicy(
    name="exclude_broken_and_shortfall",
    exclude_broken=True,
    exclude_shortfall=True,
)

# Today's behavior, kept nameable so "policy off" is an explicit choice.
ADMIT_ALL_POLICY = EligibilityPolicy(name="admit_all")

# Only tasks with a recorded PASS on gfx950.
STRICT_HARDWARE_VERIFIED_POLICY = EligibilityPolicy(
    name="strict_hardware_verified",
    exclude_broken=True,
    exclude_shortfall=True,
    exclude_near_gate=True,
    exclude_infra=True,
    require_verdict=True,
)

NAMED_POLICIES: Mapping[str, EligibilityPolicy] = MappingProxyType({
    policy.name: policy
    for policy in (
        DEFAULT_ELIGIBILITY_POLICY,
        ADMIT_ALL_POLICY,
        STRICT_HARDWARE_VERIFIED_POLICY,
    )
})


def resolve_policy(policy: Optional[EligibilityPolicy | str] = None) -> EligibilityPolicy:
    """Coerce ``None``/a policy name/a policy into an :class:`EligibilityPolicy`."""

    if policy is None:
        return DEFAULT_ELIGIBILITY_POLICY
    if isinstance(policy, EligibilityPolicy):
        return policy
    name = str(policy).strip()
    if name not in NAMED_POLICIES:
        raise VerificationError(
            f"unknown eligibility policy {policy!r}; known={sorted(NAMED_POLICIES)}"
        )
    return NAMED_POLICIES[name]


@dataclass(frozen=True)
class EligibilityDecision:
    """Why one task is or is not selectable for training under a policy."""

    task_id: str
    eligible: bool
    reason: str
    verdict: HardwareVerdict
    policy: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "eligible": self.eligible,
            "reason": self.reason,
            "policy": self.policy,
            "status": self.verdict.status,
            "band": self.verdict.band,
            "snr_db": self.verdict.snr_db,
            "threshold_db": self.verdict.threshold_db,
        }


ADMITTED_VERIFIED = "hardware_pass"
ADMITTED_UNVERIFIED = "hardware_unverified_admitted"
# Kept distinct from an admitted correctness failure so no report can read a
# harness OOM as a defect the task is responsible for.
ADMITTED_INFRA = "hardware_infra_admitted"
ADMITTED_BAND = "hardware_failure_admitted"


def eligibility(
    task_id: str,
    policy: Optional[EligibilityPolicy | str] = None,
) -> EligibilityDecision:
    """Decide one task under ``policy`` (default :data:`DEFAULT_ELIGIBILITY_POLICY`)."""

    resolved = resolve_policy(policy)
    verdict = verdict_for(task_id)
    exclusion = resolved.exclusion_reason(verdict)
    if exclusion is not None:
        return EligibilityDecision(
            verdict.task_id, False, exclusion, verdict, resolved.name
        )
    if verdict.is_pass:
        reason = ADMITTED_VERIFIED
    elif verdict.status == STATUS_UNKNOWN:
        reason = ADMITTED_UNVERIFIED
    elif verdict.is_infra:
        reason = ADMITTED_INFRA
    else:
        reason = ADMITTED_BAND
    return EligibilityDecision(verdict.task_id, True, reason, verdict, resolved.name)


def exclusions(
    task_ids: Iterable[str],
    policy: Optional[EligibilityPolicy | str] = None,
) -> Mapping[str, str]:
    """``{task_id: exclusion_reason}`` for the rejected subset of ``task_ids``."""

    resolved = resolve_policy(policy)
    out: dict[str, str] = {}
    for task_id in task_ids:
        decision = eligibility(task_id, resolved)
        if not decision.eligible:
            out[decision.task_id] = decision.reason
    return MappingProxyType(dict(sorted(out.items())))


def coverage(task_ids: Iterable[str]) -> dict[str, Any]:
    """How much of ``task_ids`` the hardware sweep actually measured."""

    ids = sorted({str(task_id).strip() for task_id in task_ids if str(task_id).strip()})
    statuses: dict[str, int] = {status: 0 for status in sorted(ALL_STATUSES)}
    bands: dict[str, int] = {band: 0 for band in ALL_BANDS}
    unknown: list[str] = []
    for task_id in ids:
        verdict = verdict_for(task_id)
        statuses[verdict.status] += 1
        bands[verdict.band] += 1
        if not verdict.is_known:
            unknown.append(task_id)
    return {
        "tasks": len(ids),
        "measured": len(ids) - len(unknown),
        "unmeasured": len(unknown),
        "status_counts": statuses,
        "band_counts": bands,
        "unmeasured_task_ids": tuple(unknown),
        "artifact_digest": report().digest,
        "architecture": VERIFIED_ARCHITECTURE,
    }


def describe(
    task_ids: Iterable[str],
    policy: Optional[EligibilityPolicy | str] = None,
) -> dict[str, Any]:
    """Machine-readable eligibility report over ``task_ids`` for docs/campaigns."""

    resolved = resolve_policy(policy)
    ids = sorted({str(task_id).strip() for task_id in task_ids if str(task_id).strip()})
    rejected = exclusions(ids, resolved)
    by_reason: dict[str, list[str]] = {}
    for task_id, reason in rejected.items():
        by_reason.setdefault(reason, []).append(task_id)
    return {
        "policy": resolved.as_dict(),
        "artifact": report().as_dict(),
        "coverage": coverage(ids),
        "considered": len(ids),
        "eligible": len(ids) - len(rejected),
        "excluded": len(rejected),
        "excluded_by_reason": {
            reason: sorted(members) for reason, members in sorted(by_reason.items())
        },
        "excluded_task_ids": dict(rejected),
    }


def with_name(policy: EligibilityPolicy, name: str) -> EligibilityPolicy:
    """Rename a policy so a campaign-specific variant still reports honestly."""

    return replace(policy, name=str(name))


__all__ = [
    "ADMITTED_BAND",
    "ADMITTED_INFRA",
    "ADMITTED_UNVERIFIED",
    "ADMITTED_VERIFIED",
    "ADMIT_ALL_POLICY",
    "ALL_BANDS",
    "BAND_BROKEN",
    "BAND_INFRA",
    "BAND_NEAR_GATE",
    "BAND_PASS",
    "BAND_SHORTFALL",
    "BAND_UNKNOWN",
    "BROKEN_SNR_DB",
    "DEFAULT_ELIGIBILITY_POLICY",
    "EXCLUSION_BROKEN",
    "EXCLUSION_INFRA",
    "EXCLUSION_NEAR_GATE",
    "EXCLUSION_SHORTFALL",
    "EXCLUSION_UNVERIFIED",
    "EligibilityDecision",
    "EligibilityPolicy",
    "HardwareVerdict",
    "NAMED_POLICIES",
    "NEAR_GATE_MARGIN_DB",
    "RECORDED_STATUSES",
    "STATUS_FAIL",
    "STATUS_INFRA",
    "STATUS_PASS",
    "STATUS_UNKNOWN",
    "STRICT_HARDWARE_VERIFIED_POLICY",
    "VERIFICATION_ARTIFACT",
    "VERIFIED_ARCHITECTURE",
    "VerificationError",
    "VerificationReport",
    "classify_band",
    "coverage",
    "describe",
    "eligibility",
    "exclusions",
    "hardware_failure_ids",
    "hardware_pass_ids",
    "load_verification",
    "report",
    "resolve_policy",
    "unknown_verdict",
    "verdict_for",
    "verdict_from_record",
    "verdicts",
    "with_name",
]
