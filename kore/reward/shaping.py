"""Evidence-gated, finite potential-based shaping.

Physics and counters are always available for diagnosis.  They become a reward
surface only when a *specific operator family* passes preregistered held-out
tests under the same fingerprinted physical model.  This module owns that gate
and the numerical bounds on ``Phi``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class ShapingThresholds:
    """Preregistered minimum evidence for a family-specific reward surface."""

    min_points: int = 20
    min_task_clusters: int = 3
    min_normalized_cv_r2: float = 0.10
    min_increment_over_baseline: float = 0.05
    min_ci95_lower: float = 0.0
    max_adjusted_p: float = 0.05


DEFAULT_SHAPING_THRESHOLDS = ShapingThresholds()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _bounded_phi(value: Any) -> float:
    if not _finite(value):
        raise ValueError(f"potential must be finite, got {value!r}")
    out = float(value)
    if not 0.0 <= out <= 1.0:
        raise ValueError(f"potential must be in [0, 1], got {out}")
    return out


@dataclass(frozen=True)
class FamilyShapingEvidence:
    """Held-out evidence and deployable normalized residual coefficients."""

    family: str
    report_fingerprint: str
    model_fingerprint: str
    n_points: int
    n_task_clusters: int
    normalized_cv_r2: float
    baseline_cv_r2: float
    ci95: tuple[float, float]
    adjusted_p: float
    coefficients: tuple[float, float, float]  # stall, occupancy deficit, intercept

    def __post_init__(self) -> None:
        if not self.family or not self.report_fingerprint or not self.model_fingerprint:
            raise ValueError("family and evidence/model fingerprints are required")
        if self.n_points < 0 or self.n_task_clusters < 0:
            raise ValueError("evidence counts must be non-negative")
        for name, value in (
            ("normalized_cv_r2", self.normalized_cv_r2),
            ("baseline_cv_r2", self.baseline_cv_r2),
            ("ci95 lower", self.ci95[0]),
            ("ci95 upper", self.ci95[1]),
            ("adjusted_p", self.adjusted_p),
            *[(f"coefficient[{i}]", v) for i, v in enumerate(self.coefficients)],
        ):
            if not _finite(value):
                raise ValueError(f"{name} must be finite")
        if self.ci95[0] > self.ci95[1]:
            raise ValueError("ci95 lower bound exceeds upper bound")
        if not 0.0 <= self.adjusted_p <= 1.0:
            raise ValueError("adjusted_p must be in [0, 1]")

    def passes(self, thresholds: ShapingThresholds = DEFAULT_SHAPING_THRESHOLDS) -> bool:
        return (
            self.n_points >= thresholds.min_points
            and self.n_task_clusters >= thresholds.min_task_clusters
            and self.normalized_cv_r2 >= thresholds.min_normalized_cv_r2
            and self.normalized_cv_r2 - self.baseline_cv_r2
            >= thresholds.min_increment_over_baseline
            and self.ci95[0] > thresholds.min_ci95_lower
            and self.adjusted_p <= thresholds.max_adjusted_p
        )

    def predict_gap_fraction(self, stall: float, occupancy: float) -> float:
        """Bounded predicted ``(T-Tmin)/T`` from validated normalized features."""
        stall = _bounded_phi(stall)
        occupancy = _bounded_phi(occupancy)
        b_stall, b_occ, intercept = self.coefficients
        raw = b_stall * stall + b_occ * (1.0 - occupancy) + intercept
        if not math.isfinite(raw):
            raise ValueError("residual prediction is non-finite")
        return max(0.0, min(1.0, raw))

    @classmethod
    def from_mapping(cls, family: str, data: Mapping[str, Any]) -> "FamilyShapingEvidence":
        ci = data.get("ci95")
        coef = data.get("coefficients")
        if not isinstance(ci, (list, tuple)) or len(ci) != 2:
            raise ValueError(f"{family}: ci95 must contain two values")
        if not isinstance(coef, (list, tuple)) or len(coef) != 3:
            raise ValueError(f"{family}: coefficients must contain three values")
        return cls(
            family=family,
            report_fingerprint=str(data.get("report_fingerprint") or ""),
            model_fingerprint=str(data.get("model_fingerprint") or ""),
            n_points=int(data.get("n_points", 0)),
            n_task_clusters=int(data.get("n_task_clusters", 0)),
            normalized_cv_r2=float(data.get("normalized_cv_r2")),
            baseline_cv_r2=float(data.get("baseline_cv_r2")),
            ci95=(float(ci[0]), float(ci[1])),
            adjusted_p=float(data.get("adjusted_p")),
            coefficients=(float(coef[0]), float(coef[1]), float(coef[2])),
        )


def _document_fingerprint(document: Mapping[str, Any]) -> str:
    payload = dict(document)
    payload.pop("evidence_fingerprint", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=16)
def _load_evidence_cached(
    path_text: str,
    expected_evidence_fingerprint: str,
) -> tuple[dict[str, FamilyShapingEvidence], str]:
    path = Path(path_text)
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load shaping evidence {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("shaping evidence must be a JSON object")
    actual = _document_fingerprint(document)
    declared = str(document.get("evidence_fingerprint") or actual)
    if declared != actual:
        raise ValueError(f"shaping evidence fingerprint is stale: {declared} != {actual}")
    if expected_evidence_fingerprint and actual != expected_evidence_fingerprint:
        raise ValueError(
            f"shaping evidence fingerprint mismatch: expected "
            f"{expected_evidence_fingerprint}, got {actual}"
        )
    raw = document.get("shaping_evidence", {}).get("families", {})
    if not isinstance(raw, dict):
        raise ValueError("shaping_evidence.families must be an object")
    parsed = {
        str(family): FamilyShapingEvidence.from_mapping(str(family), values)
        for family, values in raw.items()
        if isinstance(values, dict)
    }
    return parsed, actual


def evidence_for_task(task, config, model_fingerprint: str) -> Optional[FamilyShapingEvidence]:
    """Passing evidence for ``task`` under ``model_fingerprint``, else ``None``."""
    path = getattr(config, "physics_shaping_evidence_path", None)
    expected = str(getattr(config, "physics_shaping_evidence_fingerprint", "") or "")
    if not path or not expected:
        return None
    try:
        from kore.eval.generalization import family_of

        family = family_of(getattr(task, "task_id", ""))
        all_evidence, _ = _load_evidence_cached(str(path), expected)
        evidence = all_evidence.get(str(family))
    except (OSError, TypeError, ValueError):
        return None
    if (
        evidence is None
        or evidence.model_fingerprint != model_fingerprint
        or not evidence.passes()
    ):
        return None
    return evidence


def shaping_terms(
    phis: Sequence[Optional[float]], gamma: float, terminal_phi: float = 0.0
) -> List[float]:
    """Finite terms ``gamma*Phi(s') - Phi(s)`` with ``None`` boundaries."""
    if not _finite(gamma) or not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")
    terminal = _bounded_phi(terminal_phi)
    checked = [None if phi is None else _bounded_phi(phi) for phi in phis]
    out: List[float] = []
    for index, current in enumerate(checked):
        following = checked[index + 1] if index + 1 < len(checked) else terminal
        out.append(0.0 if current is None or following is None else float(gamma) * following - current)
    return out


def shaped_turn_rewards(
    turn_rewards: Sequence[float],
    phis: Sequence[Optional[float]],
    gamma: float,
    weight: float = 1.0,
    terminal_phi: float = 0.0,
) -> List[float]:
    """Apply bounded shaping; non-finite rewards/weights are rejected."""
    if len(turn_rewards) != len(phis):
        raise ValueError(
            f"turn_rewards ({len(turn_rewards)}) and phis ({len(phis)}) must be the same length"
        )
    if not _finite(weight) or float(weight) < 0.0:
        raise ValueError("shaping weight must be finite and non-negative")
    rewards = []
    for reward in turn_rewards:
        if not _finite(reward):
            raise ValueError(f"turn reward must be finite, got {reward!r}")
        rewards.append(float(reward))
    terms = shaping_terms(phis, gamma, terminal_phi)
    result = [reward + float(weight) * term for reward, term in zip(rewards, terms)]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("shaped reward became non-finite")
    return result


def discounted_shaping_sum(
    phis: Sequence[Optional[float]], gamma: float, terminal_phi: float = 0.0
) -> float:
    terms = shaping_terms(phis, gamma, terminal_phi)
    return float(sum((float(gamma) ** index) * term for index, term in enumerate(terms)))


__all__ = [
    "DEFAULT_SHAPING_THRESHOLDS",
    "FamilyShapingEvidence",
    "ShapingThresholds",
    "discounted_shaping_sum",
    "evidence_for_task",
    "shaped_turn_rewards",
    "shaping_terms",
]
