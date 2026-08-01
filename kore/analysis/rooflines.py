"""Deprecated operator-level facade over :mod:`kore.analysis.roofline`.

Historically KORE carried two independent roofline implementations.  This module
now contains adapters only: work estimates, peaks, runtimes, and fingerprints all
come from the singular model in :mod:`kore.analysis.roofline`.

SKU identity is never guessed silently.  ``gfx950`` covers two boards with
different compute ceilings (MI350X 2.30 PF/s bf16 at 2.2 GHz / 1000 W, MI355X
2.50 PF/s bf16 at 2.4 GHz / 1400 W), so an 8.7% peak error -- and therefore an
8% error in every ``eta`` and every SoL integrity floor -- rides on the choice.
Callers should pass ``sku=`` explicitly; when they do not, :func:`resolve_sku`
*observes* the device (``rocminfo`` / ``rocm-smi`` marketing name) instead of
assuming, records where the answer came from, and warns when it has to fall back
to :data:`DEFAULT_SKU`.  :func:`verify_runtime_sku` turns a configured-vs-observed
mismatch into a loud :class:`~kore.analysis.roofline.ModelError`.

Calibrated (measured-achievable) peaks are applied through the ONE supported
path: a ``kore.runtime-calibration.v1`` document passed as ``calibration=`` and
pinned with ``expected_fingerprint=``.  The legacy ``KORE_PEAK_*`` process
globals are unsupported; because they were silently ignored (a documented
reproduce path that quietly ran on datasheet peaks), :func:`resolve_peaks` now
warns when it sees them set.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from kore.analysis import roofline as _core

DEFAULT_ARCH = "gfx950"

# MI350X, not MI355X: this node reports Marketing Name "AMD Instinct MI350X"
# (rocminfo), a 1000 W package power cap, and a maximum sclk of 2200 MHz --
# 256 CU x 2.2 GHz x 4096 bf16 FLOP/clk = 2.31 PF/s, the MI350X datasheet peak.
# The Slurm `--gres=gpu:mi355x:8` label is a cluster-side alias for the gfx950
# partition and is NOT evidence of the silicon.  Overriding to mi355x would
# divide by a 2.50 PF/s ceiling the board cannot reach, which also lowers the
# SoL integrity floor by 8% and would admit physically impossible timings.
DEFAULT_SKU = "mi350x"

# Fallback SKU per architecture, used only when the device cannot be observed.
_ARCH_FALLBACK_SKU: dict[str, str] = {"gfx950": DEFAULT_SKU, "gfx942": "mi300x"}

CALIBRATION_SCHEMA = "kore.runtime-calibration.v1"

# Unsupported process-global peak overrides.  Kept named here so the silent
# no-op that shipped with the documented reproduce path becomes a loud warning.
LEGACY_PEAK_ENV_VARS: tuple[str, ...] = (
    "KORE_PEAK_HBM_BW",
    "KORE_PEAK_BF16",
    "KORE_PEAK_FP8",
)

# Supported, fingerprinted replacements for the legacy globals.
CALIBRATION_ENV_VAR = "KORE_PHYSICS_CALIBRATION"
FINGERPRINT_ENV_VAR = "KORE_PHYSICS_MODEL_FINGERPRINT"

_PROBE_CACHE: dict[str, Optional[dict[str, str]]] = {}


def legacy_peak_env_overrides() -> dict[str, str]:
    """Legacy ``KORE_PEAK_*`` variables that are set but have no effect."""
    return {name: os.environ[name] for name in LEGACY_PEAK_ENV_VARS if os.environ.get(name)}


def warn_on_legacy_peak_env() -> dict[str, str]:
    """Warn (loudly) when the dead ``KORE_PEAK_*`` overrides are exported.

    They were removed because they were invisible, unfingerprinted global
    calibration.  Silently ignoring them is worse: a reproduce path that exports
    them believes it is running on measured peaks while it runs on datasheet
    peaks (a ~2x shift in ``eta``).  Returns the offending mapping.
    """
    found = legacy_peak_env_overrides()
    if found:
        warnings.warn(
            f"{', '.join(sorted(found))} is set but has NO effect: process-global peak "
            "overrides are unsupported because they are unfingerprinted. Emit a "
            f"{CALIBRATION_SCHEMA} document (python -m kore.analysis.calibrate_peaks "
            f"--out data/calibration.json) and pass it via calibration=/"
            f"{CALIBRATION_ENV_VAR}, pinned with {FINGERPRINT_ENV_VAR}.",
            RuntimeWarning,
            stacklevel=3,
        )
    return found


def _probe_device() -> Optional[dict[str, str]]:
    """Memoized device probe.  Never runs at import; at most one probe per process."""
    if "value" not in _PROBE_CACHE:
        try:
            _PROBE_CACHE["value"] = _core.detect_runtime_device()
        except Exception:  # noqa: BLE001 - a probe failure must never break analysis
            _PROBE_CACHE["value"] = None
    return _PROBE_CACHE["value"]


def observed_sku(arch: Optional[str] = None) -> Optional[str]:
    """The SKU the device reports, or ``None`` when it cannot be observed.

    ``arch`` (when given) must agree with the probed architecture; a device that
    disagrees is reported as unobservable rather than trusted.
    """
    detected = _probe_device() or {}
    sku = str(detected.get("sku") or "").lower()
    if not sku:
        return None
    try:
        spec = _core.hardware_spec(sku)
    except _core.ModelError:
        return None
    if arch and spec.architecture.lower() != str(arch).lower():
        return None
    return spec.sku.lower()


def resolve_sku(arch: Optional[str] = None, sku: Optional[str] = None) -> tuple[str, str]:
    """Return ``(sku, source)`` for ``arch``, preferring explicit then observed.

    ``source`` is one of ``explicit`` (caller passed it), ``runtime-probe`` (the
    device reported it), or ``arch-fallback`` (nothing could be observed, so the
    documented default is used and a warning is emitted).  An unknown SKU, or a
    SKU whose architecture contradicts ``arch``, raises ``ModelError`` instead of
    quietly selecting the neighbouring board.
    """
    architecture = str(arch or DEFAULT_ARCH)
    if sku:
        selected = _core.hardware_spec(sku)
        if selected.architecture.lower() != architecture.lower():
            raise _core.ModelError(
                f"SKU {selected.sku} is {selected.architecture}, not requested {architecture}"
            )
        return selected.sku.lower(), "explicit"
    detected = observed_sku(architecture)
    if detected:
        return detected, "runtime-probe"
    fallback = _ARCH_FALLBACK_SKU.get(architecture.lower())
    if fallback is None:
        raise _core.ModelError(
            f"unsupported architecture {architecture!r}; pass sku= explicitly "
            f"(known SKUs: {_core.available_skus()})"
        )
    warnings.warn(
        f"hardware SKU for {architecture} was neither passed nor observable; falling back to "
        f"{fallback!r}. {architecture} spans boards with different compute peaks, so pass "
        "sku= explicitly (or run where rocminfo/rocm-smi is available) to keep eta and the "
        "SoL floor on the right ceiling.",
        RuntimeWarning,
        stacklevel=3,
    )
    return fallback, "arch-fallback"


def verify_runtime_sku(sku: str, *, arch: Optional[str] = None) -> dict[str, Any]:
    """Fail loudly when a configured SKU contradicts the observed device.

    Returns the verification record (``{sku, observed, status}``).  ``status`` is
    ``verified`` when the device confirms the SKU, ``unobservable`` when there is
    no device to ask (CPU-only hosts; not an error), and the call raises
    ``ModelError`` when the device reports a DIFFERENT gfx950 board -- the case
    that silently mis-scales every ``eta`` by 8.7%.
    """
    spec = _core.hardware_spec(sku)
    architecture = str(arch or spec.architecture)
    if spec.architecture.lower() != architecture.lower():
        raise _core.ModelError(
            f"SKU {spec.sku} is {spec.architecture}, not requested {architecture}"
        )
    detected = observed_sku()
    if detected is None:
        return {"sku": spec.sku, "observed": None, "status": "unobservable"}
    if detected != spec.sku.lower():
        raise _core.ModelError(
            f"configured SKU {spec.sku} contradicts the observed device "
            f"{_core.hardware_spec(detected).sku}: peaks differ, so eta and the SoL "
            "integrity floor would be computed against the wrong ceiling"
        )
    return {"sku": spec.sku, "observed": _core.hardware_spec(detected).sku, "status": "verified"}


def _sku_for_arch(arch: str, sku: Optional[str] = None) -> str:
    """Backwards-compatible wrapper over :func:`resolve_sku`."""
    return resolve_sku(arch, sku)[0]


def _peak_mapping(sku: str) -> dict[str, float]:
    spec = _core.hardware_spec(sku)
    out = {"hbm_bytes_per_s": spec.hbm_bytes_per_s}
    out.update({f"{dtype}_flops_per_s": value for dtype, value in spec.compute_flops_per_s.items()})
    return out


# Static catalog for old report/calibration code.  It is not an active runtime
# model and no environment/GPU probe occurs at import.  The ``gfx950`` row is
# DEFAULT_SKU's datasheet; arch->peaks is ambiguous on gfx950, so prefer the
# SKU-keyed rows.
PEAKS: dict[str, dict[str, float]] = {
    "gfx950": _peak_mapping(DEFAULT_SKU),
    "mi350x": _peak_mapping("mi350x"),
    "mi355x": _peak_mapping("mi355x"),
    "gfx942": _peak_mapping("mi300x"),
    "mi300x": _peak_mapping("mi300x"),
}


def detect_arch(default: Optional[str] = None) -> str:
    """Runtime probe retained for compatibility; never called at import."""
    detected = _probe_device()
    if detected and detected.get("architecture"):
        return detected["architecture"]
    if default:
        return default
    raise _core.ModelError("GPU architecture unavailable; pass architecture/SKU explicitly")


# --------------------------------------------------------------------------- #
# The one supported calibration path: kore.runtime-calibration.v1 documents.
# --------------------------------------------------------------------------- #
def calibration_document(
    sku: str,
    *,
    hbm_bytes_per_s: float,
    compute_flops_per_s: Mapping[str, float],
    calibration_id: str,
    runtime: Mapping[str, Any],
    source: str = "runtime-measured",
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build a validated ``kore.runtime-calibration.v1`` document.

    This is the only format :func:`kore.analysis.roofline.make_physical_model`
    accepts, and the only way measured peaks reach ``T_min``.  The document is
    validated by building the model from it, so a document that cannot be
    applied is rejected here rather than at the next reproduce attempt.  The
    resulting ``model_fingerprint`` is embedded so the artifact can be pinned.

    ``runtime`` must identify the stack the numbers were measured on (ROCm /
    torch / driver versions, host); an anonymous calibration is not auditable
    and ``make_physical_model`` rejects it.
    """
    spec = _core.hardware_spec(sku)
    if not str(calibration_id or "").strip():
        raise _core.ModelError("calibration_id must be a non-empty identifier")
    runtime_map = {str(key): str(value) for key, value in dict(runtime).items() if value is not None}
    if not runtime_map:
        raise _core.ModelError(
            "calibration runtime metadata is mandatory (ROCm/torch/driver identity)"
        )
    peaks = {str(dtype): float(value) for dtype, value in dict(compute_flops_per_s).items()}
    document: dict[str, Any] = {
        "schema": CALIBRATION_SCHEMA,
        "architecture": spec.architecture,
        "sku": spec.sku,
        "calibration_id": str(calibration_id),
        "source": str(source),
        "runtime": runtime_map,
        "hbm_bytes_per_s": float(hbm_bytes_per_s),
        "compute_flops_per_s": peaks,
    }
    if extra:
        for key, value in dict(extra).items():
            if key not in document:
                document[key] = value
    model = _core.make_physical_model(spec.sku.lower(), document)
    document["model_fingerprint"] = model.fingerprint
    return document


def load_calibration(path: Union[str, Path]) -> dict[str, Any]:
    """Read a calibration document, rejecting anything that is not v1 schema."""
    target = Path(path)
    try:
        data = json.loads(target.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise _core.ModelError(f"cannot read calibration {target}: {exc}") from exc
    if not isinstance(data, dict):
        raise _core.ModelError("calibration must be a JSON object")
    schema = str(data.get("schema") or "")
    if schema and schema != CALIBRATION_SCHEMA:
        raise _core.ModelError(
            f"calibration schema {schema!r} is not {CALIBRATION_SCHEMA!r}"
        )
    return data


def resolve_model(
    *,
    sku: str,
    calibration: Optional[Union[str, Path, Mapping[str, Any]]] = None,
    expected_fingerprint: Optional[str] = None,
    verify_runtime: bool = False,
) -> _core.PhysicalModel:
    """Build the fingerprinted model for an explicit SKU and optional calibration.

    ``expected_fingerprint`` is the fail-closed pin: an unfingerprinted or
    mismatched calibration raises instead of silently applying.  Set
    ``verify_runtime=True`` to additionally require that the observed device
    matches ``sku``.
    """
    if verify_runtime:
        verify_runtime_sku(sku)
    if isinstance(calibration, (str, Path)):
        calibration = load_calibration(calibration)
    return _core.make_physical_model(
        sku, calibration, expected_fingerprint=expected_fingerprint
    )


def resolve_peaks(
    arch: Optional[str] = None,
    *,
    sku: Optional[str] = None,
    calibration: Optional[Union[str, Path, Mapping[str, Any]]] = None,
    expected_fingerprint: Optional[str] = None,
    verify_runtime: bool = False,
) -> dict[str, Any]:
    """Compatibility peak mapping for a SKU plus an optional calibration.

    Environment overrides were removed because they were invisible,
    unfingerprinted global calibration.  Pass ``calibration`` (a
    ``kore.runtime-calibration.v1`` document or its path) instead; exported
    ``KORE_PEAK_*`` variables are reported as a warning so the dead path is never
    silent again.  The returned mapping records ``sku_source`` and the
    calibration identity so a report can state which ceiling it divided by.
    """
    warn_on_legacy_peak_env()
    architecture = arch or DEFAULT_ARCH
    selected_sku, sku_source = resolve_sku(architecture, sku)
    model = resolve_model(
        sku=selected_sku,
        calibration=calibration,
        expected_fingerprint=expected_fingerprint,
        verify_runtime=verify_runtime,
    )
    out: dict[str, Any] = {
        "hbm_bytes_per_s": model.hbm_bytes_per_s,
        "architecture": model.architecture,
        "sku": model.sku,
        "sku_source": sku_source,
        "calibration_source": model.calibration_source,
        "calibration_id": model.calibration_id,
        "model_fingerprint": model.fingerprint,
        "runtime": dict(model.runtime),
    }
    out.update(
        {f"{dtype}_flops_per_s": value for dtype, value in model.compute_flops_per_s.items()}
    )
    return out


dtype_bytes = _core.dtype_bytes


def peak_flops(peaks: Mapping[str, Any], dtype: str) -> Optional[float]:
    canonical = _core.canonical_dtype(dtype)
    if canonical is None:
        return None
    value = peaks.get(f"{canonical}_flops_per_s")
    return float(value) if isinstance(value, (int, float)) else None


def flops_bytes(
    operation: str, dims: dict[str, int], dtype: str
) -> Optional[tuple[float, float]]:
    """Compatibility tuple from the canonical typed work estimator."""
    work = _core.estimate_work(operation, dims, dtype)
    return (work.flops, work.bytes) if work else None


@dataclass(frozen=True)
class Roofline:
    task_id: str
    operation: str
    dtype: str
    shape: str
    arch: str
    sku: str
    model_fingerprint: str
    flops: float
    bytes: float
    arithmetic_intensity: float
    t_compute_ms: float
    t_mem_ms: float
    t_min_ms: float
    bound: str
    work_model: str
    work_assumptions: tuple[str, ...] = ()


def roofline(
    task_id: str,
    operation: str,
    dtype: str,
    shape_str: str,
    dims: dict[str, int],
    peaks: Optional[Mapping[str, Any]] = None,
    arch: Optional[str] = None,
    *,
    model: Optional[_core.PhysicalModel] = None,
    sku: Optional[str] = None,
) -> Optional[Roofline]:
    """Old operator signature, evaluated by the canonical model."""
    work = _core.estimate_work(operation, dims, dtype)
    if work is None:
        return None
    if model is None:
        architecture = arch or str((peaks or {}).get("architecture") or DEFAULT_ARCH)
        selected_sku = _sku_for_arch(
            architecture, sku or (str((peaks or {}).get("sku") or "").lower() or None)
        )
        if peaks is None:
            model = _core.make_physical_model(selected_sku)
        else:
            model = _core.model_from_peak_mapping(
                peaks,
                sku=selected_sku,
                source=str(peaks.get("calibration_source") or "legacy-explicit-mapping"),
            )
    result = _core.evaluate_roofline(work, model)
    if result is None:
        return None
    return Roofline(
        task_id=task_id,
        operation=operation,
        dtype=work.dtype,
        shape=shape_str,
        arch=model.architecture,
        sku=model.sku,
        model_fingerprint=model.fingerprint,
        flops=work.flops,
        bytes=work.bytes,
        arithmetic_intensity=result.arithmetic_intensity_flops_per_byte,
        t_compute_ms=result.t_compute_ms,
        t_mem_ms=result.t_memory_ms,
        t_min_ms=result.t_min_ms,
        bound=result.bound,
        work_model=work.model_kind,
        work_assumptions=work.assumptions,
    )


def shape_to_str(dims: Mapping[str, int]) -> str:
    return ",".join(f"{key}={value}" for key, value in dims.items())


__all__ = [
    "CALIBRATION_ENV_VAR",
    "CALIBRATION_SCHEMA",
    "DEFAULT_ARCH",
    "DEFAULT_SKU",
    "FINGERPRINT_ENV_VAR",
    "LEGACY_PEAK_ENV_VARS",
    "PEAKS",
    "Roofline",
    "calibration_document",
    "detect_arch",
    "dtype_bytes",
    "flops_bytes",
    "legacy_peak_env_overrides",
    "load_calibration",
    "observed_sku",
    "peak_flops",
    "resolve_model",
    "resolve_peaks",
    "resolve_sku",
    "roofline",
    "shape_to_str",
    "verify_runtime_sku",
    "warn_on_legacy_peak_env",
]
