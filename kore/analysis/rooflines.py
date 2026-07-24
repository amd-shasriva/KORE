"""Deprecated operator-level facade over :mod:`kore.analysis.roofline`.

Historically KORE carried two independent roofline implementations.  This module
now contains adapters only: work estimates, peaks, runtimes, and fingerprints all
come from the singular model in :mod:`kore.analysis.roofline`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from kore.analysis import roofline as _core

DEFAULT_ARCH = "gfx950"
DEFAULT_SKU = "mi350x"


def _sku_for_arch(arch: str, sku: Optional[str] = None) -> str:
    if sku:
        selected = _core.hardware_spec(sku)
        if selected.architecture != arch:
            raise _core.ModelError(
                f"SKU {selected.sku} is {selected.architecture}, not requested {arch}"
            )
        return sku
    # Legacy disambiguation only.  Canonical callers select SKU explicitly.
    if arch == "gfx950":
        return "mi350x"
    if arch == "gfx942":
        return "mi300x"
    raise _core.ModelError(f"unsupported architecture {arch!r}")


def _peak_mapping(sku: str) -> dict[str, float]:
    spec = _core.hardware_spec(sku)
    out = {"hbm_bytes_per_s": spec.hbm_bytes_per_s}
    out.update({f"{dtype}_flops_per_s": value for dtype, value in spec.compute_flops_per_s.items()})
    return out


# Static catalog for old report/calibration code.  It is not an active runtime
# model and no environment/GPU probe occurs at import.
PEAKS: dict[str, dict[str, float]] = {
    "gfx950": _peak_mapping("mi350x"),
    "mi350x": _peak_mapping("mi350x"),
    "mi355x": _peak_mapping("mi355x"),
    "gfx942": _peak_mapping("mi300x"),
    "mi300x": _peak_mapping("mi300x"),
}


def detect_arch(default: Optional[str] = None) -> str:
    """Runtime probe retained for compatibility; never called at import."""
    detected = _core.detect_runtime_device()
    if detected and detected.get("architecture"):
        return detected["architecture"]
    if default:
        return default
    raise _core.ModelError("GPU architecture unavailable; pass architecture/SKU explicitly")


def resolve_model(
    *,
    sku: str,
    calibration: Optional[Union[str, Path, Mapping[str, Any]]] = None,
    expected_fingerprint: Optional[str] = None,
) -> _core.PhysicalModel:
    return _core.make_physical_model(
        sku, calibration, expected_fingerprint=expected_fingerprint
    )


def resolve_peaks(
    arch: Optional[str] = None,
    *,
    sku: Optional[str] = None,
    calibration: Optional[Union[str, Path, Mapping[str, Any]]] = None,
    expected_fingerprint: Optional[str] = None,
) -> dict[str, Any]:
    """Compatibility peak mapping.

    Environment overrides were removed because they were invisible,
    unfingerprinted global calibration.  Pass a calibration object/path instead.
    """
    architecture = arch or DEFAULT_ARCH
    selected_sku = _sku_for_arch(architecture, sku)
    model = resolve_model(
        sku=selected_sku,
        calibration=calibration,
        expected_fingerprint=expected_fingerprint,
    )
    out: dict[str, Any] = {
        "hbm_bytes_per_s": model.hbm_bytes_per_s,
        "architecture": model.architecture,
        "sku": model.sku,
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
    "DEFAULT_ARCH",
    "DEFAULT_SKU",
    "PEAKS",
    "Roofline",
    "detect_arch",
    "dtype_bytes",
    "flops_bytes",
    "peak_flops",
    "resolve_model",
    "resolve_peaks",
    "roofline",
    "shape_to_str",
]
