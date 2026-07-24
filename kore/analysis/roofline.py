"""Authoritative, unit-safe roofline and counter semantics.

The canonical API is deliberately explicit:

* :func:`make_physical_model` selects an architecture/SKU and optional runtime
  calibration.  A model is immutable, validated, and fingerprinted.
* :func:`estimate_work` returns a typed :class:`WorkEstimate`, or ``None`` when
  an operation/dtype cannot be defended.  There is no generic elementwise
  fallback.
* :func:`evaluate_roofline` combines those two typed values.

The small scalar helpers at the bottom are compatibility adapters.  They build
the documented MI350X datasheet model on each call when no model is supplied;
there is no import-time GPU detection or mutable "active architecture".
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union


class ModelError(ValueError):
    """A physical-model or unit contract is invalid."""


class CounterUnit(str, Enum):
    INSTRUCTIONS = "instructions"
    QCYCLES = "qcycles"
    CYCLES = "cycles"
    MOPS_512_FMA = "mops_512_fma"
    PERCENT = "percent"
    REQUESTS = "requests"
    BYTES = "bytes"
    WAVES = "waves"
    COUNT = "count"


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _positive(name: str, value: Any, *, allow_zero: bool = False) -> float:
    if not _finite(value):
        raise ModelError(f"{name} must be finite, got {value!r}")
    out = float(value)
    if out < 0.0 or (out == 0.0 and not allow_zero):
        op = "non-negative" if allow_zero else "positive"
        raise ModelError(f"{name} must be {op}, got {value!r}")
    return out


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelError(f"{name} must be a positive integer, got {value!r}")
    return value


_DTYPE_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("fp4", ("mxfp4", "fp4", "e2m1")),
    ("fp6", ("mxfp6", "fp6", "e2m3", "e3m2")),
    ("fp8", ("mxfp8", "fp8", "float8", "e4m3", "e5m2")),
    ("int8", ("int8",)),
    ("bf16", ("bf16", "bfloat16")),
    ("fp16", ("fp16", "float16", "half")),
    ("tf32", ("tf32", "xf32")),
    ("fp64", ("fp64", "float64", "double")),
    ("fp32", ("fp32", "float32")),
)

_DTYPE_BYTES = {
    "fp4": 0.5,
    "fp6": 0.75,
    "fp8": 1.0,
    "int8": 1.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "tf32": 4.0,
    "fp32": 4.0,
    "fp64": 8.0,
}


def canonical_dtype(dtype: str) -> Optional[str]:
    """Return the canonical dtype name, or ``None`` when unsupported."""
    d = str(dtype or "").strip().lower()
    if d in _DTYPE_BYTES:
        return d
    for canonical, aliases in _DTYPE_ALIASES:
        if any(alias in d for alias in aliases):
            return canonical
    return None


def dtype_bytes(dtype: str) -> Optional[float]:
    """Packed storage bytes/element, or ``None`` for an unknown dtype."""
    canonical = canonical_dtype(dtype)
    return _DTYPE_BYTES.get(canonical) if canonical else None


@dataclass(frozen=True)
class HardwareSpec:
    """Vendor upper-bound specification with units encoded in field names."""

    architecture: str
    sku: str
    hbm_bytes_per_s: float
    compute_flops_per_s: Mapping[str, float]
    num_cus: int
    peak_clock_hz: float
    infinity_cache_bytes: int
    lds_bytes_per_cu: int
    vgpr_per_simd: int = 512
    vgpr_alloc_granularity: int = 8
    max_waves_per_simd: int = 8
    simds_per_cu: int = 4

    def __post_init__(self) -> None:
        if not self.architecture or not self.sku:
            raise ModelError("architecture and SKU must be explicit")
        object.__setattr__(self, "hbm_bytes_per_s", _positive("hbm_bytes_per_s", self.hbm_bytes_per_s))
        peaks: dict[str, float] = {}
        for dtype, value in dict(self.compute_flops_per_s).items():
            canonical = canonical_dtype(dtype)
            if canonical is None or canonical != dtype:
                raise ModelError(f"compute peak key must be canonical, got {dtype!r}")
            peaks[canonical] = _positive(f"{dtype}_flops_per_s", value)
        if not peaks:
            raise ModelError("at least one compute peak is required")
        object.__setattr__(self, "compute_flops_per_s", peaks)
        for name in (
            "num_cus",
            "infinity_cache_bytes",
            "lds_bytes_per_cu",
            "vgpr_per_simd",
            "vgpr_alloc_granularity",
            "max_waves_per_simd",
            "simds_per_cu",
        ):
            object.__setattr__(self, name, _positive_int(name, getattr(self, name)))
        object.__setattr__(self, "peak_clock_hz", _positive("peak_clock_hz", self.peak_clock_hz))


_HARDWARE_SPECS: dict[str, HardwareSpec] = {
    "mi350x": HardwareSpec(
        architecture="gfx950",
        sku="MI350X",
        hbm_bytes_per_s=8.0e12,
        compute_flops_per_s={
            "bf16": 2.30e15,
            "fp16": 2.30e15,
            "fp8": 4.60e15,
            "int8": 4.60e15,
            "fp6": 9.20e15,
            "fp4": 9.20e15,
            "tf32": 1.153e15,
            "fp32": 1.442e14,
            "fp64": 7.21e13,
        },
        num_cus=256,
        peak_clock_hz=2.2e9,
        infinity_cache_bytes=256 * 1024 * 1024,
        lds_bytes_per_cu=160 * 1024,
    ),
    "mi355x": HardwareSpec(
        architecture="gfx950",
        sku="MI355X",
        hbm_bytes_per_s=8.0e12,
        compute_flops_per_s={
            "bf16": 2.50e15,
            "fp16": 2.50e15,
            "fp8": 5.00e15,
            "int8": 5.00e15,
            "fp6": 10.10e15,
            "fp4": 10.10e15,
            "tf32": 1.2583e15,
            "fp32": 1.573e14,
            "fp64": 7.86e13,
        },
        num_cus=256,
        peak_clock_hz=2.4e9,
        infinity_cache_bytes=256 * 1024 * 1024,
        lds_bytes_per_cu=160 * 1024,
    ),
    # Compatibility target.  Native fp4/fp6 are intentionally absent: using an
    # fp8 peak for those dtypes fabricated support in the previous model.
    "mi300x": HardwareSpec(
        architecture="gfx942",
        sku="MI300X",
        hbm_bytes_per_s=5.325e12,
        compute_flops_per_s={
            "bf16": 1.3074e15,
            "fp16": 1.3074e15,
            "fp8": 2.6149e15,
            "int8": 2.6149e15,
            "tf32": 6.537e14,
            "fp32": 1.634e14,
            "fp64": 1.634e14,
        },
        num_cus=304,
        peak_clock_hz=2.1e9,
        infinity_cache_bytes=256 * 1024 * 1024,
        lds_bytes_per_cu=64 * 1024,
        vgpr_alloc_granularity=16,
    ),
}


def available_skus() -> tuple[str, ...]:
    return tuple(sorted(_HARDWARE_SPECS))


def hardware_spec(sku: str) -> HardwareSpec:
    key = str(sku or "").strip().lower()
    aliases = {"mi350": "mi350x", "mi355": "mi355x", "mi300": "mi300x"}
    key = aliases.get(key, key)
    if key not in _HARDWARE_SPECS:
        raise ModelError(f"unsupported SKU {sku!r}; choose one of {available_skus()}")
    return _HARDWARE_SPECS[key]


@dataclass(frozen=True)
class PhysicalModel:
    """A selected hardware model and runtime calibration."""

    spec: HardwareSpec
    hbm_bytes_per_s: float
    compute_flops_per_s: Mapping[str, float]
    calibration_source: str
    calibration_id: str
    runtime: Mapping[str, str] = field(default_factory=dict)
    integrity_upper_bound: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "hbm_bytes_per_s", _positive("hbm_bytes_per_s", self.hbm_bytes_per_s))
        peaks: dict[str, float] = {}
        for dtype, value in dict(self.compute_flops_per_s).items():
            canonical = canonical_dtype(dtype)
            if canonical is None or canonical != dtype:
                raise ModelError(f"calibrated peak key must be canonical, got {dtype!r}")
            if canonical not in self.spec.compute_flops_per_s:
                raise ModelError(f"{self.spec.sku} does not support {canonical}")
            peaks[canonical] = _positive(f"{dtype}_flops_per_s", value)
        if not peaks:
            raise ModelError("calibration exposes no supported compute peak")
        object.__setattr__(self, "compute_flops_per_s", peaks)
        if not self.calibration_source or not self.calibration_id:
            raise ModelError("calibration source and id must be explicit")
        runtime = {str(k): str(v) for k, v in dict(self.runtime).items() if v is not None}
        object.__setattr__(self, "runtime", runtime)

    @property
    def architecture(self) -> str:
        return self.spec.architecture

    @property
    def sku(self) -> str:
        return self.spec.sku

    def peak_flops_per_s(self, dtype: str) -> Optional[float]:
        canonical = canonical_dtype(dtype)
        return self.compute_flops_per_s.get(canonical) if canonical else None

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "schema": "kore.physical-model.v1",
            "architecture": self.architecture,
            "sku": self.sku,
            "calibration_source": self.calibration_source,
            "calibration_id": self.calibration_id,
            "integrity_upper_bound": self.integrity_upper_bound,
            "hbm_bytes_per_s": self.hbm_bytes_per_s,
            "compute_flops_per_s": dict(sorted(self.compute_flops_per_s.items())),
            "runtime": dict(sorted(self.runtime.items())),
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.fingerprint_payload(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {**self.fingerprint_payload(), "fingerprint": self.fingerprint}

    def require_fingerprint(self, expected: Optional[str]) -> None:
        if expected and expected != self.fingerprint:
            raise ModelError(
                f"physical-model fingerprint mismatch: expected {expected}, got {self.fingerprint}"
            )

    def for_integrity(self) -> "PhysicalModel":
        """Conservative datasheet-or-higher upper bounds for rejection/pruning.

        Empirical achievable calibration is never used as a physical limit:
        a future kernel may legitimately exceed it.  Taking the larger bandwidth
        and compute peak yields the smallest defensible ``T_min``.
        """
        peaks = dict(self.spec.compute_flops_per_s)
        for dtype, value in self.compute_flops_per_s.items():
            peaks[dtype] = max(peaks.get(dtype, 0.0), value)
        return PhysicalModel(
            spec=self.spec,
            hbm_bytes_per_s=max(self.spec.hbm_bytes_per_s, self.hbm_bytes_per_s),
            compute_flops_per_s=peaks,
            calibration_source="integrity-upper-bound",
            calibration_id=f"{self.calibration_id}:integrity",
            runtime=self.runtime,
            integrity_upper_bound=True,
        )


def _read_calibration(calibration: Union[str, Path, Mapping[str, Any]]) -> Mapping[str, Any]:
    if isinstance(calibration, Mapping):
        return dict(calibration)
    path = Path(calibration)
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelError(f"cannot read calibration {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ModelError("calibration must be a JSON object")
    return data


def make_physical_model(
    sku: str,
    calibration: Optional[Union[str, Path, Mapping[str, Any]]] = None,
    *,
    expected_fingerprint: Optional[str] = None,
) -> PhysicalModel:
    """Build a validated, fingerprinted model for an explicit SKU.

    Calibration schema ``kore.runtime-calibration.v1`` requires ``architecture``,
    ``sku``, ``calibration_id``, ``runtime``, ``hbm_bytes_per_s``, and a
    ``compute_flops_per_s`` mapping.  Legacy files lacking SKU/runtime identity
    are rejected rather than silently attached to the wrong gfx950 board.
    """
    spec = hardware_spec(sku)
    if calibration is None:
        model = PhysicalModel(
            spec=spec,
            hbm_bytes_per_s=spec.hbm_bytes_per_s,
            compute_flops_per_s=spec.compute_flops_per_s,
            calibration_source="vendor-datasheet",
            calibration_id=f"{spec.sku.lower()}-datasheet-v1",
            runtime={},
        )
    else:
        data = _read_calibration(calibration)
        required = {
            "architecture",
            "sku",
            "calibration_id",
            "runtime",
            "hbm_bytes_per_s",
            "compute_flops_per_s",
        }
        missing = sorted(required - set(data))
        if missing:
            raise ModelError(
                "calibration is not fingerprint-safe; missing " + ", ".join(missing)
            )
        if str(data["architecture"]).lower() != spec.architecture.lower():
            raise ModelError(
                f"calibration architecture {data['architecture']!r} does not match {spec.architecture}"
            )
        if str(data["sku"]).lower() != spec.sku.lower():
            raise ModelError(f"calibration SKU {data['sku']!r} does not match {spec.sku}")
        model = PhysicalModel(
            spec=spec,
            hbm_bytes_per_s=data["hbm_bytes_per_s"],
            compute_flops_per_s=data["compute_flops_per_s"],
            calibration_source=str(data.get("source") or "runtime-measured"),
            calibration_id=str(data["calibration_id"]),
            runtime=data["runtime"],
        )
    model.require_fingerprint(expected_fingerprint)
    return model


def model_from_peak_mapping(
    peaks: Mapping[str, Any],
    *,
    sku: str,
    source: str = "legacy-explicit-mapping",
) -> PhysicalModel:
    """Compatibility adapter for old ``{..._flops_per_s}`` dictionaries."""
    spec = hardware_spec(sku)
    compute: dict[str, float] = {}
    for dtype in _DTYPE_BYTES:
        value = peaks.get(f"{dtype}_flops_per_s")
        if value is None and dtype == "bf16":
            value = peaks.get("peak_flops_bf16")
        if value is not None and dtype in spec.compute_flops_per_s:
            compute[dtype] = value
    hbm = peaks.get("hbm_bytes_per_s", peaks.get("hbm_bw_bytes_per_s"))
    if hbm is None:
        raise ModelError("peak mapping has no hbm_bytes_per_s")
    return PhysicalModel(
        spec=spec,
        hbm_bytes_per_s=hbm,
        compute_flops_per_s=compute,
        calibration_source=source,
        calibration_id=str(peaks.get("calibration_id") or source),
        runtime=peaks.get("runtime") or {},
    )


@dataclass(frozen=True)
class WorkEstimate:
    """Mandatory operator work.  Units are FLOPs and bytes."""

    operation: str
    dtype: str
    flops: float
    bytes: float
    model_kind: str
    assumptions: tuple[str, ...] = ()
    integrity_safe: bool = True

    def __post_init__(self) -> None:
        canonical = canonical_dtype(self.dtype)
        if canonical is None:
            raise ModelError(f"unsupported dtype {self.dtype!r}")
        object.__setattr__(self, "dtype", canonical)
        object.__setattr__(self, "flops", _positive("flops", self.flops))
        object.__setattr__(self, "bytes", _positive("bytes", self.bytes))
        if self.model_kind not in {"exact", "mandatory-lower-bound"}:
            raise ModelError(f"invalid work model kind {self.model_kind!r}")


ShapeLike = Union[Mapping[str, int], tuple[int, ...], list[int], int]


def _dims_from_shape(shape: ShapeLike) -> Optional[dict[str, int]]:
    if isinstance(shape, Mapping):
        dims: dict[str, int] = {}
        for key, value in shape.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                continue
            dims[str(key)] = value
        return dims or None
    if isinstance(shape, int) and not isinstance(shape, bool) and shape > 0:
        return {"numel": shape}
    if isinstance(shape, (tuple, list)):
        vals = [v for v in shape if isinstance(v, int) and not isinstance(v, bool) and v > 0]
        if not vals:
            return None
        keys = ("M", "N", "K") if len(vals) <= 3 else ("B", "M", "N", "K")
        return dict(zip(keys, vals))
    return None


def _dim(dims: Mapping[str, int], *names: str, default: Optional[int] = None) -> Optional[int]:
    for name in names:
        value = dims.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return default


def _numel(dims: Mapping[str, int]) -> Optional[int]:
    ignored = {"topk", "k", "num_warps"}
    values = [
        value
        for key, value in dims.items()
        if key not in ignored and isinstance(value, int) and not isinstance(value, bool) and value > 0
    ]
    if not values:
        return None
    out = 1
    for value in values:
        out *= value
    return out


def estimate_work(
    operation: str,
    shape: ShapeLike,
    dtype: str,
    *,
    n_tensors: Optional[int] = None,
    flops_per_element: Optional[float] = None,
) -> Optional[WorkEstimate]:
    """Defensible FLOPs and mandatory bytes, else ``None``.

    Attention, MoE, top-k, backward, unknown fused operations, and low-precision
    formats with unmodeled scale traffic are deliberately unavailable.
    """
    op = str(operation or "").strip().lower()
    canonical = canonical_dtype(dtype)
    dims = _dims_from_shape(shape)
    elem = dtype_bytes(dtype)
    if not op or canonical is None or dims is None or elem is None:
        return None
    if "backward" in op or op.endswith("_bwd") or "attn" in op or "attention" in op or "moe" in op:
        return None
    if "topk" in op or "router" in op:
        return None

    def work(flops: float, by: float, kind: str, *assumptions: str) -> WorkEstimate:
        return WorkEstimate(op, canonical, flops, by, kind, tuple(assumptions))

    # GEMM: low-precision block scales/metadata are not yet modeled.
    if "gemm" in op or "matmul" in op or op in {"mm", "bmm"}:
        if canonical in {"fp4", "fp6", "int8"}:
            return None
        m, n, k = _dim(dims, "M", "m"), _dim(dims, "N", "n"), _dim(dims, "K", "k")
        if not (m and n and k):
            return None
        batch = _dim(dims, "B", "batch", "num_batches", default=1) or 1
        flops = 2.0 * batch * m * n * k
        if canonical == "fp8":
            by = batch * ((m * k + k * n) * 1.0 + m * n * 2.0)
            return work(
                flops,
                by,
                "mandatory-lower-bound",
                "packed fp8 inputs",
                "bf16 output",
                "scale metadata excluded; use only where scales are resident/explicitly negligible",
            )
        by = float(batch) * (m * k + k * n + m * n) * elem
        return work(flops, by, "mandatory-lower-bound", "read A/B once; write C once")

    m, n = _dim(dims, "M", "m", "rows"), _dim(dims, "N", "n", "cols")
    if "fused_add_rmsnorm" in op or "fused_add_rms_norm" in op:
        if not (m and n):
            return None
        return work(5.0 * m * n, (4 * m * n + n) * elem, "mandatory-lower-bound")
    if op in {"rmsnorm", "rms_norm", "rmsnorm_forward"}:
        if not (m and n):
            return None
        return work(4.0 * m * n, (2 * m * n + n) * elem, "mandatory-lower-bound")
    if op in {"layernorm", "layer_norm", "layernorm_forward"}:
        if not (m and n):
            return None
        return work(6.0 * m * n, (2 * m * n + 2 * n) * elem, "mandatory-lower-bound")
    if op in {"softmax", "softmax_forward"}:
        if not (m and n):
            return None
        return work(5.0 * m * n, 2 * m * n * elem, "mandatory-lower-bound")
    if op in {"silu_and_mul", "silu_mul", "swiglu"}:
        if not (m and n):
            return None
        return work(4.0 * m * n, 3 * m * n * elem, "mandatory-lower-bound")
    if op in {"gelu", "gelu_tanh", "relu", "sigmoid", "tanh"}:
        if not (m and n):
            return None
        flop_rate = 8.0 if "gelu" in op else (1.0 if op == "relu" else 4.0)
        return work(flop_rate * m * n, 2 * m * n * elem, "mandatory-lower-bound")
    if op in {"sum", "mean", "reduce", "reduction", "row_sum", "row_mean", "row_max", "row_min"}:
        if n is None:
            return None
        m = m or 1
        return work(float(m * n), (m * n + m) * elem, "mandatory-lower-bound")
    if op in {"rope", "rope_gptj", "rope_partial"}:
        s = _dim(dims, "S", "seqlen")
        b = _dim(dims, "B", "batch")
        h = _dim(dims, "H", "heads")
        d = _dim(dims, "D", "head_dim")
        if not (s and b and h and d):
            return None
        count = s * b * h * d
        return work(6.0 * count, (2 * count + s * d) * elem, "mandatory-lower-bound")
    if op in {"quant_fp8_pertoken", "quant_fp8_per_token"}:
        if not (m and n) or canonical != "fp8":
            return None
        return work(
            2.0 * m * n,
            m * n * 2.0 + m * n * 1.0 + m * 4.0,
            "mandatory-lower-bound",
            "bf16 input",
            "fp8 output",
            "fp32 row scale",
        )

    # Explicit compatibility family only; unknown names never receive a model.
    if op in {"elementwise", "pointwise", "activation", "ew"}:
        count = _numel(dims)
        if count is None:
            return None
        tensors = n_tensors if n_tensors is not None else 2
        if isinstance(tensors, bool) or not isinstance(tensors, int) or tensors < 2:
            raise ModelError("n_tensors must be an integer >= 2")
        fpe = 1.0 if flops_per_element is None else _positive(
            "flops_per_element", flops_per_element
        )
        return work(fpe * count, tensors * count * elem, "mandatory-lower-bound")
    return None


@dataclass(frozen=True)
class RooflineResult:
    work: WorkEstimate
    model_fingerprint: str
    architecture: str
    sku: str
    peak_flops_per_s: float
    peak_bandwidth_bytes_per_s: float
    arithmetic_intensity_flops_per_byte: float
    ridge_point_flops_per_byte: float
    attainable_flops_per_s: float
    t_compute_ms: float
    t_memory_ms: float
    t_min_ms: float
    bound: str

    @property
    def flops(self) -> float:
        return self.work.flops

    @property
    def bytes(self) -> float:
        return self.work.bytes

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_fingerprint": self.model_fingerprint,
            "architecture": self.architecture,
            "sku": self.sku,
            "operation": self.work.operation,
            "dtype": self.work.dtype,
            "flops": self.flops,
            "bytes": self.bytes,
            "work_model": self.work.model_kind,
            "work_assumptions": list(self.work.assumptions),
            "arithmetic_intensity": self.arithmetic_intensity_flops_per_byte,
            "ridge_point": self.ridge_point_flops_per_byte,
            "peak_attainable_flops": self.attainable_flops_per_s,
            "peak_flops": self.peak_flops_per_s,
            "peak_bandwidth_bytes_per_s": self.peak_bandwidth_bytes_per_s,
            "t_compute_ms": self.t_compute_ms,
            "t_mem_ms": self.t_memory_ms,
            "t_min_ms": self.t_min_ms,
            "bound": self.bound,
        }


def evaluate_roofline(work: WorkEstimate, model: PhysicalModel) -> Optional[RooflineResult]:
    peak = model.peak_flops_per_s(work.dtype)
    if peak is None:
        return None
    bandwidth = model.hbm_bytes_per_s
    t_compute_s = work.flops / peak
    t_memory_s = work.bytes / bandwidth
    intensity = work.flops / work.bytes
    ridge = peak / bandwidth
    return RooflineResult(
        work=work,
        model_fingerprint=model.fingerprint,
        architecture=model.architecture,
        sku=model.sku,
        peak_flops_per_s=peak,
        peak_bandwidth_bytes_per_s=bandwidth,
        arithmetic_intensity_flops_per_byte=intensity,
        ridge_point_flops_per_byte=ridge,
        attainable_flops_per_s=min(peak, intensity * bandwidth),
        t_compute_ms=t_compute_s * 1e3,
        t_memory_ms=t_memory_s * 1e3,
        t_min_ms=max(t_compute_s, t_memory_s) * 1e3,
        bound="compute" if t_compute_s >= t_memory_s else "memory",
    )


def attainment(measured_ms: Any, result: RooflineResult) -> Optional[float]:
    """``T_min / measured`` as a fraction, or ``None`` for invalid timing."""
    if not _finite(measured_ms) or float(measured_ms) <= 0.0:
        return None
    value = result.t_min_ms / float(measured_ms)
    return value if math.isfinite(value) and value >= 0.0 else None


def _counter_key(counters: Mapping[str, Any], name: str) -> Optional[str]:
    target = name.upper()
    return next((str(key) for key in counters if str(key).upper() == target), None)


def counter_unit(name: str) -> Optional[CounterUnit]:
    n = str(name or "").upper()
    if n in {"MEMUNITSTALLED", "MEMUNITSTALLED_PCT", "MEM_UNIT_STALLED",
             "OCCUPANCYPERCENT", "OCCUPANCY_PERCENT", "MFMAUTIL"}:
        return CounterUnit.PERCENT
    if "MFMA_MOPS" in n or n == "MFMA_MOPS":
        return CounterUnit.MOPS_512_FMA
    if n.startswith("SQ_WAIT_INST_") or n == "SQ_ACTIVE_INST_VMEM":
        return CounterUnit.QCYCLES
    if n in {"SQ_BUSY_CYCLES", "SQ_VALU_MFMA_BUSY_CYCLES", "GRBM_GUI_ACTIVE", "GRBM_COUNT"}:
        return CounterUnit.CYCLES
    if n.startswith("SQ_INSTS_"):
        return CounterUnit.INSTRUCTIONS
    if n.startswith("TCC_"):
        return CounterUnit.REQUESTS
    if n == "SQ_WAVES":
        return CounterUnit.WAVES
    if n in {"LDS_BYTES", "SCRATCH_BYTES"}:
        return CounterUnit.BYTES
    if n in {"VGPR_COUNT", "NUM_WARPS"}:
        return CounterUnit.COUNT
    return None


def counter_value(
    counters: Mapping[str, Any], name: str, unit: CounterUnit
) -> Optional[float]:
    """Validated counter value when its declared unit matches."""
    key = _counter_key(counters, name)
    if key is None or counter_unit(key) != unit:
        return None
    value = counters[key]
    if not _finite(value) or float(value) < 0.0:
        return None
    return float(value)


def _first_counter(
    counters: Mapping[str, Any], unit: CounterUnit, *names: str
) -> Optional[float]:
    for name in names:
        value = counter_value(counters, name, unit)
        if value is not None:
            return value
    return None


def mfma_flops(counters: Mapping[str, Any]) -> Optional[float]:
    """FLOPs from MOPS counters only; issue counters are never mixed in."""
    values = [
        float(value)
        for key, value in counters.items()
        if counter_unit(str(key)) == CounterUnit.MOPS_512_FMA
        and _finite(value)
        and float(value) >= 0.0
    ]
    if not values:
        return None
    return 512.0 * 2.0 * sum(values)


def mfma_instruction_count(counters: Mapping[str, Any]) -> Optional[float]:
    values: list[float] = []
    for key, value in counters.items():
        name = str(key).upper()
        if (
            "MFMA" in name
            and "MOPS" not in name
            and name != "SQ_INSTS_VALU"
            and counter_unit(name) == CounterUnit.INSTRUCTIONS
            and _finite(value)
            and float(value) >= 0.0
        ):
            values.append(float(value))
    return sum(values) if values else None


def issued_instructions(counters: Mapping[str, Any]) -> Optional[float]:
    """Sum disjoint instruction families without MFMA double-counting."""
    values = [
        _first_counter(counters, CounterUnit.INSTRUCTIONS, name)
        for name in ("SQ_INSTS_VALU", "SQ_INSTS_SALU", "SQ_INSTS_VMEM", "SQ_INSTS_LDS")
    ]
    present = [value for value in values if value is not None]
    if present:
        return sum(present)
    return mfma_instruction_count(counters)


def derived_percent(counters: Mapping[str, Any], *names: str) -> Optional[float]:
    value = _first_counter(counters, CounterUnit.PERCENT, *names)
    if value is None or value > 100.0:
        return None
    return value / 100.0


def l2_hit_rate(counters: Mapping[str, Any]) -> Optional[float]:
    hit = _first_counter(counters, CounterUnit.REQUESTS, "TCC_HIT_sum", "TCC_HIT")
    miss = _first_counter(counters, CounterUnit.REQUESTS, "TCC_MISS_sum", "TCC_MISS")
    if hit is None or miss is None or hit + miss <= 0.0:
        return None
    return hit / (hit + miss)


def hbm_bytes(counters: Mapping[str, Any]) -> Optional[float]:
    """Exact HBM bytes; unavailable when transaction-size splits are absent."""
    rd = _first_counter(
        counters, CounterUnit.REQUESTS, "TCC_EA0_RDREQ_sum", "TCC_EA_RDREQ_sum"
    )
    rd32 = _first_counter(
        counters, CounterUnit.REQUESTS, "TCC_EA0_RDREQ_32B_sum", "TCC_EA_RDREQ_32B_sum"
    )
    wr = _first_counter(
        counters, CounterUnit.REQUESTS, "TCC_EA0_WRREQ_sum", "TCC_EA_WRREQ_sum"
    )
    wr64 = _first_counter(
        counters, CounterUnit.REQUESTS, "TCC_EA0_WRREQ_64B_sum", "TCC_EA_WRREQ_64B_sum"
    )
    if rd is None and wr is None:
        return None
    if (rd is not None and rd32 is None) or (wr is not None and wr64 is None):
        return None
    if rd32 is not None and rd is not None and rd32 > rd:
        return None
    if wr64 is not None and wr is not None and wr64 > wr:
        return None
    read_bytes = 0.0 if rd is None else 32.0 * (rd32 or 0.0) + 64.0 * (rd - (rd32 or 0.0))
    write_bytes = 0.0 if wr is None else 64.0 * (wr64 or 0.0) + 32.0 * (wr - (wr64 or 0.0))
    return read_bytes + write_bytes


@dataclass(frozen=True)
class OccupancyEstimate:
    waves_per_simd: float
    occupancy: float
    limiter: str
    workgroups_per_cu: int
    num_warps: int


def est_occupancy(
    vgpr: Optional[int] = None,
    lds: Optional[int] = None,
    num_warps: Optional[int] = None,
    *,
    model: Optional[PhysicalModel] = None,
) -> OccupancyEstimate:
    """Resource-limited occupancy using the selected model's explicit SKU."""
    selected = model or make_physical_model("mi350x")
    spec = selected.spec
    warps = int(num_warps) if num_warps and num_warps > 0 else 4
    if vgpr and vgpr > 0:
        alloc = int(math.ceil(vgpr / spec.vgpr_alloc_granularity) * spec.vgpr_alloc_granularity)
        waves_vgpr = min(spec.max_waves_per_simd, spec.vgpr_per_simd // alloc)
    else:
        waves_vgpr = spec.max_waves_per_simd
    wg_vgpr = waves_vgpr * spec.simds_per_cu // warps
    wg_lds = (
        spec.lds_bytes_per_cu // int(lds)
        if lds and lds > 0
        else spec.max_waves_per_simd * spec.simds_per_cu // warps
    )
    wg_slots = spec.max_waves_per_simd * spec.simds_per_cu // warps
    workgroups = max(0, min(wg_vgpr, wg_lds, wg_slots))
    waves = workgroups * warps / spec.simds_per_cu
    limiter = "none"
    if workgroups == wg_vgpr and waves_vgpr < spec.max_waves_per_simd:
        limiter = "vgpr"
    elif workgroups == wg_lds and lds and lds > 0:
        limiter = "lds"
    elif workgroups == wg_slots:
        limiter = "wave_slots"
    return OccupancyEstimate(
        waves_per_simd=waves,
        occupancy=max(0.0, min(1.0, waves / spec.max_waves_per_simd)),
        limiter=limiter,
        workgroups_per_cu=workgroups,
        num_warps=warps,
    )


BOTTLENECK_GROUNDING_TERMS: dict[str, tuple[str, ...]] = {
    "memory-bound": ("memory", "vmem", "bandwidth", "hbm", "l2", "cache"),
    "l2-bound": ("l2", "cache", "reuse", "tile", "blocking"),
    "lds-bound": ("lds", "shared memory", "bank conflict"),
    "no-matrix-cores": ("mfma", "tl.dot", "matrix core"),
    "occupancy-bound": ("occupancy", "waves", "vgpr", "register", "lds"),
    "compute-bound": ("compute", "mfma", "matrix", "pipeline"),
    "unknown": (),
}

_CANONICAL_LABEL = {
    "l2-bound": "memory-bound",
    "occupancy-bound": "compute-bound",
}


def canonicalize_label(label: str) -> str:
    return _CANONICAL_LABEL.get(label, label)


def bottleneck_from_counters(
    counters: Mapping[str, Any],
    vgpr: Optional[int] = None,
    lds: Optional[int] = None,
    num_warps: Optional[int] = None,
    *,
    model: Optional[PhysicalModel] = None,
) -> tuple[str, str]:
    """Dimensionally valid bottleneck diagnosis; unknown evidence stays unknown."""
    counters = counters or {}
    if not counters and vgpr is None and lds is None:
        return "unknown", "no counters collected"
    mfma_ops = mfma_flops(counters)
    mfma_issues = mfma_instruction_count(counters)
    mfma_present = any("MFMA" in str(key).upper() for key in counters)
    valu = _first_counter(counters, CounterUnit.INSTRUCTIONS, "SQ_INSTS_VALU")
    vmem = _first_counter(counters, CounterUnit.INSTRUCTIONS, "SQ_INSTS_VMEM")
    wait_any = _first_counter(counters, CounterUnit.QCYCLES, "SQ_WAIT_INST_ANY")
    wait_lds = _first_counter(counters, CounterUnit.QCYCLES, "SQ_WAIT_INST_LDS")
    wait_vmem = _first_counter(counters, CounterUnit.QCYCLES, "SQ_WAIT_INST_VMEM")
    hit = l2_hit_rate(counters)
    traffic = hbm_bytes(counters)
    mfma_util = derived_percent(counters, "MfmaUtil")
    if mfma_util is None:
        busy = _first_counter(counters, CounterUnit.CYCLES, "SQ_VALU_MFMA_BUSY_CYCLES")
        active = _first_counter(counters, CounterUnit.CYCLES, "GRBM_GUI_ACTIVE", "SQ_BUSY_CYCLES")
        if busy is not None and active and busy <= active:
            mfma_util = busy / active
    occupancy = derived_percent(counters, "OccupancyPercent", "Occupancy")
    if occupancy is None and (vgpr is not None or lds is not None):
        occupancy = est_occupancy(vgpr, lds, num_warps, model=model).occupancy

    if mfma_present and (mfma_ops == 0.0 or mfma_issues == 0.0) and valu and valu > 0.0:
        return "no-matrix-cores", "MFMA counter is present and zero while VALU instructions execute"
    if occupancy is not None and occupancy <= 0.25:
        return "occupancy-bound", f"occupancy {occupancy:.0%}; resource limits leave too few resident waves"
    if wait_any and wait_lds is not None and wait_lds / wait_any >= 0.30:
        return "lds-bound", f"SQ_WAIT_INST_LDS is {wait_lds / wait_any:.0%} of wait qcycles"
    if hit is not None and hit < 0.50:
        traffic_text = f", exact HBM traffic {traffic / 1e6:.1f} MB" if traffic is not None else ""
        return "memory-bound", f"L2 hit-rate {hit:.0%}{traffic_text}"
    if wait_any and wait_vmem is not None and wait_vmem / wait_any >= 0.50:
        return "memory-bound", f"SQ_WAIT_INST_VMEM is {wait_vmem / wait_any:.0%} of wait qcycles"
    if hit is not None and hit >= 0.90 and vmem and vmem > 0.0:
        return "l2-bound", f"L2 hit-rate {hit:.0%} with vector-memory instructions present"
    if mfma_util is not None and mfma_util >= 0.70:
        return "compute-bound", f"MFMA busy/utilization {mfma_util:.0%}"
    if mfma_ops is not None and mfma_ops > 0.0:
        return "unknown", "MFMA work observed, but no same-unit utilization denominator"
    if vmem and vmem > 0.0:
        return "memory-bound", f"SQ_INSTS_VMEM={vmem:.0f}; stronger traffic/stall counters unavailable"
    return "unknown", "counters are valid but inconclusive"


# ------------------------------------------------------------------------- #
# Compatibility adapters.  Canonical callers should pass WorkEstimate/model.
# ------------------------------------------------------------------------- #
def peak_flops(dtype: str, model: Optional[PhysicalModel] = None) -> Optional[float]:
    return (model or make_physical_model("mi350x")).peak_flops_per_s(dtype)


def roofline(
    flops: float,
    bytes: float,
    dtype: str = "bf16",
    model: Optional[PhysicalModel] = None,
) -> Optional[dict[str, Any]]:
    canonical = canonical_dtype(dtype)
    if canonical is None:
        return None
    work = WorkEstimate("compat-explicit-work", canonical, flops, bytes, "exact")
    result = evaluate_roofline(work, model or make_physical_model("mi350x"))
    return result.as_dict() if result else None


def attained_fraction(
    measured_ms: float,
    flops: float,
    bytes: float,
    dtype: str = "bf16",
    model: Optional[PhysicalModel] = None,
) -> Optional[float]:
    result_dict = roofline(flops, bytes, dtype, model)
    if result_dict is None or not _finite(measured_ms) or float(measured_ms) <= 0.0:
        return None
    return 100.0 * result_dict["t_min_ms"] / float(measured_ms)


def attained_metrics(
    measured_ms: float,
    flops: float,
    bytes: float,
    dtype: str = "bf16",
    model: Optional[PhysicalModel] = None,
) -> Optional[dict[str, float]]:
    result = roofline(flops, bytes, dtype, model)
    if result is None or not _finite(measured_ms) or float(measured_ms) <= 0.0:
        return None
    seconds = float(measured_ms) / 1e3
    achieved_flops = float(flops) / seconds
    achieved_bw = float(bytes) / seconds
    return {
        "pct_of_roofline": 100.0 * result["t_min_ms"] / float(measured_ms),
        "achieved_flops": achieved_flops,
        "achieved_bw": achieved_bw,
        "pct_of_peak_flops": 100.0 * achieved_flops / result["peak_flops"],
        "pct_of_peak_bw": 100.0 * achieved_bw / result["peak_bandwidth_bytes_per_s"],
    }


def op_flop_bytes(
    op_family: str,
    shape: ShapeLike,
    dtype: str = "bf16",
    n_tensors: int = 2,
    flops_per_elem: float = 1.0,
) -> Optional[tuple[float, float]]:
    work = estimate_work(
        op_family,
        shape,
        dtype,
        n_tensors=n_tensors,
        flops_per_element=flops_per_elem,
    )
    return (work.flops, work.bytes) if work else None


def detect_runtime_device() -> Optional[dict[str, str]]:
    """Best-effort device identity.  Never runs at import and never selects a model."""
    for command in (["rocminfo"], ["rocm-smi", "--showproductname"]):
        executable = shutil.which(command[0])
        if not executable:
            continue
        try:
            output = subprocess.run(
                command, capture_output=True, text=True, timeout=20, check=False
            ).stdout.lower()
        except (OSError, subprocess.SubprocessError):
            continue
        sku = "mi355x" if "mi355" in output else ("mi350x" if "mi350" in output else None)
        if sku is None and ("mi300" in output or "gfx942" in output):
            sku = "mi300x"
        arch = "gfx950" if "gfx950" in output else ("gfx942" if "gfx942" in output else None)
        if sku or arch:
            return {"sku": sku or "", "architecture": arch or hardware_spec(sku).architecture}
    return None


def _legacy_bundle(sku: str) -> dict[str, Any]:
    spec = hardware_spec(sku)
    out = {
        "name": spec.sku,
        "arch": spec.architecture,
        "hbm_bw_bytes_per_s": spec.hbm_bytes_per_s,
        "num_cus": spec.num_cus,
        "peak_clock_hz": spec.peak_clock_hz,
        "infinity_cache_bytes": spec.infinity_cache_bytes,
        "lds_bytes_per_cu": spec.lds_bytes_per_cu,
        "vgpr_per_simd": spec.vgpr_per_simd,
        "max_waves_per_simd": spec.max_waves_per_simd,
    }
    out.update({f"peak_flops_{dtype}": value for dtype, value in spec.compute_flops_per_s.items()})
    return out


MI300X = _legacy_bundle("mi300x")
MI350X = _legacy_bundle("mi350x")
MI355X = _legacy_bundle("mi355x")


__all__ = [
    "BOTTLENECK_GROUNDING_TERMS",
    "CounterUnit",
    "HardwareSpec",
    "MI300X",
    "MI350X",
    "MI355X",
    "ModelError",
    "OccupancyEstimate",
    "PhysicalModel",
    "RooflineResult",
    "ShapeLike",
    "WorkEstimate",
    "attained_fraction",
    "attained_metrics",
    "attainment",
    "available_skus",
    "bottleneck_from_counters",
    "canonical_dtype",
    "canonicalize_label",
    "counter_unit",
    "counter_value",
    "derived_percent",
    "detect_runtime_device",
    "dtype_bytes",
    "est_occupancy",
    "estimate_work",
    "evaluate_roofline",
    "hardware_spec",
    "hbm_bytes",
    "issued_instructions",
    "l2_hit_rate",
    "make_physical_model",
    "mfma_flops",
    "mfma_instruction_count",
    "model_from_peak_mapping",
    "op_flop_bytes",
    "peak_flops",
    "roofline",
]
