"""Fail-closed, model-load-free resource preflight utilities.

The preflight consumes a resolved :class:`~kore.policy.model_spec.ModelSpec`, so
all memory arithmetic is based on exact safetensors metadata rather than a model
name. Analytical values are explicitly lower bounds. A positive fit assertion
requires an environment-bound measured peak profile.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from kore.policy.model_spec import UNRESOLVED, ModelSpec, canonical_profile_hash


MEASURE = UNRESOLVED


class ResourcePreflightError(RuntimeError):
    """Base class for resource preflight failures."""


class UnresolvedProductionFieldError(ResourcePreflightError):
    """Raised when a production preflight still contains ``MEASURE`` fields."""


class InsufficientResourcesError(ResourcePreflightError):
    """Raised when an exact lower bound or measured peak cannot fit."""


PreflightError = ResourcePreflightError
UnresolvedMeasurementError = UnresolvedProductionFieldError


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _unresolved_paths(value: Any, prefix: str = "") -> list[str]:
    if is_dataclass(value):
        value = asdict(value)
    if value is None or value == MEASURE or value == "":
        return [prefix or "<root>"]
    if isinstance(value, Mapping):
        paths: list[str] = []
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_unresolved_paths(item, child))
        return paths
    if isinstance(value, (tuple, list)):
        paths = []
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            paths.extend(_unresolved_paths(item, child))
        return paths
    return []


def reject_unresolved_production_fields(value: Any, *, context: str) -> None:
    unresolved = _unresolved_paths(value)
    if unresolved:
        raise UnresolvedProductionFieldError(
            f"{context} has unresolved production fields: "
            + ", ".join(unresolved[:20])
        )


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResourcePreflightError(
            f"{field_name} must be a non-negative integer, got {value!r}"
        )
    return value


@dataclass(frozen=True)
class GPUDevice:
    index: int
    name: str
    uuid: str
    pci_bus_id: str
    numa_node: int | str
    total_hbm_bytes: int | str
    free_hbm_bytes: int | str
    visible: bool = True

    @property
    def identity(self) -> str:
        return self.uuid if self.uuid != MEASURE else self.pci_bus_id


@dataclass(frozen=True)
class FilesystemCapacity:
    role: str
    path: str
    device_id: int | str
    total_bytes: int | str
    free_bytes: int | str


_REQUIRED_SOFTWARE = (
    "python",
    "kernel",
    "rocm",
    "amdgpu",
    "torch",
    "transformers",
    "safetensors",
    "accelerate",
    "trl",
    "peft",
)


@dataclass(frozen=True)
class ResourceSnapshot:
    """Exact capacity/free-space inventory for one preflight instant."""

    gpus: tuple[GPUDevice, ...]
    gpu_topology: Mapping[str, Any] | str
    visible_devices: str
    host_ram_total_bytes: int | str
    host_ram_available_bytes: int | str
    filesystems: tuple[FilesystemCapacity, ...]
    software_versions: Mapping[str, str]
    source: str = "local-probe"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResourceSnapshot":
        try:
            gpus = tuple(
                sorted(
                    (GPUDevice(**dict(item)) for item in payload.get("gpus", [])),
                    key=lambda gpu: gpu.index,
                )
            )
            filesystems = tuple(
                sorted(
                    (
                        FilesystemCapacity(**dict(item))
                        for item in payload.get("filesystems", [])
                    ),
                    key=lambda filesystem: filesystem.role,
                )
            )
            return cls(
                gpus=gpus,
                gpu_topology=payload.get("gpu_topology", MEASURE),
                visible_devices=str(payload.get("visible_devices", MEASURE)),
                host_ram_total_bytes=payload.get("host_ram_total_bytes", MEASURE),
                host_ram_available_bytes=payload.get(
                    "host_ram_available_bytes", MEASURE
                ),
                filesystems=filesystems,
                software_versions=dict(payload.get("software_versions", {})),
                source=str(payload.get("source", "profile")),
            )
        except (TypeError, ValueError) as exc:
            raise ResourcePreflightError(
                f"invalid resource profile payload: {exc}"
            ) from exc

    def unresolved_fields(self) -> list[str]:
        unresolved = _unresolved_paths(
            {
                "gpus": self.gpus,
                "gpu_topology": self.gpu_topology,
                "visible_devices": self.visible_devices,
                "host_ram_total_bytes": self.host_ram_total_bytes,
                "host_ram_available_bytes": self.host_ram_available_bytes,
                "filesystems": self.filesystems,
                "software_versions": {
                    key: self.software_versions.get(key, MEASURE)
                    for key in _REQUIRED_SOFTWARE
                },
            }
        )
        if not self.gpus:
            unresolved.append("gpus")
        if not isinstance(self.gpu_topology, Mapping) or not self.gpu_topology:
            unresolved.append("gpu_topology")
        roles = {filesystem.role for filesystem in self.filesystems}
        for role in ("model", "scratch"):
            if role not in roles:
                unresolved.append(f"filesystems.{role}")
        return sorted(set(unresolved))

    def validate_resolved(self) -> None:
        unresolved = self.unresolved_fields()
        if unresolved:
            raise UnresolvedProductionFieldError(
                "resource snapshot has unresolved production fields: "
                + ", ".join(unresolved[:20])
            )
        if len({gpu.index for gpu in self.gpus}) != len(self.gpus):
            raise ResourcePreflightError("GPU indices must be unique")
        if len({gpu.identity for gpu in self.gpus}) != len(self.gpus):
            raise ResourcePreflightError("GPU identities must be unique")
        for gpu in self.gpus:
            total = _non_negative_int(gpu.total_hbm_bytes, "gpu.total_hbm_bytes")
            free = _non_negative_int(gpu.free_hbm_bytes, "gpu.free_hbm_bytes")
            if free > total:
                raise ResourcePreflightError(
                    f"GPU {gpu.index} free HBM exceeds total HBM"
                )
        roles = [filesystem.role for filesystem in self.filesystems]
        if len(set(roles)) != len(roles):
            raise ResourcePreflightError("filesystem roles must be unique")
        total_ram = _non_negative_int(
            self.host_ram_total_bytes, "host_ram_total_bytes"
        )
        available_ram = _non_negative_int(
            self.host_ram_available_bytes, "host_ram_available_bytes"
        )
        if available_ram > total_ram:
            raise ResourcePreflightError(
                "host_ram_available_bytes exceeds host_ram_total_bytes"
            )
        for filesystem in self.filesystems:
            fs_total = _non_negative_int(
                filesystem.total_bytes, f"filesystems.{filesystem.role}.total_bytes"
            )
            fs_free = _non_negative_int(
                filesystem.free_bytes, f"filesystems.{filesystem.role}.free_bytes"
            )
            if fs_free > fs_total:
                raise ResourcePreflightError(
                    f"filesystem {filesystem.role!r} free space exceeds total space"
                )

    @property
    def profile_hash(self) -> str:
        return canonical_profile_hash(
            {"schema_version": 1, "kind": "resource-snapshot", **_jsonable(self)}
        )

    @property
    def environment_hash(self) -> str:
        """Stable measured-profile binding, excluding momentary free capacity."""

        return canonical_profile_hash(
            {
                "schema_version": 1,
                "kind": "resource-environment",
                "gpus": [
                    {
                        "index": gpu.index,
                        "name": gpu.name,
                        "uuid": gpu.uuid,
                        "pci_bus_id": gpu.pci_bus_id,
                        "numa_node": gpu.numa_node,
                        "total_hbm_bytes": gpu.total_hbm_bytes,
                        "visible": gpu.visible,
                    }
                    for gpu in self.gpus
                ],
                "gpu_topology": self.gpu_topology,
                "visible_devices": self.visible_devices,
                "host_ram_total_bytes": self.host_ram_total_bytes,
                "filesystems": [
                    {
                        "role": filesystem.role,
                        "device_id": filesystem.device_id,
                        "total_bytes": filesystem.total_bytes,
                    }
                    for filesystem in self.filesystems
                ],
                "software_versions": self.software_versions,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **_jsonable(self),
            "profile_hash": self.profile_hash,
            "environment_hash": self.environment_hash,
            "unresolved_fields": self.unresolved_fields(),
        }


@dataclass(frozen=True)
class AnalyticalLowerBounds:
    """Persistent-state arithmetic only; never a peak-memory fit claim."""

    label: str
    exact_parameter_count: int
    checkpoint_tensor_bytes: int
    bf16_weights_bytes: int
    bf16_gradients_bytes: int
    fp32_master_weights_bytes: int
    fp32_adam_moments_bytes: int
    full_finetune_persistent_state_bytes: int
    assumptions: tuple[str, ...]
    exclusions: tuple[str, ...]

    @property
    def profile_hash(self) -> str:
        return canonical_profile_hash(
            {"schema_version": 1, "kind": "analytical-lower-bounds", **asdict(self)}
        )


def compute_analytical_lower_bounds(model_spec: ModelSpec) -> AnalyticalLowerBounds:
    """Compute exact-count lower bounds without loading any model tensors."""

    parameters = model_spec.parameter_count
    bf16_weights = parameters * 2
    bf16_gradients = parameters * 2
    fp32_master = parameters * 4
    adam_moments = parameters * 8
    return AnalyticalLowerBounds(
        label=(
            "ANALYTICAL LOWER BOUNDS ONLY — excludes runtime peaks and does not "
            "assert that the workload fits"
        ),
        exact_parameter_count=parameters,
        checkpoint_tensor_bytes=model_spec.checkpoint.tensor_storage_bytes,
        bf16_weights_bytes=bf16_weights,
        bf16_gradients_bytes=bf16_gradients,
        fp32_master_weights_bytes=fp32_master,
        fp32_adam_moments_bytes=adam_moments,
        full_finetune_persistent_state_bytes=(
            bf16_weights + bf16_gradients + fp32_master + adam_moments
        ),
        assumptions=(
            "BF16 model weights and gradients",
            "FP32 master weights",
            "two FP32 Adam-family optimizer moments",
            "no parameter replication beyond one logical copy",
        ),
        exclusions=(
            "activations and temporary buffers",
            "attention and generation KV caches",
            "allocator fragmentation and communication workspaces",
            "framework, kernels, and graph memory",
            "checkpoint save/load transients",
        ),
    )


analytical_lower_bounds = compute_analytical_lower_bounds


@dataclass(frozen=True)
class MeasuredPeakProfile:
    """Measured workload peaks bound to one model and software/topology profile."""

    model_profile_hash: str
    environment_hash: str
    workload: str
    gpu_peak_hbm_bytes: tuple[int | str, ...]
    host_peak_ram_bytes: int | str
    filesystem_peak_bytes: Mapping[str, int | str]
    source: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MeasuredPeakProfile":
        try:
            raw_gpu = payload.get("gpu_peak_hbm_bytes", ())
            if not isinstance(raw_gpu, (list, tuple)):
                raise TypeError("gpu_peak_hbm_bytes must be a list")
            raw_filesystem = payload.get("filesystem_peak_bytes", {})
            if not isinstance(raw_filesystem, Mapping):
                raise TypeError("filesystem_peak_bytes must be an object")
            return cls(
                model_profile_hash=str(
                    payload.get("model_profile_hash", MEASURE)
                ),
                environment_hash=str(payload.get("environment_hash", MEASURE)),
                workload=str(payload.get("workload", MEASURE)),
                gpu_peak_hbm_bytes=tuple(raw_gpu),
                host_peak_ram_bytes=payload.get("host_peak_ram_bytes", MEASURE),
                filesystem_peak_bytes=dict(raw_filesystem),
                source=str(payload.get("source", MEASURE)),
            )
        except (TypeError, ValueError) as exc:
            raise ResourcePreflightError(
                f"invalid measured peak profile: {exc}"
            ) from exc

    @classmethod
    def from_json(cls, path: str | Path) -> "MeasuredPeakProfile":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ResourcePreflightError(
                f"cannot read measured profile {path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ResourcePreflightError(
                f"measured profile {path} must contain a JSON object"
            )
        return cls.from_dict(payload)

    def unresolved_fields(
        self, *, filesystem_roles: tuple[str, ...] = ("model", "scratch")
    ) -> list[str]:
        payload = {
            "model_profile_hash": self.model_profile_hash,
            "environment_hash": self.environment_hash,
            "workload": self.workload,
            "gpu_peak_hbm_bytes": self.gpu_peak_hbm_bytes,
            "host_peak_ram_bytes": self.host_peak_ram_bytes,
            "filesystem_peak_bytes": {
                role: self.filesystem_peak_bytes.get(role, MEASURE)
                for role in filesystem_roles
            },
            "source": self.source,
        }
        return sorted(set(_unresolved_paths(payload)))

    def validate_resolved(
        self, *, filesystem_roles: tuple[str, ...] = ("model", "scratch")
    ) -> None:
        unresolved = self.unresolved_fields(filesystem_roles=filesystem_roles)
        if unresolved:
            raise UnresolvedProductionFieldError(
                "measured peak profile has unresolved production fields: "
                + ", ".join(unresolved[:20])
            )
        for index, peak in enumerate(self.gpu_peak_hbm_bytes):
            _non_negative_int(peak, f"gpu_peak_hbm_bytes[{index}]")
        _non_negative_int(self.host_peak_ram_bytes, "host_peak_ram_bytes")
        for role in filesystem_roles:
            _non_negative_int(
                self.filesystem_peak_bytes[role],
                f"filesystem_peak_bytes.{role}",
            )

    @property
    def profile_hash(self) -> str:
        return canonical_profile_hash(
            {"schema_version": 1, "kind": "measured-peak-profile", **_jsonable(self)}
        )

    def to_dict(self) -> dict[str, Any]:
        return {**_jsonable(self), "profile_hash": self.profile_hash}


@dataclass(frozen=True)
class PreflightReport:
    """Auditable preflight result with an explicit fit-assertion state."""

    model_spec: ModelSpec
    resources: ResourceSnapshot
    analytical_lower_bounds: AnalyticalLowerBounds
    measured_profile: Optional[MeasuredPeakProfile]
    status: str
    fit_asserted: bool
    reasons: tuple[str, ...]
    headroom_fraction: float

    @property
    def production_ready(self) -> bool:
        return self.status == "measured_pass" and self.fit_asserted

    @property
    def profile_hash(self) -> str:
        return canonical_profile_hash(
            {
                "schema_version": 1,
                "kind": "resource-preflight-report",
                "model_profile_hash": self.model_spec.profile_hash,
                "resource_profile_hash": self.resources.profile_hash,
                "analytical_profile_hash": self.analytical_lower_bounds.profile_hash,
                "measured_profile_hash": (
                    self.measured_profile.profile_hash
                    if self.measured_profile is not None
                    else None
                ),
                "status": self.status,
                "fit_asserted": self.fit_asserted,
                "reasons": self.reasons,
                "headroom_fraction": self.headroom_fraction,
            }
        )

    def assert_production_ready(self) -> None:
        if not self.production_ready:
            raise ResourcePreflightError(
                "resource fit is not established by a matching measured peak "
                f"profile (status={self.status!r}): " + "; ".join(self.reasons)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model_spec": self.model_spec.to_dict(),
            "resources": self.resources.to_dict(),
            "analytical_lower_bounds": {
                **asdict(self.analytical_lower_bounds),
                "profile_hash": self.analytical_lower_bounds.profile_hash,
            },
            "measured_profile": (
                self.measured_profile.to_dict()
                if self.measured_profile is not None
                else None
            ),
            "status": self.status,
            "fit_asserted": self.fit_asserted,
            "production_ready": self.production_ready,
            "reasons": list(self.reasons),
            "headroom_fraction": self.headroom_fraction,
            "profile_hash": self.profile_hash,
        }


def _required_with_headroom(value: int, headroom_fraction: float) -> int:
    return int(value * (1.0 + headroom_fraction) + 0.999999)


def evaluate_resource_preflight(
    model_spec: ModelSpec,
    resources: ResourceSnapshot,
    measured_profile: Optional[MeasuredPeakProfile] = None,
    *,
    headroom_fraction: float = 0.10,
) -> PreflightReport:
    """Evaluate resources without turning an incomplete result into a pass."""

    if not 0.0 <= headroom_fraction < 1.0:
        raise ResourcePreflightError(
            "headroom_fraction must be in the half-open interval [0, 1)"
        )
    bounds = compute_analytical_lower_bounds(model_spec)
    unresolved_resources = resources.unresolved_fields()
    if unresolved_resources:
        return PreflightReport(
            model_spec=model_spec,
            resources=resources,
            analytical_lower_bounds=bounds,
            measured_profile=measured_profile,
            status="unresolved",
            fit_asserted=False,
            reasons=(
                "resource snapshot contains MEASURE fields: "
                + ", ".join(unresolved_resources[:20]),
            ),
            headroom_fraction=headroom_fraction,
        )

    resources.validate_resolved()
    visible_gpus = tuple(gpu for gpu in resources.gpus if gpu.visible)
    reasons: list[str] = []
    if not visible_gpus:
        reasons.append("no visible GPUs were recorded")
    else:
        # This is only a necessary condition. It cannot establish fit because it
        # ignores placement, runtime buffers, and topology.
        aggregate_free = sum(int(gpu.free_hbm_bytes) for gpu in visible_gpus)
        weights_lower_bound = max(
            bounds.checkpoint_tensor_bytes, bounds.bf16_weights_bytes
        )
        if aggregate_free < weights_lower_bound:
            reasons.append(
                "aggregate free HBM is below the exact weights-only analytical "
                f"lower bound ({aggregate_free} < {weights_lower_bound})"
            )

    if reasons:
        return PreflightReport(
            model_spec=model_spec,
            resources=resources,
            analytical_lower_bounds=bounds,
            measured_profile=measured_profile,
            status="insufficient",
            fit_asserted=False,
            reasons=tuple(reasons),
            headroom_fraction=headroom_fraction,
        )

    if measured_profile is None:
        return PreflightReport(
            model_spec=model_spec,
            resources=resources,
            analytical_lower_bounds=bounds,
            measured_profile=None,
            status="analytical_only",
            fit_asserted=False,
            reasons=(
                "analytical lower bounds are necessary conditions only; ingest a "
                "matching measured peak profile to assert fit",
            ),
            headroom_fraction=headroom_fraction,
        )

    filesystem_roles = tuple(
        sorted(filesystem.role for filesystem in resources.filesystems)
    )
    unresolved_measured = measured_profile.unresolved_fields(
        filesystem_roles=filesystem_roles
    )
    if unresolved_measured:
        return PreflightReport(
            model_spec=model_spec,
            resources=resources,
            analytical_lower_bounds=bounds,
            measured_profile=measured_profile,
            status="unresolved",
            fit_asserted=False,
            reasons=(
                "measured profile contains MEASURE fields: "
                + ", ".join(unresolved_measured[:20]),
            ),
            headroom_fraction=headroom_fraction,
        )
    measured_profile.validate_resolved(filesystem_roles=filesystem_roles)

    if measured_profile.model_profile_hash != model_spec.profile_hash:
        reasons.append("measured profile model hash does not match this checkpoint")
    if measured_profile.environment_hash != resources.environment_hash:
        reasons.append(
            "measured profile environment hash does not match GPU topology/software"
        )
    if len(measured_profile.gpu_peak_hbm_bytes) != len(visible_gpus):
        reasons.append(
            "measured GPU peak count does not match the visible GPU count "
            f"({len(measured_profile.gpu_peak_hbm_bytes)} != {len(visible_gpus)})"
        )
    else:
        measured_hbm = sum(
            int(peak) for peak in measured_profile.gpu_peak_hbm_bytes
        )
        weights_lower_bound = max(
            bounds.checkpoint_tensor_bytes, bounds.bf16_weights_bytes
        )
        if measured_hbm < weights_lower_bound:
            reasons.append(
                "measured aggregate HBM peak is below the exact weights-only "
                f"lower bound ({measured_hbm} < {weights_lower_bound})"
            )
        for gpu, peak_value in zip(
            visible_gpus, measured_profile.gpu_peak_hbm_bytes
        ):
            peak = int(peak_value)
            required = _required_with_headroom(peak, headroom_fraction)
            if required > int(gpu.free_hbm_bytes):
                reasons.append(
                    f"GPU {gpu.index} free HBM is insufficient for measured peak "
                    f"plus headroom ({gpu.free_hbm_bytes} < {required})"
                )

    required_host = _required_with_headroom(
        int(measured_profile.host_peak_ram_bytes), headroom_fraction
    )
    if required_host > int(resources.host_ram_available_bytes):
        reasons.append(
            "available host RAM is insufficient for measured peak plus headroom "
            f"({resources.host_ram_available_bytes} < {required_host})"
        )

    filesystems = {filesystem.role: filesystem for filesystem in resources.filesystems}
    for role in filesystem_roles:
        required = _required_with_headroom(
            int(measured_profile.filesystem_peak_bytes[role]),
            headroom_fraction,
        )
        free = int(filesystems[role].free_bytes)
        if required > free:
            reasons.append(
                f"{role} filesystem free capacity is insufficient for measured "
                f"peak plus headroom ({free} < {required})"
            )

    status = "insufficient" if reasons else "measured_pass"
    return PreflightReport(
        model_spec=model_spec,
        resources=resources,
        analytical_lower_bounds=bounds,
        measured_profile=measured_profile,
        status=status,
        fit_asserted=not reasons,
        reasons=tuple(reasons) if reasons else ("matching measured peaks fit with headroom",),
        headroom_fraction=headroom_fraction,
    )


def run_resource_preflight(
    model_spec: ModelSpec,
    resources: ResourceSnapshot,
    measured_profile: Optional[MeasuredPeakProfile] = None,
    *,
    production: bool = True,
    require_measured: bool = False,
    headroom_fraction: float = 0.10,
) -> PreflightReport:
    """Run preflight and reject unresolved/insufficient production inputs."""

    report = evaluate_resource_preflight(
        model_spec,
        resources,
        measured_profile,
        headroom_fraction=headroom_fraction,
    )
    if production and report.status == "unresolved":
        raise UnresolvedProductionFieldError("; ".join(report.reasons))
    if report.status == "insufficient":
        raise InsufficientResourcesError("; ".join(report.reasons))
    if require_measured and not report.production_ready:
        raise UnresolvedProductionFieldError(
            "production fit requires a resolved, matching measured peak profile"
        )
    return report


preflight_resources = run_resource_preflight


def _read_text(path: Path) -> Optional[str]:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return value or None


def _read_host_ram() -> tuple[int | str, int | str]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            fields = raw.strip().split()
            if fields:
                values[key] = int(fields[0]) * 1024
    except (OSError, ValueError):
        return MEASURE, MEASURE
    return values.get("MemTotal", MEASURE), values.get("MemAvailable", MEASURE)


def _existing_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _filesystem_capacity(path: str | Path, role: str) -> FilesystemCapacity:
    existing = _existing_path(path)
    try:
        usage = shutil.disk_usage(existing)
        device_id: int | str = existing.stat().st_dev
        return FilesystemCapacity(
            role=role,
            path=str(existing),
            device_id=device_id,
            total_bytes=usage.total,
            free_bytes=usage.free,
        )
    except OSError:
        return FilesystemCapacity(
            role=role,
            path=str(existing),
            device_id=MEASURE,
            total_bytes=MEASURE,
            free_bytes=MEASURE,
        )


def collect_software_versions() -> dict[str, str]:
    """Collect package versions without importing GPU/framework packages."""

    versions = {
        "python": platform.python_version(),
        "kernel": platform.release(),
    }
    rocm = None
    for path in (
        Path("/opt/rocm/.info/version"),
        Path("/opt/rocm/.info/version-dev"),
        Path("/opt/rocm/.info/version-utils"),
    ):
        rocm = _read_text(path)
        if rocm:
            break
    versions["rocm"] = rocm or MEASURE
    module_version = _read_text(Path("/sys/module/amdgpu/version"))
    versions["amdgpu"] = module_version or f"kernel-module@{platform.release()}"
    for package in ("torch", "transformers", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = MEASURE
    for package in ("accelerate", "trl", "peft"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = MEASURE
    return versions


def _visible_device_filter() -> str:
    for key in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        value = os.environ.get(key)
        if value is not None:
            return f"{key}={value}"
    return "all"


def _visible_indices(filter_value: str) -> Optional[set[int]]:
    if filter_value == "all":
        return None
    _, _, raw = filter_value.partition("=")
    pieces = [piece.strip() for piece in raw.split(",") if piece.strip()]
    if not pieces:
        return set()
    if all(piece.isdigit() for piece in pieces):
        return {int(piece) for piece in pieces}
    # UUID filtering cannot be mapped safely until UUIDs are collected.
    return None


def _probe_rocm_topology() -> Mapping[str, Any] | str:
    command = shutil.which("rocm-smi")
    if not command:
        return MEASURE
    try:
        result = subprocess.run(
            [command, "--showtopo", "--json"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        topology = json.loads(result.stdout)
    except (
        OSError,
        subprocess.SubprocessError,
        json.JSONDecodeError,
    ):
        return MEASURE
    return topology if isinstance(topology, Mapping) and topology else MEASURE


def _parse_uevent(path: Path) -> dict[str, str]:
    text = _read_text(path)
    if not text:
        return {}
    values: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def collect_amd_gpu_devices(
    sysfs_root: str | Path = "/sys/class/drm",
) -> tuple[GPUDevice, ...]:
    """Read AMD HBM capacity/free values directly from sysfs."""

    root = Path(sysfs_root)
    visible_filter = _visible_device_filter()
    visible_indices = _visible_indices(visible_filter)
    devices: list[GPUDevice] = []
    for card in sorted(
        root.glob("card[0-9]*"),
        key=lambda path: int(path.name.removeprefix("card")),
    ):
        device = card / "device"
        vendor = _read_text(device / "vendor")
        if vendor is not None and vendor.lower() != "0x1002":
            continue
        total_raw = _read_text(device / "mem_info_vram_total")
        used_raw = _read_text(device / "mem_info_vram_used")
        if total_raw is None:
            continue
        try:
            total: int | str = int(total_raw)
            used = int(used_raw) if used_raw is not None else None
            free: int | str = max(0, total - used) if used is not None else MEASURE
        except ValueError:
            total, free = MEASURE, MEASURE
        index = len(devices)
        try:
            pci_bus_id = device.resolve().name
        except OSError:
            pci_bus_id = MEASURE
        uevent = _parse_uevent(device / "uevent")
        name = (
            _read_text(device / "product_name")
            or uevent.get("PCI_ID")
            or MEASURE
        )
        uuid = _read_text(device / "unique_id") or MEASURE
        numa_raw = _read_text(device / "numa_node")
        try:
            numa_node: int | str = int(numa_raw) if numa_raw is not None else MEASURE
        except ValueError:
            numa_node = MEASURE
        devices.append(
            GPUDevice(
                index=index,
                name=name,
                uuid=uuid,
                pci_bus_id=pci_bus_id,
                numa_node=numa_node,
                total_hbm_bytes=total,
                free_hbm_bytes=free,
                visible=visible_indices is None or index in visible_indices,
            )
        )
    return tuple(devices)


def collect_resource_snapshot(
    model_path: str | Path,
    scratch_path: str | Path,
    *,
    sysfs_root: str | Path = "/sys/class/drm",
) -> ResourceSnapshot:
    """Collect capacity, free-space, topology, and software without GPU init."""

    host_total, host_available = _read_host_ram()
    return ResourceSnapshot(
        gpus=collect_amd_gpu_devices(sysfs_root),
        gpu_topology=_probe_rocm_topology(),
        visible_devices=_visible_device_filter(),
        host_ram_total_bytes=host_total,
        host_ram_available_bytes=host_available,
        filesystems=tuple(
            sorted(
                (
                    _filesystem_capacity(model_path, "model"),
                    _filesystem_capacity(scratch_path, "scratch"),
                ),
                key=lambda filesystem: filesystem.role,
            )
        ),
        software_versions=collect_software_versions(),
    )


def load_resource_snapshot(path: str | Path) -> ResourceSnapshot:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResourcePreflightError(
            f"cannot read resource profile {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ResourcePreflightError(
            f"resource profile {path} must contain a JSON object"
        )
    return ResourceSnapshot.from_dict(payload)


__all__ = [
    "MEASURE",
    "AnalyticalLowerBounds",
    "FilesystemCapacity",
    "GPUDevice",
    "InsufficientResourcesError",
    "MeasuredPeakProfile",
    "PreflightError",
    "PreflightReport",
    "ResourcePreflightError",
    "ResourceSnapshot",
    "UnresolvedMeasurementError",
    "UnresolvedProductionFieldError",
    "analytical_lower_bounds",
    "collect_amd_gpu_devices",
    "collect_resource_snapshot",
    "collect_software_versions",
    "compute_analytical_lower_bounds",
    "evaluate_resource_preflight",
    "load_resource_snapshot",
    "preflight_resources",
    "reject_unresolved_production_fields",
    "run_resource_preflight",
]
