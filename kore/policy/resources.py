"""Fail-closed, model-load-free resource preflight utilities.

The preflight consumes a resolved :class:`~kore.policy.model_spec.ModelSpec`, so
all memory arithmetic is based on exact safetensors metadata rather than a model
name. Analytical values are explicitly lower bounds. A positive fit assertion
requires an environment-bound measured peak profile.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from kore.policy.model_spec import UNRESOLVED, ModelSpec, canonical_profile_hash


MEASURE = UNRESOLVED
ABSENT = "ABSENT"
NOT_APPLICABLE = "NOT_APPLICABLE"


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
    """Cross-source identity for one physical GPU.

    DRM/sysfs and HIP each report the BDF/UUID independently. Keeping both
    prevents an ordinal from being silently paired with the wrong card.
    """

    drm_card: str
    render_node: str
    pci_bdf: str
    hip_reported_pci_bdf: str
    uuid: str
    hip_reported_uuid: str
    physical_card: int | str
    hip_ordinal: int | str
    slurm_gres_id: str
    slurm_allocated: bool | str
    name: str
    numa_node: int | str
    total_hbm_bytes: int | str
    free_hbm_bytes: int | str
    visible: bool | str

    @property
    def identity(self) -> str:
        return self.uuid if self.uuid != MEASURE else self.pci_bdf


@dataclass(frozen=True)
class FilesystemCapacity:
    role: str
    path: str
    device_id: int | str
    total_bytes: int | str
    free_bytes: int | str


_SOFTWARE_COMPONENTS = (
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
    "datasets",
)
_CORE_SOFTWARE = ("python", "kernel", "rocm", "amdgpu")
_BDF_RE = re.compile(r"^(?:[0-9a-fA-F]{4}:)?[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$")
_FINGERPRINT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


def _normalize_bdf(value: str) -> str:
    raw = value.strip().lower()
    if not _BDF_RE.fullmatch(raw):
        raise ResourcePreflightError(f"invalid PCI BDF {value!r}")
    return raw if raw.count(":") == 2 else f"0000:{raw}"


def _normalize_uuid(value: str) -> str:
    return value.strip().lower().removeprefix("0x")


@dataclass(frozen=True)
class ResourceSnapshot:
    """Exact capacity/free-space inventory for one preflight instant."""

    gpus: tuple[GPUDevice, ...]
    gpu_topology: Mapping[str, Any] | str
    visible_device_policy: Mapping[str, Any] | str
    slurm_allocation: Mapping[str, Any] | str
    host_ram_total_bytes: int | str
    host_ram_available_bytes: int | str
    filesystems: tuple[FilesystemCapacity, ...]
    software_versions: Mapping[str, str]
    code_fingerprint: str
    source: str = "local-probe"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResourceSnapshot":
        try:
            gpus = tuple(
                sorted(
                    (GPUDevice(**dict(item)) for item in payload.get("gpus", [])),
                    key=lambda gpu: str(gpu.drm_card),
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
                visible_device_policy=payload.get(
                    "visible_device_policy", MEASURE
                ),
                slurm_allocation=payload.get("slurm_allocation", MEASURE),
                host_ram_total_bytes=payload.get("host_ram_total_bytes", MEASURE),
                host_ram_available_bytes=payload.get(
                    "host_ram_available_bytes", MEASURE
                ),
                filesystems=filesystems,
                software_versions=dict(payload.get("software_versions", {})),
                code_fingerprint=str(
                    payload.get("code_fingerprint", MEASURE)
                ),
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
                "visible_device_policy": self.visible_device_policy,
                "slurm_allocation": self.slurm_allocation,
                "host_ram_total_bytes": self.host_ram_total_bytes,
                "host_ram_available_bytes": self.host_ram_available_bytes,
                "filesystems": self.filesystems,
                "software_versions": {
                    key: self.software_versions.get(key, MEASURE)
                    for key in _SOFTWARE_COMPONENTS
                },
                "code_fingerprint": self.code_fingerprint,
            }
        )
        if not self.gpus:
            unresolved.append("gpus")
        if not isinstance(self.gpu_topology, Mapping) or not self.gpu_topology:
            unresolved.append("gpu_topology")
        if (
            not isinstance(self.visible_device_policy, Mapping)
            or not self.visible_device_policy
        ):
            unresolved.append("visible_device_policy")
        if (
            not isinstance(self.slurm_allocation, Mapping)
            or not self.slurm_allocation
        ):
            unresolved.append("slurm_allocation")
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
        unique_fields = (
            "drm_card",
            "render_node",
            "pci_bdf",
            "uuid",
            "physical_card",
            "hip_ordinal",
        )
        for field_name in unique_fields:
            values = [getattr(gpu, field_name) for gpu in self.gpus]
            if len(set(values)) != len(values):
                raise ResourcePreflightError(
                    f"GPU {field_name} values must be unique"
                )
        if len({gpu.identity for gpu in self.gpus}) != len(self.gpus):
            raise ResourcePreflightError("GPU identities must be unique")
        for gpu in self.gpus:
            if not re.fullmatch(r"card[0-9]+", gpu.drm_card):
                raise ResourcePreflightError(
                    f"invalid DRM card node {gpu.drm_card!r}"
                )
            if not re.fullmatch(r"renderD[0-9]+", gpu.render_node):
                raise ResourcePreflightError(
                    f"invalid DRM render node {gpu.render_node!r}"
                )
            if _normalize_bdf(gpu.pci_bdf) != _normalize_bdf(
                gpu.hip_reported_pci_bdf
            ):
                raise ResourcePreflightError(
                    f"HIP ordinal {gpu.hip_ordinal} BDF does not match "
                    f"{gpu.drm_card}"
                )
            if _normalize_uuid(gpu.uuid) != _normalize_uuid(
                gpu.hip_reported_uuid
            ):
                raise ResourcePreflightError(
                    f"HIP ordinal {gpu.hip_ordinal} UUID does not match "
                    f"{gpu.drm_card}"
                )
            _non_negative_int(gpu.physical_card, "gpu.physical_card")
            _non_negative_int(gpu.hip_ordinal, "gpu.hip_ordinal")
            if not isinstance(gpu.visible, bool):
                raise ResourcePreflightError("gpu.visible must be boolean")
            total = _non_negative_int(gpu.total_hbm_bytes, "gpu.total_hbm_bytes")
            free = _non_negative_int(gpu.free_hbm_bytes, "gpu.free_hbm_bytes")
            if free > total:
                raise ResourcePreflightError(
                    f"GPU {gpu.hip_ordinal} free HBM exceeds total HBM"
                )
        authority = self.visible_device_policy.get("authority")
        raw_ordinals = self.visible_device_policy.get("hip_ordinals")
        if authority not in ("HIP_VISIBLE_DEVICES", "unmasked"):
            raise ResourcePreflightError(
                "visibility policy must use HIP_VISIBLE_DEVICES or be unmasked"
            )
        if not isinstance(raw_ordinals, (list, tuple)):
            raise ResourcePreflightError(
                "visible_device_policy.hip_ordinals must be a list"
            )
        policy_ordinals = tuple(
            _non_negative_int(value, "visible_device_policy.hip_ordinals")
            for value in raw_ordinals
        )
        if len(set(policy_ordinals)) != len(policy_ordinals):
            raise ResourcePreflightError("visibility policy has duplicate ordinals")
        policy_raw = self.visible_device_policy.get("raw")
        if authority == "HIP_VISIBLE_DEVICES":
            try:
                parsed_raw = _parse_ordinal_list(str(policy_raw))
            except ValueError as exc:
                raise ResourcePreflightError(
                    "visibility policy raw HIP mask is invalid"
                ) from exc
            if parsed_raw != policy_ordinals:
                raise ResourcePreflightError(
                    "visibility policy raw mask and ordinal mapping differ"
                )
        elif policy_raw != NOT_APPLICABLE:
            raise ResourcePreflightError(
                "unmasked visibility must be explicitly NOT_APPLICABLE"
            )
        visible_ordinals = tuple(
            sorted(int(gpu.hip_ordinal) for gpu in self.gpus if gpu.visible)
        )
        if tuple(sorted(policy_ordinals)) != visible_ordinals:
            raise ResourcePreflightError(
                "visibility policy ordinals do not match explicitly mapped GPUs"
            )

        slurm_mode = self.slurm_allocation.get("mode")
        if slurm_mode not in ("none", "slurm"):
            raise ResourcePreflightError("slurm_allocation.mode is invalid")
        if slurm_mode == "none":
            for gpu in self.gpus:
                if (
                    gpu.slurm_gres_id != NOT_APPLICABLE
                    or gpu.slurm_allocated != NOT_APPLICABLE
                ):
                    raise ResourcePreflightError(
                        "non-Slurm inventory must mark GRES fields not applicable"
                    )
        else:
            allocated_cards = self.slurm_allocation.get("physical_cards")
            allocated_ordinals = self.slurm_allocation.get("hip_ordinals")
            if not isinstance(allocated_cards, (list, tuple)) or not isinstance(
                allocated_ordinals, (list, tuple)
            ):
                raise ResourcePreflightError(
                    "Slurm allocation must resolve physical cards and HIP ordinals"
                )
            expected_cards = {
                int(gpu.physical_card) for gpu in self.gpus if gpu.visible
            }
            expected_ordinals = {
                int(gpu.hip_ordinal) for gpu in self.gpus if gpu.visible
            }
            if {int(value) for value in allocated_cards} != expected_cards:
                raise ResourcePreflightError(
                    "Slurm/GRES physical-card allocation does not match visible GPUs"
                )
            if {int(value) for value in allocated_ordinals} != expected_ordinals:
                raise ResourcePreflightError(
                    "Slurm/GRES HIP-ordinal allocation does not match visible GPUs"
                )
            for gpu in self.gpus:
                should_be_allocated = int(gpu.physical_card) in {
                    int(value) for value in allocated_cards
                }
                if gpu.slurm_allocated is not should_be_allocated:
                    raise ResourcePreflightError(
                        f"GPU {gpu.hip_ordinal} Slurm allocation flag is inconsistent"
                    )
                if should_be_allocated and gpu.slurm_gres_id == NOT_APPLICABLE:
                    raise ResourcePreflightError(
                        f"GPU {gpu.hip_ordinal} lacks an explicit GRES identity"
                    )
                if gpu.visible and not should_be_allocated:
                    raise ResourcePreflightError(
                        f"visible GPU {gpu.hip_ordinal} is outside Slurm allocation"
                    )
        if not _FINGERPRINT_RE.fullmatch(self.code_fingerprint):
            raise ResourcePreflightError(
                "code_fingerprint must be a clean 40/64-hex revision"
            )
        for component in _CORE_SOFTWARE:
            if self.software_versions.get(component) == ABSENT:
                raise ResourcePreflightError(
                    f"core software component {component!r} cannot be ABSENT"
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
            {"schema_version": 2, "kind": "resource-snapshot", **_jsonable(self)}
        )

    @property
    def environment_hash(self) -> str:
        """Stable measured-profile binding, excluding momentary free capacity."""

        return canonical_profile_hash(
            {
                "schema_version": 2,
                "kind": "resource-environment",
                "gpus": [
                    {
                        "drm_card": gpu.drm_card,
                        "render_node": gpu.render_node,
                        "name": gpu.name,
                        "uuid": gpu.uuid,
                        "pci_bdf": gpu.pci_bdf,
                        "physical_card": gpu.physical_card,
                        "hip_ordinal": gpu.hip_ordinal,
                        "slurm_gres_id": gpu.slurm_gres_id,
                        "slurm_allocated": gpu.slurm_allocated,
                        "numa_node": gpu.numa_node,
                        "total_hbm_bytes": gpu.total_hbm_bytes,
                        "visible": gpu.visible,
                    }
                    for gpu in self.gpus
                ],
                "gpu_topology": self.gpu_topology,
                "visible_device_policy": self.visible_device_policy,
                "slurm_allocation": self.slurm_allocation,
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
                "code_fingerprint": self.code_fingerprint,
            }
        )

    @property
    def topology_hash(self) -> str:
        return canonical_profile_hash(
            {
                "gpus": [
                    {
                        "drm_card": gpu.drm_card,
                        "render_node": gpu.render_node,
                        "pci_bdf": gpu.pci_bdf,
                        "uuid": gpu.uuid,
                        "physical_card": gpu.physical_card,
                        "hip_ordinal": gpu.hip_ordinal,
                    }
                    for gpu in self.gpus
                ],
                "gpu_topology": self.gpu_topology,
                "visibility": self.visible_device_policy,
                "slurm": self.slurm_allocation,
            }
        )

    @property
    def dependency_profile_hash(self) -> str:
        return canonical_profile_hash(self.software_versions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            **_jsonable(self),
            "profile_hash": self.profile_hash,
            "environment_hash": self.environment_hash,
            "topology_hash": self.topology_hash,
            "dependency_profile_hash": self.dependency_profile_hash,
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
class WorkloadSpec:
    """Fully resolved workload/config identity for reusable peak evidence."""

    stage: str
    global_batch_size: int
    microbatch_size: int
    gradient_accumulation_steps: int
    sequence_lengths: Mapping[str, int]
    precision: str
    sharding: str
    offload: str
    backend: str
    world_size: int
    topology_hash: str
    optimizer: str
    optimizer_initialized: bool
    model_copies: int
    reference_copies: int
    rollout_copies: int
    resolved_config: Mapping[str, Any]
    code_fingerprint: str
    dependency_profile_hash: str
    model_profile_hash: str
    required_dependencies: tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WorkloadSpec":
        if not isinstance(payload, Mapping):
            raise ResourcePreflightError(
                "workload must be a resolved object, not a free-form string"
            )
        sequence_lengths = payload.get("sequence_lengths", {})
        resolved_config = payload.get("resolved_config", {})
        required_dependencies = payload.get("required_dependencies", ())
        if not isinstance(sequence_lengths, Mapping):
            raise ResourcePreflightError("sequence_lengths must be an object")
        if not isinstance(resolved_config, Mapping):
            raise ResourcePreflightError("resolved_config must be an object")
        if not isinstance(required_dependencies, (list, tuple)):
            raise ResourcePreflightError(
                "required_dependencies must be a list"
            )
        return cls(
            stage=str(payload.get("stage", MEASURE)),
            global_batch_size=payload.get("global_batch_size", MEASURE),
            microbatch_size=payload.get("microbatch_size", MEASURE),
            gradient_accumulation_steps=payload.get(
                "gradient_accumulation_steps", MEASURE
            ),
            sequence_lengths=dict(sequence_lengths),
            precision=str(payload.get("precision", MEASURE)),
            sharding=str(payload.get("sharding", MEASURE)),
            offload=str(payload.get("offload", MEASURE)),
            backend=str(payload.get("backend", MEASURE)),
            world_size=payload.get("world_size", MEASURE),
            topology_hash=str(payload.get("topology_hash", MEASURE)),
            optimizer=str(payload.get("optimizer", MEASURE)),
            optimizer_initialized=payload.get(
                "optimizer_initialized", MEASURE
            ),
            model_copies=payload.get("model_copies", MEASURE),
            reference_copies=payload.get("reference_copies", MEASURE),
            rollout_copies=payload.get("rollout_copies", MEASURE),
            resolved_config=dict(resolved_config),
            code_fingerprint=str(payload.get("code_fingerprint", MEASURE)),
            dependency_profile_hash=str(
                payload.get("dependency_profile_hash", MEASURE)
            ),
            model_profile_hash=str(payload.get("model_profile_hash", MEASURE)),
            required_dependencies=tuple(str(item) for item in required_dependencies),
        )

    def validate_resolved(self) -> None:
        reject_unresolved_production_fields(self, context="workload spec")
        for field_name in (
            "global_batch_size",
            "microbatch_size",
            "gradient_accumulation_steps",
            "world_size",
        ):
            value = _non_negative_int(getattr(self, field_name), field_name)
            if value == 0:
                raise ResourcePreflightError(f"{field_name} must be positive")
        for field_name in ("model_copies", "reference_copies", "rollout_copies"):
            _non_negative_int(getattr(self, field_name), field_name)
        if self.model_copies + self.reference_copies + self.rollout_copies == 0:
            raise ResourcePreflightError(
                "workload must resolve at least one model/reference/rollout copy"
            )
        if not self.sequence_lengths:
            raise ResourcePreflightError("sequence_lengths cannot be empty")
        for name, length in self.sequence_lengths.items():
            if not name or _non_negative_int(length, f"sequence_lengths.{name}") == 0:
                raise ResourcePreflightError(
                    "sequence lengths must have named positive values"
                )
        if not self.resolved_config:
            raise ResourcePreflightError("resolved_config cannot be empty")
        if not isinstance(self.optimizer_initialized, bool):
            raise ResourcePreflightError("optimizer_initialized must be boolean")
        if self.stage in {"midtrain", "sft", "dpo", "rft", "grpo"}:
            if not self.optimizer_initialized or self.optimizer.lower() == "none":
                raise ResourcePreflightError(
                    "training workload must include initialized optimizer state"
                )
        for field_name in (
            "topology_hash",
            "dependency_profile_hash",
            "model_profile_hash",
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", getattr(self, field_name)):
                raise ResourcePreflightError(
                    f"{field_name} must be a 64-hex profile hash"
                )
        if not _FINGERPRINT_RE.fullmatch(self.code_fingerprint):
            raise ResourcePreflightError(
                "workload code_fingerprint must be a clean 40/64-hex revision"
            )
        if len(set(self.required_dependencies)) != len(
            self.required_dependencies
        ):
            raise ResourcePreflightError(
                "required_dependencies contains duplicates"
            )

    @property
    def config_fingerprint(self) -> str:
        return canonical_profile_hash(self.resolved_config)

    @property
    def profile_hash(self) -> str:
        return canonical_profile_hash(
            {"schema_version": 2, "kind": "resolved-workload", **asdict(self)}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            **_jsonable(self),
            "config_fingerprint": self.config_fingerprint,
            "profile_hash": self.profile_hash,
        }


@dataclass(frozen=True)
class MeasurementProvenance:
    command: tuple[str, ...]
    tool: str
    tool_version: str
    hostname: str
    started_at_utc: str
    artifact_sha256: str
    exit_code: int

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MeasurementProvenance":
        command = payload.get("command", ())
        if isinstance(command, str) or not isinstance(command, (list, tuple)):
            raise ResourcePreflightError(
                "measurement command must be an argv list, not shell text"
            )
        return cls(
            command=tuple(str(item) for item in command),
            tool=str(payload.get("tool", MEASURE)),
            tool_version=str(payload.get("tool_version", MEASURE)),
            hostname=str(payload.get("hostname", MEASURE)),
            started_at_utc=str(payload.get("started_at_utc", MEASURE)),
            artifact_sha256=str(payload.get("artifact_sha256", MEASURE)),
            exit_code=payload.get("exit_code", MEASURE),
        )

    def validate_resolved(self) -> None:
        reject_unresolved_production_fields(
            self, context="measurement provenance"
        )
        if not self.command or any(not argument for argument in self.command):
            raise ResourcePreflightError(
                "measurement provenance requires a non-empty argv command"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", self.artifact_sha256):
            raise ResourcePreflightError(
                "measurement artifact_sha256 must be 64 lowercase hex"
            )
        if self.exit_code != 0:
            raise ResourcePreflightError(
                "measurement command did not exit successfully"
            )


@dataclass(frozen=True)
class RankPeakReport:
    rank: int
    hip_ordinal: int
    pci_bdf: str
    run_peak_hbm_bytes: tuple[int, ...]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RankPeakReport":
        runs = payload.get("run_peak_hbm_bytes", ())
        if not isinstance(runs, (list, tuple)):
            raise ResourcePreflightError(
                "run_peak_hbm_bytes must be a repeated-run list"
            )
        return cls(
            rank=payload.get("rank", MEASURE),
            hip_ordinal=payload.get("hip_ordinal", MEASURE),
            pci_bdf=str(payload.get("pci_bdf", MEASURE)),
            run_peak_hbm_bytes=tuple(runs),
        )

    @property
    def peak_bytes(self) -> int:
        return max(self.run_peak_hbm_bytes)

    @property
    def peak_variance_fraction(self) -> float:
        return _relative_peak_stddev(self.run_peak_hbm_bytes)

    def validate_resolved(self, *, repeat_count: int) -> None:
        reject_unresolved_production_fields(self, context="rank peak report")
        _non_negative_int(self.rank, "rank")
        _non_negative_int(self.hip_ordinal, "hip_ordinal")
        _normalize_bdf(self.pci_bdf)
        if len(self.run_peak_hbm_bytes) != repeat_count:
            raise ResourcePreflightError(
                "every rank must report every repeated measurement run"
            )
        for peak in self.run_peak_hbm_bytes:
            _non_negative_int(peak, "run_peak_hbm_bytes")


def _relative_peak_stddev(values: tuple[int, ...]) -> float:
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / mean


@dataclass(frozen=True)
class PhaseEvidence:
    phase: str
    rank_reports: tuple[RankPeakReport, ...]
    host_peak_runs_bytes: tuple[int, ...]
    filesystem_peak_runs_bytes: Mapping[str, tuple[int, ...]]
    safety_margin_fraction: float
    max_peak_variance_fraction: float
    optimizer_initialized: bool
    provenance: MeasurementProvenance

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PhaseEvidence":
        rank_reports = payload.get("rank_reports", ())
        host_runs = payload.get("host_peak_runs_bytes", ())
        filesystem_runs = payload.get("filesystem_peak_runs_bytes", {})
        provenance = payload.get("provenance", {})
        if not isinstance(rank_reports, (list, tuple)):
            raise ResourcePreflightError("rank_reports must be a list")
        if not isinstance(host_runs, (list, tuple)):
            raise ResourcePreflightError("host_peak_runs_bytes must be a list")
        if not isinstance(filesystem_runs, Mapping):
            raise ResourcePreflightError(
                "filesystem_peak_runs_bytes must be an object"
            )
        if not isinstance(provenance, Mapping):
            raise ResourcePreflightError("provenance must be an object")
        return cls(
            phase=str(payload.get("phase", MEASURE)),
            rank_reports=tuple(
                RankPeakReport.from_dict(item) for item in rank_reports
            ),
            host_peak_runs_bytes=tuple(host_runs),
            filesystem_peak_runs_bytes={
                str(role): tuple(values)
                for role, values in filesystem_runs.items()
            },
            safety_margin_fraction=payload.get(
                "safety_margin_fraction", MEASURE
            ),
            max_peak_variance_fraction=payload.get(
                "max_peak_variance_fraction", MEASURE
            ),
            optimizer_initialized=payload.get(
                "optimizer_initialized", MEASURE
            ),
            provenance=MeasurementProvenance.from_dict(provenance),
        )

    def validate_resolved(
        self,
        *,
        world_size: int,
        filesystem_roles: tuple[str, ...],
    ) -> None:
        reject_unresolved_production_fields(self, context=f"phase {self.phase}")
        if len(self.host_peak_runs_bytes) < 3:
            raise ResourcePreflightError(
                f"phase {self.phase!r} requires at least three repeated runs"
            )
        repeat_count = len(self.host_peak_runs_bytes)
        if len(self.rank_reports) != world_size:
            raise ResourcePreflightError(
                f"phase {self.phase!r} lacks per-rank reports"
            )
        expected_ranks = set(range(world_size))
        actual_ranks = {report.rank for report in self.rank_reports}
        if actual_ranks != expected_ranks:
            raise ResourcePreflightError(
                f"phase {self.phase!r} rank reports are incomplete"
            )
        for report in self.rank_reports:
            report.validate_resolved(repeat_count=repeat_count)
        for value in self.host_peak_runs_bytes:
            _non_negative_int(value, "host_peak_runs_bytes")
        for role in filesystem_roles:
            values = self.filesystem_peak_runs_bytes.get(role)
            if values is None or len(values) != repeat_count:
                raise ResourcePreflightError(
                    f"phase {self.phase!r} lacks repeated {role} filesystem peaks"
                )
            for value in values:
                _non_negative_int(value, f"filesystem_peak_runs_bytes.{role}")
        if not isinstance(self.optimizer_initialized, bool):
            raise ResourcePreflightError(
                "phase optimizer_initialized must be boolean"
            )
        if not isinstance(self.safety_margin_fraction, (int, float)) or not (
            0.0 < self.safety_margin_fraction < 1.0
        ):
            raise ResourcePreflightError(
                "safety_margin_fraction must be in (0, 1)"
            )
        if not isinstance(
            self.max_peak_variance_fraction, (int, float)
        ) or not (0.0 <= self.max_peak_variance_fraction < 1.0):
            raise ResourcePreflightError(
                "max_peak_variance_fraction must be in [0, 1)"
            )
        spreads = [
            report.peak_variance_fraction for report in self.rank_reports
        ]
        spreads.append(_relative_peak_stddev(self.host_peak_runs_bytes))
        spreads.extend(
            _relative_peak_stddev(self.filesystem_peak_runs_bytes[role])
            for role in filesystem_roles
        )
        observed_spread = max(spreads)
        if observed_spread > self.max_peak_variance_fraction:
            raise ResourcePreflightError(
                f"phase {self.phase!r} repeated-run peak variance exceeds policy"
            )
        if self.safety_margin_fraction < observed_spread:
            raise ResourcePreflightError(
                f"phase {self.phase!r} safety margin is below observed variance"
            )
        self.provenance.validate_resolved()

    @property
    def profile_hash(self) -> str:
        return canonical_profile_hash(
            {"schema_version": 2, "kind": "phase-evidence", **asdict(self)}
        )


_TRAINING_STAGES = {"midtrain", "sft", "dpo", "rft", "grpo"}


def required_measurement_phases(stage: str) -> tuple[str, ...]:
    normalized = stage.strip().lower()
    if normalized in _TRAINING_STAGES:
        return (
            "steady_step",
            "synchronization",
            "checkpoint_save",
            "checkpoint_restore",
        )
    if normalized == "soup":
        return ("soup_load", "checkpoint_save", "checkpoint_restore")
    if normalized in {"eval", "retention", "serve"}:
        return ("eval_load", "synchronization")
    raise ResourcePreflightError(
        f"no audited measurement-phase policy for stage {stage!r}"
    )


@dataclass(frozen=True)
class MeasuredPeakProfile:
    """Repeated, per-rank, phase-complete peak evidence."""

    workload: WorkloadSpec
    environment_hash: str
    phases: tuple[PhaseEvidence, ...]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MeasuredPeakProfile":
        workload = payload.get("workload", {})
        phases = payload.get("phases", ())
        if not isinstance(workload, Mapping):
            raise ResourcePreflightError(
                "measured workload must be a resolved object, not free-form text"
            )
        if not isinstance(phases, (list, tuple)):
            raise ResourcePreflightError("measured phases must be a list")
        return cls(
            workload=WorkloadSpec.from_dict(workload),
            environment_hash=str(payload.get("environment_hash", MEASURE)),
            phases=tuple(PhaseEvidence.from_dict(item) for item in phases),
        )

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

    def unresolved_fields(self) -> list[str]:
        return sorted(set(_unresolved_paths(self)))

    def validate_resolved(
        self, *, filesystem_roles: tuple[str, ...]
    ) -> None:
        unresolved = self.unresolved_fields()
        if unresolved:
            raise UnresolvedProductionFieldError(
                "measured peak profile has unresolved MEASURE production fields: "
                + ", ".join(unresolved[:20])
            )
        self.workload.validate_resolved()
        if not re.fullmatch(r"[0-9a-f]{64}", self.environment_hash):
            raise ResourcePreflightError(
                "environment_hash must be a 64-hex profile hash"
            )
        phase_names = [phase.phase for phase in self.phases]
        if len(set(phase_names)) != len(phase_names):
            raise ResourcePreflightError("measurement phases must be unique")
        missing = set(required_measurement_phases(self.workload.stage)) - set(
            phase_names
        )
        if missing:
            raise ResourcePreflightError(
                "measured profile omits required separate phases: "
                + ", ".join(sorted(missing))
            )
        for phase in self.phases:
            phase.validate_resolved(
                world_size=self.workload.world_size,
                filesystem_roles=filesystem_roles,
            )
            if (
                self.workload.optimizer_initialized
                and not phase.optimizer_initialized
            ):
                raise ResourcePreflightError(
                    f"phase {phase.phase!r} was measured before optimizer initialization"
                )

    @property
    def profile_hash(self) -> str:
        return canonical_profile_hash(
            {"schema_version": 2, "kind": "measured-peak-profile", **_jsonable(self)}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "workload": self.workload.to_dict(),
            "environment_hash": self.environment_hash,
            "phases": [
                {**_jsonable(phase), "profile_hash": phase.profile_hash}
                for phase in self.phases
            ],
            "profile_hash": self.profile_hash,
        }


@dataclass(frozen=True)
class PreflightReport:
    """Auditable preflight result with an explicit fit-assertion state."""

    model_spec: ModelSpec
    resources: ResourceSnapshot
    analytical_lower_bounds: AnalyticalLowerBounds
    expected_workload: Optional[WorkloadSpec]
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
                "schema_version": 2,
                "kind": "resource-preflight-report",
                "model_profile_hash": self.model_spec.profile_hash,
                "resource_profile_hash": self.resources.profile_hash,
                "analytical_profile_hash": self.analytical_lower_bounds.profile_hash,
                "expected_workload_hash": (
                    self.expected_workload.profile_hash
                    if self.expected_workload is not None
                    else None
                ),
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
            "schema_version": 2,
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
            "expected_workload": (
                self.expected_workload.to_dict()
                if self.expected_workload is not None
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
    expected_workload: Optional[WorkloadSpec] = None,
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
            expected_workload=expected_workload,
            measured_profile=measured_profile,
            status="unresolved",
            fit_asserted=False,
            reasons=(
                "resource snapshot contains MEASURE fields: "
                + ", ".join(unresolved_resources[:20]),
            ),
            headroom_fraction=headroom_fraction,
        )

    try:
        resources.validate_resolved()
    except ResourcePreflightError as exc:
        return PreflightReport(
            model_spec=model_spec,
            resources=resources,
            analytical_lower_bounds=bounds,
            expected_workload=expected_workload,
            measured_profile=measured_profile,
            status="unresolved",
            fit_asserted=False,
            reasons=(f"resource identity/allocation validation failed: {exc}",),
            headroom_fraction=headroom_fraction,
        )
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
            expected_workload=expected_workload,
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
            expected_workload=expected_workload,
            measured_profile=None,
            status="analytical_only",
            fit_asserted=False,
            reasons=(
                "analytical lower bounds are necessary conditions only; ingest a "
                "matching measured peak profile to assert fit",
            ),
            headroom_fraction=headroom_fraction,
        )

    if expected_workload is None:
        return PreflightReport(
            model_spec=model_spec,
            resources=resources,
            analytical_lower_bounds=bounds,
            expected_workload=None,
            measured_profile=measured_profile,
            status="unresolved",
            fit_asserted=False,
            reasons=(
                "measured evidence requires an exact expected WorkloadSpec; "
                "free-form workload labels are not accepted",
            ),
            headroom_fraction=headroom_fraction,
        )
    filesystem_roles = tuple(
        sorted(filesystem.role for filesystem in resources.filesystems)
    )
    evidence_reasons: list[str] = []
    try:
        expected_workload.validate_resolved()
    except ResourcePreflightError as exc:
        evidence_reasons.append(f"expected workload is unresolved: {exc}")
    if expected_workload.model_profile_hash != model_spec.profile_hash:
        evidence_reasons.append(
            "expected workload model fingerprint is stale or mismatched"
        )
    if expected_workload.topology_hash != resources.topology_hash:
        evidence_reasons.append(
            "expected workload topology/world mapping is stale or mismatched"
        )
    if expected_workload.dependency_profile_hash != resources.dependency_profile_hash:
        evidence_reasons.append(
            "expected workload dependency/software fingerprint is stale"
        )
    if expected_workload.code_fingerprint != resources.code_fingerprint:
        evidence_reasons.append("expected workload code fingerprint is stale")
    if expected_workload.world_size != len(visible_gpus):
        evidence_reasons.append(
            "expected workload world_size differs from visible mapped GPUs"
        )
    for dependency in expected_workload.required_dependencies:
        version = resources.software_versions.get(dependency, MEASURE)
        if version == ABSENT:
            evidence_reasons.append(
                f"required dependency {dependency!r} is explicitly ABSENT"
            )
        elif version in (MEASURE, None, ""):
            evidence_reasons.append(
                f"required dependency {dependency!r} is unresolved"
            )

    try:
        measured_profile.validate_resolved(filesystem_roles=filesystem_roles)
    except ResourcePreflightError as exc:
        evidence_reasons.append(f"measured phase evidence is invalid: {exc}")
    if measured_profile.workload.profile_hash != expected_workload.profile_hash:
        evidence_reasons.append(
            "measured workload/config fingerprint does not match expected workload"
        )
    if measured_profile.environment_hash != resources.environment_hash:
        evidence_reasons.append(
            "measured environment hash does not match current resources"
        )
    if evidence_reasons:
        return PreflightReport(
            model_spec=model_spec,
            resources=resources,
            analytical_lower_bounds=bounds,
            expected_workload=expected_workload,
            measured_profile=measured_profile,
            status="unresolved",
            fit_asserted=False,
            reasons=tuple(evidence_reasons),
            headroom_fraction=headroom_fraction,
        )

    capacity_reasons: list[str] = []
    visible_by_ordinal = {
        int(gpu.hip_ordinal): gpu for gpu in visible_gpus
    }
    filesystems = {
        filesystem.role: filesystem for filesystem in resources.filesystems
    }
    weights_lower_bound = max(
        bounds.checkpoint_tensor_bytes, bounds.bf16_weights_bytes
    )
    copy_weights_lower_bound = weights_lower_bound * (
        expected_workload.model_copies
        + expected_workload.reference_copies
        + expected_workload.rollout_copies
    )
    if (
        expected_workload.optimizer_initialized
        and "adam" in expected_workload.optimizer.lower()
    ):
        resident_state_lower_bound = (
            bounds.full_finetune_persistent_state_bytes
            * expected_workload.model_copies
            + weights_lower_bound
            * (
                expected_workload.reference_copies
                + expected_workload.rollout_copies
            )
        )
    else:
        resident_state_lower_bound = copy_weights_lower_bound
    no_offload = expected_workload.offload.lower() in {
        "none",
        "false",
        "disabled",
        "no",
    }
    for phase in measured_profile.phases:
        phase_margin = max(
            headroom_fraction, float(phase.safety_margin_fraction)
        )
        phase_total_hbm = 0
        for rank_report in phase.rank_reports:
            gpu = visible_by_ordinal.get(rank_report.hip_ordinal)
            if gpu is None:
                capacity_reasons.append(
                    f"phase {phase.phase!r} rank {rank_report.rank} reports an "
                    "unmapped HIP ordinal"
                )
                continue
            if _normalize_bdf(rank_report.pci_bdf) != _normalize_bdf(gpu.pci_bdf):
                capacity_reasons.append(
                    f"phase {phase.phase!r} rank/BDF mapping differs from inventory"
                )
                continue
            peak = rank_report.peak_bytes
            phase_total_hbm += peak
            required = _required_with_headroom(peak, phase_margin)
            if required > int(gpu.free_hbm_bytes):
                capacity_reasons.append(
                    f"phase {phase.phase!r} HIP ordinal {gpu.hip_ordinal} free HBM "
                    f"is insufficient ({gpu.free_hbm_bytes} < {required})"
                )
        host_peak = max(phase.host_peak_runs_bytes)
        measured_resident_bytes = (
            phase_total_hbm if no_offload else phase_total_hbm + host_peak
        )
        if measured_resident_bytes < resident_state_lower_bound:
            capacity_reasons.append(
                f"phase {phase.phase!r} measured resident state is below the "
                "analytical model/reference/rollout/optimizer lower bound"
            )
        required_host = _required_with_headroom(
            host_peak, phase_margin
        )
        if required_host > int(resources.host_ram_available_bytes):
            capacity_reasons.append(
                f"phase {phase.phase!r} host RAM is insufficient "
                f"({resources.host_ram_available_bytes} < {required_host})"
            )
        for role in filesystem_roles:
            required = _required_with_headroom(
                max(phase.filesystem_peak_runs_bytes[role]), phase_margin
            )
            free = int(filesystems[role].free_bytes)
            if required > free:
                capacity_reasons.append(
                    f"phase {phase.phase!r} {role} filesystem is insufficient "
                    f"({free} < {required})"
                )

    status = "insufficient" if capacity_reasons else "measured_pass"
    return PreflightReport(
        model_spec=model_spec,
        resources=resources,
        analytical_lower_bounds=bounds,
        expected_workload=expected_workload,
        measured_profile=measured_profile,
        status=status,
        fit_asserted=not capacity_reasons,
        reasons=(
            tuple(capacity_reasons)
            if capacity_reasons
            else ("all required measured phases fit with validated margins",)
        ),
        headroom_fraction=headroom_fraction,
    )


def run_resource_preflight(
    model_spec: ModelSpec,
    resources: ResourceSnapshot,
    measured_profile: Optional[MeasuredPeakProfile] = None,
    *,
    expected_workload: Optional[WorkloadSpec] = None,
    production: bool = True,
    require_measured: bool = False,
    headroom_fraction: float = 0.10,
) -> PreflightReport:
    """Run preflight and reject unresolved/insufficient production inputs."""

    report = evaluate_resource_preflight(
        model_spec,
        resources,
        measured_profile,
        expected_workload=expected_workload,
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
    """Collect versions without importing GPU/framework packages.

    ``ABSENT`` is a known fact and differs from ``MEASURE`` (unknown). A workload
    declares which optional dependencies are required; absent unrelated packages
    do not by themselves make the host inventory unresolved.
    """

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
    for package in (
        "torch",
        "transformers",
        "safetensors",
        "accelerate",
        "trl",
        "peft",
        "datasets",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = ABSENT
        except Exception:
            versions[package] = MEASURE
    return versions


def _parse_ordinal_list(raw: str) -> tuple[int, ...]:
    pieces = [piece.strip() for piece in raw.split(",")]
    if not pieces or any(not piece or not piece.isdigit() for piece in pieces):
        raise ValueError("ordinals must be comma-separated non-negative integers")
    values = tuple(int(piece) for piece in pieces)
    if len(set(values)) != len(values):
        raise ValueError("ordinals contain duplicates")
    return values


def _visibility_policy(
    all_hip_ordinals: tuple[int, ...],
    environ: Mapping[str, str],
) -> Mapping[str, Any]:
    hip = environ.get("HIP_VISIBLE_DEVICES")
    rocr = environ.get("ROCR_VISIBLE_DEVICES")
    if hip is not None and rocr is not None:
        return {
            "authority": MEASURE,
            "raw": f"HIP={hip};ROCR={rocr}",
            "hip_ordinals": MEASURE,
            "error": "double masks",
        }
    if rocr is not None:
        return {
            "authority": MEASURE,
            "raw": f"ROCR_VISIBLE_DEVICES={rocr}",
            "hip_ordinals": MEASURE,
            "error": "non-authoritative ROCR mask",
        }
    if hip is None:
        return {
            "authority": "unmasked",
            "raw": NOT_APPLICABLE,
            "hip_ordinals": list(sorted(all_hip_ordinals)),
        }
    try:
        ordinals = _parse_ordinal_list(hip)
    except ValueError:
        return {
            "authority": MEASURE,
            "raw": f"HIP_VISIBLE_DEVICES={hip}",
            "hip_ordinals": MEASURE,
            "error": "invalid HIP mask",
        }
    if not set(ordinals).issubset(set(all_hip_ordinals)):
        return {
            "authority": MEASURE,
            "raw": f"HIP_VISIBLE_DEVICES={hip}",
            "hip_ordinals": MEASURE,
            "error": "mask contains unmapped HIP ordinals",
        }
    return {
        "authority": "HIP_VISIBLE_DEVICES",
        "raw": hip,
        "hip_ordinals": list(ordinals),
    }


def _parse_slurm_cards(raw: str) -> tuple[int, ...]:
    cards: list[int] = []
    for piece in (item.strip() for item in raw.split(",")):
        if not piece:
            raise ValueError("empty Slurm GPU token")
        if "-" in piece:
            start_raw, end_raw = piece.split("-", 1)
            if not start_raw.isdigit() or not end_raw.isdigit():
                raise ValueError("non-numeric Slurm GPU range")
            start, end = int(start_raw), int(end_raw)
            if end < start:
                raise ValueError("descending Slurm GPU range")
            cards.extend(range(start, end + 1))
        elif piece.isdigit():
            cards.append(int(piece))
        else:
            raise ValueError("non-numeric Slurm GPU token")
    if not cards or len(set(cards)) != len(cards):
        raise ValueError("Slurm GPU list is empty or duplicated")
    return tuple(cards)


def _slurm_allocation(
    hip_inventory: tuple[Mapping[str, Any], ...],
    environ: Mapping[str, str],
) -> Mapping[str, Any]:
    active = any(
        environ.get(key)
        for key in (
            "SLURM_JOB_ID",
            "SLURM_STEP_ID",
            "SLURM_JOB_GPUS",
            "SLURM_STEP_GPUS",
            "SLURM_JOB_GRES",
            "SLURM_STEP_GRES",
        )
    )
    if not active:
        return {
            "mode": "none",
            "job_id": NOT_APPLICABLE,
            "step_id": NOT_APPLICABLE,
            "gres": NOT_APPLICABLE,
            "physical_cards": [],
            "hip_ordinals": [],
        }
    raw_cards = environ.get("SLURM_STEP_GPUS") or environ.get("SLURM_JOB_GPUS")
    raw_gres = environ.get("SLURM_STEP_GRES") or environ.get("SLURM_JOB_GRES")
    if not raw_cards or not raw_gres:
        return {
            "mode": "slurm",
            "job_id": environ.get("SLURM_JOB_ID", MEASURE),
            "step_id": environ.get("SLURM_STEP_ID", NOT_APPLICABLE),
            "gres": raw_gres or MEASURE,
            "physical_cards": MEASURE,
            "hip_ordinals": MEASURE,
        }
    try:
        physical_cards = _parse_slurm_cards(raw_cards)
    except ValueError:
        physical_cards = ()
    by_physical: dict[int, int] = {}
    for entry in hip_inventory:
        physical = entry.get("physical_card")
        ordinal = entry.get("hip_ordinal")
        if isinstance(physical, int) and isinstance(ordinal, int):
            if physical in by_physical:
                return {
                    "mode": "slurm",
                    "job_id": environ.get("SLURM_JOB_ID", MEASURE),
                    "step_id": environ.get("SLURM_STEP_ID", NOT_APPLICABLE),
                    "gres": raw_gres,
                    "physical_cards": list(physical_cards),
                    "hip_ordinals": MEASURE,
                }
            by_physical[physical] = ordinal
    if not physical_cards or any(card not in by_physical for card in physical_cards):
        hip_ordinals: list[int] | str = MEASURE
    else:
        hip_ordinals = [by_physical[card] for card in physical_cards]
    return {
        "mode": "slurm",
        "job_id": environ.get("SLURM_JOB_ID", MEASURE),
        "step_id": environ.get("SLURM_STEP_ID", NOT_APPLICABLE),
        "gres": raw_gres,
        "physical_cards": list(physical_cards) if physical_cards else MEASURE,
        "hip_ordinals": hip_ordinals,
    }


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
    *,
    hip_inventory: Optional[tuple[Mapping[str, Any], ...]] = None,
    slurm_allocation: Optional[Mapping[str, Any]] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> tuple[GPUDevice, ...]:
    """Join DRM nodes to an explicit HIP/physical-card inventory by PCI BDF.

    No ordinal is inferred from discovery order. Missing or ambiguous joins are
    retained as ``MEASURE`` so production validation fails with an audit trail.
    """

    root = Path(sysfs_root)
    env = dict(os.environ if environ is None else environ)
    inventory = tuple(hip_inventory or ())
    allocation = dict(
        slurm_allocation
        if slurm_allocation is not None
        else _slurm_allocation(inventory, env)
    )
    all_ordinals = tuple(
        sorted(
            entry["hip_ordinal"]
            for entry in inventory
            if isinstance(entry.get("hip_ordinal"), int)
        )
    )
    visibility = _visibility_policy(all_ordinals, env)
    visible_ordinals_raw = visibility.get("hip_ordinals")
    visible_ordinals = (
        set(visible_ordinals_raw)
        if isinstance(visible_ordinals_raw, (list, tuple))
        else None
    )

    render_by_bdf: dict[str, list[str]] = {}
    for render in root.glob("renderD[0-9]*"):
        try:
            bdf = _normalize_bdf((render / "device").resolve().name)
        except (OSError, ResourcePreflightError):
            continue
        render_by_bdf.setdefault(bdf, []).append(render.name)

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
        try:
            pci_bdf = _normalize_bdf(device.resolve().name)
        except (OSError, ResourcePreflightError):
            pci_bdf = MEASURE
        uevent = _parse_uevent(device / "uevent")
        name = (
            _read_text(device / "product_name")
            or uevent.get("PCI_ID")
            or MEASURE
        )
        uuid = _read_text(device / "unique_id") or MEASURE
        matching_inventory = []
        if pci_bdf != MEASURE:
            for entry in inventory:
                try:
                    if _normalize_bdf(str(entry.get("pci_bdf", ""))) == pci_bdf:
                        matching_inventory.append(entry)
                except ResourcePreflightError:
                    continue
        hip_entry = matching_inventory[0] if len(matching_inventory) == 1 else {}
        hip_ordinal = hip_entry.get("hip_ordinal", MEASURE)
        hip_bdf = str(hip_entry.get("pci_bdf", MEASURE))
        hip_uuid = str(hip_entry.get("uuid", MEASURE))
        physical_card = hip_entry.get("physical_card", MEASURE)
        render_nodes = (
            render_by_bdf.get(pci_bdf, []) if pci_bdf != MEASURE else []
        )
        render_node = render_nodes[0] if len(render_nodes) == 1 else MEASURE
        visible: bool | str = (
            hip_ordinal in visible_ordinals
            if visible_ordinals is not None and isinstance(hip_ordinal, int)
            else MEASURE
        )
        if allocation.get("mode") == "none":
            slurm_allocated: bool | str = NOT_APPLICABLE
            slurm_gres_id = NOT_APPLICABLE
        else:
            allocated_cards = allocation.get("physical_cards")
            if isinstance(allocated_cards, (list, tuple)) and isinstance(
                physical_card, int
            ):
                slurm_allocated = physical_card in allocated_cards
                slurm_gres_id = (
                    f"{allocation.get('gres')}@physical:{physical_card}"
                    if slurm_allocated
                    else NOT_APPLICABLE
                )
            else:
                slurm_allocated = MEASURE
                slurm_gres_id = MEASURE
        numa_raw = _read_text(device / "numa_node")
        try:
            numa_node: int | str = int(numa_raw) if numa_raw is not None else MEASURE
        except ValueError:
            numa_node = MEASURE
        devices.append(
            GPUDevice(
                drm_card=card.name,
                render_node=render_node,
                pci_bdf=pci_bdf,
                hip_reported_pci_bdf=hip_bdf,
                uuid=uuid,
                hip_reported_uuid=hip_uuid,
                physical_card=physical_card,
                hip_ordinal=hip_ordinal,
                slurm_gres_id=slurm_gres_id,
                slurm_allocated=slurm_allocated,
                name=name,
                numa_node=numa_node,
                total_hbm_bytes=total,
                free_hbm_bytes=free,
                visible=visible,
            )
        )
    return tuple(devices)


def _hip_inventory_from_env(
    environ: Mapping[str, str],
) -> tuple[Mapping[str, Any], ...]:
    raw = environ.get("KORE_HIP_INVENTORY_JSON")
    if not raw:
        return ()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return ()
    if not isinstance(payload, list) or any(
        not isinstance(item, Mapping) for item in payload
    ):
        return ()
    return tuple(dict(item) for item in payload)


def collect_code_fingerprint(code_root: Optional[str | Path] = None) -> str:
    """Return a clean git commit fingerprint; dirty/unknown trees are unresolved."""

    root = (
        Path(code_root).expanduser().resolve()
        if code_root is not None
        else Path(__file__).resolve().parents[2]
    )
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip().lower()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return MEASURE
    if dirty or not _FINGERPRINT_RE.fullmatch(revision):
        return MEASURE
    return revision


def collect_resource_snapshot(
    model_path: str | Path,
    scratch_path: str | Path,
    *,
    sysfs_root: str | Path = "/sys/class/drm",
    hip_inventory: Optional[tuple[Mapping[str, Any], ...]] = None,
    gpu_topology: Optional[Mapping[str, Any]] = None,
    environ: Optional[Mapping[str, str]] = None,
    code_root: Optional[str | Path] = None,
) -> ResourceSnapshot:
    """Collect capacity, free-space, topology, and software without GPU init."""

    env = dict(os.environ if environ is None else environ)
    inventory = tuple(
        hip_inventory
        if hip_inventory is not None
        else _hip_inventory_from_env(env)
    )
    slurm = _slurm_allocation(inventory, env)
    devices = collect_amd_gpu_devices(
        sysfs_root,
        hip_inventory=inventory,
        slurm_allocation=slurm,
        environ=env,
    )
    mapped_ordinals = tuple(
        sorted(
            int(gpu.hip_ordinal)
            for gpu in devices
            if isinstance(gpu.hip_ordinal, int)
        )
    )
    host_total, host_available = _read_host_ram()
    return ResourceSnapshot(
        gpus=devices,
        gpu_topology=(
            dict(gpu_topology)
            if gpu_topology is not None
            else _probe_rocm_topology()
        ),
        visible_device_policy=_visibility_policy(mapped_ordinals, env),
        slurm_allocation=slurm,
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
        code_fingerprint=collect_code_fingerprint(code_root),
    )


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a JSON report and fsync both file and directory."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n"
    )
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o644)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(destination.parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


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
    "ABSENT",
    "MEASURE",
    "NOT_APPLICABLE",
    "AnalyticalLowerBounds",
    "FilesystemCapacity",
    "GPUDevice",
    "InsufficientResourcesError",
    "MeasurementProvenance",
    "MeasuredPeakProfile",
    "PhaseEvidence",
    "PreflightError",
    "PreflightReport",
    "ResourcePreflightError",
    "ResourceSnapshot",
    "RankPeakReport",
    "UnresolvedMeasurementError",
    "UnresolvedProductionFieldError",
    "WorkloadSpec",
    "analytical_lower_bounds",
    "atomic_write_json",
    "collect_amd_gpu_devices",
    "collect_code_fingerprint",
    "collect_resource_snapshot",
    "collect_software_versions",
    "compute_analytical_lower_bounds",
    "evaluate_resource_preflight",
    "load_resource_snapshot",
    "preflight_resources",
    "reject_unresolved_production_fields",
    "required_measurement_phases",
    "run_resource_preflight",
]
