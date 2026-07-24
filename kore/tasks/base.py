"""Task ABI: a KORE task = a driver contract + shapes + a production baseline.

Each task lives in ``kore/tasks/<task_id>/`` with:
    task.yaml      - metadata (schema below)
    reference.py   - torch-fp32 oracle for correctness
    seed_triton.py - a compiling (not necessarily fast) seed kernel
    driver.py      - the KernelForge verifier contract:
        correctness mode : writes candidate to kernel.py, prints
                           ``SNR: <db>`` / ``allclose: <bool>`` / ``max_diff: <x>``
        --bench-mode --impl {reference|candidate} : prints ``median_ms: <x>``
                           where ``reference`` == the REAL production op (AITER/hipBLASLt).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass(frozen=True)
class Shape:
    name: str
    dims: dict[str, int]

    def as_args(self) -> list[str]:
        # drivers parse a single ``--shape "K1=V1,K2=V2"`` string.
        if not self.dims:
            return []
        spec = ",".join(f"{k}={v}" for k, v in self.dims.items())
        return ["--shape", spec]


@dataclass
class Task:
    task_id: str
    operation: str
    dtype: str
    backend: str
    gpu_target: str
    dir: Path
    seed_kernel_name: str
    snr_threshold: float
    comparison_baseline: str
    shapes: list[Shape] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    provenance_root: str = ""

    @property
    def driver_path(self) -> Path:
        return self.dir / "driver.py"

    @property
    def reference_path(self) -> Path:
        return self.dir / "reference.py"

    @property
    def seed_path(self) -> Path:
        return self.dir / self.seed_kernel_name

    @property
    def seed_source(self) -> str:
        return self.seed_path.read_text()

    def shape(self, name: str) -> Optional[Shape]:
        for s in self.shapes:
            if s.name == name:
                return s
        return None

    @property
    def source_family(self) -> Optional[str]:
        """Generator-native family metadata (not the canonical taxonomy leaf)."""
        value = self.raw.get("op_family")
        return str(value) if value is not None else None

    @classmethod
    def from_dir(cls, d: Path) -> "Task":
        d = Path(d)
        yaml_path = d / "task.yaml"
        if not yaml_path.is_file():
            raise ValueError(f"{d}: missing task.yaml")
        meta = yaml.safe_load(yaml_path.read_text())
        if not isinstance(meta, dict):
            raise ValueError(f"{yaml_path}: top-level YAML must be a mapping")

        required = (
            "task_id",
            "operation",
            "dtype",
            "backend",
            "gpu_target",
            "seed_kernel_name",
            "snr_threshold",
            "shapes",
            "targets",
        )
        missing = [key for key in required if key not in meta]
        if missing:
            raise ValueError(f"{yaml_path}: missing required keys {missing}")

        def required_string(key: str) -> str:
            value = meta.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{yaml_path}: {key} must be a non-empty string")
            return value.strip()

        task_id = required_string("task_id")
        operation = required_string("operation")
        dtype = required_string("dtype")
        backend = required_string("backend")
        gpu_target = required_string("gpu_target")
        seed_kernel_name = required_string("seed_kernel_name")
        if task_id != d.name:
            raise ValueError(
                f"{yaml_path}: task_id {task_id!r} collides with directory {d.name!r}"
            )

        shapes: list[Shape] = []
        raw_shapes = meta.get("shapes")
        if not isinstance(raw_shapes, dict) or not raw_shapes:
            raise ValueError(f"{yaml_path}: shapes must be a non-empty mapping")

        def validated_dims(name: str, value: Any) -> dict[str, int]:
            if not isinstance(value, dict) or not value:
                raise ValueError(f"{yaml_path}: shape {name!r} must be a non-empty mapping")
            dims: dict[str, int] = {}
            for key, dim in value.items():
                if not isinstance(key, str) or not key:
                    raise ValueError(f"{yaml_path}: shape {name!r} has an invalid dimension key")
                if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
                    raise ValueError(
                        f"{yaml_path}: shape {name!r} dimension {key!r} "
                        f"must be a positive integer, got {dim!r}"
                    )
                dims[key] = dim
            return dims

        for key, val in raw_shapes.items():
            if key == "validation" and isinstance(val, list):
                for i, dims in enumerate(val):
                    shapes.append(
                        Shape(f"validation_{i}", validated_dims(f"validation_{i}", dims))
                    )
            elif isinstance(val, dict):
                shapes.append(Shape(str(key), validated_dims(str(key), val)))
            else:
                raise ValueError(f"{yaml_path}: invalid shape entry {key!r}")
        if not shapes:
            raise ValueError(f"{yaml_path}: no concrete shapes declared")

        targets = meta.get("targets")
        if not isinstance(targets, dict):
            raise ValueError(f"{yaml_path}: targets must be a mapping")
        comparison_baseline = targets.get("comparison_baseline")
        if not isinstance(comparison_baseline, str) or not comparison_baseline.strip():
            raise ValueError(
                f"{yaml_path}: targets.comparison_baseline must be a non-empty string"
            )
        try:
            snr_threshold = float(meta["snr_threshold"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{yaml_path}: snr_threshold must be numeric") from exc

        for artifact in ("driver.py", "reference.py", seed_kernel_name):
            if not (d / artifact).is_file():
                raise ValueError(f"{yaml_path}: missing required artifact {artifact}")

        provenance = meta.get("provenance")
        provenance_root = meta.get("provenance_root") or meta.get("lineage_root")
        if not provenance_root and isinstance(provenance, dict):
            provenance_root = provenance.get("root")
        provenance_root = str(provenance_root or task_id).strip()
        if not provenance_root:
            raise ValueError(f"{yaml_path}: provenance root must be non-empty")

        return cls(
            task_id=task_id,
            operation=operation,
            dtype=dtype,
            backend=backend,
            gpu_target=gpu_target,
            dir=d,
            seed_kernel_name=seed_kernel_name,
            snr_threshold=snr_threshold,
            comparison_baseline=comparison_baseline.strip(),
            shapes=shapes,
            raw=meta,
            provenance_root=provenance_root,
        )
