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

    @classmethod
    def from_dir(cls, d: Path) -> "Task":
        meta = yaml.safe_load((d / "task.yaml").read_text())
        shapes: list[Shape] = []
        raw_shapes = meta.get("shapes", {}) or {}
        for key, val in raw_shapes.items():
            if key == "validation" and isinstance(val, list):
                for i, dims in enumerate(val):
                    shapes.append(Shape(f"validation_{i}", dict(dims)))
            elif isinstance(val, dict):
                shapes.append(Shape(key, dict(val)))
        targets = meta.get("targets", {}) or {}
        return cls(
            task_id=meta["task_id"],
            operation=meta.get("operation", meta["task_id"]),
            dtype=meta.get("dtype", "fp32"),
            backend=meta.get("backend", "triton"),
            gpu_target=meta.get("gpu_target", "gfx950"),  # KORE target = CDNA4 (MI350X)
            dir=d,
            seed_kernel_name=meta.get("seed_kernel_name", "seed_triton.py"),
            snr_threshold=float(meta.get("snr_threshold", targets.get("snr_db", 30.0))),
            comparison_baseline=targets.get("comparison_baseline", "torch"),
            shapes=shapes,
            raw=meta,
        )
