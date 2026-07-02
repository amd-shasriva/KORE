"""Parser for rocprofv3 CSV output."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class KernelPMC:
    """PMC counter data for a single kernel dispatch."""

    kernel_name: str
    counters: dict[str, int] = field(default_factory=dict)

    @property
    def mfma_count(self) -> int:
        """Total MFMA instructions (sum across all MFMA counter variants)."""
        return sum(
            v for k, v in self.counters.items()
            if "MFMA" in k.upper()
        )

    @property
    def vmem_count(self) -> int:
        return self.counters.get("SQ_INSTS_VMEM", 0)

    @property
    def wait_any(self) -> int:
        return self.counters.get("SQ_WAIT_INST_ANY", 0)

    @property
    def wait_lds(self) -> int:
        return self.counters.get("SQ_WAIT_INST_LDS", 0)

    @property
    def wait_mfma_ratio(self) -> float:
        """wait/MFMA ratio: <5 compute-bound, 5-10 balanced, >10 memory-bound."""
        mfma = self.mfma_count
        if mfma == 0:
            return float("inf")
        return self.wait_any / mfma

    @property
    def diagnosis(self) -> str:
        ratio = self.wait_mfma_ratio
        if ratio < 5:
            return "COMPUTE-BOUND (good MFMA utilization)"
        elif ratio < 10:
            return "BALANCED (some memory/LDS pressure)"
        else:
            return "MEMORY-BOUND (optimize data movement)"

    def summary(self) -> str:
        lines = [f"Kernel: {self.kernel_name}"]
        for k, v in sorted(self.counters.items()):
            lines.append(f"  {k}: {v:,}")
        lines.append(f"  wait/MFMA ratio: {self.wait_mfma_ratio:.2f}")
        lines.append(f"  Diagnosis: {self.diagnosis}")
        return "\n".join(lines)


def parse_rocprofv3_csv(csv_path: str | Path) -> list[KernelPMC]:
    """Parse rocprofv3 CSV output into structured KernelPMC objects.

    rocprofv3 CSV format varies by version. Common layouts:
    - Header row with counter names as columns
    - One row per kernel dispatch
    - KernelName column identifies the kernel
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"PMC CSV not found: {path}")

    results = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Find kernel name column (varies: KernelName, Kernel_Name, kernel_name)
            kernel_name = ""
            for key in ("KernelName", "Kernel_Name", "kernel_name", "Name"):
                if key in row:
                    kernel_name = row[key]
                    break

            if not kernel_name:
                continue

            # Extract counter values (skip non-counter columns)
            skip_cols = {
                "KernelName", "Kernel_Name", "kernel_name", "Name",
                "gpu-id", "GPU_ID", "queue-id", "queue-pos",
                "pid", "tid", "Index", "Dispatch_ID",
                "Start_Timestamp", "End_Timestamp", "Correlation_ID",
            }
            counters = {}
            for key, val in row.items():
                if key in skip_cols:
                    continue
                try:
                    counters[key] = int(val)
                except (ValueError, TypeError):
                    try:
                        counters[key] = int(float(val))
                    except (ValueError, TypeError):
                        pass

            if counters:
                results.append(KernelPMC(kernel_name=kernel_name, counters=counters))

    return results
