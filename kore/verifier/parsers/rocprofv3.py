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


def _kernel_name(row: dict) -> str:
    for key in ("KernelName", "Kernel_Name", "kernel_name", "Name"):
        if key in row and row[key]:
            return row[key]
    return ""


def _to_int(val) -> int | None:
    try:
        return int(val)
    except (ValueError, TypeError):
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return None


def parse_rocprofv3_csv(csv_path: str | Path) -> list[KernelPMC]:
    """Parse rocprofv3 CSV output into structured KernelPMC objects.

    Handles BOTH layouts rocprofv3 emits across versions:

    * LONG (rocprofv3 1.x ``*_counter_collection.csv``): one row per
      (dispatch, counter) with ``Kernel_Name`` / ``Counter_Name`` /
      ``Counter_Value``. Rows are grouped by dispatch (``Dispatch_Id`` +
      ``Kernel_Name``) and the counters folded into one KernelPMC per dispatch.
    * WIDE (older/other exports): counter names are columns, one row per dispatch.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"PMC CSV not found: {path}")

    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return []

    cols = set(rows[0].keys())

    # LONG format: explicit Counter_Name / Counter_Value columns.
    if {"Counter_Name", "Counter_Value"} <= cols:
        grouped: dict[tuple, KernelPMC] = {}
        order: list[tuple] = []
        for row in rows:
            kname = _kernel_name(row)
            cname = row.get("Counter_Name")
            if not kname or not cname:
                continue
            cval = _to_int(row.get("Counter_Value"))
            if cval is None:
                continue
            key = (row.get("Dispatch_Id") or row.get("Dispatch_ID") or "", kname)
            pmc = grouped.get(key)
            if pmc is None:
                pmc = KernelPMC(kernel_name=kname, counters={})
                grouped[key] = pmc
                order.append(key)
            pmc.counters[cname] = pmc.counters.get(cname, 0) + cval
        return [grouped[k] for k in order if grouped[k].counters]

    # WIDE format: counters as columns.
    skip_cols = {
        "KernelName", "Kernel_Name", "kernel_name", "Name",
        "gpu-id", "GPU_ID", "queue-id", "queue-pos", "Queue_Id",
        "pid", "tid", "Process_Id", "Thread_Id", "Index",
        "Dispatch_ID", "Dispatch_Id", "Correlation_Id", "Correlation_ID",
        "Agent_Id", "Grid_Size", "Kernel_Id", "Workgroup_Size",
        "LDS_Block_Size", "Scratch_Size", "VGPR_Count", "Accum_VGPR_Count",
        "SGPR_Count", "Start_Timestamp", "End_Timestamp",
    }
    results = []
    for row in rows:
        kernel_name = _kernel_name(row)
        if not kernel_name:
            continue
        counters = {}
        for key, val in row.items():
            if key in skip_cols:
                continue
            iv = _to_int(val)
            if iv is not None:
                counters[key] = iv
        if counters:
            results.append(KernelPMC(kernel_name=kernel_name, counters=counters))
    return results
