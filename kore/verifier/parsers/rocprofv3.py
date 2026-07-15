"""Parser for rocprofv3 CSV output (gfx942 / MI300X).

Besides the hardware ``counters`` dict, rocprofv3 emits per-dispatch kernel
resource metadata (VGPR/SGPR/AccumVGPR usage, LDS block size, scratch size).
These are NOT hardware counters but they are exactly what occupancy /
register-pressure reasoning needs, so we parse them into typed fields on
:class:`KernelPMC` instead of discarding them. Timestamps are still dropped (a
kernel's wall time comes from the verifier's own timing, not the profiler).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class KernelPMC:
    """PMC counter data + kernel resource usage for a single kernel dispatch."""

    kernel_name: str
    counters: dict[str, int] = field(default_factory=dict)
    # --- per-dispatch kernel resource metadata (rocprofv3 kernel-dispatch cols) --
    # Kept out of ``counters`` (they are not accumulating HW counters) so existing
    # counter consumers are unaffected; enables occupancy / register-pressure reasoning.
    vgpr_count: Optional[int] = None        # architected VGPRs / lane (VGPR_Count)
    accum_vgpr_count: Optional[int] = None  # MFMA accumulator VGPRs / lane (Accum_VGPR_Count)
    sgpr_count: Optional[int] = None        # SGPRs / wave (SGPR_Count)
    lds_bytes: Optional[int] = None         # LDS bytes / workgroup (LDS_Block_Size)
    scratch_size: Optional[int] = None      # private scratch bytes / work-item (Scratch_Size)
    workgroup_size: Optional[int] = None    # threads / workgroup (Workgroup_Size); enables num_warps

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
    def total_vgpr(self) -> Optional[int]:
        """Total VGPRs/lane (architected + MFMA accumulator) - the occupancy input.

        On CDNA3 the register file is shared by regular and accumulator VGPRs, so
        occupancy is driven by their sum. None if neither was reported.
        """
        if self.vgpr_count is None and self.accum_vgpr_count is None:
            return None
        return (self.vgpr_count or 0) + (self.accum_vgpr_count or 0)

    @property
    def num_warps(self) -> Optional[int]:
        """Wavefronts per workgroup (``ceil(workgroup_size / 64)``) - the occupancy
        input. None if the workgroup size was not reported. CDNA is wave64."""
        if not self.workgroup_size or self.workgroup_size <= 0:
            return None
        return (self.workgroup_size + 63) // 64

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
        if self.total_vgpr is not None:
            lines.append(f"  VGPRs (arch+acc): {self.total_vgpr}"
                         f" (arch={self.vgpr_count}, acc={self.accum_vgpr_count})")
        if self.sgpr_count is not None:
            lines.append(f"  SGPRs: {self.sgpr_count}")
        if self.lds_bytes is not None:
            lines.append(f"  LDS bytes/workgroup: {self.lds_bytes:,}")
        if self.scratch_size is not None:
            lines.append(f"  Scratch bytes/work-item: {self.scratch_size:,}")
        if self.workgroup_size is not None:
            lines.append(f"  Workgroup size: {self.workgroup_size} ({self.num_warps} waves)")
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


# Kernel resource columns rocprofv3 emits per dispatch, with the naming variants
# seen across ROCm versions. Kept OUT of the counters dict; parsed into fields.
_RESOURCE_COLS: dict[str, tuple[str, ...]] = {
    "vgpr_count": ("VGPR_Count", "Arch_VGPR_Count", "VGPRs", "vgpr_count"),
    "accum_vgpr_count": ("Accum_VGPR_Count", "AccumVGPR_Count", "AGPR_Count",
                         "accum_vgpr_count"),
    "sgpr_count": ("SGPR_Count", "SGPRs", "sgpr_count"),
    "lds_bytes": ("LDS_Block_Size", "LDS_Allocation", "LDS_Size", "Group_Segment_Size",
                  "lds_block_size"),
    "scratch_size": ("Scratch_Size", "Private_Segment_Size", "Scratch_Memory_Size",
                     "scratch_size"),
    "workgroup_size": ("Workgroup_Size", "workgroup_size", "Work_Group_Size"),
}

# Timestamp columns are intentionally ignored (wall time comes from the verifier).
_TIMESTAMP_COLS = {"Start_Timestamp", "End_Timestamp", "start_timestamp", "end_timestamp"}

# Non-counter identity/metadata columns skipped in the WIDE layout. Superset of the
# resource + timestamp columns so they never leak into ``counters``.
_SKIP_COLS = {
    "KernelName", "Kernel_Name", "kernel_name", "Name",
    "gpu-id", "GPU_ID", "queue-id", "queue-pos", "Queue_Id",
    "pid", "tid", "Pid", "Tid", "Process_Id", "Thread_Id", "Index",
    "Dispatch_ID", "Dispatch_Id", "Correlation_Id", "Correlation_ID",
    "Agent_Id", "Grid_Size", "Kernel_Id", "Workgroup_Size",
}
for _names in _RESOURCE_COLS.values():
    _SKIP_COLS.update(_names)
_SKIP_COLS.update(_TIMESTAMP_COLS)


def _resource_fields(row: dict) -> dict[str, int]:
    """Extract present kernel-resource columns from a row as {field: int}."""
    out: dict[str, int] = {}
    for field_name, aliases in _RESOURCE_COLS.items():
        for col in aliases:
            if col in row and row[col] not in (None, ""):
                iv = _to_int(row[col])
                if iv is not None:
                    out[field_name] = iv
                    break
    return out


def _apply_resources(pmc: KernelPMC, row: dict) -> None:
    """Fill any still-unset resource fields on ``pmc`` from ``row`` (idempotent).

    In LONG layout the resource columns repeat on every (dispatch, counter) row, so
    we set each field once from the first row that carries it.
    """
    for field_name, value in _resource_fields(row).items():
        if getattr(pmc, field_name) is None:
            setattr(pmc, field_name, value)


def parse_rocprofv3_csv(csv_path: str | Path) -> list[KernelPMC]:
    """Parse rocprofv3 CSV output into structured KernelPMC objects.

    Handles BOTH layouts rocprofv3 emits across versions:

    * LONG (rocprofv3 1.x ``*_counter_collection.csv``): one row per
      (dispatch, counter) with ``Kernel_Name`` / ``Counter_Name`` /
      ``Counter_Value`` (+ repeated per-dispatch resource columns). Rows are
      grouped by dispatch (``Dispatch_Id`` + ``Kernel_Name``) and the counters
      folded into one KernelPMC per dispatch; VGPR/SGPR/LDS/scratch are captured.
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
            _apply_resources(pmc, row)
        return [grouped[k] for k in order if grouped[k].counters]

    # WIDE format: counters as columns.
    results = []
    for row in rows:
        kernel_name = _kernel_name(row)
        if not kernel_name:
            continue
        counters = {}
        for key, val in row.items():
            if key in _SKIP_COLS:
                continue
            iv = _to_int(val)
            if iv is not None:
                counters[key] = iv
        if counters:
            pmc = KernelPMC(kernel_name=kernel_name, counters=counters)
            _apply_resources(pmc, row)
            results.append(pmc)
    return results
