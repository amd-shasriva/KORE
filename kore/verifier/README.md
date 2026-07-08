# `kore/verifier` — PMC counters & toolchain parsers

Schema + parsing for AMD GPU profiling. This package defines the **rocprofv3 performance-counter sets** that `KoreEnv` collects and parses their output (and hipcc/clang register/occupancy output) into typed objects. Collection itself lives in [`kore/env`](../env/README.md); the physics interpretation (stall/occupancy → residual) lives in [`kore/reward`](../reward/README.md) and [`kore/analysis`](../analysis/README.md).

---

## Files

| File | Purpose |
| --- | --- |
| `pmc.py` | Named rocprofv3 counter sets (`standard` / `full` / `memory` / `compute`) |
| `parsers/rocprofv3.py` | `KernelPMC` + `parse_rocprofv3_csv` (both LONG and WIDE layouts) |
| `parsers/compiler_output.py` | hipcc/clang register + occupancy heuristics (CDNA3/CDNA4) |

---

## Counter sets

```python
COUNTER_SETS = {
  "standard": ["SQ_INSTS_VALU_MFMA_BF16", "SQ_INSTS_VMEM", "SQ_WAIT_INST_LDS", "SQ_WAIT_INST_ANY"],
  "full":     [...VALU/MFMA/VMEM/SALU + SQ_WAIT_INST_{LDS,VMEM,ANY}],
  "memory":   ["SQ_INSTS_VMEM", "SQ_WAIT_INST_VMEM", "TCP_TCC_READ_REQ_sum", "TCP_TCC_WRITE_REQ_sum"],
  "compute":  ["SQ_INSTS_VALU", "SQ_INSTS_VALU_MFMA_{BF16,F16,F32}", "SQ_INSTS_SALU"],
}
```

`KoreEnv._collect_profile` uses `COUNTER_SETS["full"]`. The **same SQ counter names** apply to gfx942 (CDNA3) and gfx950 (CDNA4); architecture-specific interpretation happens downstream. (On gfx950 the P0 study uses rocprofv3 *derived metrics* `OccupancyPercent`, `MemUnitStalled`, `MfmaUtil` because the raw counters were renamed.)

---

## Parsing

```python
@dataclass
class KernelPMC:
    kernel_name: str
    counters: dict[str, int]
    # derived: mfma_count, vmem_count, wait_any, wait_mfma_ratio, diagnosis

def parse_rocprofv3_csv(csv_path) -> list[KernelPMC]
```

Handles **both** CSV layouts rocprofv3 emits:

| Layout | Detection | Handling |
| --- | --- | --- |
| LONG (rocprofv3 1.x `*_counter_collection.csv`) | has `Counter_Name` + `Counter_Value` | fold rows per `(Dispatch_Id, Kernel_Name)` |
| WIDE (older) | counters as columns | one row per dispatch; skip metadata columns |

`wait_mfma_ratio`: `<5` compute-bound, `5–10` balanced, `>10` memory-bound.

**Compiler output** (`parsers/compiler_output.py`): `parse_register_info` extracts VGPR/AGPR/SGPR/LDS/spill/occupancy. Heuristics (gfx942/gfx950): VGPR ≤ 256 → occupancy ≥ 2 possible; LDS ≤ 80 KB → dual-occupancy OK.

See also: [`env`](../env/README.md) (collection), [`reward/profile_reward`](../reward/README.md), [`analysis`](../analysis/README.md).
