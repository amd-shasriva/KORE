# `kore/verifier` — PMC counters & toolchain parsers

Schema and parsing for AMD GPU profiling. This package defines the **rocprofv3 performance-counter sets** that `KoreEnv` collects, the pure CPU-testable formulas that turn raw counters into the bottleneck-grounding metrics KORE reasons about (L2 hit rate, HBM bytes, occupancy), and the parsers that decode rocprofv3 CSV and hipcc/clang register output into typed objects. Counter collection lives in [`kore/env`](../env/README.md); the physics interpretation (stall / occupancy → residual) lives in [`kore/reward`](../reward/README.md) and [`kore/analysis`](../analysis/README.md).

The target is **gfx950 / CDNA4 (MI350X / MI355X)** by default, with **gfx942 / CDNA3 (MI300X)** supported. The `SQ_*` / `GRBM_*` / `TCC_*` counter family is shared across both architectures; architecture-specific interpretation happens downstream.

---

## Files

| File | Purpose |
| --- | --- |
| `pmc.py` | Named counter sets (`standard` / `full` / `memory` / `compute` / `grounding`), multi-pass grouping, derived metrics, and the arch-selected occupancy model |
| `parsers/rocprofv3.py` | `KernelPMC` + `parse_rocprofv3_csv` (both LONG and WIDE layouts) |
| `parsers/compiler_output.py` | hipcc/clang register + occupancy parsing (CDNA3/CDNA4) |

---

## Counter sets

```python
COUNTER_SETS = {
  "standard":  ["SQ_INSTS_VALU_MFMA_BF16", "SQ_INSTS_VMEM", "SQ_WAIT_INST_LDS", "SQ_WAIT_INST_ANY"],
  "full":      [...VALU/MFMA_{BF16,F16}/VMEM/SALU + SQ_WAIT_INST_{LDS,VMEM,ANY}],   # 8 SQ counters
  "memory":    ["SQ_INSTS_VMEM", "SQ_WAIT_INST_VMEM", "TCP_TCC_READ_REQ_sum", "TCP_TCC_WRITE_REQ_sum"],
  "compute":   ["SQ_INSTS_VALU", "SQ_INSTS_VALU_MFMA_{BF16,F16,F32}", "SQ_INSTS_SALU"],
  "grounding": [...SQ + GRBM + TCC roofline-bottleneck counters...],
}
```

`KoreEnv._collect_profile` collects `COUNTER_SETS["full"]` in a single hardware pass and drives the dense profiler reward. rocprofv3 fails a `--pmc` job if the whole set cannot be scheduled in one pass (limited counter slots per block), so the larger `grounding` set — which spans SQ + GRBM + TCC and grounds the roofline residual (waves, active/busy cycles, MFMA op mix, L2 hit/miss, EA↔HBM traffic) — is **multi-pass**: collect it with `counter_passes("grounding")` / `GROUNDING_PASSES` (one `--pmc` invocation per pass) and merge the per-pass `{counter: value}` dicts, which share no keys.

The gfx950/CDNA4 low-precision MFMA op counters (`SQ_INSTS_VALU_MFMA_MOPS_{F8,F6F4,XF32}`, for OCP-FP8 / MXFP6 / MXFP4 / XF32 matrix ops) exist **only** on gfx950 and are collected in their own pass; on a gfx942 node that pass yields no CSV and is skipped, so collection is arch-safe. `COUNTER_META` documents the human meaning and unit of every counter KORE requests.

---

## Derived metrics (pure, CPU-testable)

| Function | Meaning |
| --- | --- |
| `l2_hit_rate(counters)` | `TCC_HIT / (TCC_HIT + TCC_MISS)` — the ROCm "L2 Cache Hit Rate" definition |
| `hbm_read_bytes` / `hbm_write_bytes` / `hbm_bytes` | HBM bytes moved from EA (L2↔HBM) request counts, using the exact 32B/64B split (`FetchSize`/`WriteSize` formula) when present, else a 64B/request upper bound |
| `est_occupancy(vgpr, lds, num_warps)` | Resource-limited waves/SIMD with the binding `limiter` (`vgpr` / `lds` / `wave_slots`) |
| `mfma_ops` / `mfma_busy_fraction` | Matrix-core activity (`SQ_VALU_MFMA_BUSY_CYCLES / GRBM_GUI_ACTIVE`) |

The occupancy model is arch-selected (`_occupancy_constants`, cited on-device values): CDNA4/gfx950 defaults to 160 KiB LDS/CU, VGPR alloc granularity 8, 512 VGPR/SIMD, and 8 waves/SIMD; CDNA3/gfx942 selects the 64 KiB-LDS / granularity-16 limits. The register file is shared by architected and MFMA-accumulator VGPRs, so occupancy is driven by their sum.

---

## Parsing

```python
@dataclass
class KernelPMC:
    kernel_name: str
    counters: dict[str, int]
    # per-dispatch resource metadata: vgpr_count, accum_vgpr_count, sgpr_count,
    #                                 lds_bytes, scratch_size, workgroup_size
    # derived: total_vgpr, num_warps, mfma_count, vmem_count, wait_any,
    #          wait_mfma_ratio, diagnosis

def parse_rocprofv3_csv(csv_path) -> list[KernelPMC]
```

`parse_rocprofv3_csv` handles **both** CSV layouts rocprofv3 emits across versions:

| Layout | Detection | Handling |
| --- | --- | --- |
| LONG (rocprofv3 1.x `*_counter_collection.csv`) | has `Counter_Name` + `Counter_Value` | fold rows per `(Dispatch_Id, Kernel_Name)` |
| WIDE (older exports) | counters as columns | one row per dispatch; skip identity/metadata columns |

Per-dispatch resource columns (VGPR/AGPR/SGPR/LDS/scratch/workgroup size, under the naming variants seen across ROCm versions) are parsed into typed fields rather than the `counters` dict, so occupancy and register-pressure reasoning is available without disturbing counter consumers. Timestamps are dropped (wall time comes from the verifier's own timing). Counter lookup is alias-tolerant (`TCC_HIT_sum` vs `TCC_HIT`, `TCC_EA0_*` vs `TCC_EA_*`), returning `None` when a family is absent so a caller can distinguish "0" from "not collected". `wait_mfma_ratio`: `<5` compute-bound, `5–10` balanced, `>10` memory-bound.

**Compiler output** (`parsers/compiler_output.py`): `parse_register_info` extracts VGPR/AGPR/SGPR/LDS/spill/occupancy from hipcc/clang verbose output or an ISA dump; `parse_compiler_errors` / `parse_compiler_warnings` pull diagnostic lines. Occupancy heuristic (gfx942/gfx950): VGPR ≤ 256 → occupancy ≥ 2 possible; LDS ≤ 80 KB → dual-occupancy OK.

See also: [`env`](../env/README.md) (collection), [`reward`](../reward/README.md) (`profile_reward`), [`analysis`](../analysis/README.md).
