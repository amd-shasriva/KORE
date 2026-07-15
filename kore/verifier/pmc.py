"""PMC counter sets + derived-metric helpers for rocprofv3 collection (gfx950/gfx942).

The actual collection lives in ``kore.env.kore_env.KoreEnv._collect_profile`` /
``KoreEnv.collect_counters`` (candidate-vs-reference, fail-safe, wired into the
dense profiler reward). This module defines the named counter sets it draws from
and the pure, CPU-testable formulas that turn raw rocprofv3 counters into the
bottleneck-grounding metrics KORE reasons about (L2 hit-rate, HBM bytes,
occupancy).

The KORE target is **gfx950 / CDNA4 (AMD Instinct MI350X / MI355X)**; the occupancy
constants are arch-selected (``_occupancy_constants``) and default to CDNA4 (160 KiB
LDS, VGPR granularity 8), with the CDNA3/gfx942 (MI300X) legacy limits available. The
``SQ_*``/``GRBM_*``/``TCC_*`` counter names follow the ROCm "MI300 and MI200 series
performance counters and metrics" reference (the SQ family is shared on CDNA4; the
gfx950-only low-precision MFMA op counters live in a separate collection pass):
    https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300-mi200-performance-counters.html

Two counter families matter for gfx942 correctness (see ``COUNTER_META``):

* **Raw hardware counters** (``SQ_*``, ``GRBM_*``, ``TCC_*``) - what ``--pmc``
  collects. TCC counters are per-L2-channel and indexed ``[0-31]``; the ``_sum``
  suffix (e.g. ``TCC_HIT_sum``) is the ROCm-provided aggregate across channels.
* **MFMA op counts** - gfx942 exposes the FLOP-weighted matrix throughput as
  ``SQ_INSTS_VALU_MFMA_MOPS_{BF16,F16,F32,F64,I8}`` ("ops in the unit of 512"),
  which - unlike the issue-count family ``SQ_INSTS_VALU_MFMA_{F16,F32,F64,I8}`` -
  DOES have a BF16 member. There is direct MFMA-busy timing via
  ``SQ_VALU_MFMA_BUSY_CYCLES``.

.. note::
   rocprofv3 fails a ``--pmc`` job if the whole set cannot be collected in a
   single hardware pass (limited counter slots per block). ``COUNTER_SETS["full"]``
   (8 SQ counters) is single-pass. The larger ``COUNTER_SETS["grounding"]`` spans
   SQ + GRBM + TCC and MUST be collected across multiple passes: use
   ``GROUNDING_PASSES`` / :func:`counter_passes` (one ``--pmc`` invocation per pass)
   and merge the resulting ``{counter: value}`` dicts (they share no keys).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from kore.config import CONFIG


# --------------------------------------------------------------------------- #
# Named counter sets.
#
# "standard"/"full"/"memory"/"compute" are UNCHANGED (kept working for the
# existing ``_collect_profile`` single-pass collection + profile_reward). Only
# "grounding" is new.
# --------------------------------------------------------------------------- #
COUNTER_SETS = {
    "standard": [
        "SQ_INSTS_VALU_MFMA_BF16",
        "SQ_INSTS_VMEM",
        "SQ_WAIT_INST_LDS",
        "SQ_WAIT_INST_ANY",
    ],
    "full": [
        "SQ_INSTS_VALU",
        "SQ_INSTS_VALU_MFMA_BF16",
        "SQ_INSTS_VALU_MFMA_F16",
        "SQ_INSTS_VMEM",
        "SQ_INSTS_SALU",
        "SQ_WAIT_INST_LDS",
        "SQ_WAIT_INST_VMEM",
        "SQ_WAIT_INST_ANY",
    ],
    "memory": [
        "SQ_INSTS_VMEM",
        "SQ_WAIT_INST_VMEM",
        "TCP_TCC_READ_REQ_sum",
        "TCP_TCC_WRITE_REQ_sum",
    ],
    "compute": [
        "SQ_INSTS_VALU",
        "SQ_INSTS_VALU_MFMA_BF16",
        "SQ_INSTS_VALU_MFMA_F16",
        "SQ_INSTS_VALU_MFMA_F32",
        "SQ_INSTS_SALU",
    ],
    # ------------------------------------------------------------------- #
    # "grounding": the REAL bottleneck counters for CDNA3/CDNA4 roofline
    # reasoning (Pillar 4). Most names are from the ROCm MI300/MI200 counter
    # reference; the low-precision MFMA op counters (F8/F6F4/XF32) are gfx950/
    # CDNA4 additions. Spans SQ + GRBM + TCC, so it is NOT single-pass - collect
    # via GROUNDING_PASSES (see module note) and merge the per-pass dicts.
    # ------------------------------------------------------------------- #
    "grounding": [
        # --- occupancy / launch ---
        "SQ_WAVES",                     # wavefronts dispatched (occupancy proxy)
        # --- active/busy cycles (utilization + timing denominators) ---
        "GRBM_GUI_ACTIVE",              # GPU active cycles
        "GRBM_COUNT",                   # free-running GPU cycles
        "SQ_BUSY_CYCLES",               # cycles the sequencer is busy
        "SQ_VALU_MFMA_BUSY_CYCLES",     # cycles the MFMA (matrix) ALU is busy
        # --- issued-instruction mix (work proxy) ---
        "SQ_INSTS_VALU",
        "SQ_INSTS_VMEM",
        "SQ_INSTS_SALU",
        "SQ_INSTS_LDS",
        # --- MFMA throughput (FLOP-weighted MOPS family). BF16/F16/F32 exist on
        #     all CDNA; F8 (OCP-FP8), F6F4 (MXFP6/MXFP4) and XF32 are gfx950/CDNA4
        #     ONLY and are required to ground MI350 low-precision matrix kernels. ---
        "SQ_INSTS_VALU_MFMA_MOPS_BF16",
        "SQ_INSTS_VALU_MFMA_MOPS_F16",
        "SQ_INSTS_VALU_MFMA_MOPS_F32",
        "SQ_INSTS_VALU_MFMA_MOPS_F8",      # gfx950: OCP-FP8 matrix ops
        "SQ_INSTS_VALU_MFMA_MOPS_F6F4",    # gfx950: MXFP6 / MXFP4 matrix ops
        "SQ_INSTS_VALU_MFMA_MOPS_XF32",    # gfx950: XF32 (TF32-equiv) matrix ops
        # --- pipeline stalls / issue latency ---
        "SQ_WAIT_INST_ANY",
        "SQ_WAIT_INST_LDS",
        "SQ_ACTIVE_INST_VMEM",          # quad-cycles working on VMEM instrs
        # --- L2 (TCC) cache hit/miss -> l2_hit_rate() ---
        "TCC_HIT_sum",
        "TCC_MISS_sum",
        # --- HBM traffic via the Efficiency Arbiter (EA<->HBM), with the 32B/64B
        #     split needed for EXACT bytes -> hbm_bytes() ---
        "TCC_EA0_RDREQ_sum",
        "TCC_EA0_RDREQ_32B_sum",
        "TCC_EA0_WRREQ_sum",
        "TCC_EA0_WRREQ_64B_sum",
    ],
}

# Conservative single-pass groupings of COUNTER_SETS["grounding"] (each fits one
# rocprofv3 hardware pass; keys are disjoint so merging the per-pass dicts is a
# plain dict.update). Sizing mirrors the proven 8-SQ "full" pass. The exact
# per-block slot budget is arch/firmware dependent -> validate on-device with
# ``rocprofv3 --list-avail`` if a pass ever fails to schedule.
GROUNDING_PASSES: list[list[str]] = [
    # pass 1 - SQ issue/stall/busy + waves, plus GRBM active cycles
    [
        "SQ_WAVES", "SQ_BUSY_CYCLES", "SQ_VALU_MFMA_BUSY_CYCLES",
        "SQ_INSTS_VALU", "SQ_INSTS_VMEM", "SQ_INSTS_SALU",
        "SQ_WAIT_INST_ANY", "SQ_WAIT_INST_LDS",
        "GRBM_GUI_ACTIVE", "GRBM_COUNT",
    ],
    # pass 2 - SQ MFMA op mix + LDS insts + VMEM active cycles
    [
        "SQ_INSTS_VALU_MFMA_MOPS_BF16", "SQ_INSTS_VALU_MFMA_MOPS_F16",
        "SQ_INSTS_VALU_MFMA_MOPS_F32", "SQ_INSTS_LDS", "SQ_ACTIVE_INST_VMEM",
    ],
    # pass 3 - TCC L2 hit/miss + read traffic
    ["TCC_HIT_sum", "TCC_MISS_sum", "TCC_EA0_RDREQ_sum", "TCC_EA0_RDREQ_32B_sum"],
    # pass 4 - TCC write traffic
    ["TCC_EA0_WRREQ_sum", "TCC_EA0_WRREQ_64B_sum"],
    # pass 5 - gfx950/CDNA4 low-precision MFMA op mix (OCP-FP8 / MXFP6 / MXFP4 /
    #          XF32). These counters exist ONLY on gfx950; on a gfx942 node the
    #          pass yields no CSV and is silently skipped (collection merges the
    #          rest - see KoreEnv.collect_counters), so this is arch-safe.
    [
        "SQ_INSTS_VALU_MFMA_MOPS_F8", "SQ_INSTS_VALU_MFMA_MOPS_F6F4",
        "SQ_INSTS_VALU_MFMA_MOPS_XF32",
    ],
]


def counter_passes(name: str = "grounding") -> list[list[str]]:
    """Counter groups to collect ``COUNTER_SETS[name]`` in, one ``--pmc`` pass each.

    "grounding" is multi-pass (returns ``GROUNDING_PASSES``); every other set is
    single-pass and returned as one group. Callers profile once per group and
    merge the ``{counter: value}`` dicts.
    """
    if name == "grounding":
        return [list(p) for p in GROUNDING_PASSES]
    return [list(COUNTER_SETS[name])]


# --------------------------------------------------------------------------- #
# Counter -> (human meaning, unit). Documents every counter we request so the
# grounded-reasoning teacher/demo can name what each number measures.
# --------------------------------------------------------------------------- #
COUNTER_META: dict[str, tuple[str, str]] = {
    # SQ - instruction issue / occupancy / stalls
    "SQ_WAVES": ("Wavefronts dispatched to the SQ (new + restored)", "waves"),
    "SQ_BUSY_CYCLES": ("Cycles the sequencer reported busy", "cycles"),
    "SQ_VALU_MFMA_BUSY_CYCLES": ("Cycles the MFMA (matrix-core) ALU was busy", "cycles"),
    "SQ_INSTS_VALU": ("VALU instructions issued (includes matrix FMA)", "instr"),
    "SQ_INSTS_VMEM": ("Vector-memory instructions issued (flat + buffer)", "instr"),
    "SQ_INSTS_VMEM_RD": ("Vector-memory READ instructions issued", "instr"),
    "SQ_INSTS_VMEM_WR": ("Vector-memory WRITE instructions issued", "instr"),
    "SQ_INSTS_SALU": ("SALU (scalar ALU) instructions issued", "instr"),
    "SQ_INSTS_LDS": ("LDS instructions issued (MI300: excludes flat)", "instr"),
    "SQ_INSTS_MFMA": ("Matrix-FMA instructions issued (all dtypes)", "instr"),
    # MFMA op counts (FLOP-weighted; each unit = 512 ops)
    "SQ_INSTS_VALU_MFMA_MOPS_BF16": ("BF16 matrix-FMA ops", "ops x512"),
    "SQ_INSTS_VALU_MFMA_MOPS_F16": ("F16 matrix-FMA ops", "ops x512"),
    "SQ_INSTS_VALU_MFMA_MOPS_F32": ("F32 matrix-FMA ops", "ops x512"),
    "SQ_INSTS_VALU_MFMA_MOPS_F64": ("F64 matrix-FMA ops", "ops x512"),
    "SQ_INSTS_VALU_MFMA_MOPS_I8": ("INT8 matrix-FMA ops", "ops x512"),
    "SQ_INSTS_VALU_MFMA_MOPS_F8": ("OCP-FP8 matrix-FMA ops (gfx950/CDNA4)", "ops x512"),
    "SQ_INSTS_VALU_MFMA_MOPS_F6F4": ("MXFP6/MXFP4 matrix-FMA ops (gfx950/CDNA4)", "ops x512"),
    "SQ_INSTS_VALU_MFMA_MOPS_XF32": ("XF32 matrix-FMA ops (gfx950/CDNA4)", "ops x512"),
    # MFMA issue-count family (gfx942 has NO ..._BF16 member here)
    "SQ_INSTS_VALU_MFMA_F16": ("F16 matrix-FMA instructions issued", "instr"),
    "SQ_INSTS_VALU_MFMA_F32": ("F32 matrix-FMA instructions issued", "instr"),
    # stalls / activity (quad-cycles)
    "SQ_WAIT_INST_ANY": ("Quad-cycles waiting to issue ANY instruction", "qcycles"),
    "SQ_WAIT_INST_LDS": ("Quad-cycles waiting to issue an LDS instruction", "qcycles"),
    "SQ_ACTIVE_INST_VMEM": ("Quad-cycles the arbiter worked on a VMEM instruction", "qcycles"),
    # GRBM - device-level cycle counters
    "GRBM_GUI_ACTIVE": ("GPU active cycles (graphics/compute engine busy)", "cycles"),
    "GRBM_COUNT": ("Free-running GPU cycles", "cycles"),
    # TCC (L2) - hits/misses
    "TCC_HIT_sum": ("L2 cache hits, summed over channels", "requests"),
    "TCC_MISS_sum": ("L2 cache misses, summed over channels", "requests"),
    "TCC_REQ_sum": ("All L2 cache requests, summed over channels", "requests"),
    # TCC EA<->HBM - request counts (32B or 64B each)
    "TCC_EA0_RDREQ_sum": ("EA->HBM read requests (32B or 64B), summed", "requests"),
    "TCC_EA0_RDREQ_32B_sum": ("EA->HBM 32B read requests, summed", "requests"),
    "TCC_EA0_WRREQ_sum": ("EA->HBM write requests (32B or 64B), summed", "requests"),
    "TCC_EA0_WRREQ_64B_sum": ("EA->HBM 64B write requests, summed", "requests"),
}


# --------------------------------------------------------------------------- #
# Counter lookup with alias tolerance.
#
# rocprofv3 CSVs land under slightly different names across ROCm versions
# (``TCC_HIT_sum`` vs channel-summed ``TCC_HIT``; ``TCC_EA0_*`` vs ``TCC_EA_*``),
# and our parser already folds LONG-format channel rows by summing identical
# Counter_Name values. ``_counter`` sums the FIRST alias family that is present
# and returns None when none are, so callers can tell "0" from "absent".
# --------------------------------------------------------------------------- #
def _counter(counters: dict, *aliases: str) -> Optional[float]:
    for name in aliases:
        if name in counters and isinstance(counters[name], (int, float)):
            return float(counters[name])
    # case-insensitive fallback
    lowered = {str(k).lower(): v for k, v in counters.items()}
    for name in aliases:
        v = lowered.get(name.lower())
        if isinstance(v, (int, float)):
            return float(v)
    return None


_RD_ALIASES = ("TCC_EA0_RDREQ_sum", "TCC_EA_RDREQ_sum", "TCC_EA0_RDREQ", "TCC_EA_RDREQ")
_RD32_ALIASES = ("TCC_EA0_RDREQ_32B_sum", "TCC_EA_RDREQ_32B_sum",
                 "TCC_EA0_RDREQ_32B", "TCC_EA_RDREQ_32B")
_WR_ALIASES = ("TCC_EA0_WRREQ_sum", "TCC_EA_WRREQ_sum", "TCC_EA0_WRREQ", "TCC_EA_WRREQ")
_WR64_ALIASES = ("TCC_EA0_WRREQ_64B_sum", "TCC_EA_WRREQ_64B_sum",
                 "TCC_EA0_WRREQ_64B", "TCC_EA_WRREQ_64B")
_HIT_ALIASES = ("TCC_HIT_sum", "TCC_HIT")
_MISS_ALIASES = ("TCC_MISS_sum", "TCC_MISS")


# --------------------------------------------------------------------------- #
# Derived metric: L2 (TCC) cache hit rate.
#   hit_rate = TCC_HIT / (TCC_HIT + TCC_MISS)
# This is exactly the ROCm/rocprofiler-compute "L2 Cache Hit Rate" definition.
# --------------------------------------------------------------------------- #
def l2_hit_rate(counters: dict) -> Optional[float]:
    """L2 hit rate in [0, 1] from ``TCC_HIT`` / ``TCC_MISS`` (None if absent)."""
    hit = _counter(counters, *_HIT_ALIASES)
    miss = _counter(counters, *_MISS_ALIASES)
    if hit is None or miss is None:
        return None
    denom = hit + miss
    if denom <= 0:
        return None
    return hit / denom


# --------------------------------------------------------------------------- #
# Derived metric: HBM bytes moved, from EA (Efficiency Arbiter) request counts.
#
# The EA is the L2<->HBM interface; each request is a 32-byte OR 64-byte
# transaction. The ROCm counters give the totals plus the size-specific splits:
#   TCC_EA_RDREQ      = #reads (32B or 64B)      TCC_EA_RDREQ_32B = #32B reads
#   TCC_EA_WRREQ      = #writes (32B or 64B)     TCC_EA_WRREQ_64B = #64B writes
# so (this is the rocprofiler-compute "FetchSize"/"WriteSize" formula):
#   read_bytes  = 32*RDREQ_32B + 64*(RDREQ - RDREQ_32B)
#   write_bytes = 64*WRREQ_64B + 32*(WRREQ - WRREQ_64B)
# If the 32B/64B split is missing we fall back to 64B/request (upper bound for
# coalesced traffic) and flag the result approximate.
# --------------------------------------------------------------------------- #
def hbm_read_bytes(counters: dict) -> Optional[float]:
    rd = _counter(counters, *_RD_ALIASES)
    if rd is None:
        return None
    rd32 = _counter(counters, *_RD32_ALIASES)
    if rd32 is None:
        return 64.0 * rd  # approximate: size split unavailable
    rd32 = min(rd32, rd)
    return 32.0 * rd32 + 64.0 * (rd - rd32)


def hbm_write_bytes(counters: dict) -> Optional[float]:
    wr = _counter(counters, *_WR_ALIASES)
    if wr is None:
        return None
    wr64 = _counter(counters, *_WR64_ALIASES)
    if wr64 is None:
        return 64.0 * wr  # approximate: size split unavailable
    wr64 = min(wr64, wr)
    return 64.0 * wr64 + 32.0 * (wr - wr64)


def hbm_bytes(counters: dict) -> Optional[float]:
    """Total HBM bytes moved (read + write) from EA request counters.

    None if neither read nor write EA counters are present. Uses the exact 32B/64B
    split when available (see module math), else a 64B/request approximation.
    """
    r = hbm_read_bytes(counters)
    w = hbm_write_bytes(counters)
    if r is None and w is None:
        return None
    return (r or 0.0) + (w or 0.0)


# --------------------------------------------------------------------------- #
# Arch-selected occupancy hardware constants (CDNA4 gfx950 default; CDNA3 gfx942).
#
# Sources (cited so the numbers are defensible / auditable):
#  * CDNA4 (gfx950 / MI350X / MI355X) cheat sheet, "verified on-device": 512 VGPR/
#    lane per SIMD (regular<=256 + accumulator<=256, ONE shared file -- NOT doubled
#    on CDNA4), ~800 SGPR/SIMD, **160 KB LDS/CU**, max 8 waves/SIMD (32/CU), **VGPR
#    alloc granularity 8** (eight-Dword groups), SGPR granularity 16:
#    ROCm blog "Occupancy Math on the AMD MI355X GPU (CDNA4)" +
#    "AMD Instinct CDNA4 ISA Reference Guide" (Aug 2025) §3.6.4:
#    https://rocm.blogs.amd.com/software-tools-optimization/occupancy-math-mi355x/README.html
#  * CDNA3 (gfx942 / MI300X / MI325X): identical EXCEPT 64 KiB LDS/CU and VGPR
#    granularity 16 (blocks of 16). CDNA4 changed exactly those two limits
#    (LDS 64->160 KiB, VGPR granularity 16->8); everything else is shared.
#  NB the register file is shared by architected VGPRs and MFMA accumulator
#  (Acc)VGPRs; ``vgpr`` here is the TOTAL (regular + accumulator) per lane.
# --------------------------------------------------------------------------- #
def _occupancy_constants(arch: str) -> dict:
    """Per-arch occupancy hardware constants (see the citation block above).

    gfx950/CDNA4 is the KORE target and the default; only gfx942/gfx90a/gfx908
    (CDNA3/CDNA2) select the 64 KiB-LDS / granularity-16 legacy values.
    """
    a = (arch or "").lower()
    is_cdna3 = ("gfx942" in a or "gfx90a" in a or "gfx908" in a)
    return {
        "vgpr_per_simd": 512,                                # unchanged CDNA3->CDNA4
        "vgpr_alloc_granularity": 16 if is_cdna3 else 8,     # CDNA4: 8-Dword groups
        "sgpr_per_simd": 800,
        "sgpr_alloc_granularity": 16,
        "max_waves_per_simd": 8,                             # 8 waves/SIMD (32/CU)
        "simds_per_cu": 4,
        "lds_bytes_per_cu": 65536 if is_cdna3 else 163840,   # 64 KiB / 160 KiB
        "wavefront_size": 64,                                # CDNA is wave64
    }


_OCC = _occupancy_constants(getattr(CONFIG, "gpu_target", "gfx950"))
VGPR_PER_SIMD = _OCC["vgpr_per_simd"]              # total VGPRs per SIMD (regular + acc)
VGPR_ALLOC_GRANULARITY = _OCC["vgpr_alloc_granularity"]
SGPR_PER_SIMD = _OCC["sgpr_per_simd"]              # ~800 SGPRs/SIMD, <=102/wave
SGPR_ALLOC_GRANULARITY = _OCC["sgpr_alloc_granularity"]
MAX_WAVES_PER_SIMD = _OCC["max_waves_per_simd"]    # instruction-buffer slots
SIMDS_PER_CU = _OCC["simds_per_cu"]
LDS_BYTES_PER_CU = _OCC["lds_bytes_per_cu"]
WAVEFRONT_SIZE = _OCC["wavefront_size"]


def _ceil_mult(x: float, mult: int) -> int:
    return int(math.ceil(x / mult) * mult)


@dataclass
class Occupancy:
    """Result of :func:`est_occupancy` (all "per-CU"/"per-SIMD" in the arch-selected
    units -- CDNA4/gfx950 by default; see ``_occupancy_constants``)."""
    waves_per_simd: float          # achieved wavefronts per SIMD
    occupancy: float               # waves_per_simd / MAX_WAVES_PER_SIMD, in [0,1]
    limiter: str                   # "vgpr" | "lds" | "wave_slots" | "none"
    workgroups_per_cu: int
    waves_by_vgpr_per_simd: int    # VGPR-limited waves/SIMD (before packing)
    wg_by_vgpr: int                # workgroups/CU allowed by VGPRs
    wg_by_lds: int                 # workgroups/CU allowed by LDS
    wg_by_wave_slots: int          # workgroups/CU allowed by the 8-wave/SIMD cap
    num_warps: int                 # wavefronts per workgroup used in the calc

    def as_dict(self) -> dict:
        return {
            "waves_per_simd": self.waves_per_simd,
            "occupancy": self.occupancy,
            "limiter": self.limiter,
            "workgroups_per_cu": self.workgroups_per_cu,
            "waves_by_vgpr_per_simd": self.waves_by_vgpr_per_simd,
            "num_warps": self.num_warps,
        }


def waves_per_simd_from_vgpr(vgpr: Optional[int]) -> int:
    """VGPR-limited wavefronts per SIMD (arch-selected granularity).

    ``floor(512 / roundup(vgpr, VGPR_ALLOC_GRANULARITY))`` capped at 8. On CDNA4/
    gfx950 the granularity is 8 (ROCm worked example: 100 VGPRs -> 104 -> floor(512/
    104) = 4 waves/SIMD); on CDNA3/gfx942 it is 16. ``vgpr`` is the total (regular +
    accumulator) VGPRs per lane; 0/None means "not VGPR-limited".
    """
    if not vgpr or vgpr <= 0:
        return MAX_WAVES_PER_SIMD
    alloc = _ceil_mult(vgpr, VGPR_ALLOC_GRANULARITY)
    if alloc <= 0:
        return MAX_WAVES_PER_SIMD
    return max(0, min(MAX_WAVES_PER_SIMD, VGPR_PER_SIMD // alloc))


def est_occupancy(vgpr: Optional[int] = None, lds: Optional[int] = None,
                  num_warps: Optional[int] = None) -> Occupancy:
    """Estimate achieved occupancy (waves/SIMD); arch-selected limits (CDNA4/gfx950
    default -- 160 KiB LDS, VGPR granularity 8; CDNA3/gfx942 -- 64 KiB, granularity 16).

    Args:
        vgpr: total VGPRs per lane the kernel uses (regular + accumulator).
        lds: LDS (shared memory) bytes per workgroup.
        num_warps: wavefronts per workgroup (Triton ``num_warps``; default 4).

    Implements the standard AMD resource-limited occupancy calc (see the constants
    block for citations). All resource limits are reduced to workgroups/CU, the
    minimum is the achieved count, then converted back to waves/SIMD::

        occ_vgpr   = floor(512 / roundup(vgpr, GRAN))             # waves/SIMD
        wg_vgpr    = floor(occ_vgpr * SIMDS_PER_CU / num_warps)    # workgroups/CU
        wg_lds     = floor(LDS_BYTES_PER_CU / lds)                # workgroups/CU
        wg_slots   = floor(8 * SIMDS_PER_CU / num_warps)          # workgroups/CU
        workgroups = min(wg_vgpr, wg_lds, wg_slots)
        waves/SIMD = workgroups * num_warps / SIMDS_PER_CU
    """
    nW = int(num_warps) if (num_warps and num_warps > 0) else 4

    occ_vgpr = waves_per_simd_from_vgpr(vgpr)
    wg_by_vgpr = (occ_vgpr * SIMDS_PER_CU) // nW

    if lds and lds > 0:
        wg_by_lds = LDS_BYTES_PER_CU // int(lds)
    else:
        wg_by_lds = (MAX_WAVES_PER_SIMD * SIMDS_PER_CU) // nW  # effectively unbounded

    wg_by_wave_slots = (MAX_WAVES_PER_SIMD * SIMDS_PER_CU) // nW

    workgroups = max(0, min(wg_by_vgpr, wg_by_lds, wg_by_wave_slots))
    waves_per_simd = workgroups * nW / SIMDS_PER_CU
    occupancy = waves_per_simd / MAX_WAVES_PER_SIMD

    # Which resource bound the result (ties resolved vgpr > lds > wave_slots).
    limiter = "none"
    if workgroups == 0:
        limiter = "vgpr" if wg_by_vgpr == 0 else ("lds" if wg_by_lds == 0 else "wave_slots")
    else:
        binder = min(wg_by_vgpr, wg_by_lds, wg_by_wave_slots)
        if wg_by_vgpr == binder and occ_vgpr < MAX_WAVES_PER_SIMD:
            limiter = "vgpr"
        elif wg_by_lds == binder and (lds and lds > 0):
            limiter = "lds"
        elif wg_by_wave_slots == binder:
            limiter = "wave_slots"

    return Occupancy(
        waves_per_simd=waves_per_simd,
        occupancy=occupancy,
        limiter=limiter,
        workgroups_per_cu=workgroups,
        waves_by_vgpr_per_simd=occ_vgpr,
        wg_by_vgpr=wg_by_vgpr,
        wg_by_lds=wg_by_lds,
        wg_by_wave_slots=wg_by_wave_slots,
        num_warps=nW,
    )


# --------------------------------------------------------------------------- #
# MFMA (matrix-core) activity helpers. Match ANY MFMA counter family by
# substring so both the FLOP-weighted MOPS names and the legacy issue-count
# names (and the historical ..._BF16 alias used elsewhere in KORE) are counted.
# --------------------------------------------------------------------------- #
def mfma_ops(counters: dict) -> int:
    """Sum of every MFMA counter present (any family, any dtype)."""
    return sum(int(v) for k, v in counters.items()
               if "MFMA" in str(k).upper() and isinstance(v, (int, float)))


def mfma_busy_fraction(counters: dict) -> Optional[float]:
    """Fraction of active GPU cycles the MFMA ALU was busy, in [0, 1].

    ``SQ_VALU_MFMA_BUSY_CYCLES / GRBM_GUI_ACTIVE`` - the most direct matrix-core
    utilization signal on gfx942. None if either counter is absent.
    """
    busy = _counter(counters, "SQ_VALU_MFMA_BUSY_CYCLES")
    active = _counter(counters, "GRBM_GUI_ACTIVE", "SQ_BUSY_CYCLES")
    if busy is None or active is None or active <= 0:
        return None
    return max(0.0, min(1.0, busy / active))


__all__ = [
    "COUNTER_SETS",
    "GROUNDING_PASSES",
    "counter_passes",
    "COUNTER_META",
    "l2_hit_rate",
    "hbm_bytes",
    "hbm_read_bytes",
    "hbm_write_bytes",
    "est_occupancy",
    "waves_per_simd_from_vgpr",
    "mfma_ops",
    "mfma_busy_fraction",
    "Occupancy",
    "VGPR_PER_SIMD",
    "VGPR_ALLOC_GRANULARITY",
    "MAX_WAVES_PER_SIMD",
    "SIMDS_PER_CU",
    "LDS_BYTES_PER_CU",
]
