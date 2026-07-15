"""CPU-only tests for the gfx942 PMC counter sets, derived-metric helpers, and the
rocprofv3 parser's register/LDS/scratch capture.

These pin the load-bearing gfx942 facts (counter names, the AMD occupancy formula,
the EA->HBM byte formula, the L2 hit-rate) so a rename or a formula regression is
caught on CPU, before any GPU run.
"""

from __future__ import annotations

import pytest

from kore.verifier import pmc
from kore.verifier.parsers.rocprofv3 import KernelPMC, parse_rocprofv3_csv


# --------------------------------------------------------------------------- #
# counter sets
# --------------------------------------------------------------------------- #
def test_full_set_unchanged_backcompat():
    # _collect_profile / profile_reward rely on "full" verbatim; must not drift.
    assert pmc.COUNTER_SETS["full"] == [
        "SQ_INSTS_VALU",
        "SQ_INSTS_VALU_MFMA_BF16",
        "SQ_INSTS_VALU_MFMA_F16",
        "SQ_INSTS_VMEM",
        "SQ_INSTS_SALU",
        "SQ_WAIT_INST_LDS",
        "SQ_WAIT_INST_VMEM",
        "SQ_WAIT_INST_ANY",
    ]
    # the other legacy sets still exist
    for k in ("standard", "memory", "compute"):
        assert k in pmc.COUNTER_SETS and pmc.COUNTER_SETS[k]


def test_grounding_set_has_real_gfx942_bottleneck_counters():
    g = pmc.COUNTER_SETS["grounding"]
    assert isinstance(g, list) and all(isinstance(x, str) for x in g)
    # memory traffic (EA<->HBM) with the 32B/64B split needed for exact bytes
    for c in ("TCC_EA0_RDREQ_sum", "TCC_EA0_RDREQ_32B_sum",
              "TCC_EA0_WRREQ_sum", "TCC_EA0_WRREQ_64B_sum"):
        assert c in g
    # L2 hit rate, waves/occupancy, active cycles, MFMA busy
    for c in ("TCC_HIT_sum", "TCC_MISS_sum", "SQ_WAVES",
              "GRBM_GUI_ACTIVE", "GRBM_COUNT", "SQ_VALU_MFMA_BUSY_CYCLES"):
        assert c in g
    # gfx942-correct MFMA op counters use the MOPS family (which HAS a BF16 member)
    assert "SQ_INSTS_VALU_MFMA_MOPS_BF16" in g
    # keeps the existing issue/wait signal
    assert "SQ_INSTS_VALU" in g and "SQ_WAIT_INST_ANY" in g
    # no accidental duplicates
    assert len(g) == len(set(g))


def test_counter_meta_documents_every_grounding_counter():
    for c in pmc.COUNTER_SETS["grounding"]:
        assert c in pmc.COUNTER_META, f"{c} missing meaning/unit"
        meaning, unit = pmc.COUNTER_META[c]
        assert meaning and unit


def test_grounding_passes_partition_the_set_and_are_single_pass():
    passes = pmc.GROUNDING_PASSES
    # each pass fits one hw pass (conservative: <=8 SQ-ish counters per group)
    assert all(1 <= len(p) <= 10 for p in passes)
    flat = [c for p in passes for c in p]
    # disjoint (merging per-pass dicts never collides) and cover the whole set
    assert len(flat) == len(set(flat))
    assert set(flat) == set(pmc.COUNTER_SETS["grounding"])


def test_counter_passes_helper():
    assert pmc.counter_passes("grounding") == pmc.GROUNDING_PASSES
    assert len(pmc.counter_passes("grounding")) > 1          # multi-pass
    assert pmc.counter_passes("full") == [pmc.COUNTER_SETS["full"]]  # single pass


# --------------------------------------------------------------------------- #
# L2 hit rate
# --------------------------------------------------------------------------- #
def test_l2_hit_rate():
    assert pmc.l2_hit_rate({"TCC_HIT_sum": 900, "TCC_MISS_sum": 100}) == pytest.approx(0.9)
    # alias tolerance: channel-summed raw names
    assert pmc.l2_hit_rate({"TCC_HIT": 3, "TCC_MISS": 1}) == pytest.approx(0.75)
    assert pmc.l2_hit_rate({}) is None
    assert pmc.l2_hit_rate({"TCC_HIT_sum": 0, "TCC_MISS_sum": 0}) is None


# --------------------------------------------------------------------------- #
# HBM bytes from EA request counts (rocprofiler-compute FetchSize/WriteSize)
# --------------------------------------------------------------------------- #
def test_hbm_bytes_exact_with_32b_64b_split():
    c = {
        "TCC_EA0_RDREQ_sum": 1000, "TCC_EA0_RDREQ_32B_sum": 200,
        "TCC_EA0_WRREQ_sum": 500, "TCC_EA0_WRREQ_64B_sum": 100,
    }
    # reads: 200*32 + 800*64 = 6400 + 51200 = 57600
    assert pmc.hbm_read_bytes(c) == pytest.approx(57600)
    # writes: 100*64 + 400*32 = 6400 + 12800 = 19200
    assert pmc.hbm_write_bytes(c) == pytest.approx(19200)
    assert pmc.hbm_bytes(c) == pytest.approx(76800)


def test_hbm_bytes_approx_without_split_assumes_64b():
    # only the aggregate present -> 64B/request upper bound
    assert pmc.hbm_read_bytes({"TCC_EA0_RDREQ_sum": 10}) == pytest.approx(640)
    assert pmc.hbm_bytes({"TCC_EA_WRREQ_sum": 5}) == pytest.approx(320)
    assert pmc.hbm_bytes({}) is None


# --------------------------------------------------------------------------- #
# Occupancy — matches the ROCm MI300X worked example.
# --------------------------------------------------------------------------- #
def test_occupancy_constants_arch_selected():
    # CDNA4 (gfx950, the KORE default) vs CDNA3 (gfx942): the ROCm MI355X cheat sheet
    # changed exactly two limits -- LDS 64->160 KiB and VGPR granularity 16->8; the
    # 512-VGPR/SIMD file is UNCHANGED (CDNA4 did not double it).
    c4 = pmc._occupancy_constants("gfx950")
    c3 = pmc._occupancy_constants("gfx942")
    assert c4["lds_bytes_per_cu"] == 163840 and c4["vgpr_alloc_granularity"] == 8
    assert c3["lds_bytes_per_cu"] == 65536 and c3["vgpr_alloc_granularity"] == 16
    assert c4["vgpr_per_simd"] == c3["vgpr_per_simd"] == 512
    # module defaults are CDNA4 (the gfx950 target)
    assert pmc.LDS_BYTES_PER_CU == 163840 and pmc.VGPR_ALLOC_GRANULARITY == 8


def test_waves_per_simd_from_vgpr_rocm_worked_example():
    # ROCm workload guide: 170 VGPRs -> round up to 176 -> floor(512/176) = 2
    # (176 = ceil under BOTH granularity 8 and 16, so this holds on either arch).
    assert pmc.waves_per_simd_from_vgpr(170) == 2
    # ROCm MI355X (CDNA4) granularity-8 example: 100 -> 104 -> floor(512/104) = 4.
    assert pmc.waves_per_simd_from_vgpr(100) == 4
    # caps / exact divisors
    assert pmc.waves_per_simd_from_vgpr(128) == 4      # 512/128
    assert pmc.waves_per_simd_from_vgpr(64) == pmc.MAX_WAVES_PER_SIMD  # 512/64=8 (capped)
    assert pmc.waves_per_simd_from_vgpr(0) == pmc.MAX_WAVES_PER_SIMD   # no VGPR limit
    assert pmc.waves_per_simd_from_vgpr(256) == 2      # 512/256


def test_est_occupancy_vgpr_limited_matches_rocm_example():
    occ = pmc.est_occupancy(vgpr=170, lds=None, num_warps=4)
    assert occ.waves_by_vgpr_per_simd == 2
    assert occ.waves_per_simd == pytest.approx(2.0)
    assert occ.occupancy == pytest.approx(0.25)        # 2 of 8 waves/SIMD
    assert occ.limiter == "vgpr"


def test_est_occupancy_lds_limited_cdna4_default():
    # CDNA4 (gfx950 default): 32 KB LDS/wg -> floor(160/32)=5 workgroups/CU -> 5
    # waves/SIMD (4 waves/wg). The 160 KiB LDS lifts the CDNA3 (2 wg) ceiling.
    occ = pmc.est_occupancy(vgpr=None, lds=32768, num_warps=4)
    assert occ.wg_by_lds == 5
    assert occ.waves_per_simd == pytest.approx(5.0)
    assert occ.limiter == "lds"


def test_est_occupancy_mxfp8_tile_matches_rocm_mi355x_worked_example():
    # ROCm MI355X blog worked example: 256-thread WG (4 waves), 128 total VGPR,
    # 32 KB LDS -> CDNA4 is register-bound at 50% (LDS headroom), where the SAME tile
    # was LDS-bound at 25% on CDNA3. The module default is gfx950 (CDNA4).
    occ = pmc.est_occupancy(vgpr=128, lds=32768, num_warps=4)
    assert occ.occupancy == pytest.approx(0.50) and occ.limiter == "vgpr"


def test_est_occupancy_unconstrained_is_full():
    occ = pmc.est_occupancy(vgpr=None, lds=None, num_warps=4)
    assert occ.occupancy == pytest.approx(1.0)
    assert occ.waves_per_simd == pytest.approx(float(pmc.MAX_WAVES_PER_SIMD))


def test_mfma_helpers():
    # substring match across families incl. the historical ..._BF16 alias
    c = {"SQ_INSTS_VALU_MFMA_MOPS_BF16": 10, "SQ_INSTS_VALU_MFMA_F16": 5}
    assert pmc.mfma_ops(c) == 15
    # MFMA busy fraction vs active cycles
    assert pmc.mfma_busy_fraction(
        {"SQ_VALU_MFMA_BUSY_CYCLES": 800, "GRBM_GUI_ACTIVE": 1000}) == pytest.approx(0.8)
    assert pmc.mfma_busy_fraction({"GRBM_GUI_ACTIVE": 1000}) is None


# --------------------------------------------------------------------------- #
# Parser: keep VGPR/SGPR/LDS/scratch, keep counters back-compat, drop timestamps.
# --------------------------------------------------------------------------- #
def test_parser_wide_captures_resources_not_in_counters(tmp_path):
    csv = tmp_path / "wide.csv"
    csv.write_text(
        "Dispatch_Id,Kernel_Name,SQ_INSTS_VALU,SQ_WAIT_INST_ANY,VGPR_Count,"
        "Accum_VGPR_Count,SGPR_Count,LDS_Block_Size,Scratch_Size,Workgroup_Size,"
        "Start_Timestamp,End_Timestamp\n"
        "1,my_kernel,1000,500,64,32,48,16384,0,256,100000,200000\n"
    )
    (k,) = parse_rocprofv3_csv(csv)
    # counters keep ONLY hardware counters (regression: resources/timestamps leaking in)
    assert k.counters == {"SQ_INSTS_VALU": 1000, "SQ_WAIT_INST_ANY": 500}
    assert "Start_Timestamp" not in k.counters and "End_Timestamp" not in k.counters
    assert "VGPR_Count" not in k.counters and "Workgroup_Size" not in k.counters
    # resources parsed into typed fields
    assert k.vgpr_count == 64 and k.accum_vgpr_count == 32
    assert k.sgpr_count == 48 and k.lds_bytes == 16384 and k.scratch_size == 0
    assert k.total_vgpr == 96
    assert k.workgroup_size == 256 and k.num_warps == 4  # 256/64 wavefronts


def test_parser_long_captures_resources(tmp_path):
    csv = tmp_path / "long.csv"
    csv.write_text(
        "Dispatch_Id,Kernel_Name,Counter_Name,Counter_Value,VGPR_Count,"
        "Accum_VGPR_Count,SGPR_Count,LDS_Block_Size,Scratch_Size,"
        "Start_Timestamp,End_Timestamp\n"
        "1,gemm_kernel,SQ_INSTS_VALU_MFMA_MOPS_BF16,4096,128,64,80,32768,256,10,20\n"
        "1,gemm_kernel,TCC_HIT_sum,900,128,64,80,32768,256,10,20\n"
        "1,gemm_kernel,TCC_MISS_sum,100,128,64,80,32768,256,10,20\n"
    )
    (k,) = parse_rocprofv3_csv(csv)
    assert k.kernel_name == "gemm_kernel"
    assert k.counters == {
        "SQ_INSTS_VALU_MFMA_MOPS_BF16": 4096, "TCC_HIT_sum": 900, "TCC_MISS_sum": 100}
    assert k.vgpr_count == 128 and k.accum_vgpr_count == 64
    assert k.sgpr_count == 80 and k.lds_bytes == 32768 and k.scratch_size == 256
    assert k.total_vgpr == 192
    # derived metrics compute straight off the parsed record
    assert pmc.l2_hit_rate(k.counters) == pytest.approx(0.9)


def test_parser_long_backcompat_no_resources(tmp_path):
    # the pre-existing minimal LONG layout still parses; resource fields stay None.
    csv = tmp_path / "min.csv"
    csv.write_text(
        "Dispatch_Id,Kernel_Name,Counter_Name,Counter_Value\n"
        "1,my_kernel,SQ_INSTS_VALU,1000\n"
        "1,my_kernel,SQ_WAIT_INST_ANY,500\n"
        "2,other_kernel,SQ_INSTS_VALU,50\n"
    )
    ks = parse_rocprofv3_csv(csv)
    assert len(ks) == 2
    k0 = next(k for k in ks if k.kernel_name == "my_kernel")
    assert k0.counters["SQ_INSTS_VALU"] == 1000
    assert k0.vgpr_count is None and k0.total_vgpr is None


def test_kernelpmc_total_vgpr_none_when_absent():
    assert KernelPMC(kernel_name="k").total_vgpr is None
    assert KernelPMC(kernel_name="k", vgpr_count=100).total_vgpr == 100
    assert KernelPMC(kernel_name="k", accum_vgpr_count=50).total_vgpr == 50


def test_kernelpmc_num_warps_from_workgroup_size():
    assert KernelPMC(kernel_name="k").num_warps is None
    assert KernelPMC(kernel_name="k", workgroup_size=256).num_warps == 4
    assert KernelPMC(kernel_name="k", workgroup_size=128).num_warps == 2
    assert KernelPMC(kernel_name="k", workgroup_size=65).num_warps == 2  # ceil(65/64)


def test_parsed_record_feeds_occupancy_end_to_end():
    # a kernel using 200 total VGPRs at 256 threads/workgroup -> VGPR-limited
    k = KernelPMC(kernel_name="k", vgpr_count=160, accum_vgpr_count=40,
                  workgroup_size=256)
    occ = pmc.est_occupancy(k.total_vgpr, k.lds_bytes, k.num_warps)
    # total 200 VGPRs -> round to 208 -> floor(512/208)=2 waves/SIMD
    assert occ.waves_by_vgpr_per_simd == 2
    assert occ.limiter == "vgpr"
