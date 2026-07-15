"""Pillar 4 - profiler-counter-grounded reasoning."""

from __future__ import annotations

from kore.data.grounded_reasoning import (
    counter_grounded_prompt,
    diagnose_bottleneck,
    verify_reasoning_grounding,
)

MEM = {"SQ_INSTS_VMEM": 1000, "SQ_INSTS_VALU_MFMA_BF16": 10,
       "SQ_WAIT_INST_VMEM": 800, "SQ_WAIT_INST_LDS": 50, "SQ_WAIT_INST_ANY": 1000}
LDS = {"SQ_INSTS_VMEM": 500, "SQ_INSTS_VALU_MFMA_BF16": 100,
       "SQ_WAIT_INST_LDS": 600, "SQ_WAIT_INST_VMEM": 100, "SQ_WAIT_INST_ANY": 1000}
NOMM = {"SQ_INSTS_VALU": 5000, "SQ_INSTS_VALU_MFMA_BF16": 0,
        "SQ_INSTS_VMEM": 200, "SQ_WAIT_INST_ANY": 100}
COMP = {"SQ_INSTS_VALU_MFMA_BF16": 9000, "SQ_INSTS_VMEM": 200,
        "SQ_WAIT_INST_VMEM": 10, "SQ_WAIT_INST_ANY": 50}


def test_diagnose_bottleneck_classes():
    assert diagnose_bottleneck(MEM)[0] == "memory-bound"
    assert diagnose_bottleneck(LDS)[0] == "lds-bound"
    assert diagnose_bottleneck(NOMM)[0] == "no-matrix-cores"
    assert diagnose_bottleneck(COMP)[0] == "compute-bound"
    assert diagnose_bottleneck({})[0] == "unknown"


def test_counter_grounded_prompt_injects_counters_and_diagnosis():
    p = counter_grounded_prompt("gemm_bias_relu", MEM, transform="stage loads through LDS")
    assert "SQ_WAIT_INST_VMEM" in p          # raw counter injected
    assert "memory-bound" in p               # diagnosis injected
    assert "stage loads through LDS" in p     # the transform to justify
    assert "cite the counter" in p.lower()   # requires grounding


def test_verify_reasoning_grounding_accepts_grounded_rejects_fabricated():
    grounded = verify_reasoning_grounding(
        "VMEM waits dominate, so I improve global-memory coalescing / bandwidth.", MEM)
    assert grounded["grounded"] and grounded["bottleneck"] == "memory-bound"
    fabricated = verify_reasoning_grounding("I bumped the block size because bigger is better.", MEM)
    assert not fabricated["grounded"]
    # citing the raw counter name is detected
    cited = verify_reasoning_grounding("SQ_WAIT_INST_VMEM is high; reduce global loads.", MEM)
    assert cited["cites_counter"]


def test_no_matrix_cores_grounding_terms():
    r = verify_reasoning_grounding("MFMA issues are zero; switch to tl.dot for the matrix cores.", NOMM)
    assert r["bottleneck"] == "no-matrix-cores" and r["grounded"]


def test_gold_win_reasoning_grounded_when_group_carries_counters():
    # BUG 7: gold_wins grounds the ANALYSIS in real counters when present on the group
    from kore.data.gold_wins import mint_gold_win
    group = {
        "task_id": "t", "operation": "gemm_bias", "arch": "gfx942",
        "candidates": [{"source": "def a(): pass", "rank": 0, "snr_db": 99, "wall_us": 5.0},
                       {"source": "def b(): pass", "rank": 1, "snr_db": 99, "wall_us": 10.0}],
        "preferences": [[0, 1]], "counters": MEM,
    }
    wr = mint_gold_win(group, snr_gate=30.0, min_speedup=1.02)
    assert wr is not None
    analysis = wr.trajectory[-1]["content"]
    assert "memory-bound" in analysis  # grounded in the measured bottleneck
    # without counters -> templated (no bottleneck claim)
    group_no = dict(group); group_no.pop("counters")
    wr2 = mint_gold_win(group_no, snr_gate=30.0, min_speedup=1.02)
    assert wr2 is not None and "bottleneck" not in wr2.trajectory[-1]["content"]
