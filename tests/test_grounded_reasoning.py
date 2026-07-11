"""Pillar 4 — profiler-counter-grounded reasoning."""

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
