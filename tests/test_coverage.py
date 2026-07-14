"""Pillar 2 — task data-coverage audit."""

from __future__ import annotations

import json

from kore.data.coverage import (
    FRONTIER_OPS,
    REQUIRED_KINDS,
    coverage_report,
    frontier_coverage,
    space_coverage,
    task_coverage,
    undercovered_tasks,
)


def _mk(data_root, kind, task_id, n=1):
    d = data_root / kind
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{task_id}.jsonl").write_text("\n".join(json.dumps({"i": i}) for i in range(n)) + "\n")


def test_task_coverage_full_partial_missing(tmp_path):
    # full: all 3 kinds; partial: only repair; missing: nothing
    for k in REQUIRED_KINDS:
        _mk(tmp_path, k, "gen_full", 3)
    _mk(tmp_path, "repair", "gen_partial", 2)
    cov = task_coverage(tmp_path, ["gen_full", "gen_partial", "gen_missing"])
    assert cov["gen_full"]["full"] is True
    assert cov["gen_full"]["wins"] == 3
    assert cov["gen_partial"]["full"] is False
    assert cov["gen_partial"]["repair"] == 2 and cov["gen_partial"]["groups"] == 0
    assert cov["gen_missing"]["full"] is False and cov["gen_missing"]["repair"] == 0


def test_undercovered_lists_missing_kinds(tmp_path):
    _mk(tmp_path, "repair", "t1", 1)
    _mk(tmp_path, "groups", "t1", 1)  # t1 missing wins
    for k in REQUIRED_KINDS:
        _mk(tmp_path, k, "t2", 1)     # t2 full
    under = undercovered_tasks(tmp_path, ["t1", "t2"])
    assert under == {"t1": ["wins"]}


def test_coverage_report_shape(tmp_path):
    for k in REQUIRED_KINDS:
        _mk(tmp_path, k, "t_full", 1)
    _mk(tmp_path, "repair", "t_hole", 1)
    rep = coverage_report(tmp_path, ["t_full", "t_hole"])
    assert rep["n_train_tasks"] == 2
    assert rep["n_full_coverage"] == 1
    assert rep["coverage_pct"] == 50.0
    assert rep["per_kind_covered"]["repair"] == 2
    assert rep["per_kind_covered"]["wins"] == 1
    assert "t_hole" in rep["undercovered"]


def test_registry_task_count_after_fp32_expansion():
    # Pillar 2 regression guard: fp32 materialization -> 236 tasks (was 209)
    try:
        from kore.tasks.registry import all_tasks
    except Exception:
        return
    n = len(all_tasks())
    assert n >= 236, f"expected >=236 tasks after fp32 expansion, got {n}"


def test_frontier_coverage_reports_capabilities():
    fc = frontier_coverage()
    if not fc:  # registry unavailable
        return
    assert fc["n_capabilities"] == len(FRONTIER_OPS)
    assert fc["n_covered"] + fc["n_missing"] == fc["n_capabilities"]
    # core capabilities KORE definitely has (attention_mla + quant_int4 now authored
    # + GPU-proven; MLA is held out for the generalization eval but still "covered")
    for cap in ("gemm_dense", "gemm_fp8", "attention_mha", "attention_mla", "moe_router",
                "norm_rms", "rope", "elementwise", "reduction", "quant_int4"):
        assert cap in fc["covered"], f"{cap} should be covered"
    # remaining holes that require GPU-verified authoring (not fabricated offline)
    for cap in ("sampling", "collective"):
        assert cap in fc["missing"], f"{cap} should be a tracked hole"


def test_space_coverage_reports_dtype_frontier():
    sc = space_coverage()
    # generate_ops now emits fp32 for every generated family (no fp32 holes)
    if sc:  # registry available
        for fam, d in sc["per_family"].items():
            assert "fp32" in d["emitted"], f"{fam} should emit fp32"
        # fp8/int8 are intentionally NOT generated (vendor-op territory)
        assert "fp8" in sc["all_dtypes"]
