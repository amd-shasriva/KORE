"""CPU tests for training the value model from real ranked-group shards (P1a)."""

from __future__ import annotations

import json

from kore.data.schemas import RankedGroupRecord
from kore.value import replay_train as rt
from kore.value.train_value import _synth_source


def _mk_group(task_id, op, dtype, quality_specs):
    """Build a RankedGroupRecord whose candidates' speedup tracks schedule quality."""
    cands = []
    for i, (bm, bn, bk, warps, stages, use_dot, fp32, su) in enumerate(quality_specs):
        cands.append({
            "source": _synth_source(bm, bn, bk, warps, stages, use_dot, fp32),
            "wall_us": 100.0 / su, "baseline_wall_us": 100.0,
            "snr_db": 40.0, "rank": i, "speedup": su,
        })
    # NB: RankedGroupRecord has no dtype field; dtype is derived from task_id
    # (e.g. gen_gemm_bf16 -> bf16), exactly as replay_train does in production.
    return RankedGroupRecord(
        task_id=task_id, parent_id="p0", candidates=cands,
        preferences=[[0, len(cands) - 1]], operation=op, shape="M=512,N=512,K=512",
    )


def test_group_rows_extraction_carries_source_and_speedup():
    rec = _mk_group("gen_gemm_bf16", "gemm", "bf16", [
        (128, 128, 64, 8, 3, True, True, 1.8),
        (96, 96, 48, 2, 1, False, False, 0.6),
    ])
    rows = rt.group_rows_from_record(rec)
    assert len(rows) == 2
    assert all("source" in r and r["source"] for r in rows)
    assert rows[0]["dtype"] == "bf16" and rows[0]["operation"] == "gemm"
    assert rows[0]["M"] == 512 and rows[0]["speedup"] == 1.8
    assert rows[0]["compiled"] is True and rows[0]["snr_pass"] is True


def test_candidate_speedup_fallbacks():
    assert rt._candidate_speedup({"speedup": 2.0}) == 2.0
    assert rt._candidate_speedup({"baseline_wall_us": 200.0, "wall_us": 100.0}) == 2.0
    assert rt._candidate_speedup({}) is None


def test_train_value_from_real_looking_groups(tmp_path):
    gdir = tmp_path / "groups"
    gdir.mkdir()
    # Write several groups where good schedules (64-multiple tiles, tl.dot, fp32) win.
    good = (128, 128, 64, 8, 3, True, True, 1.9)
    mid = (128, 64, 64, 4, 2, True, False, 1.1)
    bad = (96, 96, 48, 2, 1, False, False, 0.5)
    with (gdir / "gen_gemm_bf16.jsonl").open("w") as f:
        for k in range(8):
            rec = _mk_group("gen_gemm_bf16", "gemm", "bf16", [good, mid, bad])
            f.write(json.dumps(rec.to_dict()) + "\n")
    with (gdir / "gen_add_fp16.jsonl").open("w") as f:
        for k in range(6):
            rec = _mk_group("gen_add_fp16", "add", "fp16", [good, bad, mid])
            f.write(json.dumps(rec.to_dict()) + "\n")

    groups = rt.load_groups_from_dir(str(gdir))
    assert len(groups) == 14 and all(len(g) == 3 for g in groups)

    out = tmp_path / "value_model.pkl"
    metrics = rt.train_value_from_groups(str(gdir), str(out))
    assert out.exists()
    assert metrics["n_groups"] == 14
    assert metrics["n_candidates"] == 42
    assert "heldout_group_rank_corr" in metrics
    # a shard-only ``_``-prefixed file is ignored
    (gdir / "_repair_pairs.jsonl").write_text("{}\n")
    assert len(rt.load_groups_from_dir(str(gdir))) == 14
