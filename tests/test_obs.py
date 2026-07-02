"""CPU-only tests for the kore.obs structured logger."""

from __future__ import annotations

import json

from kore import obs


def test_jsonl_and_levels(tmp_path):
    obs.configure(run_dir=tmp_path, level="DEBUG", color=False)
    log = obs.get_logger("test")
    log.info("hello", a=1, b=2.5)
    log.event("thing_happened", n=3)
    log.metric("m", loss=0.5)
    recs = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]
    kinds = {r["kind"] for r in recs}
    assert {"log", "event", "metric"} <= kinds
    hello = [r for r in recs if r["msg"] == "hello"][0]
    assert hello["fields"] == {"a": 1, "b": 2.5} and "elapsed_s" in hello


def test_stage_timer_and_stack(tmp_path):
    obs.configure(run_dir=tmp_path, level="DEBUG", color=False)
    log = obs.get_logger("t")
    with log.stage("outer"):
        with log.stage("inner"):
            log.info("mid")
    recs = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]
    stage_recs = [r for r in recs if r["kind"] == "stage"]
    # start+done for both stages
    assert sum(1 for r in stage_recs if "stage start" in r["msg"]) == 2
    assert sum(1 for r in stage_recs if "stage done" in r["msg"]) == 2
    mid = [r for r in recs if r["msg"] == "mid"][0]
    assert mid["stage"] == "inner"  # innermost stage attributed


def test_stage_records_failure(tmp_path):
    obs.configure(run_dir=tmp_path, level="INFO", color=False)
    log = obs.get_logger("t")
    try:
        with log.stage("boom"):
            raise ValueError("x")
    except ValueError:
        pass
    recs = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert any("stage FAILED" in r["msg"] for r in recs)


def test_progress_eta(tmp_path):
    obs.configure(run_dir=tmp_path, level="INFO", color=False)
    log = obs.get_logger("t")
    import time
    t0 = time.time() - 1.0
    log.progress(5, 10, "items", t_start=t0)
    rec = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()][-1]
    assert rec["kind"] == "progress" and "5/10" in rec["msg"]
    assert "rate_per_s" in rec["fields"] and "eta" in rec["fields"]
