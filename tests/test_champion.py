"""CPU tests for the champion re-evaluation anti-hack gate (verdict core + loader)."""

from __future__ import annotations

from kore.data.schemas import WinRecord, stamp_source_only_record, write_jsonl
from kore.eval.champion import (
    Champion,
    ChampionReport,
    champion_verdict,
    load_champions,
)


def test_certified_when_survives():
    v = champion_verdict("t", claimed_speedup=1.5, measured_speedup=1.45,
                         correct=True, hack_free=True, high_variance=False)
    assert v.certified and not v.collapsed


def test_collapse_detected():
    # claimed 3.13x, re-measures 1.49x -> collapse (the classic hack signature).
    v = champion_verdict("t", claimed_speedup=3.13, measured_speedup=1.49,
                         correct=True, hack_free=True, high_variance=False,
                         collapse_ratio=0.7)
    assert v.collapsed and not v.certified
    assert "collapsed" in v.reason


def test_incorrect_rejected():
    v = champion_verdict("t", 2.0, 2.0, correct=False, hack_free=True,
                         high_variance=False)
    assert not v.certified and "incorrect" in v.reason


def test_hack_flagged_rejected():
    v = champion_verdict("t", 2.0, 2.0, correct=True, hack_free=False,
                         high_variance=False)
    assert not v.certified and "hack" in v.reason


def test_not_faster_rejected():
    v = champion_verdict("t", None, 0.8, correct=True, hack_free=True,
                         high_variance=False, min_speedup=1.0)
    assert not v.certified and "faster" in v.reason


def test_high_variance_rejected():
    v = champion_verdict("t", 2.0, 2.0, correct=True, hack_free=True,
                         high_variance=True)
    assert not v.certified and "variance" in v.reason


def test_no_claim_still_certifiable():
    # a champion with no claimed speedup is certified purely on the re-measurement.
    v = champion_verdict("t", None, 1.3, correct=True, hack_free=True,
                         high_variance=False)
    assert v.certified and not v.collapsed


def test_load_champions_keeps_best_per_task(tmp_path):
    p = tmp_path / "wins.jsonl"
    records = [
        WinRecord(task_id="a", trajectory=[], initial_wall_us=None,
                  final_wall_us=None, speedup=1.2, final_source="src_a_slow"),
        WinRecord(task_id="a", trajectory=[], initial_wall_us=None,
                  final_wall_us=None, speedup=1.9, final_source="src_a_fast"),
        WinRecord(task_id="b", trajectory=[], initial_wall_us=None,
                  final_wall_us=None, speedup=2.5, final_source="src_b"),
    ]
    write_jsonl(p, [
        stamp_source_only_record(
            record,
            provenance_id="champion-test",
            evaluation_id=f"champion-test:{index}",
            source_status="verified_external",
        )
        for index, record in enumerate(records)
    ])
    champs = {c.task_id: c for c in load_champions(str(p))}
    assert set(champs) == {"a", "b"}
    assert champs["a"].claimed_speedup == 1.9 and champs["a"].source == "src_a_fast"
    assert champs["b"].claimed_speedup == 2.5


def test_report_summary_counts():
    verdicts = [
        champion_verdict("a", 1.5, 1.5, correct=True, hack_free=True, high_variance=False),
        champion_verdict("b", 3.0, 1.0, correct=True, hack_free=True, high_variance=False),
    ]
    rep = ChampionReport(n_champions=2,
                         n_certified=sum(v.certified for v in verdicts),
                         n_collapsed=sum(v.collapsed for v in verdicts),
                         verdicts=verdicts)
    assert rep.n_certified == 1 and rep.n_collapsed == 1
    assert "CERTIFIED" in rep.summary() and "COLLAPSED" in rep.summary()
