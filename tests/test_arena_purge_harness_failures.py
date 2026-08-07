"""The purge must remove exactly the harness-scored rows and nothing else.

This tool edits the only record of work already paid for, so its failure modes are
asymmetric: purging too little wastes a re-run, purging too much destroys verdicts
that cost GPU-hours and, worse, can discard a win. These pin both directions.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "purge_arena_harness_failures.py"

sys.path.insert(0, str(SCRIPT.parent))

from purge_arena_harness_failures import (  # noqa: E402
    SIGNATURES, _is_missing_extension_contract, _is_triton_collision)


def _row(**kw):
    base = {"task_id": "x/y", "task_type": "torch2hip", "compiled": False,
            "correct": False, "score": 0.0, "error": ""}
    base.update(kw)
    return base


# ---- predicates -----------------------------------------------------------

def test_triton_collision_matches_the_real_error_text():
    assert _is_triton_collision(_row(
        task_type="torch2flydsl",
        error="FAIL: e16_t128 - aiter gluon kernels require triton>=3.6.0, found 3.5.1"))


def test_triton_collision_does_not_match_an_unrelated_aiter_error():
    assert not _is_triton_collision(_row(
        task_type="torch2flydsl", error="aiter: ROCm version file not found"))


def test_missing_contract_matches_an_uncompiled_torch2hip():
    assert _is_missing_extension_contract(_row(compiled=False))


def test_missing_contract_ignores_a_compiled_torch2hip():
    """Compiled-but-wrong is a verdict about the kernel, not the harness."""
    assert not _is_missing_extension_contract(_row(compiled=True))


def test_missing_contract_ignores_other_categories():
    """hip2hip targets ship non-empty, so the contract was never missing there."""
    assert not _is_missing_extension_contract(_row(task_type="hip2hip",
                                                  compiled=False))
    assert not _is_missing_extension_contract(_row(task_type="triton2triton",
                                                  compiled=False))


# ---- end to end -----------------------------------------------------------

def _write_ledger(d: Path, arm: str, rows: list[dict], shard: int = 0) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"results_{arm}.shard{shard}of8.partial.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def _run(out: Path, arm: str, *extra):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out), "--arm", arm, *extra],
        capture_output=True, text=True, timeout=120)


def test_dry_run_reports_but_writes_nothing(tmp_path):
    led = _write_ledger(tmp_path, "v4", [
        _row(task_id="torch2hip/a"),
        _row(task_id="triton2triton/b", task_type="triton2triton", compiled=True),
    ])
    before = led.read_text()
    r = _run(tmp_path, "v4")
    assert r.returncode == 0, r.stderr
    assert "1 attributable" in r.stdout
    assert "dry run" in r.stdout
    assert led.read_text() == before


def test_apply_removes_only_the_matching_rows(tmp_path):
    keep_a = _row(task_id="triton2triton/keep", task_type="triton2triton",
                  compiled=True, correct=True, score=120.0)
    keep_b = _row(task_id="hip2hip/keep", task_type="hip2hip", compiled=False)
    drop = _row(task_id="torch2hip/drop")
    led = _write_ledger(tmp_path, "v4", [keep_a, keep_b, drop])
    r = _run(tmp_path, "v4", "--apply")
    assert r.returncode == 0, r.stderr
    ids = {json.loads(l)["task_id"] for l in led.read_text().splitlines() if l.strip()}
    assert ids == {"triton2triton/keep", "hip2hip/keep"}


def test_a_correct_row_is_never_purged_even_if_it_matches(tmp_path):
    """The asymmetric failure: re-rolling a win can only lose points."""
    win = _row(task_id="torch2hip/win", compiled=False, correct=True, score=120.0)
    led = _write_ledger(tmp_path, "v4", [win])
    r = _run(tmp_path, "v4", "--apply")
    assert r.returncode == 0, r.stderr
    assert "never re-roll a win" in r.stdout
    assert json.loads(led.read_text().strip())["task_id"] == "torch2hip/win"


def test_originals_are_backed_up_before_rewrite(tmp_path):
    led = _write_ledger(tmp_path, "v4", [_row(task_id="torch2hip/a")])
    original = led.read_text()
    _run(tmp_path, "v4", "--apply")
    backups = list(tmp_path.glob("ledger_backup_v4_*/" + led.name))
    assert len(backups) == 1
    assert backups[0].read_text() == original


def test_all_shard_ledgers_are_processed(tmp_path):
    a = _write_ledger(tmp_path, "v4", [_row(task_id="torch2hip/a")], shard=0)
    b = _write_ledger(tmp_path, "v4", [_row(task_id="torch2hip/b")], shard=3)
    r = _run(tmp_path, "v4", "--apply")
    assert "2 attributable" in r.stdout
    assert a.read_text().strip() == ""
    assert b.read_text().strip() == ""


def test_signature_filter_restricts_what_is_purged(tmp_path):
    led = _write_ledger(tmp_path, "v4", [
        _row(task_id="torch2hip/a"),
        _row(task_id="torch2flydsl/b", task_type="torch2flydsl", compiled=True,
             error="aiter gluon kernels require triton>=3.6.0, found 3.5.1"),
    ])
    _run(tmp_path, "v4", "--signature", "triton_collision", "--apply")
    ids = {json.loads(l)["task_id"] for l in led.read_text().splitlines() if l.strip()}
    assert ids == {"torch2hip/a"}


def test_torn_last_line_does_not_abort_the_purge(tmp_path):
    """Ledgers are killed mid-append routinely; a partial line must be tolerated."""
    p = tmp_path / "results_v4.shard0of8.partial.jsonl"
    tmp_path.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_row(task_id="torch2hip/a")) + "\n" + '{"task_id": "trunc')
    r = _run(tmp_path, "v4", "--apply")
    assert r.returncode == 0, r.stderr
    assert "1 attributable" in r.stdout


def test_missing_ledger_dir_is_an_error_not_a_silent_success(tmp_path):
    r = _run(tmp_path / "nope", "v4")
    assert r.returncode == 1
    assert "no ledgers" in r.stdout


def test_every_signature_has_a_human_reason():
    for name, (pred, reason) in SIGNATURES.items():
        assert callable(pred)
        assert reason and len(reason) > 20, name
