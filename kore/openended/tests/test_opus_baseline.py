"""CPU tests for the competitor-anchored Opus baseline.

Two halves, both torch-free where possible:

  * ``build_opus_scores`` (pure/stdlib): mines a tiny SYNTHETIC ``data_root``
    (``wins/`` + ``groups/`` JSONL matching :mod:`kore.data.schemas`) into a
    per-task ``regret_vs_opus`` map in ``[0, 1]``; checks the mapping is sensible
    (monotone in Opus speedup, clamped, strongest task saturates at 1.0), that
    verification gating + the SNR gate work, that caching round-trips, and that
    every malformed/missing input degrades to an inert ``{}``.

  * ``CoevolutionController`` wiring: with a real (registered) task menu, an
    ``opus_scores`` map (or JSON path) concentrates the proposed batch on the
    high-regret tasks, while ``None`` / ``{}`` is byte-identical to the default
    curriculum. (Builds real descriptors, so this half imports torch lazily.)

Run: ``python -m pytest kore/openended/tests/test_opus_baseline.py -q``
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kore.openended import opus_baseline as ob
from kore.openended.opus_baseline import (build_opus_scores, load_opus_scores,
                                          save_opus_scores, summarize_opus_scores)


# --------------------------------------------------------------------------- #
# synthetic corpus helpers
# --------------------------------------------------------------------------- #
def _write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _make_corpus(root: Path) -> Path:
    """A tiny Opus-teacher corpus: task A strong (8x), C mid (4x verified), B mild.

    Group task C also has an INCORRECT 9x candidate (must be ignored) and a slow
    0.5x correct candidate (must not lower the max)."""
    _write_jsonl(root / "wins" / "a.jsonl", [
        {"type": "win", "task_id": "gen_a_bf16", "speedup": 6.0, "snr_db": 55.0},
        {"type": "win", "task_id": "gen_a_bf16", "speedup": 8.0, "snr_db": 60.0},  # best
    ])
    _write_jsonl(root / "wins" / "b.jsonl", [
        {"type": "win", "task_id": "gen_b_bf16", "speedup": 1.2, "snr_db": 40.0},
    ])
    _write_jsonl(root / "groups" / "c.jsonl", [
        {"type": "ranked_group", "task_id": "gen_c_bf16", "candidates": [
            {"speedup": 4.0, "correct": True, "snr_db": 50.0},   # best verified
            {"speedup": 9.0, "correct": False, "snr_db": 10.0},  # NOT verified -> ignored
            {"speedup": 0.5, "correct": True, "snr_db": 48.0},   # slow, doesn't lower max
        ]},
    ])
    return root


# --------------------------------------------------------------------------- #
# 1. build_opus_scores: sensible per-task [0,1] regret
# --------------------------------------------------------------------------- #
def test_build_opus_scores_basic_monotone_and_bounded(tmp_path):
    root = _make_corpus(tmp_path / "data")
    scores = build_opus_scores(root)

    assert set(scores) == {"gen_a_bf16", "gen_b_bf16", "gen_c_bf16"}
    assert all(0.0 <= v <= 1.0 for v in scores.values())          # in range
    a, b, c = scores["gen_a_bf16"], scores["gen_b_bf16"], scores["gen_c_bf16"]
    # monotone in Opus's best verified speedup: A(8) > C(4) > B(1.2)
    assert a > c > b >= 0.0
    # the strongest Opus task saturates at 1.0 (>= the percentile reference)
    assert a == pytest.approx(1.0)
    # a task where Opus barely beat baseline carries little regret
    assert b < 0.2


def test_build_opus_scores_takes_best_verified_speedup_per_task(tmp_path):
    root = _make_corpus(tmp_path / "data")
    # require_correct=True (default): C's verified best is 4.0, NOT the 9x incorrect one
    strict = build_opus_scores(root)
    # require_correct=False: the 9x candidate now counts -> C becomes the strongest
    loose = build_opus_scores(root, require_correct=False)

    assert loose["gen_c_bf16"] > strict["gen_c_bf16"]
    assert loose["gen_c_bf16"] == pytest.approx(1.0)               # 9x is the new max
    assert loose["gen_c_bf16"] > loose["gen_a_bf16"]              # C(9) now beats A(8)


def test_build_opus_scores_snr_gate_drops_low_snr(tmp_path):
    root = _make_corpus(tmp_path / "data")
    # B's only win has snr 40; gate at 45 drops it entirely (A/C survive).
    scores = build_opus_scores(root, min_snr_db=45.0)
    assert "gen_b_bf16" not in scores
    assert set(scores) == {"gen_a_bf16", "gen_c_bf16"}


def test_build_opus_scores_source_toggles(tmp_path):
    root = _make_corpus(tmp_path / "data")
    wins_only = build_opus_scores(root, include_groups=False)
    groups_only = build_opus_scores(root, include_wins=False)
    assert set(wins_only) == {"gen_a_bf16", "gen_b_bf16"}          # C is groups-only
    assert set(groups_only) == {"gen_c_bf16"}                      # A/B are wins-only


# --------------------------------------------------------------------------- #
# 2. fail-safe: missing / malformed / empty -> {} (feature inert)
# --------------------------------------------------------------------------- #
def test_build_opus_scores_missing_or_invalid_root_is_inert(tmp_path):
    assert build_opus_scores(tmp_path / "does_not_exist") == {}
    assert build_opus_scores(None) == {}
    # a *file* (not a directory) is not a valid data_root
    f = tmp_path / "afile"
    f.write_text("hi", encoding="utf-8")
    assert build_opus_scores(f) == {}


def test_build_opus_scores_empty_corpus_is_inert(tmp_path):
    root = tmp_path / "data"
    (root / "wins").mkdir(parents=True)
    (root / "groups").mkdir(parents=True)
    assert build_opus_scores(root) == {}                          # dirs exist but empty


def test_build_opus_scores_skips_malformed_lines_and_records(tmp_path):
    root = tmp_path / "data"
    win = root / "wins" / "w.jsonl"
    win.parent.mkdir(parents=True)
    with win.open("w", encoding="utf-8") as f:
        f.write("{ this is not json\n")                            # malformed line
        f.write("\n")                                              # blank line
        f.write(json.dumps({"type": "win", "task_id": "gen_ok_bf16",
                            "speedup": 3.0}) + "\n")               # valid
        f.write(json.dumps({"type": "win", "speedup": 5.0}) + "\n")  # missing task_id
        f.write(json.dumps({"type": "win", "task_id": "gen_bad_bf16",
                            "speedup": "NaN"}) + "\n")             # bad speedup
        f.write(json.dumps({"type": "win", "task_id": "gen_neg_bf16",
                            "speedup": -2.0}) + "\n")              # non-positive
    scores = build_opus_scores(root)
    assert set(scores) == {"gen_ok_bf16"}                          # only the clean record
    assert 0.0 <= scores["gen_ok_bf16"] <= 1.0


def test_build_opus_scores_ignores_non_speedup_record_types(tmp_path):
    root = tmp_path / "data"
    # a repair record living in groups/ has no candidates -> skipped, not crashed
    _write_jsonl(root / "groups" / "r.jsonl", [
        {"type": "repair", "task_id": "gen_x_bf16", "failure_class": "compile_fail"},
    ])
    assert build_opus_scores(root) == {}


# --------------------------------------------------------------------------- #
# 3. mapping-primitive unit tests (pure, no IO)
# --------------------------------------------------------------------------- #
def test_speedups_to_regret_is_monotone_clamped_and_floored():
    best = {"lo": 1.0, "mid": 4.0, "hi": 16.0, "slow": 0.5}
    out = ob._speedups_to_regret(best, ref_percentile=95.0, min_ref_speedup=2.0)
    assert set(out) == set(best)
    assert all(0.0 <= v <= 1.0 for v in out.values())
    # Opus at/below baseline (speedup <= 1) -> zero competitive headroom
    assert out["lo"] == 0.0 and out["slow"] == 0.0
    # strictly increasing in speedup for the >1 tasks
    assert out["hi"] > out["mid"] > out["lo"]
    assert out["hi"] == pytest.approx(1.0)                         # strongest saturates


def test_speedups_to_regret_empty_is_empty():
    assert ob._speedups_to_regret({}, ref_percentile=95.0, min_ref_speedup=2.0) == {}


def test_opus_percentile_small_inputs():
    assert ob._percentile([], 95.0) == 0.0
    assert ob._percentile([3.0], 95.0) == 3.0
    assert ob._percentile([1.0, 2.0, 3.0], 0.0) == 1.0
    assert ob._percentile([1.0, 2.0, 3.0], 100.0) == 3.0
    assert ob._percentile([1.0, 3.0], 50.0) == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# 4. caching to a JSON file (coevolve_opus_scores_path)
# --------------------------------------------------------------------------- #
def test_build_opus_scores_writes_and_reuses_cache(tmp_path):
    root = _make_corpus(tmp_path / "data")
    cache = tmp_path / "cache" / "opus_scores.json"

    first = build_opus_scores(root, cache_path=cache)
    assert first and cache.is_file()                               # computed + written

    # cache hit: even with a BROKEN data_root, the cached map is returned unchanged
    reused = build_opus_scores(tmp_path / "gone", cache_path=cache)
    assert reused == first

    # the on-disk cache round-trips through the loader
    assert load_opus_scores(cache) == first


def test_opus_load_and_save_round_trip_and_sanitize(tmp_path):
    p = tmp_path / "s.json"
    assert save_opus_scores({"gen_a_bf16": 0.7, "gen_b_bf16": 0.1}, p) == p
    assert load_opus_scores(p) == {"gen_a_bf16": 0.7, "gen_b_bf16": 0.1}
    # loader sanitizes: non-str keys / out-of-range / non-finite dropped or clamped
    p.write_text(json.dumps({"ok": 1.5, "lo": -1.0, "bad": "x", "3": 0.4}),
                 encoding="utf-8")
    loaded = load_opus_scores(p)
    assert loaded == {"ok": 1.0, "lo": 0.0, "3": 0.4}


def test_load_opus_scores_missing_or_bad_file_is_inert(tmp_path):
    assert load_opus_scores(tmp_path / "nope.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_opus_scores(bad) == {}
    arr = tmp_path / "arr.json"
    arr.write_text("[1, 2, 3]", encoding="utf-8")                  # not a dict
    assert load_opus_scores(arr) == {}


def test_summarize_opus_scores():
    assert summarize_opus_scores({})["tasks"] == 0
    s = summarize_opus_scores({"a": 1.0, "b": 0.0, "c": 0.5})
    assert s["tasks"] == 3 and s["max"] == 1.0 and s["min"] == 0.0
    assert s["mean"] == pytest.approx(0.5)
    assert s["top"][0][0] == "a"


# --------------------------------------------------------------------------- #
# 5. CoevolutionController wiring (real registered menu; torch imported lazily)
# --------------------------------------------------------------------------- #
def _distinct_family_task_ids(n):
    """``n`` registered task_ids each from a DISTINCT operator family (=> distinct
    MAP-Elites niche, since ``family`` is the first niche field), so the proposer's
    per-niche diversity cap can't interfere with the concentration assertions."""
    from kore.openended import task_space as ts
    picks: dict = {}
    for d in ts.enumerate_descriptors(include_vendor=True):
        picks.setdefault(d.family, d.task_id)
        if len(picks) >= n:
            break
    return list(picks.values())[:n]


def _fresh_ctrl(picks, *, batch, **kwargs):
    """A controller over ``picks`` with each task seeded to peak learnability
    (solve_rate 0.5) so the (learnability-multiplied) opus term is active and all
    base scores are equal (the archive stays empty => constant novelty)."""
    from kore.openended.controller import CoevolutionController
    from kore.openended.proposer import DescriptorStats
    ctrl = CoevolutionController(picks, seed=0, batch=batch, **kwargs)
    for tid in picks:
        desc = ctrl.by_task.get(tid)
        if desc is not None:
            ctrl.history[desc] = DescriptorStats(solve_rate=0.5, headroom_regret=0.0,
                                                 attempts=4)
    return ctrl


def _drain(ctrl, n):
    return [ctrl.next_task_id() for _ in range(n)]


def test_controller_opus_scores_concentrate_on_high_regret():
    picks = _distinct_family_task_ids(5)
    assert len(picks) >= 4                                         # sanity for the env
    n = len(picks)

    # baseline (anchor OFF) service order
    off = _fresh_ctrl(picks, batch=n)
    off_order = _drain(off, n)

    # boost exactly the two tasks the default curriculum served LAST -> they must
    # jump to the front once the competitor anchor is applied.
    boost = set(off_order[-2:])
    opus = {t: 0.95 for t in boost}
    on = _fresh_ctrl(picks, batch=n, opus_scores=opus)
    on_order = _drain(on, n)

    pos = {t: on_order.index(t) for t in on_order}
    # every boosted (high-regret) task precedes every non-boosted task
    assert max(pos[t] for t in boost) < min(pos[t] for t in on_order if t not in boost)
    assert set(on_order[:2]) == boost                             # batch front == boosted
    assert on_order != off_order                                  # curriculum changed
    assert on.opus_scores is not None and len(on.opus_scores) == 2
    assert on.report()["opus_anchored"] is True
    assert on.report()["opus_tasks"] == 2


def test_controller_opus_none_and_empty_are_byte_identical():
    picks = _distinct_family_task_ids(5)
    n = len(picks)
    default = _fresh_ctrl(picks, batch=n)
    none_c = _fresh_ctrl(picks, batch=n, opus_scores=None)
    empty_c = _fresh_ctrl(picks, batch=n, opus_scores={})

    seq = _drain(default, n)
    assert seq == _drain(none_c, n) == _drain(empty_c, n)         # unchanged selection
    # {} and None both normalize to the inert (None) anchor
    assert default.opus_scores is None
    assert none_c.opus_scores is None
    assert empty_c.opus_scores is None
    assert default.report()["opus_anchored"] is False
    assert default.report()["opus_tasks"] == 0


def test_controller_loads_opus_scores_from_path(tmp_path):
    picks = _distinct_family_task_ids(5)
    n = len(picks)
    off_order = _drain(_fresh_ctrl(picks, batch=n), n)
    boost = set(off_order[-2:])

    path = tmp_path / "opus_scores.json"
    save_opus_scores({t: 0.95 for t in boost}, path)

    on = _fresh_ctrl(picks, batch=n, opus_scores_path=str(path))
    on_order = _drain(on, n)
    assert on.opus_scores is not None and len(on.opus_scores) == 2
    assert set(on_order[:2]) == boost                             # same concentration


def test_controller_bad_opus_path_is_inert(tmp_path):
    picks = _distinct_family_task_ids(5)
    n = len(picks)
    baseline = _drain(_fresh_ctrl(picks, batch=n), n)
    # a non-existent path leaves the anchor inert -> identical to the default
    on = _fresh_ctrl(picks, batch=n, opus_scores_path=str(tmp_path / "nope.json"))
    assert on.opus_scores is None
    assert _drain(on, n) == baseline


def test_controller_end_to_end_build_then_anchor(tmp_path):
    """The orchestrator flow: build_opus_scores(data_root) -> controller anchor."""
    picks = _distinct_family_task_ids(5)
    n = len(picks)
    off_order = _drain(_fresh_ctrl(picks, batch=n), n)
    strong = off_order[-2:]                                       # will be given big Opus speedups
    weak = off_order[:-2]

    root = tmp_path / "data"
    for i, tid in enumerate(strong):
        _write_jsonl(root / "wins" / f"s{i}.jsonl",
                     [{"type": "win", "task_id": tid, "speedup": 10.0 - i}])
    for i, tid in enumerate(weak):
        _write_jsonl(root / "wins" / f"w{i}.jsonl",
                     [{"type": "win", "task_id": tid, "speedup": 1.02}])

    scores = build_opus_scores(root)
    assert set(scores) == set(picks)
    assert all(scores[t] > 0.5 for t in strong)                   # strong Opus -> high regret
    assert all(scores[t] < 0.2 for t in weak)                     # weak Opus -> low regret

    on_order = _drain(_fresh_ctrl(picks, batch=n, opus_scores=scores), n)
    assert set(on_order[:2]) == set(strong)                       # anchored batch front
