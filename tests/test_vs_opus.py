"""CPU-only tests for the KORE vs Opus head-to-head (no GPU, no network).

Everything here runs with STUB policies: the KORE side is a plain
``messages -> str`` callable, the Opus side is a ``StubTeacher`` (deterministic,
dependency-free), and every measurement is a precomputed ``Observation`` fed
through ``evaluate_policy``'s ``dry_run`` path. So the win-rate, aggregation, and
CI logic is exercised end to end without touching a GPU or the gateway.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

from kore.data.teacher import StubTeacher
from kore.reward.reward import Observation
from kore.eval import vs_opus


# A minimal, contract-shaped completion. In dry_run mode the returned kernel is
# never measured (the fabricated Observation is used), so any non-empty response
# that parses is fine; this exercises build_transcript + parse_response for real.
CANNED = "ANALYSIS: tune it.\n\nFULL_KERNEL:\n```python\ndef k():\n    pass\n```\n"


# --------------------------------------------------------------------------- #
# Fabricated observations (mirror tests/test_eval.py conventions).
# --------------------------------------------------------------------------- #
def _obs(speedup: float, snr: float = 90.0, dtype: str = "bf16") -> Observation:
    """A correct, cleanly-timed kernel with the given speedup vs the baseline."""
    wall = 1.0 / speedup
    return Observation(
        compiled=True, snr_db=snr, wall_ms=wall, baseline_ms=1.0,
        wall_by_shape={"s": wall}, baseline_by_shape={"s": 1.0},
        snr_by_shape={"s": snr}, validation_passed=True, dtype=dtype,
    )


def _obs_bad(dtype: str = "bf16") -> Observation:
    """Compiled but INCORRECT (SNR below the gate) -> contributes no speedup."""
    return Observation(
        compiled=True, snr_db=5.0, wall_ms=0.5, baseline_ms=1.0,
        wall_by_shape={"s": 0.5}, baseline_by_shape={"s": 1.0},
        snr_by_shape={"s": 5.0}, validation_passed=True, dtype=dtype,
    )


def _kore_gen(counter: list | None = None):
    """A stub KORE served-model generate callable (messages, **kw) -> str."""
    def gen(messages, **kw):
        if counter is not None:
            counter.append(1)
        return CANNED
    return gen


def _stub_teacher() -> StubTeacher:
    return StubTeacher(fn=lambda messages: CANNED)


# --------------------------------------------------------------------------- #
# PURE win logic: winner_for_task
# --------------------------------------------------------------------------- #
def _rec(correct: bool, su):
    return {"task_id": "t", "correct": correct, "best_speedup": su}


def test_winner_for_task_all_branches():
    # both correct, KORE faster
    assert vs_opus.winner_for_task(_rec(True, 2.0), _rec(True, 1.5)) == "kore"
    # both correct, Opus faster
    assert vs_opus.winner_for_task(_rec(True, 1.2), _rec(True, 3.0)) == "opus"
    # equal speedups -> tie (margin=1.0 means strictly faster to win)
    assert vs_opus.winner_for_task(_rec(True, 2.0), _rec(True, 2.0)) == "tie"
    # only KORE correct -> KORE wins even if Opus "would" be fast
    assert vs_opus.winner_for_task(_rec(True, 1.1), _rec(False, 9.0)) == "kore"
    # only Opus correct
    assert vs_opus.winner_for_task(_rec(False, 9.0), _rec(True, 1.1)) == "opus"
    # neither correct
    assert vs_opus.winner_for_task(_rec(False, 9.0), _rec(False, 9.0)) == "neither"
    # a correct kernel with no measured speedup does not compete
    assert vs_opus.winner_for_task(_rec(True, None), _rec(True, 2.0)) == "opus"


def test_winner_for_task_margin():
    # 2.05 barely beats 2.0 at margin=1.0 ...
    assert vs_opus.winner_for_task(_rec(True, 2.05), _rec(True, 2.0)) == "kore"
    # ... but a 10% margin turns that into a tie (neither clears the other by 10%)
    assert vs_opus.winner_for_task(_rec(True, 2.05), _rec(True, 2.0), margin=1.1) == "tie"


# --------------------------------------------------------------------------- #
# PURE win-rate aggregation: head_to_head_winrate
# --------------------------------------------------------------------------- #
def test_head_to_head_winrate_counts_and_matching():
    kore = [
        {"task_id": "t1", "correct": True, "best_speedup": 2.0},   # beats opus 1.5
        {"task_id": "t2", "correct": True, "best_speedup": 3.0},   # opus incorrect
        {"task_id": "t3", "correct": False, "best_speedup": None}, # both incorrect
        {"task_id": "t4", "correct": True, "best_speedup": 2.0},   # tie with opus
    ]
    opus = [
        {"task_id": "t1", "correct": True, "best_speedup": 1.5},
        {"task_id": "t2", "correct": False, "best_speedup": None},
        {"task_id": "t3", "correct": False, "best_speedup": None},
        {"task_id": "t4", "correct": True, "best_speedup": 2.0},
    ]
    wr = vs_opus.head_to_head_winrate(kore, opus)
    assert wr["counts"] == {"kore": 2, "opus": 0, "tie": 1, "neither": 1}
    assert wr["n"] == 4
    assert wr["win_rate"] == 0.5             # KORE wins t1, t2
    assert wr["opus_win_rate"] == 0.0
    assert wr["tie_rate"] == 0.25            # t4
    assert wr["both_incorrect_rate"] == 0.25 # t3
    # per-task breakdown carries the matched, gated speedups
    by_id = {r["task_id"]: r for r in wr["per_task"]}
    assert by_id["t2"]["winner"] == "kore" and by_id["t2"]["opus_speedup"] is None
    assert by_id["t4"]["winner"] == "tie"


def test_winrate_denominator_is_uncorrected():
    # 1 KORE win but a 3-task split -> 1/3, not 1/1 (unattempted tasks count).
    kore = [
        {"task_id": "t1", "correct": True, "best_speedup": 2.0},
        {"task_id": "t2", "correct": False, "best_speedup": None},
        {"task_id": "t3", "correct": False, "best_speedup": None},
    ]
    opus = [{"task_id": "t1", "correct": False, "best_speedup": None}]
    wr = vs_opus.head_to_head_winrate(kore, opus)
    assert wr["n"] == 3
    assert abs(wr["win_rate"] - 1.0 / 3.0) < 1e-12


# --------------------------------------------------------------------------- #
# head_to_head end to end (stub KORE gen + StubTeacher + dry_run observations)
# --------------------------------------------------------------------------- #
def test_head_to_head_scores_both_sides_identically():
    tasks = ["t1", "t2", "t3", "t4"]
    kore_calls: list = []
    teacher = _stub_teacher()

    kore_dry = {"t1": [_obs(2.0)], "t2": [_obs(3.0)], "t3": [_obs_bad()], "t4": [_obs(2.0)]}
    opus_dry = {"t1": [_obs(1.5)], "t2": [_obs_bad()], "t3": [_obs_bad()], "t4": [_obs(2.0)]}

    res = vs_opus.head_to_head(
        tasks, _kore_gen(kore_calls), teacher, budget=1, seeds=[0],
        kore_dry_run=kore_dry, opus_dry_run=opus_dry,
    )

    assert res["skipped"] is False
    assert res["n"] == 4
    # Both sides really went through the model_policy path: KORE gen + the teacher
    # were each invoked once per task.
    assert len(kore_calls) == 4
    assert len(teacher.calls) == 4

    # KORE: t1,t2,t4 correct & > 1x -> fast_1 = 3/4; Opus: t1,t4 -> 2/4.
    kore_fp1 = res["kore"]["fast_p_mean_ci"][1.0]["mean"]
    opus_fp1 = res["opus"]["fast_p_mean_ci"][1.0]["mean"]
    assert abs(kore_fp1 - 0.75) < 1e-9
    assert abs(opus_fp1 - 0.5) < 1e-9

    # Win-rate: KORE wins t1 (2x>1.5x) and t2 (opus incorrect); t4 tie; t3 neither.
    assert abs(res["win_rate_mean_ci"]["mean"] - 0.5) < 1e-9
    assert abs(res["opus_win_rate_mean_ci"]["mean"] - 0.0) < 1e-9
    assert abs(res["tie_rate_mean_ci"]["mean"] - 0.25) < 1e-9
    # fast_1 delta = 0.75 - 0.5 = 0.25 in KORE's favor.
    assert abs(res["fast_p_delta_mean_ci"][1.0]["mean"] - 0.25) < 1e-9


def test_head_to_head_multiseed_confidence_interval():
    tasks = ["t1", "t2"]

    def kore_seed(sd):
        if sd == 0:
            return {"t1": [_obs(2.0)], "t2": [_obs(2.0)]}   # wins both
        return {"t1": [_obs(2.0)], "t2": [_obs(1.5)]}        # wins t1, loses t2

    def opus_seed(sd):
        if sd == 0:
            return {"t1": [_obs(1.5)], "t2": [_obs(1.5)]}
        return {"t1": [_obs(1.5)], "t2": [_obs(2.0)]}

    res = vs_opus.head_to_head(
        tasks, _kore_gen(), _stub_teacher(), budget=1, seeds=[0, 1, 2],
        seed_kore_dry_run=kore_seed, seed_opus_dry_run=opus_seed,
    )
    assert res["skipped"] is False
    # win_rate per seed: seed0 -> 1.0, seed1 & seed2 -> 0.5 => mean = 2/3, CI > 0.
    mc = res["win_rate_mean_ci"]
    assert abs(mc["mean"] - 2.0 / 3.0) < 1e-9
    assert mc["ci95"] > 0.0
    assert mc["n"] == 3
    assert len(res["per_seed_winrate"]) == 3


# --------------------------------------------------------------------------- #
# Graceful degradation (like the retention gate): never crash on an absent teacher
# --------------------------------------------------------------------------- #
def test_head_to_head_skips_cleanly_when_teacher_is_none():
    tasks = ["t1", "t2"]
    res = vs_opus.head_to_head(
        tasks, _kore_gen(), None, budget=1, seeds=[0],
        kore_dry_run={"t1": [_obs(2.0)], "t2": [_obs(2.0)]},
    )
    assert res["skipped"] is True
    assert res["skip_reason"] and "teacher" in res["skip_reason"].lower()
    assert res["opus"] is None
    assert res["win_rate_mean_ci"] is None
    # KORE numbers are still reported (fast_1 = 1.0, both correct + 2x).
    assert abs(res["kore"]["fast_p_mean_ci"][1.0]["mean"] - 1.0) < 1e-9


def test_head_to_head_skips_cleanly_on_teacher_outage_midrun():
    tasks = ["t1"]

    def _boom(messages):
        raise RuntimeError("gateway 503 (simulated sustained outage)")

    teacher = StubTeacher(fn=_boom)
    # KORE side succeeds via dry_run; the Opus policy calls the teacher, which
    # raises -> head_to_head must catch it and SKIP, not crash.
    res = vs_opus.head_to_head(
        tasks, _kore_gen(), teacher, budget=1, seeds=[0],
        kore_dry_run={"t1": [_obs(2.0)]}, opus_dry_run={"t1": [_obs(1.5)]},
    )
    assert res["skipped"] is True
    assert "503" in res["skip_reason"] or "RuntimeError" in res["skip_reason"]
    assert res["opus"] is None
    assert abs(res["kore"]["fast_p_mean_ci"][1.0]["mean"] - 1.0) < 1e-9


def test_make_opus_teacher_returns_none_on_failure():
    # 'nope' is an unknown teacher kind -> make_teacher raises -> we get None
    # (loud warning), never an exception.
    assert vs_opus.make_opus_teacher("nope") is None


# --------------------------------------------------------------------------- #
# build_policies + reporting
# --------------------------------------------------------------------------- #
def test_build_policies_includes_opus_only_when_teacher_present():
    with_teacher = vs_opus.build_policies(_kore_gen(), _stub_teacher())
    assert set(with_teacher) == {"kore", "opus"}
    assert callable(with_teacher["kore"]) and callable(with_teacher["opus"])

    without = vs_opus.build_policies(_kore_gen(), None)
    assert set(without) == {"kore"}


def _assert_ascii_dashes(text: str):
    """No em-dash / en-dash / figure-dash / horizontal-bar (U+2012..U+2015)."""
    for ch in text:
        assert not (0x2012 <= ord(ch) <= 0x2015), f"non-ASCII dash U+{ord(ch):04X} in report"


def test_report_renders_and_is_ascii_dash_free():
    tasks = ["t1", "t2"]
    res = vs_opus.head_to_head(
        tasks, _kore_gen(), _stub_teacher(), budget=1, seeds=[0, 1],
        seed_kore_dry_run=lambda sd: {"t1": [_obs(2.0)], "t2": [_obs(2.0)]},
        seed_opus_dry_run=lambda sd: {"t1": [_obs(1.5)], "t2": [_obs_bad()]},
    )
    md = vs_opus.format_vs_opus_report(res)
    assert "KORE vs Opus" in md
    assert "Win-rate" in md and "Verdict" in md
    assert "fast_p" in md
    _assert_ascii_dashes(md)


def test_report_renders_skip_path():
    res = vs_opus.head_to_head(
        ["t1"], _kore_gen(), None, budget=1, seeds=[0],
        kore_dry_run={"t1": [_obs(2.0)]},
    )
    md = vs_opus.format_vs_opus_report(res)
    assert "SKIPPED" in md
    assert "KORE-only" in md
    _assert_ascii_dashes(md)


# --------------------------------------------------------------------------- #
# Import-safety: the CPU dry-run path must NOT import torch / the serving backend
# --------------------------------------------------------------------------- #
def test_cpu_path_does_not_import_torch_or_serving_backend():
    import kore.eval.vs_opus as vo

    repo_root = os.path.abspath(os.path.join(os.path.dirname(vo.__file__), "..", ".."))
    script = textwrap.dedent(
        """
        import sys
        from kore.eval.vs_opus import head_to_head
        from kore.data.teacher import StubTeacher
        from kore.reward.reward import Observation

        CANNED = "FULL_KERNEL:\\n```python\\ndef k():\\n    pass\\n```\\n"

        def obs(su):
            w = 1.0 / su
            return Observation(compiled=True, snr_db=90.0, wall_ms=w, baseline_ms=1.0,
                               wall_by_shape={"s": w}, baseline_by_shape={"s": 1.0},
                               snr_by_shape={"s": 90.0}, validation_passed=True, dtype="bf16")

        res = head_to_head(
            ["t1"], (lambda messages, **kw: CANNED), StubTeacher(fn=lambda m: CANNED),
            budget=1, seeds=[0],
            kore_dry_run={"t1": [obs(2.0)]}, opus_dry_run={"t1": [obs(1.5)]},
        )
        assert res["skipped"] is False
        assert "torch" not in sys.modules, "torch imported on the CPU dry-run path"
        assert "kore.policy.serve" not in sys.modules, "serving backend imported on CPU path"
        print("CPU_SAFE_OK")
        """
    )
    r = subprocess.run([sys.executable, "-c", script], cwd=repo_root,
                       capture_output=True, text=True)
    assert r.returncode == 0, f"subprocess failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    assert "CPU_SAFE_OK" in r.stdout
