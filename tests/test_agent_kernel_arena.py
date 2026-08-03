"""AgentKernelArena adapter: score the way AKA scores, or the number is a lie.

This is the only benchmark where "we beat Opus" is checkable, because AMD
publishes frontier-agent numbers on gfx950 for the same tasks. That only holds
if we reproduce their contract exactly: the same gate order, the same scoring
formula, and no invented speedups. A number that looks comparable and is not is
worse than no number.

We re-implement the runner because this cluster has no container runtime at all,
so AKA's Docker path is unavailable. The tasks declare their own argv, so the
contract is portable even though the image is not.
"""

from __future__ import annotations


from pathlib import Path

import pytest

from kore.eval.agent_kernel_arena import (PUBLISHED_OPUS_MEAN_SPEEDUP, ArenaResult,
                                          _parse_speedup, discover_tasks,
                                          evaluate_task, load_task, score_result,
                                          summarize)


def _task_dir(tmp_path: Path, name: str, task_type: str, *,
              arch=None, status=None, perf_out=None,
              compile_ok=True, correct_ok=True) -> Path:
    d = tmp_path / "tasks" / task_type / name
    d.mkdir(parents=True)
    (d / "kernel.py").write_text("# kernel\n")
    plat = ""
    if arch or status:
        plat = "platform_support:\n"
        if arch: plat += f"  required_arch: {arch}\n"
        if status: plat += f"  status: {status}\n"
    lines = [
        f"task_type: {task_type}",
        "source_file_path:",
        "- kernel.py",
        "target_kernel_functions:",
        "- _k",
        "compile_command:",
        f'- python3 -c "exit({0 if compile_ok else 1})"',
        "correctness_command:",
        f'- python3 -c "exit({0 if correct_ok else 1})"',
    ]
    if perf_out is not None:
        (d / "perf.py").write_text(f"print({perf_out!r})\n")
        lines += ["performance_command:", "- python3 perf.py"]
    if plat:
        lines.append(plat.rstrip("\n"))
    (d / "config.yaml").write_text("\n".join(lines) + "\n")
    return d


# --------------------------------------------------------------- scoring ---
def test_scoring_matches_the_published_policy():
    # compile fails -> 0 ; correctness fails -> 20 ; both -> 120 + speedup*100
    assert score_result(False, False, None) == 0
    assert score_result(True, False, None) == 20
    assert score_result(True, True, None) == 120
    assert score_result(True, True, 2.13) == pytest.approx(333.0)


def test_speedup_only_counts_when_correctness_passed():
    # Scoring a fast-but-wrong kernel would reward the exact shortcut the
    # training filter exists to remove.
    assert score_result(True, False, 9.0) == 20


# --------------------------------------------------------------- parsing ---
def test_parses_speedup_from_json_line():
    b, o, s = _parse_speedup('noise\n{"baseline_time": 2.0, "optimized_time": 0.5, "speedup_ratio": 4.0}')
    assert (b, o, s) == (2.0, 0.5, 4.0)


def test_parses_speedup_from_labelled_text():
    _, _, s = _parse_speedup("...\nspeedup: 3.25\n")
    assert s == 3.25


def test_unparseable_output_yields_no_speedup_rather_than_a_guess():
    assert _parse_speedup("benchmark finished")[2] is None


def test_nonpositive_or_nan_speedup_is_rejected():
    # A failed measurement must not enter the score as a real number.
    assert _parse_speedup('{"speedup_ratio": 0}')[2] is None
    assert _parse_speedup('{"speedup_ratio": -1}')[2] is None


# ------------------------------------------------------------- discovery ---
def test_discovery_filters_by_arch_and_skip(tmp_path):
    _task_dir(tmp_path, "a", "triton2triton")
    _task_dir(tmp_path, "b", "triton2triton", arch="gfx942")
    _task_dir(tmp_path, "c", "triton2triton", status="skip")
    _task_dir(tmp_path, "d", "hip2hip", arch="gfx950")
    found = {t.task_id: t for t in discover_tasks(tmp_path, gpu_arch="gfx950")}
    assert set(found) == {"triton2triton/a", "hip2hip/d"}


def test_discovery_can_select_task_types(tmp_path):
    _task_dir(tmp_path, "a", "triton2triton")
    _task_dir(tmp_path, "d", "hip2hip")
    got = discover_tasks(tmp_path, task_types=["triton2triton"], gpu_arch="gfx950")
    assert [t.task_type for t in got] == ["triton2triton"]


def test_load_task_reads_the_contract(tmp_path):
    d = _task_dir(tmp_path, "a", "triton2triton", perf_out="{}")
    t = load_task(d / "config.yaml")
    assert t.task_type == "triton2triton"
    assert t.target_functions == ["_k"]
    assert t.compile_command and t.correctness_command and t.performance_command


# ---------------------------------------------------------------- gating ---
def test_compile_failure_short_circuits(tmp_path):
    d = _task_dir(tmp_path, "a", "triton2triton", compile_ok=False)
    r = evaluate_task(load_task(d / "config.yaml"), d, timeout=60)
    assert (r.compiled, r.correct, r.score) == (False, False, 0.0)


def test_correctness_failure_scores_twenty_and_skips_timing(tmp_path):
    d = _task_dir(tmp_path, "a", "triton2triton", correct_ok=False,
                  perf_out='{"speedup_ratio": 99}')
    r = evaluate_task(load_task(d / "config.yaml"), d, timeout=60)
    assert r.compiled and not r.correct
    assert r.score == 20.0
    assert r.speedup is None, "a failed kernel must never be timed into the score"


def test_full_pass_scores_with_speedup(tmp_path):
    d = _task_dir(tmp_path, "a", "triton2triton",
                  perf_out='{"baseline_time": 4.0, "optimized_time": 2.0, "speedup_ratio": 2.0}')
    r = evaluate_task(load_task(d / "config.yaml"), d, timeout=60)
    assert r.compiled and r.correct and r.speedup == 2.0
    assert r.score == pytest.approx(320.0)


# --------------------------------------------------------------- summary ---
def test_summary_compares_against_the_published_opus_bar():
    rs = [
        ArenaResult("t1", "triton2triton", True, True, speedup=3.0, score=420),
        ArenaResult("t2", "triton2triton", True, True, speedup=3.0, score=420),
    ]
    s = summarize(rs)["by_type"]["triton2triton"]
    assert s["opus_published_mean_speedup"] == 2.13
    assert s["beats_opus"] is True


def test_summary_does_not_claim_a_win_without_speedups():
    rs = [ArenaResult("t1", "hip2hip", True, False, score=20)]
    s = summarize(rs)["by_type"]["hip2hip"]
    assert s["mean_speedup"] is None
    assert s["beats_opus"] is False


def test_published_bars_are_the_paper_numbers():
    # Guard against these drifting: they are the claim we measure against.
    assert PUBLISHED_OPUS_MEAN_SPEEDUP == {
        "torch2hip": 6.89, "hip2hip": 6.69, "triton2triton": 2.13,
    }
