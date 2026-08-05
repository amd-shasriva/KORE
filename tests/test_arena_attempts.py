"""Multi-attempt evaluation must match how the published numbers were produced.

AKA's reference agents run with ``max_iterations: 3`` and full tool access in the
workspace: they compile, read the compiler error, and try again. Scoring a single
shot against those numbers compares two procedures, not two models, and the gap
falls hardest on the categories gated on compiling a translation unit correctly
first time -- torch2hip, which carries the highest bar of all.

Two properties are load-bearing and easy to get subtly wrong, so both are pinned:
the harness's verdict must actually reach the next attempt, and the BEST attempt
must win rather than the last.
"""

import pathlib
import sys
import tempfile
import types

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import run_agent_kernel_arena as R  # noqa: E402
from kore.eval.agent_kernel_arena import ArenaResult  # noqa: E402


def _run_attempts(results, attempts=3):
    """Drive _attempt_task against a scripted sequence of harness verdicts."""
    seen = {"n": 0, "feedback": []}

    def fake_eval(task, ws, timeout=0, reference_latency=None):
        r = results[min(seen["n"], len(results) - 1)]
        seen["n"] += 1
        return r

    def policy(prompt, feedback=None):
        seen["feedback"].append(feedback)
        return "```python\nkernel\n```"

    orig_eval, orig_write, orig_health = (
        R.evaluate_task, R._write_answer, R._generation_health)
    R.evaluate_task = fake_eval
    R._write_answer = lambda p, c: None
    R._generation_health = lambda a, b: ""
    try:
        best = R._attempt_task(
            types.SimpleNamespace(task_id="t", task_type="triton2triton"),
            pathlib.Path(tempfile.mkdtemp()), "kernel.py", "PROMPT", policy,
            types.SimpleNamespace(attempts=attempts, timeout=60), None)
    finally:
        R.evaluate_task, R._write_answer, R._generation_health = (
            orig_eval, orig_write, orig_health)
    return best, seen


def _res(**kw):
    base = dict(task_id="t", task_type="triton2triton")
    return ArenaResult(**{**base, **kw})


def test_the_best_attempt_wins_not_the_last():
    """A later attempt can regress -- a model chasing speed breaks correctness it
    already had. Reporting the final state would score the regression."""
    best, seen = _run_attempts([
        _res(compiled=False, correct=False, score=0.0, error="undefined symbol"),
        _res(compiled=True, correct=True, speedup=1.5, score=270.0),
        _res(compiled=True, correct=False, score=20.0, error="SNR 3dB"),
    ])
    assert seen["n"] == 3
    assert best.score == 270.0 and best.correct
    assert best.detail["attempt"] == 2


def test_the_compiler_error_reaches_the_next_attempt():
    """Retrying without telling the model what broke is just resampling."""
    _, seen = _run_attempts([
        _res(compiled=False, correct=False, score=0.0,
             error="exit 1: no member named 'getCurrentHIPStream'"),
        _res(compiled=True, correct=True, score=220.0),
    ])
    assert seen["feedback"][0] is None, "first attempt must start clean"
    assert "getCurrentHIPStream" in str(seen["feedback"][1])


def test_a_correct_kernel_still_gets_another_attempt():
    """Score is 120 + speedup*100, so stopping at 'correct' leaves the entire
    speed component unexplored. The feedback renderer asks for one further
    optimization precisely so a passing kernel keeps improving."""
    best, seen = _run_attempts([
        _res(compiled=True, correct=True, speedup=1.1, score=230.0),
        _res(compiled=True, correct=True, speedup=3.0, score=420.0),
        _res(compiled=True, correct=True, speedup=2.0, score=320.0),
    ])
    assert seen["n"] == 3
    assert best.score == 420.0


@pytest.mark.parametrize("n", [1, 2, 5])
def test_attempt_budget_is_honoured_exactly(n):
    _, seen = _run_attempts([_res(compiled=False, score=0.0)], attempts=n)
    assert seen["n"] == n


def test_single_shot_remains_available_and_makes_one_call():
    """A deliberate single-shot measurement is still a valid thing to ask for."""
    best, seen = _run_attempts([_res(compiled=True, correct=True, score=220.0)],
                               attempts=1)
    assert seen["n"] == 1 and seen["feedback"] == [None]
    assert best.score == 220.0
