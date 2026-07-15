"""CPU-only tests for CONVERGENT win reconstruction (KORE Stage 3 quality gate).

The audited failures these lock down: raw greedy search was stored verbatim as a
"win" even when the wall oscillated and never converged, the footer metrics did not
multiply out (initial/final != speedup), and analyses described optimizations the
emitted code never implemented. These tests assert:

  * non-convergent trajectories are PRUNED to the strictly-improving path;
  * footer metrics are recomputed from the kept turns and multiply out exactly;
  * every per-turn "speedup=Xx" is consistent with its "wall=Yus" and the walls on
    the improving path decrease monotonically;
  * turns that "describe but don't implement" their claim are dropped;
  * a real regression is optionally kept as an explicit "tried X, slower, reverted"
    lesson without changing the final kernel;
  * a search that never improves yields no win.

No GPU, no teacher model, no torch/triton. Pure reconstruction + a scripted stub.
"""

from __future__ import annotations

import re

from kore.data.gen_wins import (
    build_convergent_trajectory,
    generate_wins,
    WinTurn,
)
from kore.data.schemas import WinRecord
from kore.policy.format import format_assistant_turn
from kore.reward.reward import Observation


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _body(tag: str) -> str:
    return f"def k():\n    x = {tag}\n    return x"


def _turn(wall, tag, analysis="Improve memory throughput.",
          proposed="Rewrite the inner section.", correct=True, snr=999.0):
    """A raw evolve turn with a vague (non-knob) claim so it survives claim-check."""
    src = _body(tag)
    return WinTurn(response=format_assistant_turn(analysis, proposed, src),
                   cand_src=src, correct=correct, wall_us=wall, snr_db=snr)


def _speedup_lines(messages):
    """(wall, speedup) pairs from every feedback string in the chat."""
    out = []
    for m in messages:
        for w, s in re.findall(r"wall=([\d.]+)us speedup=([\d.]+)x", m["content"]):
            out.append((float(w), float(s)))
    return out


def _assert_consistent(messages, initial):
    for w, s in _speedup_lines(messages):
        assert abs(initial / w - s) < 5e-3, (w, s, initial / w)


# --------------------------------------------------------------------------- #
# 1. Prune a non-convergent (oscillating) search to the improving path.
# --------------------------------------------------------------------------- #
def test_prunes_to_monotonic_improving_path():
    # raw walls: 100(seed) -> 90 keep -> 95 regress -> 70 keep -> BAD -> 72 regress -> 60 keep
    turns = [_turn(90, "a"), _turn(95, "b"), _turn(70, "c"),
             _turn(0, "d", correct=False), _turn(72, "e"), _turn(60, "f")]
    built = build_convergent_trajectory(_body("seed"), 100.0, 999.0, turns,
                                        include_regression_lesson=False)
    assert built is not None
    msgs = built["messages"]
    assert [m["role"] for m in msgs] == \
        ["system", "user", "assistant", "user", "assistant", "user", "assistant"]

    text = " ".join(m["content"] for m in msgs)
    # only the improving kernels survive; regressions / incorrect turns are gone
    assert "x = a" in text and "x = c" in text and "x = f" in text
    assert "x = b" not in text and "x = d" not in text and "x = e" not in text

    # walls quoted in the feedback strictly decrease along the improving path
    walls = [w for w, _ in _speedup_lines(msgs)]
    assert walls == [100.0, 90.0, 70.0]
    assert all(walls[i] > walls[i + 1] for i in range(len(walls) - 1))


def test_footer_recomputed_and_multiplies_out():
    turns = [_turn(90, "a"), _turn(95, "b"), _turn(70, "c"), _turn(60, "f")]
    built = build_convergent_trajectory(_body("seed"), 100.0, 999.0, turns,
                                        include_regression_lesson=False)
    assert built["initial_wall_us"] == 100.0
    assert built["final_wall_us"] == 60.0          # best KEPT wall, not raw last
    assert abs(built["speedup"] - 100.0 / 60.0) < 1e-9
    # the footer multiplies out (the audited "76.9/18.9/1.37" inconsistency is gone)
    assert abs(built["initial_wall_us"]
               - built["final_wall_us"] * built["speedup"]) < 1e-9
    assert built["final_source"] == _body("f")
    _assert_consistent(built["messages"], 100.0)


def test_every_feedback_speedup_is_consistent():
    turns = [_turn(80, "a"), _turn(64, "b"), _turn(40, "c")]
    built = build_convergent_trajectory(_body("seed"), 100.0, 999.0, turns,
                                        include_regression_lesson=False)
    # 100/80=1.25, 100/64=1.5625, footer 100/40=2.5 - all internally consistent
    _assert_consistent(built["messages"], 100.0)
    assert abs(built["speedup"] - 2.5) < 1e-9


# --------------------------------------------------------------------------- #
# 2. Drop turns whose ANALYSIS claim is not implemented in the code diff.
# --------------------------------------------------------------------------- #
def test_drops_describe_but_dont_implement_turn():
    seed = "def k():\n    BLOCK_M = 64\n    num_stages = 1\n    return 0"
    # t0 CLAIMS num_warps but only changes BLOCK_M -> unsupported -> dropped
    t0_src = seed.replace("BLOCK_M = 64", "BLOCK_M = 128")
    t0 = WinTurn(response=format_assistant_turn(
        "Increase num_warps to improve occupancy.", "Set num_warps=8.", t0_src),
        cand_src=t0_src, correct=True, wall_us=80.0, snr_db=999.0)
    # t1 claims num_stages AND changes it -> supported -> kept
    t1_src = seed.replace("num_stages = 1", "num_stages = 2")
    t1 = WinTurn(response=format_assistant_turn(
        "Deepen the software pipeline.", "Set num_stages=2.", t1_src),
        cand_src=t1_src, correct=True, wall_us=60.0, snr_db=999.0)

    built = build_convergent_trajectory(seed, 100.0, 999.0, [t0, t1],
                                        include_regression_lesson=False)
    text = " ".join(m["content"] for m in built["messages"])
    assert "BLOCK_M = 128" not in text          # the unsupported-claim turn is gone
    assert "num_stages = 2" in text             # the supported-claim turn survives
    assert built["final_source"] == t1_src


def test_keeps_turn_when_concrete_claim_is_implemented():
    seed = "def k():\n    num_warps = 4\n    return 0"
    src = seed.replace("num_warps = 4", "num_warps = 8")
    t = WinTurn(response=format_assistant_turn(
        "Raise num_warps for more occupancy.", "Set num_warps=8.", src),
        cand_src=src, correct=True, wall_us=50.0, snr_db=999.0)
    built = build_convergent_trajectory(seed, 100.0, 999.0, [t],
                                        include_regression_lesson=False)
    assert built is not None and built["final_source"] == src


# --------------------------------------------------------------------------- #
# 3. Regression -> explicit 2-turn "tried X, slower, reverted" lesson.
# --------------------------------------------------------------------------- #
def test_regression_kept_as_reverted_lesson():
    turns = [_turn(70, "p"), _turn(85, "r"), _turn(50, "q")]
    built = build_convergent_trajectory(_body("seed"), 100.0, 999.0, turns,
                                        include_regression_lesson=True)
    msgs = built["messages"]
    text = " ".join(m["content"] for m in msgs)
    # the lesson names the slower measurement and reverts
    assert "SLOWER" in text
    assert "Revert" in text or "Reverting" in text
    assert "x = r" in text                      # the regression candidate is shown
    # ...but the final kernel is still the fast one, and the footer is unchanged
    assert built["final_source"] == _body("q")
    assert abs(built["speedup"] - 2.0) < 1e-9
    # even with the (intentionally slower) lesson line, each speedup matches its wall
    _assert_consistent(msgs, 100.0)
    # the revert lesson quotes the regression wall (85us) and the pivot wall (70us)
    assert "85.0us" in text and "70.0us" in text


def test_no_lesson_when_no_regression_available():
    turns = [_turn(70, "p"), _turn(50, "q")]
    with_lesson = build_convergent_trajectory(_body("seed"), 100.0, 999.0, turns,
                                              include_regression_lesson=True)
    without = build_convergent_trajectory(_body("seed"), 100.0, 999.0, turns,
                                          include_regression_lesson=False)
    # no correct regression to teach from -> identical trajectories
    assert len(with_lesson["messages"]) == len(without["messages"])


# --------------------------------------------------------------------------- #
# 4. No net improvement -> not a win.
# --------------------------------------------------------------------------- #
def test_no_win_when_nothing_improves():
    turns = [_turn(100, "a"), _turn(101, "b"), _turn(100, "c")]
    assert build_convergent_trajectory(_body("seed"), 100.0, 999.0, turns) is None


def test_no_win_when_all_incorrect():
    turns = [_turn(50, "a", correct=False), _turn(40, "b", correct=False)]
    assert build_convergent_trajectory(_body("seed"), 100.0, 999.0, turns) is None


# --------------------------------------------------------------------------- #
# 5. End-to-end generate_wins with a scripted teacher + a marker-driven env.
# --------------------------------------------------------------------------- #
def _kernel(wall, tag, correct=True):
    c = "1" if correct else "0"
    return (f"def k():\n    # wall={wall} snr=999 correct={c}\n"
            f"    x = {tag}\n    return x")


def _resp(wall, tag, analysis="Improve throughput.", proposed="Adjust the kernel.",
          correct=True):
    return format_assistant_turn(analysis, proposed, _kernel(wall, tag, correct))


def _meta(src, key):
    m = re.search(rf"{key}=([\d.]+)", src or "")
    return float(m.group(1)) if m else None


class _MarkerEnv:
    """Verifier stub: reads wall/snr/correct markers embedded in the kernel."""

    def step(self, source, full_validation=True, multi_shape=True):
        wall = _meta(source, "wall")
        snr = _meta(source, "snr") or 999.0
        correct = _meta(source, "correct") != 0.0
        return Observation(
            compiled=True, validation_passed=correct, snr_db=snr,
            snr_by_shape={"s": snr},
            wall_ms=(wall / 1000.0 if wall is not None else None),
            baseline_ms=1.0, dtype="bf16")


class _SeqTeacher:
    def __init__(self, responses):
        self._responses = list(responses)
        self.i = 0
        self.calls: list[list[dict]] = []

    def generate(self, messages):
        self.calls.append(list(messages))
        r = self._responses[self.i]
        self.i += 1
        return r


class _Task:
    task_id = "gen_row_sum_bf16"
    operation = "row_sum"
    dtype = "bf16"
    gpu_target = "gfx942"

    def __init__(self, seed_wall=100):
        self.seed_source = _kernel(seed_wall, "seed")


def test_generate_wins_end_to_end_is_convergent_and_consistent():
    task = _Task(seed_wall=100)
    # raw search: improve->regress->improve->plateau
    teacher = _SeqTeacher([
        _resp(70, "a"),      # 100 -> 70 (keep)
        _resp(90, "b"),      # regress (drop, but eligible for the lesson)
        _resp(50, "c"),      # 70 -> 50 (keep, final)
        _resp(55, "d"),      # plateau/regress (drop)
    ])
    recs = generate_wins(task, teacher, _MarkerEnv(), gens=4)
    assert len(recs) == 1
    w = recs[0]
    assert isinstance(w, WinRecord)
    assert w.initial_wall_us == 100.0 and w.final_wall_us == 50.0
    assert abs(w.speedup - 2.0) < 1e-9
    # footer multiplies out and the final kernel is the fast one
    assert abs(w.initial_wall_us - w.final_wall_us * w.speedup) < 1e-9
    assert "x = c" in w.final_source
    # trajectory is a clean chat and every quoted speedup matches its wall
    assert w.trajectory[0]["role"] == "system"
    assert w.trajectory[-1]["role"] == "assistant"
    _assert_consistent(w.trajectory, 100.0)
    # the dropped plateau kernel never appears; the lesson keeps the 90us regression
    traj_text = " ".join(m["content"] for m in w.trajectory)
    assert "x = d" not in traj_text
    assert "SLOWER" in traj_text and "x = b" in traj_text


def test_generate_wins_no_lesson_is_strictly_monotonic():
    task = _Task(seed_wall=100)
    teacher = _SeqTeacher([_resp(70, "a"), _resp(90, "b"), _resp(50, "c")])
    recs = generate_wins(task, teacher, _MarkerEnv(), gens=3,
                         include_regression_lesson=False)
    assert len(recs) == 1
    w = recs[0]
    walls = [wall for wall, _ in _speedup_lines(w.trajectory)]
    assert walls == sorted(walls, reverse=True)
    assert all(walls[i] > walls[i + 1] for i in range(len(walls) - 1))
    traj_text = " ".join(m["content"] for m in w.trajectory)
    assert "x = b" not in traj_text  # the regression is dropped entirely


def test_generate_wins_returns_empty_when_no_speedup():
    task = _Task(seed_wall=100)
    teacher = _SeqTeacher([_resp(120, "a"), _resp(110, "b")])
    recs = generate_wins(task, teacher, _MarkerEnv(), gens=2)
    assert recs == []
