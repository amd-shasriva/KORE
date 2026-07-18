"""CPU-only tests for the KORE-vs-Opus head-to-head (opus_policy + head_to_head).

Entirely on CPU with a MOCK Opus policy + fabricated Observations (deterministic
fake speedups); no GPU, no network, no API key, no trained checkpoint. Covers:

  * kore.eval.opus_policy  - the teacher->PolicyFn adapter conforms to the eval
    interface (via a dependency-free StubTeacher), the multi-turn vs single-shot
    variants behave, and the no-API-key path errors CLEANLY (a single
    OpusUnavailableError; try_opus_policy degrades to (None, reason) not a crash).
  * kore.eval.head_to_head - both sides scored through evaluate_policy under a
    matched budget; per-task KORE-minus-Opus deltas, fast_p (per side + delta), the
    paired-stats battery (bootstrap CI + sign + Wilcoxon), and the geomean speedup
    ratio are correct; the JSON report is written; and a missing Opus side degrades
    to a KORE-only report instead of crashing.
"""

from __future__ import annotations

import json
import math

import pytest

from kore.data.teacher import StubTeacher
from kore.eval import head_to_head as hh
from kore.eval import opus_policy as opm
from kore.reward.reward import Observation


# --------------------------------------------------------------------------- #
# Shared fakes (CPU-only; no GPU, no network).
# --------------------------------------------------------------------------- #
def _benign_policy(task, feedback=None):
    # A neutral kernel string that trips NONE of the anti-hack patterns (no torch
    # ops, no @ operator, no vendor imports, no oracle call).
    return "def kernel(*args):\n    return compute(*args)\n"


def _obs(speedup: float, snr: float = 90.0) -> Observation:
    # Correct + timed: baseline_ms=1.0, wall=1/speedup -> worst-shape speedup exact.
    return Observation(
        compiled=True, snr_db=snr, wall_ms=1.0 / speedup, baseline_ms=1.0,
        wall_by_shape={"s": 1.0 / speedup}, baseline_by_shape={"s": 1.0},
        snr_by_shape={"s": snr}, validation_passed=True,
    )


def _obs_incorrect() -> Observation:
    # Compiled but fails the SNR gate (bf16 threshold is 25 dB) -> not correct, so
    # this side does not compete on the task.
    return Observation(compiled=True, snr_db=1.0, snr_by_shape={"s": 1.0},
                       validation_passed=False)


# Held-out tasks are plain ids (bakeoff._task_id handles strings; dtype defaults to
# bf16). Deterministic per-task fake speedups for each side:
#
#   task | KORE | Opus            | delta (KORE-Opus, non-competing=0)
#   t1   | 2.0  | 1.5             | +0.5
#   t2   | 1.4  | 1.6             | -0.2   (Opus wins this one)
#   t3   | 0.8  | incorrect (0.0) | +0.8   (KORE correct-but-slow, Opus fails)
#   t4   | 3.0  | 1.2             | +1.8
_TASKS = ["t1", "t2", "t3", "t4"]


def _kore_dry():
    return {"t1": [_obs(2.0)], "t2": [_obs(1.4)], "t3": [_obs(0.8)], "t4": [_obs(3.0)]}


def _opus_dry():
    return {"t1": [_obs(1.5)], "t2": [_obs(1.6)], "t3": [_obs_incorrect()], "t4": [_obs(1.2)]}


# =========================================================================== #
# 1. opus_policy: adapter + interface conformance (StubTeacher, no network)
# =========================================================================== #
def test_opus_policy_from_stub_teacher_conforms_to_interface():
    captured: dict = {}

    def fake_gen(messages):
        captured["messages"] = messages
        return ("ANALYSIS: memory bound\n\nPROPOSED_CHANGE: widen loads\n\n"
                "FULL_KERNEL:\n```python\ndef k():\n    return 1\n```\n")

    pol = opm.opus_policy(teacher=StubTeacher(fn=fake_gen), temperature=0.0)
    assert callable(pol)
    out = pol("t1", None)
    # Returns the PARSED FULL_KERNEL source (same contract as model_policy), not the
    # raw teacher text.
    assert out.strip() == "def k():\n    return 1"
    # It went through model_policy's transcript builder (system + user present).
    roles = [m["role"] for m in captured["messages"]]
    assert roles[0] == "system" and "user" in roles


def test_opus_multi_turn_accumulates_feedback_but_single_shot_does_not():
    seen_multi: list[list[str]] = []
    seen_single: list[list[str]] = []

    def gen_multi(messages):
        seen_multi.append([m["role"] for m in messages])
        return "FULL_KERNEL:\n```python\ndef k():\n    return 1\n```\n"

    def gen_single(messages):
        seen_single.append([m["role"] for m in messages])
        return "FULL_KERNEL:\n```python\ndef k():\n    return 1\n```\n"

    multi = opm.opus_policy(teacher=StubTeacher(fn=gen_multi), multi_turn=True)
    multi("t1", None)                                   # fresh: system+user
    multi("t1", {"correct": True, "speedup": 2.0})       # refine: +assistant +user(feedback)
    assert seen_multi[0] == ["system", "user"]
    assert "assistant" in seen_multi[1] and seen_multi[1].count("user") >= 2

    single = opm.opus_policy(teacher=StubTeacher(fn=gen_single), multi_turn=False)
    single("t1", None)
    single("t1", {"correct": True, "speedup": 2.0})      # feedback is IGNORED
    # Single-shot: every call is a fresh 2-message transcript (no cross-turn memory).
    assert all(roles == ["system", "user"] for roles in seen_single)


def test_opus_policy_no_api_key_errors_cleanly(monkeypatch):
    # Simulate the real no-key failure (ClaudeTeacher raises this exact message).
    def boom(*a, **k):
        raise RuntimeError("AMD_LLM_API_KEY not set (put it in .env.local)")

    monkeypatch.setattr("kore.data.teacher.make_teacher", boom)

    # opus_policy (no teacher) raises a SINGLE, clear OpusUnavailableError - not a
    # raw SDK/RuntimeError - and the message is actionable (names the missing key).
    with pytest.raises(opm.OpusUnavailableError) as ei:
        opm.opus_policy()
    assert "AMD_LLM_API_KEY" in str(ei.value)

    # try_opus_policy degrades to (None, reason) WITHOUT raising.
    pol, reason = opm.try_opus_policy()
    assert pol is None
    assert reason and "AMD_LLM_API_KEY" in reason


def test_build_opus_teacher_wraps_failure(monkeypatch):
    def boom(*a, **k):
        raise ImportError("No module named 'anthropic'")

    monkeypatch.setattr("kore.data.teacher.make_teacher", boom)
    with pytest.raises(opm.OpusUnavailableError) as ei:
        opm.build_opus_teacher()
    # The original cause is preserved and the message points at the fix.
    assert "anthropic" in str(ei.value)
    assert isinstance(ei.value.__cause__, ImportError)


# =========================================================================== #
# 2. head_to_head_vs_opus: paired deltas + fast_p + paired stats
# =========================================================================== #
def test_head_to_head_paired_deltas_fastp_and_stats(tmp_path):
    res = hh.head_to_head_vs_opus(
        _benign_policy, _TASKS, budget=1, mode="serial",
        opus_policy=_benign_policy,                # MOCK Opus policy
        kore_dry_run=_kore_dry(), opus_dry_run=_opus_dry(),
        seed=0, out=tmp_path / "eval" / "head_to_head_vs_opus",
    )
    assert res["opus_skipped"] is False
    assert res["n"] == 4

    # ---- per-task deltas (competing speedup = best_speedup if correct else 0) ----
    by_id = {r["task_id"]: r for r in res["per_task"]}
    assert by_id["t1"]["delta"] == pytest.approx(2.0 - 1.5)
    assert by_id["t2"]["delta"] == pytest.approx(1.4 - 1.6)
    assert by_id["t3"]["delta"] == pytest.approx(0.8 - 0.0)     # Opus failed t3
    assert by_id["t4"]["delta"] == pytest.approx(3.0 - 1.2)
    assert by_id["t3"]["opus_correct"] is False
    assert by_id["t3"]["opus_speedup"] is None
    assert res["deltas"] == pytest.approx([0.5, -0.2, 0.8, 1.8])

    # ---- fast_p per side (uncorrected denominator n=4) ----
    fp = res["fast_p"]
    assert fp["kore"][1.0] == pytest.approx(0.75)   # 2.0,1.4,3.0 > 1x  (0.8 not)
    assert fp["kore"][1.5] == pytest.approx(0.50)   # 2.0,3.0 > 1.5x
    assert fp["kore"][2.0] == pytest.approx(0.25)   # only 3.0 > 2x
    assert fp["opus"][1.0] == pytest.approx(0.75)   # 1.5,1.6,1.2 > 1x
    assert fp["opus"][1.5] == pytest.approx(0.25)   # only 1.6 > 1.5x
    assert fp["opus"][2.0] == pytest.approx(0.0)
    # per-threshold fast_p delta.
    assert fp["delta"][1.5] == pytest.approx(0.25)
    assert fp["delta"][1.0] == pytest.approx(0.0)

    # ---- paired stats on the KORE-minus-Opus per-task deltas ----
    pd = res["paired_delta"]
    assert pd["n"] == 4
    assert pd["effect_size"] == pytest.approx(0.725)            # mean of the deltas
    assert pd["direction"] == "kore_better"
    assert pd["ci"][0] <= pd["effect_size"] <= pd["ci"][1]      # CI brackets effect
    # exact two-sided sign test: 3 pos / 1 neg -> 2 * P(X<=1; n=4, p=0.5) = 0.625.
    assert pd["sign"]["p_value"] == pytest.approx(0.625)
    # the whole battery is present (bootstrap + sign + Wilcoxon).
    assert {"bootstrap", "sign", "wilcoxon", "p_bootstrap", "p_sign",
            "p_wilcoxon"} <= set(pd)

    # ---- geomean speedup RATIO on the both-correct tasks (t1, t2, t4) ----
    ratio = res["paired_speedup_ratio"]
    assert ratio is not None and ratio["n"] == 3
    expected = math.exp((math.log(2.0 / 1.5) + math.log(1.4 / 1.6)
                         + math.log(3.0 / 1.2)) / 3.0)
    assert ratio["effect_size"] == pytest.approx(expected, rel=1e-6)
    assert ratio["effect_kind"] == "geomean_speedup_ratio"
    assert res["n_both_correct"] == 3

    # ---- win / loss / tie tally (margin=1.0) ----
    assert res["winners"] == {"kore": 3, "opus": 1, "tie": 0, "both_incorrect": 0}
    assert by_id["t2"]["winner"] == "opus"   # 1.6 (Opus) vs 1.4 (KORE)

    # ---- JSON report persisted + parseable ----
    jpath = tmp_path / "eval" / "head_to_head_vs_opus.json"
    mdpath = tmp_path / "eval" / "head_to_head_vs_opus.md"
    assert jpath.exists() and mdpath.exists()
    loaded = json.loads(jpath.read_text())
    assert loaded["verdict"] and "KORE WINS" in loaded["verdict"]
    assert "head-to-head" in mdpath.read_text()


def test_head_to_head_with_real_opus_policy_object_via_stub(tmp_path):
    # Build the REAL Opus-as-policy object (adapter + model_policy) from a
    # dependency-free StubTeacher, then bench it head-to-head under dry_run
    # measurements - proving the teacher flows through the IDENTICAL eval path.
    opus_pol = opm.opus_policy(teacher=StubTeacher(), temperature=0.0)
    res = hh.head_to_head_vs_opus(
        _benign_policy, _TASKS, budget=1,
        opus_policy=opus_pol,
        kore_dry_run=_kore_dry(), opus_dry_run=_opus_dry(),
        out=tmp_path / "ht",
    )
    assert res["opus_skipped"] is False
    assert res["paired_delta"]["direction"] == "kore_better"
    assert res["winners"]["kore"] == 3


def test_head_to_head_reports_when_opus_wins(tmp_path):
    # Sanity: the paired verdict flips when Opus is uniformly faster (direction and
    # winners must both reflect Opus dominance).
    kore_dry = {"t1": [_obs(1.1)], "t2": [_obs(1.0)], "t3": [_obs(1.2)], "t4": [_obs(0.9)]}
    opus_dry = {"t1": [_obs(2.0)], "t2": [_obs(1.9)], "t3": [_obs(2.1)], "t4": [_obs(2.2)]}
    res = hh.head_to_head_vs_opus(
        _benign_policy, _TASKS, budget=1,
        opus_policy=_benign_policy,
        kore_dry_run=kore_dry, opus_dry_run=opus_dry, seed=1,
    )
    assert res["paired_delta"]["direction"] == "baseline_better"   # Opus is the baseline
    assert res["winners"]["opus"] == 4
    assert res["paired_speedup_ratio"]["effect_size"] < 1.0


# =========================================================================== #
# 3. Graceful degradation: no Opus -> KORE-only report, no crash
# =========================================================================== #
def test_head_to_head_degrades_without_opus(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise RuntimeError("AMD_LLM_API_KEY not set")

    monkeypatch.setattr("kore.data.teacher.make_teacher", boom)

    res = hh.head_to_head_vs_opus(
        _benign_policy, _TASKS, budget=1,
        kore_dry_run=_kore_dry(),
        out=tmp_path / "eval" / "ht",
    )
    # Opus SKIPPED (no crash), but the KORE side is fully reported.
    assert res["opus_skipped"] is True
    assert res["skip_reason"] and "AMD_LLM_API_KEY" in res["skip_reason"]
    assert res["opus"] is None
    assert res["paired_delta"] is None and res["paired_speedup_ratio"] is None
    # KORE-only fast_p is still present and correct.
    assert res["kore"]["fast_p"][1.0] == pytest.approx(0.75)
    # The KORE-only report is still written.
    assert (tmp_path / "eval" / "ht.json").exists()


def test_head_to_head_degrades_on_teacher_outage_mid_eval(tmp_path):
    # A teacher that authenticates but then fails EVERY generate() (e.g. a sustained
    # gateway outage) must SKIP the Opus side mid-eval, not crash the head-to-head.
    def always_raise(messages):
        raise RuntimeError("gateway 503 (sustained outage)")

    # The Opus policy IS invoked (its dry_run supplies measurements), but the teacher
    # raises when the policy calls generate() -> evaluate_policy propagates ->
    # head_to_head catches it and skips gracefully (KORE numbers survive).
    opus_pol = opm.opus_policy(teacher=StubTeacher(fn=always_raise))
    res = hh.head_to_head_vs_opus(
        _benign_policy, _TASKS, budget=1,
        opus_policy=opus_pol,
        kore_dry_run=_kore_dry(), opus_dry_run=_opus_dry(),
    )
    assert res["opus_skipped"] is True
    assert "teacher failed during eval" in res["skip_reason"]
    assert "gateway 503" in res["skip_reason"]
    assert res["kore"]["fast_p"][1.0] == pytest.approx(0.75)
