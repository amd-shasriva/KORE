"""CPU tests for the KernelForge-derived win-finder upgrades (Tiers 1-3).

  * Tier 1 - the AMD-Triton playbook is injected into the TEACHER's LIVE context
    but NEVER into the STORED SFT trajectory (no train/deploy contract drift).
  * Tier 2 - measured rocprofv3 counters map to a targeted next-move hint, and the
    win feedback only appends it for a correct+profiled turn (fail-safe when absent).
  * Tier 3 - the experience ledger distills/dedups/bounds do-NOT-repeat constraints
    and, when shared, carries them ACROSS a task's trajectories.

No GPU, no teacher model, no torch/triton.
"""

from __future__ import annotations

import re

from kore.data import amd_knowledge as ak
from kore.data.gen_wins import _bottleneck_feedback, _feedback, generate_wins
from kore.data.prompts import SYSTEM_PROMPT
from kore.policy.format import format_assistant_turn
from kore.reward.reward import Observation, compute_reward


# --------------------------------------------------------------------------- #
# Tier 1: playbook + live_system_prompt
# --------------------------------------------------------------------------- #
def test_playbook_loads_with_core_rules():
    pb = ak.playbook()
    assert len(pb) > 1500
    for token in ("num_warps", "MFMA", "tl.dot", "occupancy", "BLOCK_M", "fp8"):
        assert token in pb, token


def test_live_system_prompt_augments_but_preserves_base():
    lsp = ak.live_system_prompt("BASE_SYSTEM")
    assert lsp.startswith("BASE_SYSTEM")
    assert len(lsp) > len("BASE_SYSTEM") + 1000
    assert "num_warps" in lsp  # the playbook is appended


# --------------------------------------------------------------------------- #
# Tier 3: experience ledger
# --------------------------------------------------------------------------- #
def test_ledger_distills_dedups_and_bounds():
    led = ak.ExperienceLedger(max_constraints=2)
    assert led.render() == "" and len(led) == 0
    led.record(error_text="triton out of resource: shared memory")  # -> LDS rule only
    assert "LDS" in led.render() and len(led) == 1
    led.record(error_text="out of resource: shared memory")  # duplicate -> no growth
    assert len(led) == 1
    led.record(error_text="register spill to scratch")        # 2nd distinct
    led.record(error_text="failed to compile: syntax error")  # 3rd distinct -> bounded
    assert len(led) == 2
    assert led.render().startswith("Known constraints (do NOT repeat")


def test_ledger_one_record_can_learn_multiple_constraints():
    # an incorrect + resource-exhausted attempt legitimately teaches BOTH lessons
    led = ak.ExperienceLedger()
    led.record(error_text="out of resource: shared memory", outcome="incorrect (snr)")
    r = led.render()
    assert "LDS" in r and "correctness" in r and len(led) == 2


def test_ledger_note_passthrough():
    led = ak.ExperienceLedger()
    led.record(note="do not fuse the permute into the upstream store")
    assert "do not fuse the permute" in led.render()


# --------------------------------------------------------------------------- #
# Tier 2: bottleneck feedback mapping
# --------------------------------------------------------------------------- #
def test_bottleneck_feedback_maps_labels():
    # "no matrix cores" is sound only when the MFMA counter was collected and is
    # explicitly zero; absence is unavailable, not zero.
    nomc = _bottleneck_feedback({
        "SQ_INSTS_VALU": 1000,
        "SQ_INSTS_VMEM": 10,
        "SQ_INSTS_VALU_MFMA_MOPS_BF16": 0,
    })
    assert "no-matrix-cores" in nomc and "tl.dot" in nomc
    mem = _bottleneck_feedback({"SQ_WAIT_INST_VMEM": 90, "SQ_WAIT_INST_ANY": 100})
    assert "memory-bound" in mem and "128-bit" in mem
    assert _bottleneck_feedback(None) == "" and _bottleneck_feedback({}) == ""


def test_feedback_appends_bottleneck_only_when_correct_and_profiled():
    ok = Observation(compiled=True, validation_passed=True, snr_db=999.0,
                     snr_by_shape={"s": 999.0}, wall_ms=0.05, baseline_ms=0.1,
                     dtype="bf16")
    rr = compute_reward(ok, "def k():\n    return 0", dtype="bf16")
    assert "HARDWARE COUNTERS" not in _feedback(ok, rr)
    withc = _feedback(ok, rr, counters={
        "SQ_INSTS_VALU": 1000,
        "SQ_INSTS_VMEM": 10,
        "SQ_INSTS_VALU_MFMA_MOPS_BF16": 0,
    })
    assert "HARDWARE COUNTERS" in withc and "tl.dot" in withc


# --------------------------------------------------------------------------- #
# gen_wins integration harness (marker-driven env + spy teacher)
# --------------------------------------------------------------------------- #
def _kernel(wall, tag, correct=True):
    c = "1" if correct else "0"
    return (f"def k():\n    # wall={wall} snr=999 correct={c}\n"
            f"    x = {tag}\n    return x")


def _resp(wall, tag, correct=True):
    return format_assistant_turn("Improve throughput.", "Adjust the kernel.",
                                 _kernel(wall, tag, correct))


def _meta(src, key):
    m = re.search(rf"{key}=([\d.]+)", src or "")
    return float(m.group(1)) if m else None


class _MarkerEnv:
    """Verifier stub. Deliberately has NO collect_counters, so Tier 2 must degrade
    to wall-only feedback (fail-safe) rather than crash."""

    def step(self, source, full_validation=True, multi_shape=True):
        wall = _meta(source, "wall")
        snr = _meta(source, "snr") or 999.0
        correct = _meta(source, "correct") != 0.0
        return Observation(compiled=True, validation_passed=correct, snr_db=snr,
                           snr_by_shape={"s": snr},
                           wall_ms=(wall / 1000.0 if wall is not None else None),
                           baseline_ms=1.0, dtype="bf16")


class _SpyTeacher:
    def __init__(self, responses):
        self._r = list(responses)
        self.i = 0
        self.calls: list[list[dict]] = []

    def generate(self, messages):
        self.calls.append([dict(m) for m in messages])
        r = self._r[self.i]
        self.i += 1
        return r


class _Task:
    task_id = "gen_row_sum_bf16"
    operation = "row_sum"
    dtype = "bf16"
    gpu_target = "gfx942"

    def __init__(self):
        self.seed_source = _kernel(100, "seed")


def test_playbook_in_live_context_but_not_stored_trajectory():
    task = _Task()
    teacher = _SpyTeacher([_resp(70, "a"), _resp(50, "c")])
    recs = generate_wins(task, teacher, _MarkerEnv(), gens=2,
                         include_regression_lesson=False)
    assert len(recs) == 1
    # LIVE: every teacher call's system message is playbook-augmented (Tier 1).
    assert teacher.calls
    for call in teacher.calls:
        assert call[0]["role"] == "system"
        assert len(call[0]["content"]) > len(SYSTEM_PROMPT)
        assert "num_warps" in call[0]["content"]
    # STORED: the trajectory keeps the CANONICAL system prompt (no drift).
    traj = recs[0].trajectory
    assert traj[0]["role"] == "system"
    assert traj[0]["content"] == SYSTEM_PROMPT
    assert "Optimization Playbook" not in traj[0]["content"]


def test_shared_ledger_carries_constraints_across_trajectories():
    led = ak.ExperienceLedger()
    task = _Task()
    # trajectory 1: an INCORRECT candidate populates the shared ledger (Tier 3).
    t1 = _SpyTeacher([_resp(70, "bad", correct=False)])
    generate_wins(task, t1, _MarkerEnv(), gens=1, ledger=led)
    assert len(led) >= 1
    # trajectory 2 (same ledger): its first user turn now carries the constraints.
    t2 = _SpyTeacher([_resp(60, "c")])
    generate_wins(task, t2, _MarkerEnv(), gens=1, ledger=led)
    first_user = next(m for m in t2.calls[0] if m["role"] == "user")
    assert "do NOT repeat" in first_user["content"]


def test_fresh_ledger_first_turn_has_no_constraints_block():
    task = _Task()
    teacher = _SpyTeacher([_resp(60, "c")])
    generate_wins(task, teacher, _MarkerEnv(), gens=1)
    first_user = next(m for m in teacher.calls[0] if m["role"] == "user")
    assert "do NOT repeat" not in first_user["content"]  # empty ledger -> no block
