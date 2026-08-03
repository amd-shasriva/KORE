"""Step-centric decomposition: keep only revisions worth imitating.

Training on whole trajectories teaches the model to imitate a search. What the
RL stage actually samples is a single revision, so what SFT should teach is
"given this kernel and this feedback, make it faster without breaking it" --
Kernel-Smith's local-improver framing, and the difference between their 3.70
average speedup and Claude-4.6-opus's 3.33.

Every test here pins a revision the model must NOT be trained on, because each
one is a distinct way of learning the wrong lesson.
"""

from __future__ import annotations

from kore.data.step_centric import Step, decompose, extract_steps


def _rec(correct, speedups, n_assistant=None, rewards=None, task_id="t1"):
    n = n_assistant if n_assistant is not None else len(correct)
    messages = []
    for i in range(n):
        messages.append({"role": "user", "content": f"feedback {i}"})
        messages.append({"role": "assistant", "content": f"kernel v{i}"})
    prov = {"turn_correct": list(correct), "turn_speedups": list(speedups)}
    if rewards is not None:
        prov["turn_rewards"] = list(rewards)
    return {"task_id": task_id, "messages": messages, "provenance": prov}


def test_keeps_a_correctness_preserving_speedup():
    steps = extract_steps(_rec([True, True], [1.0, 2.0]))
    assert len(steps) == 1
    assert steps[0].kind == "speedup"
    assert steps[0].gain == 1.0


def test_keeps_the_revision_that_fixes_a_broken_kernel():
    # Going from incorrect to correct is the single most valuable lesson in the
    # trajectory even when it is not faster.
    steps = extract_steps(_rec([False, True], [None, 0.9]))
    assert [s.kind for s in steps] == ["fix"]


def test_drops_a_revision_that_breaks_correctness():
    # A faster-but-wrong kernel is the worst thing to imitate: it looks like
    # progress on the metric while destroying the only hard constraint.
    assert extract_steps(_rec([True, False], [1.0, 5.0])) == []


def test_drops_a_regression():
    assert extract_steps(_rec([True, True], [2.0, 1.0])) == []


def test_drops_noise_sized_gains():
    # Cold-cache timing on a shared GPU moves a couple of percent run to run.
    # A 2% "gain" trains the model on measurement noise.
    assert extract_steps(_rec([True, True], [1.00, 1.02])) == []
    assert len(extract_steps(_rec([True, True], [1.00, 1.30]))) == 1


def test_drops_the_reward_hacking_signature():
    # No fused Triton kernel is 1500x its Torch reference; that is a decoy
    # kernel that never runs, or skipped computation. Training on it teaches
    # the model to cheat on the metric RL will later optimise.
    assert extract_steps(_rec([True, True], [1.0, 1541.94])) == []


def test_a_failed_measurement_is_not_a_zero_speedup():
    # None means the measurement failed. Treating it as 0.0 would make the next
    # turn look like an infinite gain.
    assert extract_steps(_rec([True, True], [None, 3.0])) == []


def test_one_trajectory_yields_multiple_independent_steps():
    steps = extract_steps(_rec([True, True, True, True], [1.0, 2.0, 2.05, 4.0]))
    # v0->v1 kept, v1->v2 is noise and dropped, v2->v3 kept.
    assert [s.turn for s in steps] == [2, 4]


def test_messages_are_truncated_at_the_revision():
    steps = extract_steps(_rec([True, True, True], [1.0, 2.0, 4.0]))
    for s in steps:
        assert s.messages[-1]["role"] == "assistant", "a step must end on the revision"
    assert len(steps[0].messages) < len(steps[1].messages)


def test_rewards_are_only_a_fallback_when_speedups_are_absent():
    # Older trajectories predate turn_speedups being persisted. Reward blends
    # terms besides runtime, so it is a weaker proxy -- usable, never preferred.
    from_rewards = extract_steps(_rec([True, True], [None, None], rewards=[1.0, 2.0]))
    assert len(from_rewards) == 1
    # When speedups exist they win, even if rewards would disagree.
    both = extract_steps(_rec([True, True], [2.0, 1.0], rewards=[1.0, 9.0]))
    assert both == [], "a reward rise must not override a measured regression"


def test_single_turn_trajectory_has_no_steps():
    assert extract_steps(_rec([True], [1.0])) == []


def test_decompose_reports_what_it_kept():
    rows, stats = decompose([
        _rec([True, True], [1.0, 2.0], task_id="a"),
        _rec([False, True], [None, 1.0], task_id="b"),
        _rec([True, False], [1.0, 9.0], task_id="c"),
    ])
    assert stats["trajectories"] == 3
    assert stats["with_steps"] == 2
    assert stats["steps"] == 2
    assert stats["fix_steps"] == 1 and stats["speedup_steps"] == 1
    assert {r["_source"] for r in rows} == {"kernel_step_centric"}
