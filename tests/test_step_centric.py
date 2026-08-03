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

from kore.data.step_centric import (
    Step,
    decompose,
    decompose_with_trajectories,
    extract_full_trajectory,
    extract_steps,
)


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


# --------------------------------------------------------------------------- #
# Full successful trajectories: the half step-centric cannot represent
# --------------------------------------------------------------------------- #
def _assistant_count(row):
    return sum(1 for m in row["messages"] if m.get("role") == "assistant")


def test_a_first_turn_success_is_emitted_as_a_whole_trajectory():
    """The class step-centric structurally cannot reach.

    A revision needs a parent to improve on, so an episode that was correct on
    turn 1 and never got faster yields no step at all. 1,576 of the overnight
    campaign's 3,475 successful trajectories are exactly this, and their median
    measured speedup is 1.58x -- so the discard was a representation gap, not a
    quality filter.
    """
    rec = _rec([True, True], [2.0, 2.0])
    assert extract_steps(rec) == [], "a no-gain second turn is not a step"

    full = extract_full_trajectory(rec)
    assert full is not None
    assert full.first_correct_turn == 1
    assert full.best_turn == 1, "with no later gain the row ends on the win"
    assert _assistant_count(full.to_row()) == 1
    assert full.to_row()["_source"] == "kernel_full_trajectory"


def test_a_never_correct_trajectory_is_never_emitted():
    """Eight turns of failure teaches failure; 8,189 episodes look like this."""
    assert extract_full_trajectory(_rec([False] * 8, [None] * 8)) is None


def test_the_row_ends_on_the_fastest_correct_turn_not_the_last_one():
    """Everything after the win is a non-improvement or a regression.

    Training through it teaches the model to keep editing a kernel that was
    already right, which is the behaviour the reseed logic exists to interrupt.
    """
    full = extract_full_trajectory(_rec([True, True, True], [1.0, 3.0, 1.2]))
    assert full is not None
    assert full.best_turn == 2
    assert full.best_speedup == 3.0
    assert _assistant_count(full.to_row()) == 2
    assert full.messages[-1]["role"] == "assistant"


def test_an_implausible_speedup_is_refused_as_a_reward_hack():
    """A three-orders-of-magnitude speedup is a decoy that never ran.

    Ending a training row on it teaches precisely the metric-cheating that the RL
    stage is then free to exploit. The campaign contains one at 69,068x.
    """
    assert extract_full_trajectory(_rec([True], [69068.0])) is None
    assert extract_full_trajectory(_rec([True], [69068.0]), max_speedup=1e6) is not None


def test_a_correct_trajectory_with_no_timing_ends_at_the_first_correct_turn():
    """With no measurement anywhere, the earliest proof of correctness is the
    only defensible end point: later turns carry no evidence they are still
    right about anything."""
    full = extract_full_trajectory(_rec([False, True, True], [None, None, None]))
    assert full is not None
    assert full.first_correct_turn == 2
    assert full.best_turn == 2
    assert full.best_speedup is None


def test_residual_mode_does_not_duplicate_a_trajectory_that_already_has_steps():
    """A step row's messages are a PREFIX of the whole trajectory.

    Emitting both puts the same tokens in the corpus twice, and the mixture
    deduplicates on exact content, so a prefix is invisible to it.
    """
    improving = _rec([True, True], [1.0, 2.0], task_id="improving")
    first_turn = _rec([True, True], [2.0, 2.0], task_id="first_turn")
    never = _rec([False, False], [None, None], task_id="never")

    rows, stats = decompose_with_trajectories([improving, first_turn, never])
    sources = [r["_source"] for r in rows]
    assert sources.count("kernel_step_centric") == 1
    assert sources.count("kernel_full_trajectory") == 1
    assert stats["full_skipped_has_steps"] == 1
    assert stats["never_correct_dropped"] == 1
    assert stats["reached_correct"] == 2
    full_ids = [r["_task_id"] for r in rows
                if r["_source"] == "kernel_full_trajectory"]
    assert full_ids == ["first_turn"]


def test_non_residual_mode_emits_every_successful_trajectory():
    improving = _rec([True, True], [1.0, 2.0], task_id="improving")
    first_turn = _rec([True, True], [2.0, 2.0], task_id="first_turn")
    rows, stats = decompose_with_trajectories(
        [improving, first_turn], only_residual=False
    )
    assert stats["full_trajectories"] == 2
    assert stats["full_skipped_has_steps"] == 0


def test_step_centric_only_decompose_is_unchanged():
    """The pre-existing entry point must keep its exact behaviour.

    ``decompose`` is what the v2 mixture and every prior receipt were built with;
    if adding a second representation moved it, the two corpora would stop being
    comparable.
    """
    records = [
        _rec([True, True], [1.0, 2.0]),
        _rec([True, True], [2.0, 2.0]),
        _rec([False] * 3, [None] * 3),
    ]
    rows, stats = decompose(records)
    assert stats == {
        "trajectories": 3, "with_steps": 1, "steps": 1,
        "fix_steps": 0, "speedup_steps": 1,
    }
    assert all(r["_source"] == "kernel_step_centric" for r in rows)
