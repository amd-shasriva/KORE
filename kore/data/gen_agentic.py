"""Generate agentic tool-use trajectories (Hermes tool-calling SFT/RL data).

Drives :class:`~kore.agent.harness.AgentHarness` with a teacher to produce
:class:`~kore.agent.schema.AgenticTrajectoryRecord`s: the teacher plans, calls
build/test/bench/pmc, reads results, and keeps/reverts. Both *successful*
trajectories (reached a correct kernel) and *repair* trajectories (recovered
after a failed build/test) are emitted for SFT in the native Hermes format.

Works end-to-end with a :class:`~kore.data.teacher.StubTeacher`, so it is
CPU-only and testable without a GPU.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any, Optional

from kore.agent.format import episode_to_chat
from kore.agent.harness import AgentEpisode, AgentHarness
from kore.agent.schema import AgenticTrajectoryRecord
from kore.agent.tools import tool_use_reward
from kore.obs import get_logger

log = get_logger("data.gen_agentic")


def _category(episode: AgentEpisode) -> str:
    """success | repair | attempt (for curriculum tagging)."""
    if episode.success:
        # A trajectory that failed a build/test before succeeding is a repair.
        for t in episode.tool_trace:
            res = t.get("result") or {}
            if t.get("name") in ("build", "test", "bench") and res.get("ok") is False:
                return "repair"
        return "success"
    return "attempt"


def episode_to_record(
    episode: AgentEpisode,
    task,
    teacher: Any = None,
    thinking: bool = True,
    extra_provenance: Optional[dict] = None,
) -> AgenticTrajectoryRecord:
    """Convert a finished episode into an :class:`AgenticTrajectoryRecord`."""
    provenance = {
        "category": _category(episode),
        "teacher": type(teacher).__name__ if teacher is not None else None,
        "turns_used": episode.turns_used,
        "n_tool_calls": len(episode.tool_trace),
        "tool_use_reward": tool_use_reward(episode),
    }
    if extra_provenance:
        provenance.update(extra_provenance)

    return AgenticTrajectoryRecord(
        task_id=episode.task_id,
        messages=episode_to_chat(episode, thinking=thinking),
        tool_trace=episode.tool_trace,
        best_kernel=episode.best_kernel or "",
        best_reward=episode.best_reward,
        turns_to_best=episode.turns_to_best,
        success=episode.success,
        provenance=provenance,
        gpu=getattr(task, "gpu_target", "gfx942"),
    )


def generate_agentic_trajectories(
    task,
    teacher,
    env,
    n: int,
    max_turns: int = 8,
    keep_only_useful: bool = False,
    thinking: bool = True,
) -> list[AgenticTrajectoryRecord]:
    """Run ``n`` agentic episodes and return their trajectory records.

    Each episode is an independent :class:`AgentHarness` run driven by
    ``teacher``. When ``keep_only_useful`` is set, only successful or repair
    trajectories are retained (attempts that never reached correctness are
    dropped) — the SFT-quality subset.
    """
    total = max(0, n)
    with log.stage("generate_agentic_trajectories", task=getattr(task, "task_id", None),
                   n=total, max_turns=max_turns, keep_only_useful=keep_only_useful):
        records: list[AgenticTrajectoryRecord] = []
        t_start = time.time()
        categories: Counter = Counter()
        for idx in range(total):
            harness = AgentHarness(task, teacher, env, max_turns=max_turns)
            episode = harness.run()
            rec = episode_to_record(episode, task, teacher=teacher, thinking=thinking)
            tool_calls = [t.get("name") for t in episode.tool_trace]
            category = rec.provenance.get("category")
            categories[category] += 1
            log.event(
                "agentic_episode", task=getattr(task, "task_id", None), idx=idx,
                turns_used=episode.turns_used, success=episode.success,
                best_reward=episode.best_reward, turns_to_best=episode.turns_to_best,
                category=category, n_tool_calls=len(tool_calls),
                tool_calls=tool_calls,
            )
            dropped = keep_only_useful and category == "attempt"
            if not dropped:
                records.append(rec)
            log.progress(idx + 1, total, "agentic", t_start=t_start,
                         kept=len(records))
        log.metric(
            "agentic_summary", task=getattr(task, "task_id", None),
            episodes=total, kept=len(records), by_category=dict(categories),
        )
        return records
