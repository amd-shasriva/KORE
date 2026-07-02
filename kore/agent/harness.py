"""AgentHarness: the multi-turn tool-orchestration loop.

The policy (or a teacher) drives its own optimize loop. Each turn:

  render messages -> model.generate() -> parse Hermes/OpenAI tool calls
    -> execute each via ToolExecutor -> append role:"tool" result -> repeat

The trajectory is scored by the BEST correct kernel reached (Kevin), and
keep/revert are recorded as first-class (trainable) decisions. The GPU-bound
work lives entirely in the injected ``env``; the harness itself is CPU-only and
deterministic given a deterministic model + env.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from kore.agent.format import (
    build_agent_system_prompt,
    parse_tool_calls,
    render_tool_result,
)
from kore.agent.tools import TOOL_SCHEMAS, ToolExecutor, validate_tool_call


@dataclass
class AgentEpisode:
    """The full record of one agentic optimize episode."""

    task_id: str
    messages: list[dict] = field(default_factory=list)
    tool_trace: list[dict] = field(default_factory=list)
    keep_decisions: list[dict] = field(default_factory=list)
    best_kernel: Optional[str] = None
    best_reward: Optional[float] = None
    turns_to_best: Optional[int] = None
    turns_used: int = 0
    committed_kernel: Optional[str] = None
    success: bool = False

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "messages": self.messages,
            "tool_trace": self.tool_trace,
            "keep_decisions": self.keep_decisions,
            "best_kernel": self.best_kernel,
            "best_reward": self.best_reward,
            "turns_to_best": self.turns_to_best,
            "turns_used": self.turns_used,
            "committed_kernel": self.committed_kernel,
            "success": self.success,
        }


def build_agent_user_prompt(task, seed_src: str = "") -> str:
    """The opening user turn: the task + the seed kernel to optimize."""
    op = getattr(task, "operation", None) or getattr(task, "task_id", "kernel")
    dtype = getattr(task, "dtype", "fp32")
    seed_block = f"\n\n## Seed kernel (optimize this)\n```python\n{seed_src}\n```" if seed_src else ""
    return (
        f"Optimize the `{op}` Triton kernel ({dtype}) for AMD MI325X (gfx942).\n"
        "Use your tools to build, test, and bench candidates. Keep improvements "
        "and revert regressions. Reach a correct kernel first, then maximize the "
        f"speedup vs the production baseline.{seed_block}"
    )


class AgentHarness:
    """Run the multi-turn tool loop for a single task with an injected env."""

    def __init__(
        self,
        task,
        model_or_teacher,
        env,
        max_turns: int = 8,
        tools: Optional[list[dict]] = None,
        seed_src: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.task = task
        self.model = model_or_teacher
        self.env = env
        self.max_turns = max_turns
        self.tools = tools or TOOL_SCHEMAS
        self.system_prompt = system_prompt or build_agent_system_prompt(self.tools)
        if seed_src is None:
            seed_src = _safe_seed(task)
        self.seed_src = seed_src or ""
        self.executor = ToolExecutor(env, task, seed_src=self.seed_src or None)

    def run(self) -> AgentEpisode:
        task_id = getattr(self.task, "task_id", "task")
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": build_agent_user_prompt(self.task, self.seed_src)},
        ]
        tool_trace: list[dict] = []
        turns_used = 0

        for turn in range(self.max_turns):
            text = self.model.generate(messages)
            messages.append({"role": "assistant", "content": text})
            turns_used = turn + 1

            calls = parse_tool_calls(text)
            if not calls:
                break  # no tool call -> the model is done

            self.executor.set_turn(turn)
            for call in calls:
                v = validate_tool_call(call)
                result = self.executor.dispatch(call, turn=turn)
                name = call.get("name") or "unknown"
                tool_trace.append({
                    "turn": turn,
                    "name": call.get("name"),
                    "arguments": call.get("arguments", {}),
                    "valid_name": v["valid_name"],
                    "valid_params": v["valid_params"],
                    "malformed": bool(call.get("malformed")),
                    "result": result,
                })
                messages.append(render_tool_result(name, result))

        ex = self.executor
        best_reward = ex.best_reward if ex.best_reward != float("-inf") else None
        return AgentEpisode(
            task_id=task_id,
            messages=messages,
            tool_trace=tool_trace,
            keep_decisions=list(ex.keep_decisions),
            best_kernel=ex.best_src,
            best_reward=best_reward,
            turns_to_best=ex.best_turn,
            turns_used=turns_used,
            committed_kernel=ex.committed_src,
            success=ex.best_src is not None,
        )


def _safe_seed(task) -> str:
    try:
        return task.seed_source
    except Exception:  # noqa: BLE001 — FakeEnv tasks may have no seed file
        return ""
