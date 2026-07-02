"""Agentic trajectory record for SFT/RL over the tool-use loop.

One :class:`AgenticTrajectoryRecord` captures a full agent episode: the Hermes
multi-turn ``messages`` (system + user + assistant tool-calls + tool results),
the compact ``tool_trace``, the best kernel + its reward, turns-to-best, the
success flag, and provenance. It follows the same dataclass + symmetric
``to_dict``/``from_dict`` pattern as :mod:`kore.data.schemas` and round-trips
through that module's ``write_jsonl``/``read_jsonl`` (which are imported here,
not re-implemented — schemas.py is left untouched).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

# Re-use the canonical JSONL IO (do NOT edit schemas.py).
from kore.data.schemas import write_jsonl, read_jsonl  # noqa: F401

GPU_DEFAULT = "gfx942"


@dataclass
class AgenticTrajectoryRecord:
    """A full agentic tool-use trajectory for one task."""

    task_id: str
    messages: list[dict]              # system + user + assistant(tool_call) + tool
    tool_trace: list[dict]            # [{turn,name,arguments,valid_*,malformed,result}]
    best_kernel: str
    best_reward: Optional[float]
    turns_to_best: Optional[int]
    success: bool
    provenance: dict = field(default_factory=dict)
    type: str = "agentic"
    gpu: str = GPU_DEFAULT

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AgenticTrajectoryRecord":
        return cls(
            task_id=d["task_id"],
            messages=list(d.get("messages", [])),
            tool_trace=list(d.get("tool_trace", [])),
            best_kernel=d.get("best_kernel", ""),
            best_reward=d.get("best_reward"),
            turns_to_best=d.get("turns_to_best"),
            success=bool(d.get("success", False)),
            provenance=dict(d.get("provenance", {}) or {}),
            type=d.get("type", "agentic"),
            gpu=d.get("gpu", GPU_DEFAULT),
        )


def record_to_dict(rec: Any) -> dict:
    if hasattr(rec, "to_dict"):
        return rec.to_dict()
    if isinstance(rec, dict):
        return rec
    raise TypeError(f"cannot serialize {type(rec)!r}")
