"""KORE data-generation record schemas (KORE.pdf Sec 4.4).

Three record types feed the capability curriculum:
  - ``RepairRecord``  (Stage 1, repair-weighted SFT): a broken -> fixed turn,
    conditioned on the exact verifier error.
  - ``RankedGroupRecord`` (Stage 2, RFT + DPO): a group of candidates for one
    parent with a ranking and the derived preference pairs.
  - ``WinRecord`` (Stage 3, multi-turn evolve): a full winning trajectory.

Every record is a plain dataclass with symmetric ``to_dict``/``from_dict`` so it
round-trips losslessly through JSONL. ``write_jsonl``/``read_jsonl`` handle the
mixed-type on-disk log (the ``type`` field selects the class on read).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Union

GPU_DEFAULT = "gfx942"


@dataclass
class RepairRecord:
    """A single repair turn: parent kernel failed, teacher fixed it."""

    task_id: str
    failure_class: str          # "compile_fail" | "snr_fail"
    parent_hash: str
    error_text: str
    messages: list[dict]        # [{"role": ..., "content": ...}, ...]
    child_snr_db: float | None = None
    type: str = "repair"
    operator: str = "repair"
    gpu: str = GPU_DEFAULT

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RepairRecord":
        return cls(
            task_id=d["task_id"],
            failure_class=d["failure_class"],
            parent_hash=d["parent_hash"],
            error_text=d.get("error_text", ""),
            messages=list(d.get("messages", [])),
            child_snr_db=d.get("child_snr_db"),
            type=d.get("type", "repair"),
            operator=d.get("operator", "repair"),
            gpu=d.get("gpu", GPU_DEFAULT),
        )


@dataclass
class RankedGroupRecord:
    """A parent plus k ranked candidates and the derived preference pairs."""

    task_id: str
    parent_id: str
    candidates: list[dict]      # [{"source", "wall_us", "snr_db", "rank"}, ...]
    preferences: list[list[int]]  # [[chosen_idx, rejected_idx], ...]
    type: str = "ranked_group"
    gpu: str = GPU_DEFAULT

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RankedGroupRecord":
        return cls(
            task_id=d["task_id"],
            parent_id=d["parent_id"],
            candidates=list(d.get("candidates", [])),
            preferences=[list(p) for p in d.get("preferences", [])],
            type=d.get("type", "ranked_group"),
            gpu=d.get("gpu", GPU_DEFAULT),
        )


@dataclass
class WinRecord:
    """A full winning multi-turn trajectory (initial -> final, wall improved)."""

    task_id: str
    trajectory: list[dict]      # list of chat messages across turns
    initial_wall_us: float | None
    final_wall_us: float | None
    speedup: float | None
    final_source: str
    snr_db: float | None = None
    type: str = "win"
    gpu: str = GPU_DEFAULT

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WinRecord":
        return cls(
            task_id=d["task_id"],
            trajectory=list(d.get("trajectory", [])),
            initial_wall_us=d.get("initial_wall_us"),
            final_wall_us=d.get("final_wall_us"),
            speedup=d.get("speedup"),
            final_source=d.get("final_source", ""),
            snr_db=d.get("snr_db"),
            type=d.get("type", "win"),
            gpu=d.get("gpu", GPU_DEFAULT),
        )


Record = Union[RepairRecord, RankedGroupRecord, WinRecord]

_TYPE_TO_CLASS = {
    "repair": RepairRecord,
    "ranked_group": RankedGroupRecord,
    "win": WinRecord,
}


def record_from_dict(d: dict) -> Record:
    """Dispatch a raw dict to the right record class by its ``type`` field."""
    t = d.get("type")
    cls = _TYPE_TO_CLASS.get(t)
    if cls is None:
        raise ValueError(f"unknown record type: {t!r}")
    return cls.from_dict(d)


def _to_dict(rec: Any) -> dict:
    if hasattr(rec, "to_dict"):
        return rec.to_dict()
    if isinstance(rec, dict):
        return rec
    raise TypeError(f"cannot serialize {type(rec)!r} to a record dict")


def write_jsonl(path: Union[str, Path], records: Iterable[Any]) -> Path:
    """Write records (dataclasses or dicts) to a JSONL file, one per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(_to_dict(rec)) + "\n")
    return path


def read_jsonl(path: Union[str, Path], typed: bool = True) -> list:
    """Read a JSONL file. If ``typed``, dispatch each line to its record class;
    otherwise return raw dicts."""
    path = Path(path)
    out: list = []
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if typed and d.get("type") in _TYPE_TO_CLASS:
                out.append(record_from_dict(d))
            else:
                out.append(d)
    return out
