"""JSONL-backed replay cache for verified (task, source) -> Observation.

Benchmarking on a GPU is the scarce resource; caching every verified outcome
keyed by a content hash makes datagen/RL restartable and cheap to re-run.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, fields
from pathlib import Path
from typing import Optional

from kore.reward.reward import Observation

# Field set of the CURRENT Observation. Replay JSONL written by older code may
# carry removed fields (e.g. occupancy/registers) or lack new ones; filtering to
# this set makes the cache forward/backward compatible instead of silently
# dropping otherwise-valid cached evaluations (bench is the scarce resource).
_OBS_FIELDS = {f.name for f in fields(Observation)}


def _obs_from_dict(rec: dict) -> Observation:
    payload = {k: v for k, v in rec.items() if k in _OBS_FIELDS}
    # Old cache entries have no paired protocol identity.  They remain readable
    # but are conservatively screening-only; historical unpaired medians cannot
    # become publication-grade merely because the schema gained new defaults.
    has_timing = bool(
        rec.get("wall_by_shape") or rec.get("baseline_by_shape")
        or rec.get("wall_ms") is not None or rec.get("baseline_ms") is not None)
    if "timing_grade" not in rec and has_timing:
        payload.update({
            "timing_grade": "screening",
            "timing_protocol": "legacy-unpaired-v0",
            "timing_protocol_version": 0,
            "performance_eligible": False,
            "timing_requested": True,
        })
    return Observation(**payload)


def kernel_hash(source: str) -> str:
    """Content hash of a kernel source (stable id used across datagen)."""
    return hashlib.sha256(source.encode()).hexdigest()


def source_key(task_id: str, source: str) -> str:
    h = hashlib.sha256()
    h.update(task_id.encode())
    h.update(b"\x00")
    h.update(source.encode())
    return h.hexdigest()


class ReplayCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._mem: dict[str, dict] = {}
        self._lock = threading.Lock()
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    self._mem[rec["key"]] = rec["obs"]
                except Exception:
                    continue

    def get(self, task_id: str, source: str) -> Optional[Observation]:
        rec = self._mem.get(source_key(task_id, source))
        return _obs_from_dict(rec) if rec is not None else None

    def put(self, task_id: str, source: str, obs: Observation) -> None:
        key = source_key(task_id, source)
        rec = asdict(obs)
        with self._lock:
            self._mem[key] = rec
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps({"key": key, "task_id": task_id, "obs": rec}) + "\n")

    def __len__(self) -> int:
        return len(self._mem)
