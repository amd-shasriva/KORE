"""JSONL-backed replay cache for verified (task, source) -> Observation.

Benchmarking on a GPU is the scarce resource; caching every verified outcome
keyed by a content hash makes datagen/RL restartable and cheap to re-run.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from kore.reward.reward import Observation


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
        return Observation(**rec) if rec is not None else None

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
