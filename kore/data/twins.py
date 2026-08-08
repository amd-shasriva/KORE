"""Which source tasks already have a twin, across every output root.

A twin is a task dir re-expressed in another language: the same oracle and
driver, a task.yaml naming the new backend, and a seed the teacher writes. The
materializers keep a ``seed_attempts.jsonl`` ledger so a killed sweep resumes
where it stopped, but that ledger lives inside the run's own ``--out``.

That scoping is the bug this module exists to close. Two roots twinning the
same source into the same language cannot see each other, so aiming a fresh
``--out`` at a source a previous run already swept restarts it from the first
task and every teacher call rewrites a file that is already on disk. Measured
on the frontier HIP root: 514 of 514 tasks it seeded were already materialized
under data/pool_hip.

Twins are counted from the directories rather than from the ledgers, for two
reasons. The directory is the artifact -- a ledger line whose task dir was
rolled back is not a twin, and a twin whose ledger line was lost to a torn
write still is. And the suffix names the language, so a FlyDSL twin is never
mistaken for a HIP one.
"""

from __future__ import annotations

import json
from pathlib import Path


def read_task_cfg(task_dir: Path) -> dict:
    """A task's task.yaml, whichever dialect of it the task speaks.

    Generated pool tasks write JSON; hand-authored registry tasks write real
    YAML with nested shape maps and comments. ``json.loads`` on the latter
    raises, which is what stopped the twin path at the registry boundary.
    """
    text = (task_dir / "task.yaml").read_text(errors="ignore")
    if text.lstrip().startswith("{"):
        return json.loads(text)
    import yaml  # noqa: PLC0415 - only registry tasks need it

    return yaml.safe_load(text) or {}

#: Suffix a twin's directory carries, per backend. Order matters within a
#: backend: the longest suffix is tried first so that ``x__hipf`` is read as
#: task ``x`` twinned functionally, not as task ``x_`` twinned as ``__hip``.
TWIN_SUFFIXES: dict[str, tuple[str, ...]] = {
    "hip": ("__hipf", "__hip"),
    "flydsl": ("__flydsl",),
}


def existing_twins(suffixes, data_dir: Path) -> set[str]:
    """Source task ids that already have a twin with one of ``suffixes``.

    Every ``<root>/tasks`` directory under ``data_dir`` is scanned, so a task
    counts as twinned no matter which run produced it.
    """
    suffixes = tuple(sorted(suffixes, key=len, reverse=True))
    seen: set[str] = set()
    if not suffixes or not data_dir.is_dir():
        return seen
    for tasks_dir in sorted(data_dir.glob("*/tasks")):
        if not tasks_dir.is_dir():
            continue
        for entry in tasks_dir.iterdir():
            name = entry.name
            for suffix in suffixes:
                if name.endswith(suffix) and entry.is_dir():
                    seen.add(name[: -len(suffix)])
                    break
    return seen
