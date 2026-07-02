"""Discover and load KORE tasks from ``kore/tasks/<id>/task.yaml``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from kore.tasks.base import Task

TASKS_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _discover() -> dict[str, Task]:
    tasks: dict[str, Task] = {}
    for yml in sorted(TASKS_DIR.glob("*/task.yaml")):
        try:
            t = Task.from_dir(yml.parent)
            tasks[t.task_id] = t
        except Exception as e:  # noqa: BLE001
            print(f"[registry] skip {yml.parent.name}: {e}")
    return tasks


def all_tasks() -> list[Task]:
    return list(_discover().values())


def task_ids() -> list[str]:
    return list(_discover().keys())


def get_task(task_id: str) -> Task:
    tasks = _discover()
    if task_id not in tasks:
        raise KeyError(f"unknown task '{task_id}'; known: {sorted(tasks)}")
    return tasks[task_id]
