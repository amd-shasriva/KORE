"""Contract for the multi-node agentic campaign's shard plan.

A partition bug is invisible at submit time and expensive at merge time: two
nodes handed the same task duplicate hours of GPU work into two shards that both
look valid, and a dropped task is simply absent from the mixture with nothing to
notice it. So the plan is checked for the two properties that matter - every task
covered exactly once, and a re-plan that only covers what is still missing.

CPU-only; the registry is injected.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load_partition_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_agentic_partition", REPO / "scripts" / "agentic_partition.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_partition(tmp_path, task_ids, *, shards, episodes, existing=None):
    out_dir = tmp_path / "out"
    shard_dir = tmp_path / "plan"
    out_dir.mkdir(parents=True, exist_ok=True)
    if existing:
        # Emulate a prior wave's durable output: one line per completed episode.
        with (out_dir / "shard_000.jsonl").open("w") as handle:
            for task_id, count in existing.items():
                for episode in range(count):
                    handle.write(json.dumps({"_work_key": f"{task_id}#{episode}"}) + "\n")

    script = REPO / "scripts" / "agentic_partition.py"
    stub = tmp_path / "stub_registry.py"
    stub.write_text(
        "class _T:\n"
        "    def __init__(self, task_id): self.task_id = task_id\n"
        f"_IDS = {task_ids!r}\n"
        "def train_tasks(): return [_T(i) for i in _IDS]\n"
    )
    conftest = tmp_path / "sitecustomize.py"
    conftest.write_text(
        "import sys, types\n"
        f"sys.path.insert(0, {str(tmp_path)!r})\n"
        "import stub_registry\n"
        "import kore.tasks.registry as reg\n"
        "reg.train_tasks = stub_registry.train_tasks\n"
    )
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": f"{tmp_path}:{REPO}",
        "PYTHONSTARTUP": "",
        "HOME": str(tmp_path),
    }
    result = subprocess.run(
        [sys.executable, str(script),
         "--out-dir", str(out_dir), "--shard-dir", str(shard_dir),
         "--shards", str(shards), "--episodes-per-task", str(episodes)],
        cwd=str(REPO), env=env, text=True, capture_output=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((shard_dir / "manifest.json").read_text())
    buckets = [
        [line for line in path.read_text().splitlines() if line.strip()]
        for path in sorted(shard_dir.glob("shard_*.txt"))
    ]
    return manifest, buckets


def test_completed_counts_reads_every_shard(tmp_path):
    module = _load_partition_module()
    (tmp_path / "shard_000.jsonl").write_text(
        json.dumps({"_work_key": "alpha#0"}) + "\n"
        + json.dumps({"_work_key": "alpha#1"}) + "\n")
    (tmp_path / "shard_001.jsonl").write_text(
        json.dumps({"_work_key": "beta#0"}) + "\n")
    counts = module.completed_counts(tmp_path)
    assert counts["alpha"] == 2
    assert counts["beta"] == 1


def test_completed_counts_handles_task_ids_containing_hash(tmp_path):
    # The key is task_id + "#" + episode, so the split has to be from the right;
    # a left split would silently mis-attribute any id containing a hash.
    module = _load_partition_module()
    (tmp_path / "shard_000.jsonl").write_text(
        json.dumps({"_work_key": "odd#name#3"}) + "\n")
    assert module.completed_counts(tmp_path) == {"odd#name": 1}


@pytest.mark.parametrize("shards", [1, 3, 8])
def test_every_task_is_planned_exactly_once(tmp_path, shards):
    ids = [f"task_{i:03d}" for i in range(20)]
    manifest, buckets = _run_partition(
        tmp_path / str(shards), ids, shards=shards, episodes=4)
    flat = [task for bucket in buckets for task in bucket]
    assert sorted(flat) == sorted(ids)
    assert len(flat) == len(set(flat)), "a task appears in two shards"
    assert manifest["n_tasks_planned"] == len(ids)
    assert manifest["planned_episodes"] == len(ids) * 4


def test_shards_are_balanced_within_one_task(tmp_path):
    ids = [f"task_{i:03d}" for i in range(20)]
    _manifest, buckets = _run_partition(tmp_path, ids, shards=6, episodes=2)
    sizes = [len(bucket) for bucket in buckets]
    assert max(sizes) - min(sizes) <= 1


def test_replanning_covers_only_the_missing_episodes(tmp_path):
    ids = [f"task_{i:03d}" for i in range(10)]
    # Five tasks already have their full quota; five are short by one.
    existing = {f"task_{i:03d}": 3 for i in range(5)}
    existing.update({f"task_{i:03d}": 2 for i in range(5, 10)})
    manifest, buckets = _run_partition(
        tmp_path, ids, shards=2, episodes=3, existing=existing)
    flat = sorted(task for bucket in buckets for task in bucket)
    assert flat == [f"task_{i:03d}" for i in range(5, 10)]
    assert manifest["n_tasks_already_complete"] == 5


def test_manifest_pins_the_commit_the_plan_was_built_from(tmp_path):
    manifest, _buckets = _run_partition(
        tmp_path, ["alpha", "beta"], shards=1, episodes=1)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    # The node job refuses to run when these disagree, so a plan built against
    # different code cannot write records under a contract it does not describe.
    assert manifest["repo_commit"] == head
