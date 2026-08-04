"""The supervisor's checkpoint pruning must be numeric and conservative.

A 30B checkpoint with optimizer state is ~488GB, so a pruner that keeps the wrong
directory throws away hours of training, and one that is too eager deletes
something it was never asked to touch. Both failure modes are silent, which is
why they are pinned here rather than left to inspection.

The function is exercised as shell, extracted from the script that actually runs
on the cluster, so this cannot drift from the deployed copy the way a
reimplementation in Python would.
"""

import pathlib
import re
import subprocess

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "supervise.sh"


def _prune(target: pathlib.Path) -> str:
    """Run just prune_checkpoints() from the real script against target."""
    body = SCRIPT.read_text()
    m = re.search(r"^prune_checkpoints\(\) \{.*?^\}", body, re.S | re.M)
    assert m, "prune_checkpoints() not found -- was supervise.sh restructured?"
    prog = "say() { echo \"$*\"; }\n" + m.group(0) + f'\nprune_checkpoints "{target}"\n'
    r = subprocess.run(["bash", "-c", prog], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return r.stdout


def _mk(root: pathlib.Path, steps) -> None:
    for s in steps:
        d = root / f"checkpoint-{s}"
        d.mkdir(parents=True)
        (d / "optimizer.pt").write_text("x")


def test_keeps_highest_step_not_lexically_largest(tmp_path):
    """Sorting must be numeric: as strings, '1950' outranks '2000'.

    A lexical sort would keep step 1950 and delete step 2000, silently discarding
    the newest checkpoint -- the exact one a resume needs.
    """
    run = tmp_path / "run"
    _mk(run, [50, 100, 150, 1950, 2000])
    _prune(run)
    left = sorted(p.name for p in run.glob("checkpoint-*"))
    assert left == ["checkpoint-2000"], left


def test_leaves_non_checkpoint_directories_alone(tmp_path):
    """Only checkpoint-<int> is ours to delete."""
    run = tmp_path / "run"
    _mk(run, [10, 20])
    (run / "checkpoint-notanumber").mkdir()
    (run / "logs").mkdir()
    _prune(run)
    left = sorted(p.name for p in run.iterdir())
    assert left == ["checkpoint-20", "checkpoint-notanumber", "logs"], left


@pytest.mark.parametrize("steps", [[], [42]])
def test_noop_when_there_is_nothing_to_rotate(tmp_path, steps):
    """One checkpoint is the steady state under save_total_limit=1, not a problem."""
    run = tmp_path / "run"
    run.mkdir()
    _mk(run, steps)
    _prune(run)
    assert sorted(p.name for p in run.glob("checkpoint-*")) == [
        f"checkpoint-{s}" for s in steps]


def test_missing_directory_is_not_an_error(tmp_path):
    """The run dir does not exist until the first save; polling it must not fail."""
    _prune(tmp_path / "does_not_exist")
