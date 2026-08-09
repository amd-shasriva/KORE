"""The unattended loops must survive the environment cron actually gives them.

The loops are started by ensure_loops.sh, and ensure_loops.sh is now run from
cron so that something restarts the keepalive wrappers themselves. cron's
environment is not a login shell's: it sets LOGNAME but not USER, and these
scripts run under ``set -u``, so a bare ``$USER`` aborts the command it is in.

That is not hypothetical either. After the watchdog was added, every supervisor
queue check became "scheduler unreachable" -- the scheduler was fine, USER was
empty -- and the v4 arena, which the supervisor is the only thing that
resubmits, stayed down for eight hours with its node reserved and idle.

gpu_slots.sh had already found this and guarded itself. Six other operational
scripts had not.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: Scripts that run unattended -- from cron, from a keepalive loop, or from a
#: batch job -- and therefore cannot assume a login environment.
UNATTENDED = [
    "supervise.sh", "frontier_pipeline.sh", "staff_datagen.sh",
    "ensure_loops.sh", "gpu_slots.sh", "keepalive.sh", "hip_pool_harvest.sh",
    "watch_and_resume.sh", "spur_pipeline_driver.sh", "spur_data_driver.sh",
    "cleanup_stale_artifacts.sh", "hip_pipeline_loop.sh",
]

GUARDED = re.compile(r'\$\{USER:-')


@pytest.mark.parametrize("name", UNATTENDED)
def test_no_bare_user_in_scheduler_calls(name):
    """A bare $USER under set -u kills the call it appears in."""
    path = REPO / "scripts" / name
    if not path.is_file():
        pytest.skip(f"{name} not present")
    offenders = [
        line.strip()
        for line in path.read_text().splitlines()
        if "squeue" in line and '"$USER"' in line and not GUARDED.search(line)
    ]
    assert not offenders, f"{name} uses a bare $USER: {offenders[:2]}"


@pytest.mark.parametrize("name", UNATTENDED)
def test_scripts_parse(name):
    path = REPO / "scripts" / name
    if not path.is_file():
        pytest.skip(f"{name} not present")
    r = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_the_guard_resolves_without_user():
    """LOGNAME, then id -un: cron supplies the first, a batch job may supply
    neither."""
    script = 'echo "${USER:-${LOGNAME:-$(id -un)}}"'
    for env in ({"LOGNAME": "someone", "PATH": "/usr/bin:/bin"},
                {"PATH": "/usr/bin:/bin"}):
        r = subprocess.run(["bash", "-uc", script], env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip(), "guard produced an empty user"


def test_supervisor_refuses_to_run_blind():
    """It already declines to start without a controller address; the same
    reasoning applies to a queue check that can never succeed."""
    src = (REPO / "scripts" / "supervise.sh").read_text()
    assert "Refusing to start blind" in src
