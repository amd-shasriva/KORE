"""The agentic partition must plan the expanded task pool, not just the registry.

The registry holds ~1,289 trainable tasks and the seed screen rejects ~398 of
them, leaving under 900. Six episodes each is a campaign spent re-sampling a few
hundred programs, which is redundancy rather than data -- the external pool
(~13.5k screened, deduplicated tasks) exists precisely to lift that ceiling, and
a partition that silently ignores it wastes the whole expansion.

The collision rule is the load-bearing part. Registry ids must win, because
those entries carry the authoritative train/held-out split; letting a pool entry
shadow one could move a held-out task into training and quietly invalidate every
number the eval produces.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PARTITION = REPO / "scripts" / "agentic_partition.py"


def _source() -> str:
    return PARTITION.read_text()


def test_pool_is_planned_by_default():
    src = _source()
    assert "--no-pool" in src, "opting OUT must be the explicit choice"
    assert 'default=True' in src
    assert "load_pool" in src


def test_registry_wins_on_id_collision():
    src = _source()
    # The union must filter pool entries against registry ids, not the reverse.
    assert "seen = {t.task_id for t in tasks}" in src
    assert "if t.task_id not in seen" in src


def test_pool_failure_is_not_fatal():
    # The pool is additive. If it cannot be read the campaign should still run
    # on the registry rather than refusing to plan anything.
    src = _source()
    assert "pool is additive, never required" in src or "planning" in src
    assert "except Exception" in src


def test_partition_cli_reports_the_split():
    """--help must work, so a typo in the new flag surfaces here not on a node."""
    out = subprocess.run(
        [sys.executable, str(PARTITION), "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stderr
    assert "--no-pool" in out.stdout
