"""An arena submission must not be cut off mid-batch.

The arena runs 12 agentic rollouts concurrently per shard, so a shard starts
twelve, works them together, and lands them as a batch roughly every two hours.
A wall clock that fires mid-batch does not save anything -- it discards twelve
nearly-finished rollouts per shard, reloads a 30B model across 16 checkpoint
shards on the next attempt, and leaves the run depending on the supervisor
noticing. v4 lost eight hours to exactly that.

The limit was 8h while the partition's MaxTime is UNLIMITED and the nodes are
held by reservation for days. It bounded nothing except our own progress.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SBATCH = REPO / "scripts" / "spur_aka_1node.sbatch"
SUPERVISE = REPO / "scripts" / "supervise.sh"


def _hms_to_seconds(spec: str) -> int:
    days, _, rest = spec.partition("-")
    if not rest:
        rest, days = days, "0"
    h, m, s = (list(map(int, rest.split(":"))) + [0, 0])[:3]
    return int(days) * 86400 + h * 3600 + m * 60 + s


def test_sbatch_time_limit_outlasts_a_batch():
    """A batch is ~2h. Anything close to that guarantees losing one."""
    m = re.search(r"^#SBATCH --time=(\S+)", SBATCH.read_text(), re.M)
    assert m, "no --time directive"
    assert _hms_to_seconds(m.group(1)) >= 24 * 3600, \
        f"--time={m.group(1)} will cut batches in half"


def test_supervisor_sets_the_limit_explicitly():
    """The directive is a default; the supervisor is what submits in practice,
    so the limit has to travel with the submission."""
    src = SUPERVISE.read_text()
    assert 'AKA_TIME_LIMIT="${AKA_TIME_LIMIT:-' in src
    assert '--time="$AKA_TIME_LIMIT"' in src, "supervisor submits without a limit"


def test_supervisor_default_also_outlasts_a_batch():
    src = SUPERVISE.read_text()
    default = re.search(r'AKA_TIME_LIMIT="\$\{AKA_TIME_LIMIT:-([^}]*)\}"', src).group(1)
    assert _hms_to_seconds(default) >= 24 * 3600, f"default {default} is too short"


@pytest.mark.parametrize("spec,expected", [
    ("08:00:00", 8 * 3600),
    ("3-23:00:00", 3 * 86400 + 23 * 3600),
    ("2-00:00:00", 2 * 86400),
])
def test_time_parsing(spec, expected):
    assert _hms_to_seconds(spec) == expected
