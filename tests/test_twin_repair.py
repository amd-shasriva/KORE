"""A failing twin should see its own error before being thrown away.

Seeding is one-shot: the gate runs the kernel on real gfx950 and a failure is
discarded. HIP can afford that -- 96.5% of pool twins pass. FlyDSL cannot: 173
of 3,974 ports passed, and 3,109 of the failures crashed before producing a
number, on things the gate reports in one precise line the model never sees
(``fx.from_torch_tensor`` lives on ``flyc``; ``as_numeric()`` takes one
argument; ``Vector.load()`` was missing three).

These pin the loop that closes that, and the two ways it would quietly waste
the teacher instead: retrying a task forever, and re-prompting on a verdict
that says nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import repair_twin_seeds as R  # noqa: E402


# ---- what is worth a repair call ------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("TypeError: as_numeric() takes 1 positional argument but 2 were given", True),
    ("AcceleratorError: HIP error: an illegal memory access was encountered", True),
    ("SNR: -999.00 dB", True),
    ("", False),
    ("ok", False),
])
def test_only_diagnosable_failures_are_retried(text, expected):
    """A verdict with no message gives the teacher nothing; re-prompting on it
    is re-rolling the dice at full price."""
    assert R._diagnosable(text) is expected


def test_attempts_are_capped_per_task(tmp_path):
    """A kernel that cannot be fixed must stop costing teacher calls."""
    led = tmp_path / "repair_attempts.jsonl"
    led.write_text("".join(
        json.dumps({"task_id": "a", "status": "failed"}) + "\n" for _ in range(3)))
    assert R.load_attempts(led)["a"] == 3


def test_error_text_keeps_the_tail_not_the_head(tmp_path):
    """A traceback starts with the harness calling the candidate -- identical
    for every task. The exception is at the end."""
    row = {"error": "harness frame\n" * 500 + "TypeError: boom"}
    assert "TypeError: boom" in R._error_text(row)


# ---- the repaired kernel has to be re-gated -------------------------------

def test_stale_verdicts_are_dropped_so_the_gate_reruns(tmp_path):
    """The gate resumes from this file. Leaving the old verdict in place means
    the repaired kernel is never looked at and the pass is a silent no-op."""
    gate = tmp_path / "g.json"
    gate.write_text(json.dumps({"rows": [
        {"task_id": "a", "status": "incorrect"},
        {"task_id": "b", "status": "pass"},
    ]}))
    assert R.drop_verdicts(gate, {"a"}) == 1
    rows = json.loads(gate.read_text())["rows"]
    assert [r["task_id"] for r in rows] == ["b"]


def test_passing_verdicts_are_never_dropped(tmp_path):
    gate = tmp_path / "g.json"
    gate.write_text(json.dumps({"rows": [{"task_id": "b", "status": "pass"}]}))
    R.drop_verdicts(gate, set())
    assert json.loads(gate.read_text())["rows"][0]["status"] == "pass"


# ---- a repair must stay the same kind of artifact -------------------------

@pytest.mark.parametrize("task_id,seed,marker", [
    ("x__flydsl", "seed_flydsl.py", "flyc.jit"),
    ("x__hip", "seed_hip.hip", "PYBIND11_MODULE"),
    ("x__hipf", "seed_hip.hip", "PYBIND11_MODULE"),
])
def test_dialect_routing(task_id, seed, marker):
    _, meta = R._suffix(task_id)
    assert meta[0] == seed and meta[1] == marker


def test_unknown_dialect_is_skipped_not_guessed():
    assert R._suffix("plain_task")[0] is None


def test_a_repair_that_drops_the_entry_point_is_rejected(tmp_path, monkeypatch):
    """The harness looks up that symbol and nothing else, so a 'fix' without it
    is worse than the failure it replaced."""
    task = tmp_path / "tasks" / "x__flydsl"
    task.mkdir(parents=True)
    original = "@flyc.jit\ndef entry():\n    pass\n" + "# pad\n" * 100
    (task / "seed_flydsl.py").write_text(original)

    class Teacher:
        def generate(self, messages):
            return "```python\ndef entry():\n    pass\n```"   # no flyc.jit

    rec = R.repair_one(("x__flydsl", "TypeError: boom"), Teacher(), tmp_path)
    assert rec["status"] == "failed"
    assert (task / "seed_flydsl.py").read_text() == original, "seed was clobbered"


def test_a_no_op_repair_is_rejected(tmp_path):
    """Rewriting the file unchanged burns a gate slot to learn nothing."""
    task = tmp_path / "tasks" / "x__flydsl"
    task.mkdir(parents=True)
    original = "@flyc.jit\ndef entry():\n    pass\n" + "# pad\n" * 100
    (task / "seed_flydsl.py").write_text(original)

    class Teacher:
        def generate(self, messages):
            return f"```python\n{original}```"

    rec = R.repair_one(("x__flydsl", "TypeError: boom"), Teacher(), tmp_path)
    assert rec["status"] == "failed" and "changed nothing" in rec["error"]


def test_a_good_repair_is_written(tmp_path):
    task = tmp_path / "tasks" / "x__flydsl"
    task.mkdir(parents=True)
    (task / "seed_flydsl.py").write_text("@flyc.jit\ndef entry():\n    old()\n"
                                         + "# pad\n" * 100)

    fixed = "@flyc.jit\ndef entry():\n    new()\n" + "# pad\n" * 100

    class Teacher:
        def generate(self, messages):
            assert "TypeError: boom" in messages[0]["content"], "error not shown"
            return f"```python\n{fixed}```"

    rec = R.repair_one(("x__flydsl", "TypeError: boom"), Teacher(), tmp_path)
    assert rec["status"] == "repaired"
    assert "new()" in (task / "seed_flydsl.py").read_text()


def test_missing_seed_is_skipped(tmp_path):
    rec = R.repair_one(("gone__flydsl", "TypeError: boom"), None, tmp_path)
    assert rec["status"] == "skipped"


# ---- and it has to actually be wired in -----------------------------------

def test_pipeline_runs_the_repair_pass():
    src = (REPO / "scripts" / "frontier_pipeline.sh").read_text()
    assert "repair_twin_seeds.py" in src, "failing twins are still a dead end"
    assert "REPAIR_MAX_ATTEMPTS" in src, "repairs are unbounded"


def test_repair_holds_no_gpu_slot():
    """It is teacher-bound. Putting it in an allocation would hold a node
    hostage to network latency, which is why the materializers run this way."""
    src = (REPO / "scripts" / "frontier_pipeline.sh").read_text()
    block = src.split("--- 1b")[1].split("--- 2.")[0]
    assert "setsid nohup" in block, "repair is not detached on the login node"
    assert "sbatch" not in block, "repair takes an allocation"


def test_repair_covers_both_dialects():
    src = (REPO / "scripts" / "frontier_pipeline.sh").read_text()
    block = src.split("--- 1b")[1].split("--- 2.")[0]
    assert "FLYDSL_ROOT" in block and "REG_HIP_ROOT" in block
