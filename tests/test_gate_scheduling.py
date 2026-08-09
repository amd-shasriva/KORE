"""The gate must be able to reach a queue that actually runs.

Gating is the step that turns a teacher-written seed into something mineable.
Until a root is gated it contributes exactly zero training rows, however many
seeds it has: 861 registry-HIP and 671 FlyDSL twins were on disk while the
promoted count for FlyDSL was still zero.

Every gate was submitted to burst, which had 114 jobs running and 35 pending,
so one sat for an hour while 15 nodes stood idle and all five of my other jobs
ran on general. The chooser that would have picked general existed, but it
lived inside staff_datagen.sh where only mining could call it, and it counted
only mining jobs against the share.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def slots() -> str:
    return (SCRIPTS / "gpu_slots.sh").read_text()


@pytest.fixture(scope="module")
def pipeline() -> str:
    return (SCRIPTS / "frontier_pipeline.sh").read_text()


@pytest.fixture(scope="module")
def staff() -> str:
    return (SCRIPTS / "staff_datagen.sh").read_text()


def test_qos_chooser_is_shared(slots, staff):
    """It must live where every submitter can reach it, not inside one of them."""
    assert "pick_qos()" in slots, "chooser is not in the shared slot module"
    assert "pick_qos()" not in staff, "staff_datagen still defines its own copy"


def test_chooser_is_per_job_kind(slots):
    """A single global share let mining spend the whole allowance."""
    assert 'local prefix="$1" max="$2"' in slots


def test_gate_uses_the_chooser_not_the_burst_default(pipeline):
    assert "pick_qos kore-gate-" in pipeline, "gate does not choose a QoS"
    assert "sbatch $QOS_ARG --job-name=\"kore-gate-" not in pipeline, \
        "gate still hardcodes the burst default"


def test_mining_still_bounded_on_general(staff):
    """Mining may use general, but not all of it -- the gate needs a slot."""
    assert 'pick_qos kore-mine- "$GENERAL_MINE_MAX"' in staff


def test_gate_share_is_reserved(pipeline):
    assert "GENERAL_GATE_MAX" in pipeline


def test_only_one_gate_is_in_flight_across_roots(pipeline):
    """A gate per root put three in burst, where they never ran and still held
    three of the eight cap slots -- with two arenas that left nothing for
    mining, so no miner could be staffed at all."""
    assert "gates_in_flight" in pipeline
    assert "MAX_GATES_IN_FLIGHT" in pipeline


def test_in_flight_count_spans_every_root(pipeline):
    """Counting only this root's own gate is what allowed four at once."""
    body = pipeline.split("gates_in_flight()")[1][:300]
    assert "'^kore-gate-'" in body, "the count is not across all gate jobs"


# ---- the gate must notice work that is not a new directory ----------------

def test_gate_triggers_on_ungated_verdicts_not_new_seeds(pipeline):
    """A repair replaces a seed in place, so the seed count does not move. The
    old trigger was seeds-minus-remembered, and 628 repaired kernels produced
    no gate at all -- the entire repair loop was a no-op."""
    assert "n_verdicts" in pipeline, "gate cannot see what is already decided"
    assert 'ungated=$(( seeds - $(n_verdicts "$root") ))' in pipeline
    assert '"$ungated" -ge "$GATE_EVERY"' in pipeline


def test_gate_bookkeeping_is_on_disk_not_in_memory(pipeline):
    """LAST_GATED lived in the loop's memory, so every restart re-gated
    everything and every repair was invisible."""
    assert "LAST_GATED" not in pipeline
