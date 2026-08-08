"""The unattended pipeline has to hold four properties or it silently stops.

Each is a failure this project has already paid for:

  * A materializer put inside an allocation holds a GPU node hostage to gateway
    latency -- 8 workers at ~1 CPU-second per 5 minutes -- while the arena queues
    behind it.
  * A stage that exits when its own sub-goal completes takes the rest of the
    pipeline with it. hip_pipeline_loop returned rc=0 after seeding finished and
    killed the staffing pass that was its only caller, and five of six mining
    streams stayed dead overnight.
  * One gate job name shared across roots means a slow root blocks a fast one.
  * A shard manifest stamped at an old commit kills every worker on the preflight
    check with NonZeroExitCode, which reads as a scheduler problem rather than a
    stale-manifest one.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PIPELINE = REPO / "scripts" / "frontier_pipeline.sh"
ENSURE = REPO / "scripts" / "ensure_loops.sh"
STAFF = REPO / "scripts" / "staff_datagen.sh"


@pytest.fixture(scope="module")
def pipeline() -> str:
    return PIPELINE.read_text()


def test_pipeline_exists_and_is_executable():
    assert PIPELINE.is_file()
    assert PIPELINE.stat().st_mode & 0o111, "must be executable"


def test_materializers_never_run_inside_an_allocation(pipeline):
    """They are gateway-bound; an sbatch here would idle a node on network waits."""
    for line in pipeline.splitlines():
        if "materialize_" in line and not line.strip().startswith("#"):
            assert "sbatch" not in line, f"materializer submitted to the scheduler: {line}"
    assert "setsid nohup" in pipeline, "materializers must be detached, not blocking"


def test_only_the_gate_consumes_a_gpu_slot(pipeline):
    """Everything else is CPU or delegated to staff_datagen."""
    submits = [l for l in pipeline.splitlines()
               if "sbatch" in l and not l.strip().startswith("#")]
    assert submits, "the gate must be submitted somewhere"
    for line in submits:
        assert "gate" in line.lower(), f"non-gate sbatch in the pipeline: {line}"


def test_the_loop_never_exits_on_its_own(pipeline):
    """The bug that killed staffing: rc=0 after a sub-goal, which keepalive
    correctly declines to restart."""
    body = pipeline.split("while :;", 1)[1]
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("#"):
            continue
        assert not re.match(r"^exit\s+\d+", s), f"pipeline exits from its loop: {s}"


def test_each_root_gets_its_own_gate_job_name(pipeline):
    """A shared name lets a slow root starve a fast one."""
    assert 'kore-gate-$tag' in pipeline or "kore-gate-${tag}" in pipeline
    assert 'tag=$(basename' in pipeline


def test_gating_is_skipped_when_that_root_already_has_a_job(pipeline):
    assert 'queued "kore-gate-$tag"' in pipeline


def test_gating_respects_the_slot_cap(pipeline):
    assert "have_slot" in pipeline, "must ask before submitting, not submit and hope"


def test_shard_stamps_are_refreshed_every_pass(pipeline):
    """Every commit invalidates every manifest; unrefreshed means instant death."""
    assert "refresh_shards.py" in pipeline


def test_held_jobs_are_purged_every_pass(pipeline):
    """A held job occupies the cap while doing no work."""
    assert "purge_held" in pipeline


def test_families_are_always_passed_to_the_materializers(pipeline):
    """Without --families these regenerate the launch-bound bulk whose median
    baseline is 17us -- the exact thing this pipeline exists to stop."""
    assert "--families" in pipeline
    assert "FRONTIER_FAMILIES" in pipeline


# ---- wiring ---------------------------------------------------------------

def test_ensure_loops_starts_the_frontier_pipeline():
    s = ENSURE.read_text()
    assert re.search(r"^start frontier_pipeline", s, re.M)


def test_ensure_loops_no_longer_starts_the_superseded_hip_loop():
    s = ENSURE.read_text()
    started = re.findall(r"^start ([a-z_]+)", s, re.M)
    assert "hip_pipeline" not in started, "the HIP-only loop is superseded"
    assert set(started) == {"frontier_pipeline", "supervise", "supervise_base"}


def test_every_started_loop_has_its_keepalive_wrapper():
    """ensure_loops matches on the wrapper, so a loop started without one is
    invisible to it and will never be restarted.

    Matched against the *expanded* command rather than the source text: in the
    source the path is quoted, so ``keepalive.sh" frontier_pipeline`` is what
    appears, while ``running()`` greps the process args where the quote is gone.
    Asserting on the raw source would fail on correct code -- as it did.
    """
    s = ENSURE.read_text()
    expanded = s.replace('"', "")
    for name in re.findall(r"^start ([a-z_]+)", s, re.M):
        assert f"keepalive.sh {name} " in expanded, \
            f"{name} is started without the wrapper running() looks for"


def test_staffing_default_matches_the_configured_streams():
    """A stale default silently staffed a retired stream when run by hand."""
    ens = re.search(r'DATAGEN_STREAMS="([^"]+)"', ENSURE.read_text()).group(1)
    default = re.search(r'STREAMS="\$\{DATAGEN_STREAMS:-\\?\n?(.*?)\}"',
                        STAFF.read_text(), re.S).group(1)
    def weights(blob):
        return {m.group(1): m.group(2)
                for m in re.finditer(r"(\w+):[^:]+:[^:]+:(\d+):", blob)}
    assert weights(ens) == weights(default)
