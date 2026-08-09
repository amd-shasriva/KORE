"""A twin that is seeded and gated must also be mined.

The twin pipeline has four stages -- materialize, gate, harvest, mine -- and
only the first two were connected. The gate writes a verdict file and nothing
else; promoting the passers into a resolvable task root and sharding them is
the harvester's job, and the pipeline's harvest step was a comment followed by
a bare ``:``. So 1,104 registry-HIP and 309 registry-FlyDSL twins reached a
verdict and stopped there, no staffed shard set contained a single ``__hip`` or
``__flydsl`` id, and the 22% of the arena that is HIP and the 25% that is
FlyDSL were being scored against training data that was never produced.

Seeding a twin nothing will mine is worse than not seeding it: it spends the
teacher and a gate slot to produce a directory no stage reads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def pipeline() -> str:
    return (SCRIPTS / "frontier_pipeline.sh").read_text()


@pytest.fixture(scope="module")
def harvest() -> str:
    return (SCRIPTS / "hip_pool_harvest.sh").read_text()


@pytest.fixture(scope="module")
def loops() -> str:
    return (SCRIPTS / "ensure_loops.sh").read_text()


@pytest.fixture(scope="module")
def staff() -> str:
    return (SCRIPTS / "staff_datagen.sh").read_text()


# ---- the harvest must actually run ----------------------------------------

def test_pipeline_calls_the_harvester(pipeline):
    assert "hip_pool_harvest.sh" in pipeline, \
        "gate verdicts are never turned into promoted tasks"


def test_harvest_step_is_not_a_no_op(pipeline):
    """It was literally `:` under a comment saying harvest happened elsewhere."""
    assert ": # harvest is owned by" not in pipeline


def test_harvest_leaves_submission_to_staffing(pipeline):
    """Two submitters means a stream gets queued twice and one must be killed."""
    assert "NO_SUBMIT=1" in pipeline


# ---- FlyDSL twins must be visible to it -----------------------------------

def test_harvest_can_see_flydsl_twins(harvest):
    """The glob matched only *__hip*, so a gated FlyDSL twin was skipped by the
    promote loop and omitted from the id list -- invisible after gating."""
    assert "TWIN_GLOBS" in harvest
    assert "list_twins" in harvest


def test_pipeline_harvests_all_three_suffixes(pipeline):
    assert "'*__hip *__hipf *__flydsl'" in pipeline


# ---- frontier twins must not be diluted by the pool -----------------------

def test_frontier_twins_get_their_own_root(pipeline, harvest):
    """data/pool_hip_ok holds 6,457 pool twins. Promoting the frontier ones
    into it would make them a few percent of the shard set and mine the
    launch-bound majority instead."""
    for var in ("HIP_PROMOTED", "HIP_DATA_ROOT", "HIP_SHARD_DIR"):
        assert var in harvest, f"{var} is not overridable"
    assert "TWIN_OK_ROOT" in pipeline
    assert "data/pool_hip_ok" not in pipeline.split("TWIN_OK_ROOT")[1][:400]


def test_harvest_filters_to_the_selection(harvest, pipeline):
    """The registry roots hold 740 twins seeded before they were narrowed to
    the frontier 482; promoting those would put them straight back in."""
    assert "HIP_TASK_LIST" in harvest
    assert 'HIP_TASK_LIST="$FRONTIER_TASK_LIST"' in pipeline


# ---- and something must mine them -----------------------------------------

def test_a_stream_is_staffed_on_the_twins(loops, staff):
    for src, name in ((loops, "ensure_loops.sh"), (staff, "staff_datagen.sh")):
        assert "frontiertwins:runs/shards_frontier_twins" in src, \
            f"{name} declares no twin mining stream"
        want = _wanted(src, "frontiertwins")
        assert int(want) > 0, f"{name} staffs the twin stream with {want} workers"


def _wanted(src: str, name: str) -> str:
    """The worker count a stream spec asks for, wherever it is quoted."""
    for token in src.replace('"', " ").replace("\\", " ").split():
        if token.startswith(name + ":"):
            return token.split(":")[3]
    raise AssertionError(f"stream {name} not declared")


def test_staffing_default_matches_the_live_config(loops, staff):
    """staff_datagen is also run by hand, and a stale default once staffed four
    miners onto a stream that had just been retired."""
    for name in ("frontier", "frontiertwins"):
        a, b = _wanted(loops, name), _wanted(staff, name)
        assert a == b, \
            f"{name}: ensure_loops wants {a}, staff_datagen default is {b}"


def test_pool_flydsl_passers_are_harvested_too(pipeline, loops, staff):
    """FlyDSL is a quarter of the arena and the registry set yields almost
    nothing -- 3 passes against 172 from the pool. Leaving the pool's passers
    ungathered repeats the original bug on the only dialect that is short."""
    assert "POOL_FLYDSL_OK_ROOT" in pipeline, "pool FlyDSL passers are not promoted"
    assert _wanted(loops, "poolflydsl") == _wanted(staff, "poolflydsl")
    assert int(_wanted(loops, "poolflydsl")) > 0, "declared but staffed with nobody"


def test_pool_flydsl_is_not_pooled_into_the_frontier_set(pipeline):
    """Difficulty must not be silently mixed: the frontier set is named for it."""
    block = pipeline.split("POOL_FLYDSL_OK_ROOT=")[1][:600]
    assert "TWIN_OK_ROOT" not in block


def test_pool_flydsl_harvest_skips_the_registry_task_list(pipeline):
    """frontier_tasks.txt holds registry ids; applying it to pool twins would
    filter out every one of them and silently harvest nothing."""
    seg = pipeline.split('HIP_PROMOTED="$REPO/$POOL_FLYDSL_OK_ROOT"')[1].split("hip_pool_harvest.sh")[0]
    assert "HIP_TASK_LIST" not in seg


def test_twin_shards_are_kept_current(pipeline):
    """A manifest older than the checkout makes every worker die on preflight,
    which in the queue looks exactly like waiting a turn."""
    refresh_block = pipeline.split("refresh_shards.py")[0]
    assert "$TWIN_SHARD_DIR" in refresh_block[-400:], \
        "the twin shard set is never refreshed against the current commit"


def test_the_dialect_with_no_data_is_staffed_first(loops, staff):
    """Streams are staffed in declaration order and general QoS has room for two
    miners, so order is priority. FlyDSL is 25% of the arena and had produced
    zero rows ever while its miner sat 8 hours in the burst queue behind streams
    that already had 11,879 and 794 rows."""
    for src, name in ((loops, "ensure_loops.sh"), (staff, "staff_datagen.sh")):
        specs = [t.split(":")[0] for t in
                 src.replace('"', " ").replace("\\", " ").split()
                 if ":runs/shards" in t]
        assert specs[0] == "poolflydsl", f"{name} staffs {specs[0]} before FlyDSL"
        assert specs.index("frontiertwins") < specs.index("frontier"), \
            f"{name} prefers Triton over its own frontier twins"


def test_lowest_value_stream_is_still_last(loops):
    specs = [t.split(":")[0] for t in
             loops.replace('"', " ").replace("\\", " ").split() if ":runs/shards" in t]
    assert specs.index("poolhip") > specs.index("frontiertwins"), \
        "launch-bound pool HIP would take a slot before the frontier twins"
