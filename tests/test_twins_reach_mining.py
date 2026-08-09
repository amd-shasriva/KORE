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


def test_pool_flydsl_passers_are_harvested_and_mined(pipeline, loops, staff):
    """FlyDSL is 25% of the arena and 0.6% of the corpus, so its gated twins
    must be both promoted and worked. Repair stays off it -- 121 HIP kernels
    rescued against zero FlyDSL -- but mining a kernel that already passes is a
    different question from trying to fix one that does not."""
    assert "POOL_FLYDSL_OK_ROOT" in pipeline, "pool FlyDSL passers are not promoted"
    assert _wanted(loops, "poolflydsl") == _wanted(staff, "poolflydsl")
    assert int(_wanted(loops, "poolflydsl")) > 0, "FlyDSL gated twins are not mined"


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


#: The dialects the arena scores that the corpus is short of. Triton is not one
#: of them: 11,884 Triton rows against 738 HIP and 229 FlyDSL, so a marginal
#: Triton row is worth close to nothing and a slot spent on it is a slot not
#: spent on the two dialects that are 47% of the arena between them.
HIP_FLYDSL_STREAMS = ("frontiertwins", "poolflydsl", "hipreg", "poolhip")
TRITON_STREAMS = ("frontier", "pooltriton")


def _stream_wants(src):
    return [(t.split(":")[0], int(t.split(":")[3]))
            for t in src.replace('"', " ").replace("\\", " ").split()
            if ":runs/shards" in t]


def test_no_triton_is_mined(loops, staff):
    """Triton mining is switched off outright, not merely deprioritised."""
    for src, name in ((loops, "ensure_loops.sh"), (staff, "staff_datagen.sh")):
        wants = dict(_stream_wants(src))
        for stream in TRITON_STREAMS:
            assert wants.get(stream, 0) == 0, \
                f"{name} still staffs {stream} with {wants[stream]} worker(s)"


def test_every_worker_goes_to_hip_or_flydsl(loops, staff):
    for src, name in ((loops, "ensure_loops.sh"), (staff, "staff_datagen.sh")):
        wants = dict(_stream_wants(src))
        staffed = {s for s, w in wants.items() if w > 0}
        assert staffed, f"{name} staffs nothing at all"
        assert staffed <= set(HIP_FLYDSL_STREAMS), \
            f"{name} staffs a non-HIP/FlyDSL stream: {staffed - set(HIP_FLYDSL_STREAMS)}"


def test_frontier_difficulty_twins_are_staffed_first(loops, staff):
    """Streams are staffed in declaration order, so order is priority. The
    frontier twins are the only HIP set whose difficulty comes from the task
    rather than the dialect."""
    for src, name in ((loops, "ensure_loops.sh"), (staff, "staff_datagen.sh")):
        order = [s for s, _ in _stream_wants(src)]
        assert order[0] == "frontiertwins", f"{name} staffs {order[0]} first"


def test_repair_budget_follows_the_dialect_it_can_actually_fix(loops):
    """Across ~9,500 repairs: 121 HIP kernels went from failing to passing and
    zero FlyDSL did. Spending half the teacher budget on FlyDSL bought nothing."""
    pipeline = (SCRIPTS / "frontier_pipeline.sh").read_text()
    assert "REPAIR_ROOTS" in pipeline
    block = pipeline.split("--- 1b")[1].split("--- 2.")[0]
    assert "$REPAIR_ROOTS" in block, "repair still walks a hardcoded root list"


def test_lowest_value_stream_is_still_last(loops):
    order = [s for s, _ in _stream_wants(loops)]
    assert order.index("poolhip") > order.index("frontiertwins"), \
        "launch-bound pool HIP would take a slot before the frontier twins"
