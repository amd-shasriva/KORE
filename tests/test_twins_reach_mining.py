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
    """The twins have to be mined by something; which stream is a plan detail.

    1c517146 merged the two HIP sources into one ``frontierhip`` stream -- the
    211 registry frontier twins and the 1,857 scored hard-pool tasks -- because
    mining them separately gave 45% attention on one and 94% GEMM on the other,
    which is only what each source happens to hold. ``frontiertwins`` went to 0
    in the same commit: its tasks are inside frontierhip now and staffing both
    would cover them twice. It stays declared so its ledger and shard set
    survive, which is what makes reviving it one number.
    """
    for src, name in ((loops, "ensure_loops.sh"), (staff, "staff_datagen.sh")):
        assert "frontiertwins:runs/shards_frontier_twins" in src, \
            f"{name} deleted the twin stream instead of retiring it, losing its ledger"
        assert "frontierhip:runs/shards_frontierhip:data/v5frontierhip" in src, \
            f"{name} declares no merged HIP stream for the twins to have been folded into"
        wants = dict(_stream_wants(src))
        carrying = {s: wants.get(s, 0) for s in ("frontierhip", "frontiertwins")}
        assert sum(carrying.values()) > 0, \
            f"{name} staffs no stream on the frontier twins: {carrying}"


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


def _pool_guarded_blocks(pipeline: str) -> list[str]:
    """The bodies of the POOL_STREAMS switch, in file order."""
    return [b.split("\n    fi")[0]
            for b in pipeline.split('if [ "$POOL_STREAMS" = "1" ]; then')[1:]]


def test_pool_flydsl_is_seeded_harvested_and_mined_together(pipeline, loops, staff):
    """Whatever the pool FlyDSL stream is doing, it must do all of it or none.

    It used to be promoted and worked because FlyDSL is 25% of the arena and
    0.6% of the corpus. dac7d257 measured it against runs/frontier_tasks.txt
    instead: 226 tasks, 0% of them on the list -- kbk_actor, kbk_mlp,
    kbk_classifier at fp32, scraped modules baselined against eager torch
    rather than AITER or hipBLASLt. It is the right dialect and the wrong
    difficulty, so it goes to 0.

    What this file exists to stop is a stage that half-runs. The whole pipeline
    stalled once because the gate wrote a verdict nothing promoted, and 1,104
    registry-HIP and 309 registry-FlyDSL twins reached that verdict and stopped
    -- so seeding a twin nothing will mine is worse than not seeding it, and it
    is worse in exactly the same way whether the missing stage is the harvest
    or the miner. Hence one switch over all three stages: POOL_STREAMS gates
    the materializer and the harvest, and the stream's own count is 0 to match.
    """
    assert 'POOL_STREAMS="${POOL_STREAMS:-0}"' in pipeline, \
        "the pool streams have no single switch any more"

    blocks = _pool_guarded_blocks(pipeline)
    assert len(blocks) == 2, f"expected a seed block and a harvest block, got {len(blocks)}"
    assert any('materialize_pool_flydsl.py "$FLYDSL_ROOT"' in b for b in blocks), \
        "the pool FlyDSL materializer runs whether or not the stream is revived"
    assert any('HIP_PROMOTED="$REPO/$POOL_FLYDSL_OK_ROOT"' in b for b in blocks), \
        "the pool FlyDSL harvest runs whether or not the stream is revived"

    # The promoted root and shard set stay declared at the top level so the
    # already-gated 172 passers survive the retirement: 01dcc807 made a point of
    # keeping them promoted and sharded so resuming is a config change, not a
    # rebuild.
    for var in ("POOL_FLYDSL_OK_ROOT=", "POOL_FLYDSL_SHARD_DIR="):
        assert var in pipeline, f"{var} was deleted, so reviving the stream is a rebuild"

    assert _wanted(loops, "poolflydsl") == _wanted(staff, "poolflydsl")
    assert int(_wanted(loops, "poolflydsl")) == 0, \
        "the stream is staffed while POOL_STREAMS leaves its seeds unharvested"


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


#: Streams whose task list was chosen by score rather than by whatever the
#: source happened to hold. select_frontier_tasks ranks both halves:
#: runs/frontier_tasks.txt is the 482 registry ids above the histogram break,
#: and --out-pool --min-score 2 is the 1,857 pool tasks at a million elements
#: or more. frontierhip mines the two interleaved, frontiertriton the registry
#: half in Triton; frontier, frontiertwins and hardpool are the predecessors
#: those two merged, kept declared at 0 so their ledgers survive.
SCORED_STREAMS = ("frontierhip", "frontiertriton", "frontiertwins", "hardpool",
                  "frontier")

#: Streams that mine a source end to end. dac7d257 measured each against
#: runs/frontier_tasks.txt: hipreg is 1% frontier -- it reads
#: runs/unmined_hip.txt, which is hip_abs_fp16 and hip_div_fp32, the generated
#: elementwise set -- and poolflydsl is 0%. poolhip and pooltriton mine the raw
#: KernelBook pool, where the median baseline is 17us and 86-92% is under
#: 100us, so no amount of it teaches tiling, LDS staging or MFMA scheduling.
BREADTH_STREAMS = ("poolflydsl", "hipreg", "poolhip", "pooltriton")


def _stream_wants(src):
    return [(t.split(":")[0], int(t.split(":")[3]))
            for t in src.replace('"', " ").replace("\\", " ").split()
            if ":runs/shards" in t]


def test_triton_mining_is_aimed_at_the_frontier_not_the_pool(loops, staff):
    """Triton mining is not switched off any more, it is aimed.

    The old rule was per-dialect -- 11,884 Triton rows against 738 HIP and 229
    FlyDSL, so a marginal Triton row was worth close to nothing -- and it was
    right about the pool and wrong about the registry. 1c517146 separated them:
    pool-Triton is 6,064 launch-bound rows, median baseline 16us, zero
    attention and zero MoE, while the registry frontier still has 354 unmined
    Triton tasks worth ~24M tokens, and triton2triton is 38% of the arena and
    15 points behind Opus. So the pool stream stays at 0 and the frontier one
    gets miners until it runs out.

    Asserting on .get(name) rather than .get(name, 0): a renamed or deleted
    stream must fail here, because the version of this test that defaulted to 0
    went on passing while three of six miners sat on Triton.
    """
    for src, name in ((loops, "ensure_loops.sh"), (staff, "staff_datagen.sh")):
        wants = dict(_stream_wants(src))
        assert wants.get("pooltriton") == 0, \
            f"{name} staffs pool-Triton with {wants.get('pooltriton')} worker(s)"
        assert wants.get("frontiertriton", 0) > 0, \
            f"{name} mines no Triton at all, with 354 registry frontier tasks left"


def test_every_worker_goes_to_a_scored_stream(loops, staff):
    """The rule was per-dialect until the dialect stopped predicting value.

    dac7d257 measured every active stream against runs/frontier_tasks.txt --
    frontiertwins 100% frontier, hipreg 1%, poolflydsl 0% -- and hipreg had
    mined 3,250 rows, more than any other stream, which made the largest part
    of the corpus its least difficult part. Four of six miners were on work the
    frontier list rejects. What decides whether a slot is worth spending is
    whether the stream's task list was scored, not which language it is in;
    that is why frontiertriton may hold miners while poolflydsl may not.
    """
    for src, name in ((loops, "ensure_loops.sh"), (staff, "staff_datagen.sh")):
        wants = dict(_stream_wants(src))
        staffed = {s for s, w in wants.items() if w > 0}
        assert staffed, f"{name} staffs nothing at all"
        assert staffed <= set(SCORED_STREAMS), \
            f"{name} staffs an unscored stream: {staffed - set(SCORED_STREAMS)}"
        # A retirement is reversed by editing one number in place, so the
        # breadth streams have to still be here under these names for the check
        # above to mean anything.
        assert set(BREADTH_STREAMS) <= set(wants), \
            f"{name} no longer declares {set(BREADTH_STREAMS) - set(wants)}, " \
            "so this test would pass however they were revived"


def test_frontier_difficulty_twins_are_staffed_first(loops, staff):
    """Streams are staffed in declaration order, so order is priority.

    The frontier twins are still the set whose difficulty comes from the task
    rather than the dialect -- primary scales from 16.7M to 68.7B elements
    against the pool's uniform 1M, and AITER and hipBLASLt baselines rather
    than eager torch -- and since 1c517146 they are mined as the registry half
    of frontierhip. So the merged HIP stream is what has to come first, and it
    has to be genuinely staffed: declaring it first at 0 workers would hand the
    first free slot to frontiertriton instead.
    """
    for src, name in ((loops, "ensure_loops.sh"), (staff, "staff_datagen.sh")):
        wants = _stream_wants(src)
        order = [s for s, _ in wants]
        assert order[0] == "frontierhip", f"{name} staffs {order[0]} first"
        assert dict(wants)["frontierhip"] > 0, \
            f"{name} declares frontierhip first and gives it no workers"


def test_repair_budget_follows_the_dialect_it_can_actually_fix(loops):
    """Across ~9,500 repairs: 121 HIP kernels went from failing to passing and
    zero FlyDSL did. Spending half the teacher budget on FlyDSL bought nothing."""
    pipeline = (SCRIPTS / "frontier_pipeline.sh").read_text()
    assert "REPAIR_ROOTS" in pipeline
    block = pipeline.split("--- 1b")[1].split("--- 2.")[0]
    assert "$REPAIR_ROOTS" in block, "repair still walks a hardcoded root list"


def test_lowest_value_stream_is_still_last(loops, staff):
    """No retired stream may sit ahead of a staffed one in the declaration.

    Comparing two retired streams to each other proves nothing -- neither takes
    a slot -- and the ordering only ever matters at the moment a retirement is
    reversed, which is a supported operation here: every retired stream is kept
    declared at 0 precisely so its ledger survives and reviving it is one
    number, edited in place. Declaring poolhip -- 6,457 pool twins, 86% under
    100us, median 17us -- above frontierhip would put it first in line for
    every slot that frees the moment somebody typed that number.
    """
    for src, name in ((loops, "ensure_loops.sh"), (staff, "staff_datagen.sh")):
        wants = _stream_wants(src)
        staffed = [i for i, (_, w) in enumerate(wants) if w > 0]
        retired = [i for i, (_, w) in enumerate(wants) if w == 0]
        assert staffed, f"{name} staffs nothing at all"
        assert retired, f"{name} retires nothing, so this check is vacuous"
        assert max(staffed) < min(retired), \
            f"{name} declares a retired stream ahead of a staffed one: " \
            f"{[(s, w) for s, w in wants]}"
