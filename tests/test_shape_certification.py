"""End-to-end proof that held-out shape certification is reachable.

The shape system is only worth anything if the hidden lane is written to disk
BEFORE training and merely consumed afterwards.  These tests exercise that whole
path: the training-time writer, the durable artifacts it publishes, the receipt
that binds them together, and the certification gate that fails closed without
them.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from kore.eval.champion import (
    Champion,
    held_out_shapes,
    load_shape_manifests,
    reeval_champion,
    run_champion_reeval,
)
from kore.policy.budget import (
    BudgetExceededError,
    BudgetLedgerV1,
    EvaluationWork,
)
from kore.tasks.base import Shape
from kore.tasks.registry import get_task
from kore.tasks.shape_policy import (
    SPLIT_INDEX_FILENAME,
    ShapeSplitIndex,
    _digest,
    freeze_shape_split,
    freeze_shape_splits,
    load_frozen_shape_splits,
    load_shape_split_index,
    manifest_filename,
    shape_key,
    validate_frozen_split,
)

# Deliberately spread over hand-authored and generated families, and over the
# transform kinds (rows, context, coupled query/key, sequence, block-quant).
CERTIFIED_TASK_IDS = (
    "softmax_bf16",
    "gemm_bf16",
    "rmsnorm_aiter",
    "flash_attn_prefill_bf16",
    "paged_attn_decode_bf16",
    "moe_biased_grouped_topk_bf16",
    "genb_attn2_decode_gqa_hd128_bf16",
    "genb_ssm_mamba2_ssd_c128_n128_bf16",
    "genb_qx_quant_fp8_block2d_fp8",
    "gen_add_bf16",
)


def _tasks(*task_ids: str):
    return [get_task(task_id) for task_id in (task_ids or CERTIFIED_TASK_IDS)]


def _rehash(value: dict) -> str:
    """Re-sign an edited artifact, as a determined tamperer would."""
    return _digest({k: v for k, v in value.items() if k != "content_hash"})


# --------------------------------------------------------------------------- #
# The writer: durable artifacts with a directory receipt
# --------------------------------------------------------------------------- #
def test_frozen_manifests_round_trip_through_the_writer(tmp_path):
    tasks = _tasks()
    index = freeze_shape_splits(tasks, tmp_path, seed=5)

    assert index.task_ids == tuple(sorted(task.task_id for task in tasks))
    assert index.seed == 5
    assert index.hidden_shapes == 8 * len(tasks)
    assert index.content_hash == index.computed_hash()
    assert (tmp_path / SPLIT_INDEX_FILENAME).exists()
    for task in tasks:
        assert (tmp_path / manifest_filename(task.task_id)).exists()

    loaded = load_frozen_shape_splits(tmp_path, tasks=tasks, require_index=True)
    assert sorted(loaded) == sorted(task.task_id for task in tasks)
    for task in tasks:
        manifest = loaded[task.task_id]
        in_memory = freeze_shape_split(
            task, seed=5, created_at=manifest.created_at)
        # The persisted split IS the split the freezer produces: same lineage
        # digests, same train universe, same hidden lane, same content hash.
        assert manifest.content_hash == in_memory.content_hash
        assert manifest.to_dict() == in_memory.to_dict()
        assert index.entry(task.task_id).content_hash == manifest.content_hash
        assert held_out_shapes(task, frozen_split=manifest) == list(
            manifest.hidden_shapes)


def test_champion_loader_reads_the_writers_directory(tmp_path):
    tasks = _tasks("softmax_bf16", "gemm_bf16")
    freeze_shape_splits(tasks, tmp_path, seed=3)
    manifests = load_shape_manifests(
        str(tmp_path), tasks=tasks, require_index=True)
    assert sorted(manifests) == ["gemm_bf16", "softmax_bf16"]
    assert manifests == load_frozen_shape_splits(tmp_path)


def test_refreezing_reuses_the_original_hidden_lane(tmp_path):
    tasks = _tasks("softmax_bf16", "gemm_bf16")
    first = freeze_shape_splits(tasks, tmp_path, seed=1)
    raw = {
        path.name: path.read_text() for path in sorted(tmp_path.glob("*.json"))
    }

    # A second freeze must not re-derive the hidden lane: it was chosen once,
    # before training, so re-running the writer changes nothing at all - not even
    # the bytes of the receipt, which a lineage digest may cover.
    second = freeze_shape_splits(tasks, tmp_path, seed=1)
    assert second == first
    assert {
        path.name: path.read_text() for path in sorted(tmp_path.glob("*.json"))
    } == raw

    # Asking for a different lane is refused rather than quietly ignored (which
    # would leave the receipt describing parameters the manifests never used).
    with pytest.raises(ValueError, match="cannot be re-parameterised"):
        freeze_shape_splits(tasks, tmp_path, seed=99)
    with pytest.raises(ValueError, match="cannot be re-parameterised"):
        freeze_shape_splits(tasks, tmp_path, seed=1, hidden_max_shapes=4)
    assert freeze_shape_splits(tasks, tmp_path, seed=1) == first

    # Replacing it is possible, but only as an explicit, auditable act.
    third = freeze_shape_splits(tasks, tmp_path, seed=99, refreeze=True)
    assert third.seed == 99
    assert third.entry("gemm_bf16").content_hash != first.entry(
        "gemm_bf16").content_hash
    assert load_frozen_shape_splits(tmp_path, tasks=tasks)[
        "gemm_bf16"].seed == 99


def test_partial_freeze_keeps_earlier_manifests_in_the_receipt(tmp_path):
    freeze_shape_splits(_tasks("softmax_bf16", "gemm_bf16"), tmp_path)
    index = freeze_shape_splits(_tasks("rmsnorm_aiter"), tmp_path)
    assert index.task_ids == ("gemm_bf16", "rmsnorm_aiter", "softmax_bf16")
    assert sorted(load_frozen_shape_splits(tmp_path, require_index=True)) == [
        "gemm_bf16", "rmsnorm_aiter", "softmax_bf16"]


def test_writer_rejects_unusable_task_ids_and_duplicates(tmp_path):
    with pytest.raises(ValueError, match="cannot name a frozen shape manifest"):
        manifest_filename("../escape")
    task = get_task("softmax_bf16")
    with pytest.raises(ValueError, match="duplicate task"):
        freeze_shape_splits([task, task], tmp_path)


# --------------------------------------------------------------------------- #
# A moved digest is rejected
# --------------------------------------------------------------------------- #
def test_moved_task_digest_is_rejected_at_load_and_at_consumption(tmp_path):
    task = SimpleNamespace(
        task_id="digest_move",
        operation="softmax",
        dtype="bf16",
        backend="triton",
        gpu_target="gfx950",
        raw={},
        shapes=[Shape("primary", {"M": 4096, "N": 4096})],
    )
    freeze_shape_splits([task], tmp_path)
    manifest = load_frozen_shape_splits(tmp_path)["digest_move"]

    task.shapes[0].dims["M"] = 8192
    with pytest.raises(ValueError, match="task digest changed"):
        load_frozen_shape_splits(tmp_path, tasks=[task])
    with pytest.raises(ValueError, match="task digest changed"):
        held_out_shapes(task, frozen_split=manifest)
    # The writer will not paper over it either: a stale manifest is an error,
    # not an invitation to pick a fresh hidden lane.
    with pytest.raises(ValueError, match="task digest changed"):
        freeze_shape_splits([task], tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("policy_digest", "0" * 64, "internal policy digest"),
        ("task_file_digest", "0" * 64, "task file digest changed"),
        ("engine_digest", "0" * 64, "policy engine digest changed"),
        ("code_identity", "another-commit", "code identity digest changed"),
    ],
)
def test_each_lineage_digest_is_load_bearing(field, value, message):
    task = get_task("softmax_bf16")
    split = freeze_shape_split(task)
    moved = replace(split, **{field: value})
    moved = replace(moved, content_hash=moved.computed_hash())
    with pytest.raises(ValueError, match=message):
        validate_frozen_split(task, moved)
    with pytest.raises(ValueError, match=message):
        held_out_shapes(task, frozen_split=moved)


def test_edited_manifest_and_receipt_are_both_rejected(tmp_path):
    tasks = _tasks("softmax_bf16", "gemm_bf16")
    index = freeze_shape_splits(tasks, tmp_path)
    path = tmp_path / manifest_filename("softmax_bf16")
    original = path.read_text()

    # 1. Editing a hidden shape breaks the manifest's own content hash.
    edited = json.loads(original)
    edited["hidden_shapes"][0]["dims"]["M"] += 2
    path.write_text(json.dumps(edited))
    with pytest.raises(ValueError, match="content hash"):
        load_frozen_shape_splits(tmp_path)

    # 2. Re-signing the edit keeps the manifest self-consistent, so the receipt
    #    is what catches the swap.
    edited["content_hash"] = _rehash(edited)
    path.write_text(json.dumps(edited))
    with pytest.raises(ValueError, match="does not match the split index"):
        load_frozen_shape_splits(tmp_path)

    # 3. Injecting a whole extra manifest is rejected as unlisted.
    path.write_text(original)
    (tmp_path / "gen_add_bf16.json").write_text(
        json.dumps(freeze_shape_split(get_task("gen_add_bf16")).to_dict()))
    with pytest.raises(ValueError, match="absent from the split index"):
        load_frozen_shape_splits(tmp_path)

    # 4. Deleting a manifest the receipt lists is rejected too.
    (tmp_path / "gen_add_bf16.json").unlink()
    (tmp_path / manifest_filename("gemm_bf16")).unlink()
    with pytest.raises(ValueError, match="missing frozen shape manifests"):
        load_frozen_shape_splits(tmp_path)

    # 5. And the receipt itself cannot be re-written by hand.
    tampered = index.to_dict()
    tampered["entries"][0]["content_hash"] = "0" * 64
    (tmp_path / SPLIT_INDEX_FILENAME).write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="index content hash mismatch"):
        load_shape_split_index(tmp_path)


def test_directory_without_a_receipt_can_be_required(tmp_path):
    freeze_shape_split(get_task("softmax_bf16")).write(
        tmp_path / "softmax_bf16.json")
    assert load_shape_split_index(tmp_path) is None
    assert sorted(load_frozen_shape_splits(tmp_path)) == ["softmax_bf16"]
    with pytest.raises(ValueError, match="was not written by freeze_shape_splits"):
        load_frozen_shape_splits(tmp_path, require_index=True)
    with pytest.raises(ValueError, match="does not exist"):
        load_frozen_shape_splits(tmp_path / "absent")


# --------------------------------------------------------------------------- #
# The hidden lane never overlaps the train lane
# --------------------------------------------------------------------------- #
def test_hidden_lane_is_disjoint_from_prompt_and_train_lanes(tmp_path):
    tasks = _tasks()
    freeze_shape_splits(tasks, tmp_path, seed=4)
    manifests = load_frozen_shape_splits(tmp_path, tasks=tasks)
    for task in tasks:
        manifest = manifests[task.task_id]
        hidden = held_out_shapes(task, max_shapes=64, frozen_split=manifest)
        keys = {shape_key(shape) for shape in hidden}
        assert hidden, task.task_id
        assert len(keys) == len(hidden), task.task_id
        assert keys.isdisjoint(manifest.train_keys), task.task_id
        assert keys.isdisjoint(manifest.prompt_keys), task.task_id
        # The declared shapes are the ones the model is prompted with, so they
        # must be inside the train lane and outside the hidden lane.
        declared = {shape_key(shape) for shape in task.shapes}
        assert declared <= manifest.train_keys, task.task_id
        assert declared.isdisjoint(keys), task.task_id


def test_hidden_shapes_cannot_be_rederived_from_a_foreign_manifest(tmp_path):
    tasks = _tasks("softmax_bf16", "gemm_bf16")
    freeze_shape_splits(tasks, tmp_path)
    manifests = load_frozen_shape_splits(tmp_path)
    with pytest.raises(ValueError, match="belongs to another task"):
        held_out_shapes(
            get_task("softmax_bf16"), frozen_split=manifests["gemm_bf16"])


# --------------------------------------------------------------------------- #
# Certification fails closed without a manifest
# --------------------------------------------------------------------------- #
def test_certification_fails_closed_without_a_manifest():
    task = get_task("softmax_bf16")
    with pytest.raises(ValueError, match="training-time frozen"):
        held_out_shapes(task)

    champ = Champion(task_id="softmax_bf16", source="def kernel(): ...",
                     claimed_speedup=2.0)
    verdict = reeval_champion(champ)
    assert not verdict.certified
    assert verdict.reason == "training-time frozen shape manifest is required"
    assert verdict.n_heldout_shapes == 0

    report = run_champion_reeval([champ])
    assert report.n_champions == 1 and report.n_certified == 0
    assert not report.verdicts[0].certified


def test_certification_fails_closed_when_the_manifest_is_stale(tmp_path):
    task = get_task("softmax_bf16")
    stale = freeze_shape_split(task, code_identity="a-previous-commit")
    stale.write(tmp_path / manifest_filename(task.task_id))
    manifests = load_frozen_shape_splits(tmp_path)

    report = run_champion_reeval(
        [Champion(task_id=task.task_id, source="def kernel(): ...")],
        shape_manifests=manifests)
    assert report.n_certified == 0
    assert "code identity digest changed" in report.verdicts[0].reason


# --------------------------------------------------------------------------- #
# Certification compute is charged to the budget ledger
# --------------------------------------------------------------------------- #
PROFILER_SECONDS = 0.005
EVALUATION_SECONDS = 0.05


@pytest.fixture
def stub_env(monkeypatch):
    """Replace the GPU environment with one that spends measurable time.

    The stub verifies and benchmarks (it returns per-shape timings) and reports
    how much of the interval its profiler passes took, which is exactly the
    evidence the real ``KoreEnv`` has at the same point.
    """
    from kore.reward.reward import Observation

    observation = Observation(
        compiled=True,
        dtype="bf16",
        validation_passed=True,
        timing_requested=True,
        wall_by_shape={"primary__hidden_small_outer_rows": 1.0},
        baseline_by_shape={"primary__hidden_small_outer_rows": 2.0},
        snr_by_shape={"primary__hidden_small_outer_rows": 60.0},
    )

    class _StubEnv:
        instances: list["_StubEnv"] = []

        def __init__(self, task, config=None, use_replay=True, **kwargs):
            self.task = task
            self.use_replay = use_replay
            self.last_profiler_seconds = 0.0
            self.shapes: list = []
            _StubEnv.instances.append(self)

        def evaluate(self, task, source, shapes=None, do_bench=True):
            self.shapes = list(shapes or ())
            time.sleep(EVALUATION_SECONDS)
            self.last_profiler_seconds = PROFILER_SECONDS
            return observation

    monkeypatch.setattr("kore.env.kore_env.KoreEnv", _StubEnv)
    return _StubEnv


def _manifests(tmp_path, *task_ids):
    tasks = _tasks(*task_ids)
    freeze_shape_splits(tasks, tmp_path)
    return load_frozen_shape_splits(tmp_path, tasks=tasks)


def test_certification_charges_the_evaluation_counters(tmp_path, stub_env):
    manifests = _manifests(tmp_path, "softmax_bf16")
    ledger = BudgetLedgerV1()
    reeval_champion(
        Champion(task_id="softmax_bf16", source="def kernel(): ..."),
        shape_manifest=manifests["softmax_bf16"], ledger=ledger)

    state = ledger.to_dict()
    assert state["correctness_calls"] == 1
    assert state["fresh_timed_calls"] == 1
    assert state["profiler_gpu_seconds"] == pytest.approx(PROFILER_SECONDS)
    # Measured wall time, with the profiler's share attributed to the profiler.
    assert state["verifier_gpu_seconds"] >= EVALUATION_SECONDS - PROFILER_SECONDS
    assert state["verifier_gpu_seconds"] < EVALUATION_SECONDS
    # Certification never replays, and the counters stay independent.
    assert state["replay_hits"] == 0
    assert state["generated_tokens"] == 0
    assert state["optimizer_tokens"] == 0
    assert state["groups_attempted"] == 0

    # A second champion accumulates rather than overwriting.
    reeval_champion(
        Champion(task_id="softmax_bf16", source="def kernel2(): ..."),
        shape_manifest=manifests["softmax_bf16"], ledger=ledger)
    assert ledger.correctness_calls == 2
    assert ledger.fresh_timed_calls == 2
    assert ledger.profiler_gpu_seconds == pytest.approx(2 * PROFILER_SECONDS)


@pytest.mark.parametrize(
    "counter",
    ["correctness_calls", "fresh_timed_calls", "verifier_gpu_seconds",
     "profiler_gpu_seconds"],
)
def test_a_zero_limit_stops_certification_before_it_spends(
        tmp_path, stub_env, counter):
    manifests = _manifests(tmp_path, "softmax_bf16")
    ledger = BudgetLedgerV1(limits={counter: 0})
    champ = Champion(task_id="softmax_bf16", source="def kernel(): ...")

    if counter in ("correctness_calls", "fresh_timed_calls"):
        # These are pre-flighted, so no evaluation runs at all.
        with pytest.raises(BudgetExceededError, match=counter):
            reeval_champion(champ, shape_manifest=manifests["softmax_bf16"],
                            ledger=ledger)
        assert getattr(ledger, counter) == 0
    else:
        with pytest.raises(BudgetExceededError, match=counter):
            reeval_champion(champ, shape_manifest=manifests["softmax_bf16"],
                            ledger=ledger)

    # A breach inside the gate is a rejection, never a silent certification.
    report = run_champion_reeval([champ], shape_manifests=manifests,
                                ledger=BudgetLedgerV1(limits={counter: 0}))
    assert report.n_certified == 0
    assert "BudgetExceededError" in report.verdicts[0].reason


def test_replay_hits_are_charged_without_any_gpu_work():
    from kore.reward.reward import Observation

    ledger = BudgetLedgerV1(limits={"replay_hits": 1})
    cached = Observation(compiled=True, validation_passed=True,
                         timing_requested=True, wall_by_shape={"primary": 1.0})
    work = EvaluationWork.from_observation(
        cached, verifier_seconds=0.002, replayed=True)
    assert work.to_dict() == {
        "correctness_calls": 0,
        "fresh_timed_calls": 0,
        "replay_hits": 1,
        "verifier_gpu_seconds": 0.0,
        "profiler_gpu_seconds": 0.0,
    }
    ledger.record_evaluation_work(work)
    assert ledger.replay_hits == 1
    with pytest.raises(BudgetExceededError, match="replay_hits"):
        ledger.record_evaluation_work(work)


def test_index_schema_is_versioned(tmp_path):
    index = freeze_shape_splits(_tasks("softmax_bf16"), tmp_path)
    value = index.to_dict()
    assert value["schema_version"] == 1
    assert value["manifest_schema_version"] == 1
    assert ShapeSplitIndex.from_dict(value) == index
    value["schema_version"] = 99
    value["content_hash"] = _rehash(value)
    with pytest.raises(ValueError, match="unsupported frozen shape split index"):
        ShapeSplitIndex.from_dict(value)
