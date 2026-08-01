"""Task-integrity gates: pretraining contamination and on-hardware seed verdicts.

Two measured defects are pinned here.

1. ``data/release/generators/gen_curriculum.py`` built its Tier-4 "reasoning
   trace" prompts from every win shard under ``data/b05factory/wins/`` with no
   held-out filter, quoting optimized kernel source for 11 held-out tasks into the
   midtrain corpus (17 curriculum chunks, containment 0.795-0.943 against
   ``kore-task:<id>:seed_triton.py``).  Those 11 must stay held out AND be struck
   from any zero-shot generalization claim -- two independent states.
2. ``data/gfx950_task_verification.json`` recorded 100 breadth tasks whose own
   seed fails its declared SNR gate on real gfx950, and nothing consumed it.

The counts below are pinned deliberately: they are measurements, so a change to
the artifact must be an explicit edit here rather than a silent re-baseline.  An
earlier sweep read 937/111/4 with 27 in the ``broken`` band; 12 of those were an
``AttributeError`` on ``tl.math.tanh`` (removed in Triton 3.6) rather than a task
defect, and replacing it recovered 11 tasks with no regressions.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kore.tasks import registry, taxonomy, verification

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = REPO_ROOT / "data" / "release" / "generators" / "gen_curriculum.py"

# The exact tasks whose optimized source reached the midtrain corpus.
CONTAMINATED = (
    "genb_cv_conv2d_1x1_s2_fp16",
    "genb_cv_depthwise_conv2d_5x5_s1_bf16",
    "genb_norm_rmsnorm_h16384_bf16",
    "genb_qx_int4_unpack_group_bf16",
    "genb_qx_quant_fp8_block2d_fp8",
    "genb_red_log_softmax_bwd_fp32",
    "genb_red_rms_bf16",
    "genb_smp_repetition_penalty_bf16",
    "genb_ssm_gated_retention_c128_bf16",
    "genb_ssm_lightning_attn_bf16",
    "genb_tr_rmsprop_centered_momentum_fp32",
)

# Recorded on real gfx950 over 1,052 breadth tasks, banded by SNR against each
# task's own declared gate.
EXPECTED_STATUS_COUNTS = {"PASS": 948, "FAIL_CORRECTNESS": 100, "INFRA": 4}
EXPECTED_BAND_COUNTS = {
    "pass": 948,
    "near_gate": 74,
    "shortfall": 11,
    "broken": 15,
    "infra": 4,
}


@pytest.fixture(scope="module")
def curriculum():
    """Import the Tier-4 generator by path (``data/`` is not an import package)."""
    spec = importlib.util.spec_from_file_location("kore_gen_curriculum", GENERATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def _win_row(task_id: str, **extra) -> dict:
    row = {
        "task_id": task_id,
        "operation": task_id.replace("genb_", ""),
        "speedup": 1.9,
        "snr_db": 84.0,
        "final_source": "import triton\n\n@triton.jit\ndef k(x_ptr):\n    pass\n",
    }
    row.update(extra)
    return row


def _shard(directory: Path, task_id: str, rows=None) -> Path:
    path = directory / f"{task_id}.jsonl"
    payload = rows if rows is not None else [_win_row(task_id)]
    path.write_text("".join(json.dumps(row) + "\n" for row in payload))
    return path


# --------------------------------------------------------------------------- #
# Defect 1: the Tier-4 generator must refuse held-out win source
# --------------------------------------------------------------------------- #
def test_tier4_generator_refuses_a_heldout_win_shard(curriculum, tmp_path):
    """Reproduces the leak: a held-out shard must be refused, not quoted.

    The old code globbed the wins directory and passed every row into ``tier4``,
    which pastes ``final_source`` verbatim into a training prompt.
    """
    heldout = sorted(task.task_id for task in registry.heldout_tasks())[0]
    shard = _shard(tmp_path, heldout)

    with pytest.raises(curriculum.HeldoutWinLeak, match=heldout):
        curriculum.load_win_shard(shard)
    with pytest.raises(curriculum.HeldoutWinLeak, match="eval-only"):
        curriculum.assert_trainable_shard(shard)

    # ... and the tier itself refuses the rows even if a loader is bypassed.
    with pytest.raises(curriculum.HeldoutWinLeak, match=heldout):
        curriculum.tier4([_win_row(heldout)], 4)


def test_tier4_generator_refuses_every_contaminated_task_shard(curriculum, tmp_path):
    for task_id in CONTAMINATED:
        with pytest.raises(curriculum.HeldoutWinLeak):
            curriculum.load_win_shard(_shard(tmp_path, task_id))


def test_win_glob_excludes_heldout_shards_and_keeps_trainable_ones(curriculum, tmp_path):
    trainable = sorted(task.task_id for task in registry.train_tasks())[0]
    _shard(tmp_path, trainable)
    for task_id in CONTAMINATED:
        _shard(tmp_path, task_id)

    rows = curriculum.load_wins(str(tmp_path / "*.jsonl"))

    assert [row["task_id"] for row in rows] == [trainable]
    specs = curriculum.tier4(rows, 3, seed=0)
    assert len(specs) == 3
    assert all(trainable.replace("genb_", "") in spec[1] for spec in specs)


def test_heldout_record_inside_a_trainable_shard_is_fatal(curriculum, tmp_path):
    """A shard whose filename lies must not launder a held-out win through it."""
    trainable = sorted(task.task_id for task in registry.train_tasks())[0]
    leaked = CONTAMINATED[0]
    _shard(tmp_path, trainable, rows=[_win_row(trainable), _win_row(leaked)])

    with pytest.raises(curriculum.HeldoutWinLeak, match=leaked):
        curriculum.load_wins(str(tmp_path / "*.jsonl"))


def test_lineage_root_of_a_heldout_task_is_also_refused(curriculum, tmp_path):
    """A derived win keeps the held-out lineage even under a trainable task id."""
    trainable = sorted(task.task_id for task in registry.train_tasks())[0]
    row = _win_row(trainable, provenance_root=CONTAMINATED[1])

    with pytest.raises(curriculum.HeldoutWinLeak, match=CONTAMINATED[1]):
        curriculum.assert_trainable_rows([row], "test")
    with pytest.raises(curriculum.HeldoutWinLeak):
        curriculum.tier4([row], 2)


def test_generator_holdout_index_never_degrades_to_an_empty_filter(
    curriculum, monkeypatch
):
    monkeypatch.setattr(registry, "heldout_tasks", lambda: [])
    with pytest.raises(curriculum.HeldoutWinLeak, match="empty held-out set"):
        curriculum.heldout_ids()


# --------------------------------------------------------------------------- #
# Defect 1: contaminated tasks stay held out AND leave the generalization claim
# --------------------------------------------------------------------------- #
def test_the_eleven_leaked_tasks_are_marked_contaminated_and_excluded():
    assert set(taxonomy.CONTAMINATED_TASK_IDS) == set(CONTAMINATED)
    by_id = {task.task_id: task for task in registry.all_tasks()}

    for task_id in CONTAMINATED:
        assert task_id in by_id, f"{task_id} left the registry with its exclusion behind"
        decision = registry.split_decision(by_id[task_id])
        # Held out (never trainable) and contaminated (not zero-shot) are separate.
        assert decision.heldout
        assert decision.contaminated
        assert not decision.generalization_eligible
        assert decision.contamination_reason == (
            taxonomy.CONTAMINATION_MIDTRAIN_CURRICULUM
        )
        record = decision.contamination
        assert record is not None
        assert record.reference_id == f"kore-task:{task_id}:seed_triton.py"
        assert record.detector == "kore.data.decontam.analyze_text_contamination"
        assert record.match_kind == "directional_containment"

    scored = {task.task_id for task in registry.generalization_tasks()}
    assert scored.isdisjoint(CONTAMINATED)
    assert len(scored) == 34
    assert scored | set(CONTAMINATED) == {
        task.task_id for task in registry.heldout_tasks()
    }


def test_contaminated_tasks_remain_permanently_held_out_of_training():
    train_ids = {task.task_id for task in registry.train_tasks()}
    heldout_ids = {task.task_id for task in registry.heldout_tasks()}

    assert set(CONTAMINATED) <= heldout_ids
    assert train_ids.isdisjoint(CONTAMINATED)
    # Excluding them from the CLAIM must not shrink the reservation itself: the
    # decontamination gate reads this set to keep the source out of training.
    assert len(heldout_ids) == 45
    assert len(registry.eligible_train_tasks(verification.ADMIT_ALL_POLICY)) == len(
        train_ids
    )


def test_a_contaminated_identity_can_never_be_classified_trainable(monkeypatch):
    """Second guard: contamination alone forces eval, without the near-probe list."""
    task_id = CONTAMINATED[0]
    monkeypatch.setattr(
        taxonomy,
        "NEAR_GENERALIZATION_TASK_IDS",
        frozenset(taxonomy.NEAR_GENERALIZATION_TASK_IDS - {task_id}),
    )
    decision = taxonomy.split_decision_for_identity(
        task_id=task_id,
        operation="cv_conv2d_1x1_s2",
        product_family="convolution",
        architecture="gfx950",
        dtype="fp16",
    )
    assert decision.split == "eval"
    assert decision.reason == taxonomy.CONTAMINATED_SPLIT_REASON
    assert decision.contaminated and not decision.generalization_eligible


def test_contaminated_lineage_descendants_inherit_the_exclusion():
    decision = taxonomy.split_decision_for_identity(
        task_id="genb_red_rms_bf16_minted_v2",
        operation="red_rms",
        product_family="reduction",
        architecture="gfx950",
        dtype="bf16",
        provenance_root="genb_red_rms_bf16",
    )
    assert decision.heldout and decision.contaminated
    assert not decision.generalization_eligible


def test_generalization_claim_fails_closed_on_a_contaminated_scope():
    heldout_ids = [task.task_id for task in registry.heldout_tasks()]

    with pytest.raises(registry.ContaminatedGeneralizationError) as excinfo:
        registry.assert_generalization_scope(heldout_ids)
    message = str(excinfo.value)
    assert all(task_id in message for task_id in CONTAMINATED)
    assert taxonomy.CONTAMINATION_MIDTRAIN_CURRICULUM in message

    # The clean scope is accepted, so the gate is not vacuously strict.
    assert registry.assert_generalization_scope(registry.generalization_eval_ids())

    kept, dropped = registry.filter_generalization_scope(heldout_ids)
    assert set(kept) == set(registry.generalization_eval_ids())
    assert dict(dropped) == {
        task_id: taxonomy.CONTAMINATION_MIDTRAIN_CURRICULUM for task_id in CONTAMINATED
    }


def test_an_unregistered_task_cannot_enter_a_generalization_claim():
    assert dict(registry.generalization_exclusions(["not_a_registered_task"])) == {
        "not_a_registered_task": "unregistered_task"
    }
    with pytest.raises(registry.ContaminatedGeneralizationError):
        registry.assert_generalization_scope(["not_a_registered_task"])
    with pytest.raises(registry.ContaminatedGeneralizationError, match="empty task id"):
        registry.assert_generalization_scope([""])


def test_generalization_claim_report_records_the_reason_and_evidence():
    report = registry.generalization_claim_report()

    assert report["requested"] == 45
    assert report["scoreable"] == 34
    assert report["excluded"] == 11
    assert sorted(report["excluded_task_ids"]) == sorted(CONTAMINATED)
    evidence = report["contamination_evidence"]
    assert evidence["hits"] == 17
    assert evidence["tasks"] == 11
    assert evidence["chunks_analyzed"] == 9956
    assert evidence["public_benchmark_hits"] == 0
    assert 0.79 < evidence["containment_min"] < evidence["containment_max"] < 0.95
    assert report["taxonomy_digest"] == registry.taxonomy_digest()


def test_split_manifest_serializes_the_contamination_exclusion():
    manifest = registry.build_split_manifest()

    assert sorted(manifest.contaminated_eval_ids) == sorted(CONTAMINATED)
    assert set(manifest.generalization_eval_ids) == set(
        registry.generalization_eval_ids()
    )
    policy = manifest.as_dict()["policy"]
    assert sorted(policy["contaminated_task_ids"]) == sorted(CONTAMINATED)
    assert len(policy["generalization_eval_ids"]) == 34
    # A manifest authored before the leak was found carries no exclusion, so it
    # must be rejected rather than silently scored over all 45 eval tasks.
    stale = manifest.as_dict()
    stale["policy"] = {
        key: value
        for key, value in stale["policy"].items()
        if not key.startswith("contaminated") and key != "generalization_eval_ids"
    }
    with pytest.raises(registry.StaleSplitManifestError, match="policy metadata"):
        registry.validate_split_manifest(stale)


def test_a_contaminated_task_cannot_be_placed_in_a_manifest_train_split():
    contaminated = registry.get_task(CONTAMINATED[0])
    train = registry.train_tasks()[:4] + [contaminated]
    # The eval side deliberately omits it, so the failure is about the train
    # placement itself rather than a train/eval collision.
    with pytest.raises(registry.SplitManifestError, match="contaminated tasks placed"):
        registry.build_split_manifest(train, registry.generalization_tasks())


# --------------------------------------------------------------------------- #
# Defect 2: the hardware verification loader
# --------------------------------------------------------------------------- #
def test_committed_artifact_matches_its_own_recorded_summary():
    report = verification.report()

    assert report.status_counts() == EXPECTED_STATUS_COUNTS
    assert dict(report.summary["counts"]) == EXPECTED_STATUS_COUNTS
    assert len(report.verdicts) == 1_052
    assert report.architecture == "gfx950"
    assert report.digest == verification.load_verification().digest
    # Every measured task is a real registry task, so no verdict is unenforceable.
    assert set(report.verdicts) <= set(registry.task_ids())


def test_failures_split_into_broken_near_gate_and_shortfall():
    report = verification.report()

    assert report.band_counts() == EXPECTED_BAND_COUNTS

    broken = report.task_ids_in_band(verification.BAND_BROKEN)
    assert len(broken) == 15
    assert sum(1 for task_id in broken if task_id.startswith("genb_attn2_window")) == 8
    assert all(
        verification.verdict_for(task_id).snr_db <= verification.BROKEN_SNR_DB
        for task_id in broken
    )

    # 73 of the 74 near-gate failures actually CLEARED their declared SNR gate and
    # were rejected only by the elementwise allclose tolerance; 1 sits 0.33 dB low.
    near = report.task_ids_in_band(verification.BAND_NEAR_GATE)
    verdicts = [verification.verdict_for(task_id) for task_id in near]
    assert sum(1 for verdict in verdicts if verdict.clears_declared_gate) == 73
    assert all(verdict.margin_db >= -verification.NEAR_GATE_MARGIN_DB for verdict in verdicts)

    shortfall = report.task_ids_in_band(verification.BAND_SHORTFALL)
    assert len(shortfall) == 11
    margins = [verification.verdict_for(task_id).margin_db for task_id in shortfall]
    assert max(margins) < -verification.NEAR_GATE_MARGIN_DB
    assert all(
        0.0 <= verification.verdict_for(task_id).snr_db < 25.0 for task_id in shortfall
    )


def test_pass_fail_infra_and_missing_are_four_distinct_states():
    passing = verification.verdict_for("genb_adaptive_avgpool2d_bf16")
    failing = verification.verdict_for("genb_attn2_window256_mha_causal_bf16")
    infra = verification.verdict_for("genb_moe_block_silu_k8_e256_bf16")
    missing = verification.verdict_for("genb_definitely_not_a_task_bf16")

    assert (passing.status, passing.band) == ("PASS", "pass")
    assert passing.is_pass and passing.is_known

    assert (failing.status, failing.band) == ("FAIL_CORRECTNESS", "broken")
    assert failing.is_correctness_failure and not failing.is_pass

    # INFRA is a harness OOM, never a task defect and never a correctness verdict.
    assert (infra.status, infra.band) == ("INFRA", "infra")
    assert infra.is_infra
    assert not infra.is_pass and not infra.is_correctness_failure

    # Missing is its own value: not PASS, not a failure, and explicitly unknown.
    assert (missing.status, missing.band) == ("UNKNOWN", "unknown")
    assert not missing.is_pass
    assert not missing.is_known
    assert not missing.is_correctness_failure
    assert missing.margin_db is None and not missing.clears_declared_gate

    assert "genb_definitely_not_a_task_bf16" not in verification.hardware_pass_ids()
    assert len(verification.hardware_pass_ids()) == 948
    assert len(verification.hardware_failure_ids()) == 100
    assert verification.hardware_pass_ids().isdisjoint(
        verification.hardware_failure_ids()
    )


def test_missing_or_malformed_evidence_raises_instead_of_reading_as_pass(tmp_path):
    with pytest.raises(verification.VerificationError, match="unreadable"):
        verification.load_verification(tmp_path / "absent.json")

    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(verification.VerificationError, match="valid JSON"):
        verification.load_verification(tmp_path / "bad.json")

    (tmp_path / "empty.json").write_text(json.dumps({"results": []}))
    with pytest.raises(verification.VerificationError, match="no results"):
        verification.load_verification(tmp_path / "empty.json")

    unknown_status = {"results": [{"task": "t", "status": "MOSTLY_FINE", "snr_db": 99.0}]}
    (tmp_path / "status.json").write_text(json.dumps(unknown_status))
    with pytest.raises(verification.VerificationError, match="status"):
        verification.load_verification(tmp_path / "status.json")

    no_id = {"results": [{"status": "PASS", "snr_db": 99.0}]}
    (tmp_path / "noid.json").write_text(json.dumps(no_id))
    with pytest.raises(verification.VerificationError, match="no task id"):
        verification.load_verification(tmp_path / "noid.json")

    # A correctness failure with no declared gate cannot be banded; guessing would
    # silently turn wrong math into an admissible near-miss.
    no_gate = {"results": [{"task": "t", "status": "FAIL_CORRECTNESS", "snr_db": 12.0}]}
    (tmp_path / "nogate.json").write_text(json.dumps(no_gate))
    with pytest.raises(verification.VerificationError, match="no threshold"):
        verification.load_verification(tmp_path / "nogate.json")

    conflict = {
        "results": [
            {"task": "t", "status": "PASS", "snr_db": 99.0, "threshold": 30.0},
            {"task": "t", "status": "FAIL_CORRECTNESS", "snr_db": -999.0, "threshold": 30.0},
        ]
    }
    (tmp_path / "conflict.json").write_text(json.dumps(conflict))
    with pytest.raises(verification.VerificationError, match="recorded twice"):
        verification.load_verification(tmp_path / "conflict.json")


def test_verdict_lookup_rejects_an_empty_identity():
    with pytest.raises(verification.VerificationError):
        verification.verdict_for("")
    with pytest.raises(verification.VerificationError):
        verification.unknown_verdict("  ")


# --------------------------------------------------------------------------- #
# Defect 2: the eligibility policy
# --------------------------------------------------------------------------- #
def test_default_policy_excludes_only_broken_and_shortfall_seeds():
    policy = verification.DEFAULT_ELIGIBILITY_POLICY
    assert policy.exclude_broken and policy.exclude_shortfall
    assert not policy.exclude_near_gate
    assert not policy.exclude_infra
    assert not policy.require_verdict

    excluded = registry.train_eligibility_exclusions()
    assert len(excluded) == 24
    reasons = sorted(set(excluded.values()))
    assert reasons == [verification.EXCLUSION_BROKEN, verification.EXCLUSION_SHORTFALL]
    assert sum(
        1 for reason in excluded.values() if reason == verification.EXCLUSION_BROKEN
    ) == 15
    # 9, not 11: two shortfall tasks are held out, so they never reach the train split.
    assert sum(
        1 for reason in excluded.values() if reason == verification.EXCLUSION_SHORTFALL
    ) == 9

    eligible = registry.eligible_train_tasks()
    assert len(eligible) == 1_265
    assert {task.task_id for task in eligible}.isdisjoint(excluded)


def test_the_train_split_itself_is_untouched_by_the_policy():
    """Eligibility is an opt-in view, so "train task" keeps its registry meaning."""
    train_ids = {task.task_id for task in registry.train_tasks()}

    assert len(train_ids) == 1_289
    assert len(registry.build_split_manifest().train_ids) == 1_289
    assert {task.task_id for task in registry.eligible_train_tasks()} < train_ids
    assert {
        task.task_id
        for task in registry.eligible_train_tasks(verification.ADMIT_ALL_POLICY)
    } == train_ids


def test_near_gate_and_infra_tasks_stay_eligible_by_default():
    # SNR 57.93 dB against a 30 dB gate, rejected only by allclose.
    near = registry.hardware_verdict("genb_attn_bwd_gqa_causal_bf16")
    assert near.band == verification.BAND_NEAR_GATE and near.clears_declared_gate
    decision = verification.eligibility(near.task_id)
    assert decision.eligible and decision.reason == verification.ADMITTED_BAND

    infra = verification.eligibility("genb_moe_fused_moe_silu_k8_e256_bf16")
    assert infra.eligible
    assert infra.reason == verification.ADMITTED_INFRA


def test_unknown_verdicts_are_admitted_but_never_reported_as_verified():
    unmeasured = sorted(
        task.task_id
        for task in registry.train_tasks()
        if not registry.hardware_verdict(task.task_id).is_known
    )
    assert len(unmeasured) == 280
    assert any(task_id.startswith("gen_") for task_id in unmeasured)
    assert any(task_id.startswith("genv_") for task_id in unmeasured)

    for task_id in unmeasured[:5]:
        decision = verification.eligibility(task_id)
        assert decision.eligible
        assert decision.reason == verification.ADMITTED_UNVERIFIED
        assert not decision.verdict.is_pass

    coverage = registry.hardware_verification_coverage()
    assert coverage["tasks"] == 1_334
    assert coverage["status_counts"]["UNKNOWN"] == 1_334 - 1_052
    assert coverage["measured"] == 1_052
    assert coverage["artifact_digest"] == verification.report().digest


def test_strict_policy_admits_only_recorded_passes():
    strict = verification.STRICT_HARDWARE_VERIFIED_POLICY
    eligible = {task.task_id for task in registry.eligible_train_tasks(strict)}

    assert eligible == verification.hardware_pass_ids() & {
        task.task_id for task in registry.train_tasks()
    }
    assert len(eligible) == 912

    excluded = registry.train_eligibility_exclusions(strict)
    assert (
        excluded["genb_moe_fused_moe_silu_k8_e256_bf16"] == verification.EXCLUSION_INFRA
    )
    assert (
        excluded["genb_attn_bwd_gqa_causal_bf16"] == verification.EXCLUSION_NEAR_GATE
    )
    unmeasured = next(
        task.task_id
        for task in registry.train_tasks()
        if not registry.hardware_verdict(task.task_id).is_known
    )
    assert excluded[unmeasured] == verification.EXCLUSION_UNVERIFIED


def test_policies_are_named_resolvable_and_never_silently_default_to_admit_all():
    assert verification.resolve_policy() is verification.DEFAULT_ELIGIBILITY_POLICY
    assert verification.resolve_policy("admit_all") is verification.ADMIT_ALL_POLICY
    assert (
        verification.resolve_policy("strict_hardware_verified")
        is verification.STRICT_HARDWARE_VERIFIED_POLICY
    )
    assert verification.DEFAULT_ELIGIBILITY_POLICY is not verification.ADMIT_ALL_POLICY

    with pytest.raises(verification.VerificationError, match="unknown eligibility"):
        verification.resolve_policy("lenient")
    with pytest.raises(verification.VerificationError, match="HardwareVerdict"):
        verification.DEFAULT_ELIGIBILITY_POLICY.exclusion_reason(
            SimpleNamespace(status="PASS")  # type: ignore[arg-type]
        )

    report = registry.hardware_eligibility_report()
    assert report["policy"]["name"] == "exclude_broken_and_shortfall"
    assert report["policy"]["near_gate_margin_db"] == verification.NEAR_GATE_MARGIN_DB
    assert report["train_tasks"] == 1_289
    assert report["eligible"] == 1_265
    assert sorted(report["excluded_by_reason"]) == [
        verification.EXCLUSION_BROKEN,
        verification.EXCLUSION_SHORTFALL,
    ]


def test_band_classification_refuses_to_invent_a_verdict():
    assert (
        verification.classify_band("PASS", 999.0, 30.0) == verification.BAND_PASS
    )
    assert (
        verification.classify_band("FAIL_CORRECTNESS", 30.0, 30.0)
        == verification.BAND_NEAR_GATE
    )
    assert (
        verification.classify_band("FAIL_CORRECTNESS", 25.0, 30.0)
        == verification.BAND_NEAR_GATE
    )
    assert (
        verification.classify_band("FAIL_CORRECTNESS", 24.9, 30.0)
        == verification.BAND_SHORTFALL
    )
    assert (
        verification.classify_band("FAIL_CORRECTNESS", -999.0, 30.0)
        == verification.BAND_BROKEN
    )
    assert (
        verification.classify_band("FAIL_CORRECTNESS", None, 30.0)
        == verification.BAND_BROKEN
    )
    assert (
        verification.classify_band("FAIL_CORRECTNESS", float("nan"), 30.0)
        == verification.BAND_BROKEN
    )
    with pytest.raises(verification.VerificationError):
        verification.classify_band("SORT_OF_PASSED", 99.0, 30.0)
