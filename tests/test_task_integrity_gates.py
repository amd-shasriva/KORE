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

Defect 2 is now CLOSED: the artifact reads 1,052/1,052 ``PASS``.  73 of the 100
were correct kernels rejected by an fp32-calibrated ``atol = rtol = 1e-2``
applied to bf16/fp16/fp8/int8 outputs, 31 were real seed or oracle defects fixed
in the generators, and the 4 ``INFRA`` were a MoE shape that needed ~360 GiB of
peak on a 252 GiB device.  See ``kore/tasks/README.md`` for the breakdown.

The counts below are pinned deliberately: they are measurements, so a change to
the artifact must be an explicit edit here rather than a silent re-baseline.
Because the corpus is clean, the tests that need a FAILING verdict build a
synthetic artifact rather than reaching for a real task -- the banding and
eligibility machinery has to keep working for the sweep that finds the next
regression, and it cannot be exercised by a corpus that has none.
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
EXPECTED_STATUS_COUNTS = {"PASS": 1_052, "FAIL_CORRECTNESS": 0, "INFRA": 0}
EXPECTED_BAND_COUNTS = {
    "pass": 1_052,
    "near_gate": 0,
    "shortfall": 0,
    "broken": 0,
    "infra": 0,
}
# Every train task the sweep measured now clears its own gate, so the default
# eligibility policy removes nobody.  Pinned so a regression cannot arrive quietly.
# +24 for the spec-synthesis family. Like the HIP family these are UNMEASURED by
# the genb_ breadth sweep, and for these the sweep could not apply even in
# principle: it verifies a task by running its declared seed, and a spec task's
# seed is a signature stub that is SUPPOSED to fail. They are proven instead by
# scripts/verify_spec_tasks_e2e.py, which requires a real solution to pass the
# same oracle AND the stub to fail.
EXPECTED_TRAIN_TASKS = 1_501
EXPECTED_ELIGIBLE_TRAIN_TASKS = 1_501
EXPECTED_STRICT_ELIGIBLE_TRAIN_TASKS = 1_009
# +188 for the HIP C++ family, none of which the gfx950 breadth sweep measured
# (it ran with the genb_ prefix), so all 188 land in UNMEASURED rather than PASS.
# UNMEASURED is not UNPROVEN here: every HIP task is proven end-to-end by
# scripts/verify_hip_tasks_e2e.py, whose evidence is data/hip_task_verification.json.
# The two artifacts are separate because they measure different things -- the
# breadth sweep bands Triton seeds by SNR, and the HIP run proves runnability
# through the whole environment including the timing-admission gate.
#
# +24 for the spec-synthesis family, which the breadth sweep could not have
# measured even if it had run: the sweep verifies a task by executing its
# declared seed, and a spec task's seed is a stub whose failure is the point.
# Their evidence is data/spec_task_verification.json.
EXPECTED_UNMEASURED_TRAIN_TASKS = 492


def _synthetic_artifact(path):
    """An artifact holding one verdict of every recorded kind.

    The committed sweep is clean, so the banding and exclusion paths have no real
    input to exercise.  They still have to work, so they are tested here against
    verdicts built by hand.
    """
    payload = {
        "summary": {"prefix": "synthetic", "total": 5},
        "results": [
            {"task": "syn_pass", "status": "PASS", "snr_db": 88.0, "threshold": 30.0},
            {"task": "syn_near", "status": "FAIL_CORRECTNESS", "snr_db": 57.9,
             "threshold": 30.0},
            {"task": "syn_shortfall", "status": "FAIL_CORRECTNESS", "snr_db": 4.7,
             "threshold": 30.0},
            {"task": "syn_broken", "status": "FAIL_CORRECTNESS", "snr_db": -999.0,
             "threshold": 30.0},
            {"task": "syn_infra", "status": "INFRA", "snr_db": None,
             "threshold": 30.0},
        ],
    }
    path.write_text(json.dumps(payload))
    return verification.load_verification(path)


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
    # The sweep writes only the statuses it actually observed; the loader fills the
    # rest in as zero.  Both views have to agree on every status either mentions.
    recorded = dict(report.summary["counts"])
    assert recorded == {k: v for k, v in EXPECTED_STATUS_COUNTS.items() if v}
    assert sum(recorded.values()) == len(report.verdicts)
    assert len(report.verdicts) == 1_052
    assert report.architecture == "gfx950"
    assert report.digest == verification.load_verification().digest
    # Every measured task is a real registry task, so no verdict is unenforceable.
    assert set(report.verdicts) <= set(registry.task_ids())


def test_the_committed_corpus_has_no_failing_or_infra_verdict():
    report = verification.report()

    assert report.band_counts() == EXPECTED_BAND_COUNTS
    for band in (verification.BAND_BROKEN, verification.BAND_NEAR_GATE,
                 verification.BAND_SHORTFALL, verification.BAND_INFRA):
        assert report.task_ids_in_band(band) == ()
    # Every recorded seed clears the gate its own task.yaml declares.
    assert all(verdict.is_pass for verdict in report.verdicts.values())
    assert len(verification.hardware_pass_ids()) == EXPECTED_STATUS_COUNTS["PASS"]
    assert verification.hardware_failure_ids() == frozenset()


def test_failures_split_into_broken_near_gate_and_shortfall(tmp_path):
    """The banding still has to work for the sweep that finds the next regression."""
    report = _synthetic_artifact(tmp_path / "synthetic.json")

    assert report.band_counts() == {
        "pass": 1, "near_gate": 1, "shortfall": 1, "broken": 1, "infra": 1
    }
    broken = report.verdicts["syn_broken"]
    assert broken.band == verification.BAND_BROKEN
    assert broken.snr_db <= verification.BROKEN_SNR_DB
    assert broken.margin_db is None and not broken.clears_declared_gate

    # A near-gate seed CLEARED its declared SNR gate and was rejected only by the
    # elementwise check -- a different claim from missing the gate.
    near = report.verdicts["syn_near"]
    assert near.band == verification.BAND_NEAR_GATE
    assert near.clears_declared_gate and near.margin_db > 0.0

    shortfall = report.verdicts["syn_shortfall"]
    assert shortfall.band == verification.BAND_SHORTFALL
    assert shortfall.margin_db < -verification.NEAR_GATE_MARGIN_DB


def test_pass_fail_infra_and_missing_are_four_distinct_states(tmp_path):
    synthetic = _synthetic_artifact(tmp_path / "synthetic.json")
    passing = verification.verdict_for("genb_adaptive_avgpool2d_bf16")
    failing = synthetic.verdicts["syn_broken"]
    infra = synthetic.verdicts["syn_infra"]
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
    assert len(verification.hardware_pass_ids()) == EXPECTED_STATUS_COUNTS["PASS"]
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
def test_default_policy_excludes_only_broken_and_shortfall_seeds(tmp_path):
    policy = verification.DEFAULT_ELIGIBILITY_POLICY
    assert policy.exclude_broken and policy.exclude_shortfall
    assert not policy.exclude_near_gate
    assert not policy.exclude_infra
    assert not policy.require_verdict

    # Both excluded bands are empty on the committed sweep, so the policy removes
    # nobody -- it is a standing rule, not a currently-active filter.
    excluded = registry.train_eligibility_exclusions()
    assert excluded == {}
    eligible = registry.eligible_train_tasks()
    assert len(eligible) == EXPECTED_ELIGIBLE_TRAIN_TASKS

    # What the rule WOULD do, on verdicts that exercise every band.
    synthetic = _synthetic_artifact(tmp_path / "synthetic.json")
    decided = {
        task_id: policy.exclusion_reason(verdict)
        for task_id, verdict in synthetic.verdicts.items()
    }
    assert decided == {
        "syn_pass": None,
        "syn_near": None,          # cleared its gate; only the elementwise check failed
        "syn_infra": None,         # a node fault is not a task defect
        "syn_broken": verification.EXCLUSION_BROKEN,
        "syn_shortfall": verification.EXCLUSION_SHORTFALL,
    }


def test_the_train_split_itself_is_untouched_by_the_policy():
    """Eligibility is an opt-in view, so "train task" keeps its registry meaning."""
    train_ids = {task.task_id for task in registry.train_tasks()}

    assert len(train_ids) == EXPECTED_TRAIN_TASKS
    assert len(registry.build_split_manifest().train_ids) == EXPECTED_TRAIN_TASKS
    # A subset always, and equal only because nothing is currently excluded.
    assert {task.task_id for task in registry.eligible_train_tasks()} <= train_ids
    assert {
        task.task_id
        for task in registry.eligible_train_tasks(verification.ADMIT_ALL_POLICY)
    } == train_ids


def test_near_gate_and_infra_tasks_stay_eligible_by_default(tmp_path):
    synthetic = _synthetic_artifact(tmp_path / "synthetic.json")
    policy = verification.DEFAULT_ELIGIBILITY_POLICY

    # Cleared its declared SNR gate; only the elementwise check rejected it.
    near = synthetic.verdicts["syn_near"]
    assert near.band == verification.BAND_NEAR_GATE and near.clears_declared_gate
    assert policy.admits(near)

    infra = synthetic.verdicts["syn_infra"]
    assert infra.band == verification.BAND_INFRA
    assert policy.admits(infra)

    # And a real PASS reports as verified rather than merely admitted.
    decision = verification.eligibility("genb_attn_bwd_gqa_causal_bf16")
    assert decision.eligible and decision.reason == verification.ADMITTED_VERIFIED


def test_unknown_verdicts_are_admitted_but_never_reported_as_verified():
    unmeasured = sorted(
        task.task_id
        for task in registry.train_tasks()
        if not registry.hardware_verdict(task.task_id).is_known
    )
    assert len(unmeasured) == EXPECTED_UNMEASURED_TRAIN_TASKS
    assert any(task_id.startswith("gen_") for task_id in unmeasured)
    assert any(task_id.startswith("genv_") for task_id in unmeasured)

    for task_id in unmeasured[:5]:
        decision = verification.eligibility(task_id)
        assert decision.eligible
        assert decision.reason == verification.ADMITTED_UNVERIFIED
        assert not decision.verdict.is_pass

    coverage = registry.hardware_verification_coverage()
    assert coverage["tasks"] == 1_546
    assert coverage["status_counts"]["UNKNOWN"] == 1_546 - 1_052
    assert coverage["measured"] == 1_052
    assert coverage["artifact_digest"] == verification.report().digest


def test_strict_policy_admits_only_recorded_passes():
    strict = verification.STRICT_HARDWARE_VERIFIED_POLICY
    eligible = {task.task_id for task in registry.eligible_train_tasks(strict)}

    assert eligible == verification.hardware_pass_ids() & {
        task.task_id for task in registry.train_tasks()
    }
    assert len(eligible) == EXPECTED_STRICT_ELIGIBLE_TRAIN_TASKS

    excluded = registry.train_eligibility_exclusions(strict)
    # Strict admits only a recorded PASS, so it drops every unmeasured task -- and
    # would drop a near-gate or infra verdict too, which the default admits.
    for verdict, reason in (
        (verification.HardwareVerdict("t", "INFRA", verification.BAND_INFRA),
         verification.EXCLUSION_INFRA),
        (verification.HardwareVerdict("t", "FAIL_CORRECTNESS",
                                      verification.BAND_NEAR_GATE, 57.9, 30.0),
         verification.EXCLUSION_NEAR_GATE),
    ):
        assert strict.exclusion_reason(verdict) == reason
        assert verification.DEFAULT_ELIGIBILITY_POLICY.admits(verdict)
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
    assert report["train_tasks"] == EXPECTED_TRAIN_TASKS
    assert report["eligible"] == EXPECTED_ELIGIBLE_TRAIN_TASKS
    assert report["excluded_by_reason"] == {}


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
