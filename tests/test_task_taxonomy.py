"""Whole-registry contract tests for the versioned task taxonomy."""

from __future__ import annotations

import copy
import json
from collections import Counter
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from kore.data import mutate
from kore.eval import generalization
from kore.openended import minter, task_space
from kore.tasks import registry, taxonomy
from kore.tasks.base import Task


# Includes the 188-task HIP C++ family (kore/tasks/generate_hip.py), every one of
# which is proven end-to-end on gfx950 before it is claimed -- see
# data/hip_task_verification.json. Its 188 tasks sit in six families: activation
# 90, fusion 36, reduction 33, elementwise 18, normalization 9, quantization 2.
# ``gemm`` contributes nothing: hip_gemm is defined and verified but is not
# timing-admissible on this host, so the generator does not materialize it.
EXPECTED_PRODUCT_COUNTS = {
    "activation": 175,
    "attention": 147,
    "convolution": 120,
    "data_movement": 2,
    "elementwise": 42,
    "fusion": 139,
    "gemm": 95,
    "mla": 1,
    "moe": 89,
    "normalization": 111,
    "paged_attention": 1,
    "positional": 7,
    "quantization": 35,
    "reduction": 196,
    "sampling": 84,
    "sequence": 94,
    "sparse": 16,
    "training": 168,
}
# Content-addressed over the versioned rules plus every live task assignment,
# which now includes each task's contamination state. Recording that the midtrain
# corpus saw 11 held-out tasks changes the payload, so the digest moves and every
# manifest authored before the finding is correctly stale. Adding the HIP C++
# family moves it again, for the same reason and with the same consequence: a
# split manifest frozen before the HIP tasks existed does not describe this
# registry and must be re-frozen rather than reused.
EXPECTED_TAXONOMY_DIGEST = (
    "92e5acc60631afae19ef209cdd167a937fe196f698f0e87d7f478f171dbeebf5"
)
CONTAMINATED_HELDOUT_TASKS = frozenset({
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
})


def test_whole_registry_has_complete_stable_classification():
    tasks = registry.all_tasks()
    # 1,334 Triton + 188 HIP C++ (kore/tasks/generate_hip.py).
    assert len(tasks) == 1_522
    counts = Counter(registry.operator_family(task) for task in tasks)
    assert dict(sorted(counts.items())) == EXPECTED_PRODUCT_COUNTS
    assert set(counts) == set(taxonomy.PRODUCT_FAMILIES)
    assert registry.taxonomy_digest() == EXPECTED_TAXONOMY_DIGEST
    assert len(registry.operation_family_map()) < len(tasks)


def test_contamination_state_is_inside_the_content_address(monkeypatch):
    """Dropping a contamination record must move the digest, not pass silently.

    The pinned digest is only meaningful if the contamination exclusions are part
    of the payload it addresses; otherwise the 11 leaked tasks could be quietly
    un-marked while every manifest still validated.
    """
    tasks = registry.all_tasks()
    payload = taxonomy.taxonomy_payload(tasks)
    assert set(payload["contaminated_tasks"]) == set(CONTAMINATED_HELDOUT_TASKS)
    assert all(
        entry["contaminated"] is (entry["task_id"] in CONTAMINATED_HELDOUT_TASKS)
        for entry in payload["assignments"]
    )

    monkeypatch.setattr(taxonomy, "CONTAMINATION_RECORDS", {})
    monkeypatch.setattr(taxonomy, "CONTAMINATED_TASK_IDS", frozenset())
    assert taxonomy.taxonomy_digest(tasks) != EXPECTED_TAXONOMY_DIGEST


def test_attention_precedence_and_task_level_near_probes():
    assert taxonomy.product_family_for_name("mla_decode_variant") == "mla"
    assert (
        taxonomy.product_family_for_name("paged_attn_decode_variant")
        == "paged_attention"
    )
    assert taxonomy.product_family_for_name("flash_attn_decode") == "attention"

    by_id = {task.task_id: task for task in registry.all_tasks()}
    assert set(taxonomy.NEAR_GENERALIZATION_TASK_IDS) <= set(by_id)
    assert len(taxonomy.NEAR_GENERALIZATION_TASK_IDS) == 43
    # Contamination is recorded alongside the reservation, never instead of it:
    # all 43 keep ``near_probe`` as the reason they are held out.
    assert CONTAMINATED_HELDOUT_TASKS <= set(taxonomy.NEAR_GENERALIZATION_TASK_IDS)
    assert all(
        registry.split_decision(by_id[task_id]).reason == "near_probe"
        for task_id in taxonomy.NEAR_GENERALIZATION_TASK_IDS
    )
    assert "attention" not in registry.heldout_families()
    assert {"mla", "paged_attention"} == registry.heldout_families()
    assert any(
        registry.operator_family(task) == "attention" and not registry.is_heldout(task)
        for task in by_id.values()
    )
    assert any(
        registry.operator_family(task) == "attention" and registry.is_heldout(task)
        for task in by_id.values()
    )
    root = next(iter(taxonomy.NEAR_GENERALIZATION_TASK_IDS))
    descendant = taxonomy.split_decision_for_identity(
        task_id=f"{root}_minted_variant",
        operation="flash_attn_variant",
        product_family="attention",
        architecture="gfx950",
        dtype="bf16",
        provenance_root=root,
    )
    assert descendant.reason == "heldout_lineage" and descendant.heldout


def test_split_manifest_is_immutable_complete_and_lineage_disjoint():
    manifest = registry.build_split_manifest()
    # The HIP family is 188 trainable tasks (168 more than the 20 it started as);
    # the held-out reservation is untouched by them, because every HIP task is a
    # new identity with its own provenance root.
    assert len(manifest.train_ids) == 1_477
    assert len(manifest.eval_ids) == 45
    # Contaminated tasks stay in the held-out reservation and leave only the
    # zero-shot claim: 45 held out, 34 of them still scoreable.
    assert set(manifest.contaminated_eval_ids) == CONTAMINATED_HELDOUT_TASKS
    assert len(manifest.generalization_eval_ids) == 34
    assert set(manifest.contaminated_eval_ids) <= set(manifest.eval_ids)
    assert isinstance(manifest.train_ids, tuple)
    assert not (set(manifest.train_ids) & set(manifest.eval_ids))
    train_roots = set(dict(manifest.train_provenance_roots).values())
    eval_roots = set(dict(manifest.eval_provenance_roots).values())
    assert train_roots.isdisjoint(eval_roots)
    assert registry.validate_split_manifest(manifest.as_dict()) == manifest
    with pytest.raises(FrozenInstanceError):
        manifest.train_ids = ()  # type: ignore[misc]


def _identity(task_id: str, operation: str, family: str):
    return SimpleNamespace(
        task_id=task_id,
        operation=operation,
        dtype="bf16",
        gpu_target="gfx950",
        provenance_root=task_id,
        raw={"generated": True, "minted": True, "op_family": family},
    )


def test_assignment_validation_rejects_duplicate_and_colliding_tasks():
    one = _identity("gen_one_bf16", "one", "activation")
    duplicate = _identity("gen_one_bf16", "two", "activation")
    with pytest.raises(taxonomy.TaxonomyError, match="duplicate"):
        taxonomy.validate_task_assignments([one, duplicate])

    activation = _identity("gen_activation_bf16", "same_op", "activation")
    fusion = _identity("gen_fusion_bf16", "same_op", "fusion")
    with pytest.raises(taxonomy.TaxonomyError, match="maps to both"):
        taxonomy.validate_task_assignments([activation, fusion])

    malformed = _identity("gen_bad_bf16", "bad", "not_a_family")
    with pytest.raises(taxonomy.TaxonomyError, match="unknown product family"):
        taxonomy.validate_task_assignments([malformed])


def test_task_loader_rejects_directory_identity_collision(tmp_path):
    task_dir = tmp_path / "directory_name"
    task_dir.mkdir()
    for artifact in ("driver.py", "reference.py", "seed_triton.py"):
        (task_dir / artifact).write_text("# test\n")
    metadata = {
        "task_id": "different_name",
        "operation": "relu",
        "dtype": "bf16",
        "backend": "triton",
        "gpu_target": "gfx950",
        "seed_kernel_name": "seed_triton.py",
        "snr_threshold": 30,
        "shapes": {"minimal": {"M": 1}},
        "targets": {"comparison_baseline": "torch"},
    }
    (task_dir / "task.yaml").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="collides with directory"):
        Task.from_dir(task_dir)


def test_foreign_arch_dtype_and_unknown_operations_are_explicit_eval_slices():
    foreign_arch = taxonomy.split_decision_for_identity(
        task_id="external_relu",
        operation="relu",
        architecture="gfx1100",
        dtype="bf16",
    )
    foreign_dtype = taxonomy.split_decision_for_identity(
        task_id="external_relu_fp64",
        operation="relu",
        architecture="gfx950",
        dtype="fp64",
    )
    unknown = taxonomy.split_decision_for_identity(
        task_id="external_unknown",
        operation="brand_new_unreviewed_op",
        architecture="gfx950",
        dtype="bf16",
    )
    assert foreign_arch.reason == "foreign_arch"
    assert foreign_dtype.reason == "foreign_dtype"
    assert unknown.reason == "unclassified_operation"
    assert foreign_arch.heldout and foreign_dtype.heldout and unknown.heldout


def test_every_consumer_uses_the_same_hierarchy():
    for task in registry.all_tasks():
        product = registry.operator_family(task)
        assert generalization.family_of(task.task_id) == taxonomy.analysis_family(product)
        assert mutate.infer_family(task.operation) == taxonomy.mutation_family(product)

    gen = task_space.TaskDescriptor("genops", "unary", "relu", "bf16")
    vendor = task_space.TaskDescriptor(
        "vendor", "vendor_rmsnorm", "rmsnorm", "bf16"
    )
    assert task_space.product_family(gen) == "activation"
    assert task_space.product_family(vendor) == "normalization"
    assert minter.is_heldout("paged_attn_candidate", "attention")
    assert not minter.is_heldout("matmul_bias_gelu", "gemm", "bf16")


def test_stale_and_malformed_manifests_are_invalidated():
    payload = registry.build_split_manifest().as_dict()

    stale = copy.deepcopy(payload)
    stale["taxonomy"]["digest"] = "0" * 64
    with pytest.raises(registry.StaleSplitManifestError, match="digest changed"):
        registry.validate_split_manifest(stale)

    duplicate = copy.deepcopy(payload)
    duplicate["train_ids"].append(duplicate["train_ids"][0])
    duplicate["train_ids"].sort()
    with pytest.raises(registry.SplitManifestError, match="duplicates"):
        registry.validate_split_manifest(duplicate)

    legacy = {"train_ids": payload["train_ids"], "eval_ids": payload["eval_ids"]}
    with pytest.raises(registry.StaleSplitManifestError, match="lacks taxonomy"):
        registry.validate_split_manifest(legacy)


def test_direct_grpo_defaults_to_train_split_only(monkeypatch):
    from kore.policy import grpo

    seen = {}
    def fake_train(_config, tasks):
        seen["tasks"] = list(tasks)
        return "checkpoint"

    monkeypatch.setattr(grpo, "_train_grpo_inprocess", fake_train)
    result = grpo.train_grpo(SimpleNamespace(model_id="test"))
    assert result == "checkpoint"
    assert set(seen["tasks"]) == {task.task_id for task in registry.train_tasks()}
    assert set(seen["tasks"]).isdisjoint(
        task.task_id for task in registry.heldout_tasks()
    )
    with pytest.raises(ValueError, match="non-empty"):
        grpo.train_grpo(SimpleNamespace(model_id="test"), tasks=[])
