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


EXPECTED_PRODUCT_COUNTS = {
    "activation": 85,
    "attention": 147,
    "convolution": 120,
    "data_movement": 2,
    "elementwise": 24,
    "fusion": 103,
    "gemm": 95,
    "mla": 1,
    "moe": 89,
    "normalization": 102,
    "paged_attention": 1,
    "positional": 7,
    "quantization": 33,
    "reduction": 163,
    "sampling": 84,
    "sequence": 94,
    "sparse": 16,
    "training": 168,
}
EXPECTED_TAXONOMY_DIGEST = (
    "4fcfe32792ef57a1e35d612c30e6955dc831b66b59cf17ae133e8aa17fcc8cd8"
)


def test_whole_registry_has_complete_stable_classification():
    tasks = registry.all_tasks()
    assert len(tasks) == 1_334
    counts = Counter(registry.operator_family(task) for task in tasks)
    assert dict(sorted(counts.items())) == EXPECTED_PRODUCT_COUNTS
    assert set(counts) == set(taxonomy.PRODUCT_FAMILIES)
    assert registry.taxonomy_digest() == EXPECTED_TAXONOMY_DIGEST
    assert len(registry.operation_family_map()) < len(tasks)


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
    assert len(manifest.train_ids) == 1_289
    assert len(manifest.eval_ids) == 45
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
