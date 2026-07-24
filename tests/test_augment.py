"""CPU-only tests for semantics-preserving shape augmentation."""

from __future__ import annotations

from types import SimpleNamespace

from kore.tasks.augment import (
    ShapeAugmentationPolicy,
    audit_registry_shapes,
    augment_shapes,
    augmentation_policy,
    boundary_regime,
    constraint_errors,
    freeze_shape_split,
    generated_shape_error,
    policy_metadata_for_shapes,
    shape_key,
)
from kore.tasks.base import Shape


def _task(
    operation: str,
    shapes: list[Shape],
    *,
    task_id: str = "test_task",
    raw: dict | None = None,
):
    return SimpleNamespace(
        task_id=task_id,
        operation=operation,
        shapes=shapes,
        raw={} if raw is None else raw,
    )


def _explicit_policy(dim: str = "M") -> ShapeAugmentationPolicy:
    return ShapeAugmentationPolicy(
        mutable_dims=(dim,),
        enabled=True,
        source="test",
        reason="explicit test policy",
        policy_id="test_independent",
        status="eligible",
        max_elements=1 << 62,
    )


def test_declared_shapes_stay_first_unchanged_even_above_max():
    base = [
        Shape("minimal", {"M": 64, "N": 512}),
        Shape("primary", {"M": 4096, "N": 4096}),
        Shape("validation_0", {"M": 8192, "N": 2048}),
        Shape("validation_1", {"M": 1024, "N": 32768}),
        Shape("validation_2", {"M": 4096, "N": 8191}),
    ]
    aug = augment_shapes(base, policy=_explicit_policy(), max_shapes=2)
    assert aug == base
    assert all(actual is declared for actual, declared in zip(aug, base))


def test_max_shapes_caps_only_generated_additions():
    base = [
        Shape("minimal", {"M": 64, "N": 512}),
        Shape("primary", {"M": 4096, "N": 4096}),
    ]
    assert len(augment_shapes(base, policy=_explicit_policy(), max_shapes=2)) == 2
    aug = augment_shapes(base, policy=_explicit_policy(), max_shapes=3)
    assert aug[:2] == base
    assert len(aug) == 3


def test_no_context_fails_closed():
    base = [Shape("primary", {"M": 4096, "N": 4096})]
    assert augment_shapes(base, max_shapes=8) == base


def test_unknown_operation_family_fails_closed():
    base = [Shape("primary", {"M": 4096, "category": 7})]
    task = _task(
        "unknown_coupled_operation",
        base,
        raw={"generated": True, "op_family": "unknown"},
    )
    policy = augmentation_policy(task)
    assert not policy.enabled
    assert policy.status == "ineligible"
    assert "absent from explicit" in policy.reason
    assert augment_shapes(base, task=task, max_shapes=8) == base
    assert freeze_shape_split(task).hidden_shapes == ()


def test_metadata_can_explicitly_enable_or_disable_augmentation():
    base = [Shape("primary", {"rows": 4096, "category": 7})]
    enabled = _task(
        "custom",
        base,
        raw={"shape_augmentation": {"mutable_dims": ["rows"]}},
    )
    disabled = _task(
        "custom",
        base,
        raw={"shape_augmentation": False},
    )
    aug = augment_shapes(base, task=enabled, max_shapes=2)
    assert len(aug) == 2 and aug[-1].dims["category"] == 7
    assert augment_shapes(base, task=disabled, max_shapes=2) == base


def test_metadata_can_author_exact_hidden_shapes_without_generic_transform():
    base = [Shape("primary", {"category": 7, "extent": 64})]
    task = _task(
        "custom_authored_hidden",
        base,
        raw={"shape_policy": {
            "schema_version": 2,
            "policy_id": "authored_only",
            "status": "eligible",
            "enabled": True,
            "source": "metadata",
            "reason": "operation author supplied exact hidden coverage",
            "transforms": [],
            "constraints": [],
            "authored_hidden_shapes": [
                {"name": "hidden_edge", "dims": {"category": 7, "extent": 65}},
            ],
            "max_elements": 1000,
        }},
    )
    split = freeze_shape_split(task)
    assert split.policy.claim_eligible
    assert split.hidden_shapes == (
        Shape("hidden_edge", {"category": 7, "extent": 65}),)


def test_generated_shapes_cover_multiple_boundary_regimes():
    base = [Shape("primary", {"M": 4096, "N": 4096})]
    generated = augment_shapes(
        base, policy=_explicit_policy(), max_shapes=None)[len(base):]
    regimes = {boundary_regime(shape) for shape in generated}
    assert {"small", "large", "non_power_of_two", "tail"} <= regimes
    tail = next(shape for shape in generated if boundary_regime(shape) == "tail")
    assert tail.dims["M"] % 8 != 0
    assert all(shape.dims["N"] == 4096 for shape in generated)


def test_decode_attention_freezes_batch_query_heads_and_head_ratio():
    base = [
        Shape("primary", {
            "B": 1, "H": 32, "HKV": 8, "SQ": 1, "SK": 4096, "D": 128,
        }),
    ]
    task = _task("attn2_decode_gqa_hd128", base, raw={
        "generated": True,
        "op_family": "breadth_attn2_decode_gqa_hd128",
    })
    generated = next(
        shape for shape in augment_shapes(base, task=task, max_shapes=None)
        if boundary_regime(shape) == "attention_tail"
    )
    assert generated.dims["SK"] % 8 != 0
    for dim in ("B", "H", "HKV", "SQ", "D"):
        assert generated.dims[dim] == base[0].dims[dim]
    assert generated.dims["H"] % generated.dims["HKV"] == 0


def test_singleton_outer_decode_regime_is_not_augmented():
    base = [Shape("primary", {"M": 1, "N": 4096, "K": 4096})]
    task = _task("gemm_fp8_a8w8_blockscale", base)
    assert augmentation_policy(task).enabled
    assert augment_shapes(base, task=task, max_shapes=8) == base


def test_paged_decode_changes_only_context_and_keeps_decode_layout():
    base = [
        Shape("primary", {
            "B": 1, "H": 32, "KV": 8, "Skv": 4096, "D": 128,
        }),
    ]
    task = _task("paged_attn_decode", base)
    generated = next(
        shape for shape in augment_shapes(base, task=task, max_shapes=None)
        if boundary_regime(shape) == "page_tail"
    )
    assert generated.dims["Skv"] % 16 != 0
    for dim in ("B", "H", "KV", "D"):
        assert generated.dims[dim] == base[0].dims[dim]


def test_self_attention_uses_atomic_equal_query_key_transform():
    base = [
        Shape("primary", {
            "B": 2, "H": 32, "HKV": 8, "SQ": 4096, "SK": 4096, "D": 128,
        }),
        Shape("validation_0", {
            "B": 1, "H": 32, "HKV": 8, "SQ": 2047, "SK": 2047, "D": 128,
        }),
    ]
    task = _task("attn_gqa_hd128_causal", base, raw={
        "generated": True,
        "op_family": "breadth_attn_gqa_hd128_causal",
    })
    policy = augmentation_policy(task)
    assert policy.claim_eligible
    assert policy.effective_transforms[0].dims == ("SQ", "SK")
    generated = augment_shapes(base, task=task, max_shapes=None)[len(base):]
    assert generated
    assert all(shape.dims["SQ"] == shape.dims["SK"] for shape in generated)
    assert all(not constraint_errors(shape.dims, policy) for shape in generated)


def test_moe_freezes_topk_expert_and_group_constraints():
    base = [
        Shape("primary", {
            "M": 4096, "E": 256, "topk": 8,
            "n_groups": 8, "topk_group": 4,
        }),
    ]
    task = _task("moe_biased_grouped_topk", base)
    generated = next(
        shape for shape in augment_shapes(base, task=task, max_shapes=None)
        if boundary_regime(shape) == "row_tail"
    )
    assert generated.dims["M"] % 8 != 0
    for dim in ("E", "topk", "n_groups", "topk_group"):
        assert generated.dims[dim] == base[0].dims[dim]
    assert generated.dims["E"] % generated.dims["n_groups"] == 0
    assert generated.dims["topk"] <= generated.dims["E"]


def test_reduction_and_quant_group_axes_remain_frozen():
    reduction = [Shape("primary", {"M": 4096, "N": 8192})]
    red_task = _task("red_topk256", reduction, raw={
        "generated": True,
        "op_family": "breadth_red_topk256",
    })
    red_generated = next(
        shape for shape in augment_shapes(reduction, task=red_task, max_shapes=None)
        if boundary_regime(shape) == "row_tail"
    )
    assert red_generated.dims["M"] % 8 != 0
    assert red_generated.dims["N"] == 8192

    quant = [Shape("primary", {"M": 4096, "N": 4096, "K": 4096})]
    quant_task = _task("gemm_int4_asym_group", quant, raw={
        "generated": True,
        "op_family": "breadth_gemm_int4_asym_group",
    })
    quant_generated = next(
        shape for shape in augment_shapes(quant, task=quant_task, max_shapes=None)
        if boundary_regime(shape) == "row_tail"
    )
    assert quant_generated.dims["M"] % 8 != 0
    assert quant_generated.dims["N"] == 4096
    assert quant_generated.dims["K"] == 4096


def test_block_quantization_preserves_atomic_block_layout():
    base = [
        Shape("primary", {"M": 4096, "K": 4096}),
        Shape("validation_0", {"M": 4224, "K": 4096}),
    ]
    task = _task("qx_quant_fp8_block2d", base, raw={
        "generated": True,
        "op_family": "breadth_qx_quant_fp8_block2d",
    })
    policy = augmentation_policy(task)
    assert policy.claim_eligible
    generated = augment_shapes(base, task=task, max_shapes=None)[len(base):]
    assert generated
    assert all(shape.dims["M"] % 128 == 0 for shape in generated)
    assert all(shape.dims["K"] % 128 == 0 for shape in generated)
    assert any(boundary_regime(shape) == "block_tail" for shape in generated)


def test_dim0_reduction_does_not_change_reduction_axis():
    base = [Shape("primary", {"M": 4096, "N": 8192})]
    task = _task("red_logsumexp_dim0", base, raw={
        "generated": True,
        "op_family": "breadth_red_logsumexp_dim0",
    })
    generated = next(
        shape for shape in augment_shapes(base, task=task, max_shapes=None)
        if boundary_regime(shape) == "tail"
    )
    assert generated.dims["M"] == 4096
    assert generated.dims["N"] % 8 != 0


def test_deterministic_generated_dedup_keeps_duplicate_declarations():
    base = [
        Shape("primary", {"M": 2048, "N": 4096}),
        Shape("validation_0", {"M": 2048, "N": 4096}),
    ]
    a = augment_shapes(base, policy=_explicit_policy(), max_shapes=8)
    b = augment_shapes(base, policy=_explicit_policy(), max_shapes=8)
    assert a == b
    assert a[:2] == base
    generated_keys = [shape_key(shape) for shape in a[len(base):]]
    assert len(generated_keys) == len(set(generated_keys))
    assert not set(generated_keys) & {shape_key(shape) for shape in base}
    assert all(
        generated_shape_error(base, shape, _explicit_policy()) is None
        for shape in a[len(base):]
    )


def test_empty_base_returns_empty():
    assert augment_shapes([]) == []


def test_state_sequence_and_multi_tensor_families_have_explicit_policies():
    from kore.tasks.registry import get_task

    state = get_task("genb_ssm_mamba2_ssd_c128_n128_bf16")
    state_policy = augmentation_policy(state)
    assert state_policy.claim_eligible
    assert state_policy.effective_transforms[0].dims == ("L",)
    assert state_policy.effective_transforms[0].misalign_dim == "chunk"

    multi = get_task("genb_tr_foreach_sgd_bf16")
    multi_policy = augmentation_policy(multi)
    assert multi_policy.claim_eligible
    assert multi_policy.effective_transforms[0].dims == ("N",)
    assert all(
        shape.dims["G"] in {base.dims["G"] for base in multi.shapes}
        for shape in augment_shapes(multi.shapes, task=multi, max_shapes=None)
    )


def test_manifest_generation_is_seeded_and_memory_bounded():
    from kore.tasks.registry import get_task

    task = get_task("genb_attn2_decode_gqa_hd128_bf16")
    first = freeze_shape_split(
        task, seed=17, created_at="2000-01-01T00:00:00+00:00")
    second = freeze_shape_split(
        task, seed=17, created_at="2000-01-01T00:00:00+00:00")
    assert first.to_dict() == second.to_dict()
    assert first.hidden_keys.isdisjoint(first.prompt_keys | first.train_keys)
    assert all(
        not constraint_errors(shape.dims, first.policy)
        for shape in (*first.train_shapes, *first.hidden_shapes)
    )


def test_source_generators_emit_serializable_explicit_policies():
    import yaml

    from kore.tasks import generate_breadth, generate_ops, generate_vendor_ops

    breadth_mod = generate_breadth._op_module_map()["ssm_mamba2_ssd_c128_n128"][1]
    generated = (
        generate_ops._yaml("add", "binary", "bf16", 30.0),
        generate_breadth._yaml(
            breadth_mod, "ssm_mamba2_ssd_c128_n128", "bf16", 30.0),
        generate_vendor_ops._yaml("softmax", "bf16", 30.0),
    )
    for document in generated:
        metadata = yaml.safe_load(document)["shape_policy"]
        assert metadata["schema_version"] >= 2
        assert metadata["status"] == "eligible"
        assert metadata["transforms"]
        assert metadata["max_elements"] > 0

    for op, (_name, module) in generate_breadth._op_module_map().items():
        metadata = policy_metadata_for_shapes(
            op, module.SHAPES[op], source=f"generator:breadth_{op}")
        assert metadata["status"] == "eligible", op
    for op in generate_vendor_ops.V.VENDOR_OPS:
        metadata = policy_metadata_for_shapes(
            op,
            generate_vendor_ops.V.VENDOR_SHAPES[op],
            source=f"generator:vendor_{op}",
        )
        assert metadata["status"] == "eligible", op


def test_whole_registry_has_explicit_status_and_hidden_family_coverage():
    report = audit_registry_shapes()
    assert report.task_count == 1334
    assert report.ok, report.failures[:10]
    assert report.explicit_status_tasks == report.task_count
    assert report.claim_eligible_tasks == report.task_count
    assert report.unsupported_tasks == 0
    assert not report.unsupported_families
    assert report.generated_candidates > 0
    assert report.odd_candidates > 0
    assert report.hidden_shapes > 0
    assert report.eligible_family_hidden_coverage
    assert sum(report.eligible_family_hidden_coverage.values()) == report.task_count
    assert all(task.originals_preserved for task in report.tasks)
    assert all(task.hidden_train_overlap == 0 for task in report.tasks)
    assert all(
        len(task.hidden_regimes) >= 3
        for task in report.tasks if task.claim_eligible
    )
