"""CPU-only tests for semantics-preserving shape augmentation."""

from __future__ import annotations

from types import SimpleNamespace

from kore.tasks.augment import (
    ShapeAugmentationPolicy,
    audit_registry_shapes,
    augment_shapes,
    augmentation_policy,
    generated_shape_error,
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
    assert not augmentation_policy(task).enabled
    assert augment_shapes(base, task=task, max_shapes=8) == base


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


def test_generated_shape_is_odd_non_power_of_two_and_misaligned():
    base = [Shape("primary", {"M": 4096, "N": 4096})]
    aug = augment_shapes(base, policy=_explicit_policy(), max_shapes=2)
    generated = aug[-1]
    value = generated.dims["M"]
    assert value % 2 == 1
    assert value & (value - 1) != 0
    assert value % 8 != 0
    assert generated.dims["N"] == 4096


def test_decode_attention_freezes_batch_query_heads_and_head_ratio():
    base = [
        Shape("primary", {
            "B": 1, "H": 32, "HKV": 8, "SQ": 1, "SK": 4096, "D": 128,
        }),
    ]
    task = _task("attn2_decode_gqa_hd128", base, raw={"generated": True})
    generated = augment_shapes(base, task=task, max_shapes=2)[-1]
    assert generated.dims["SK"] % 2 == 1
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
    generated = augment_shapes(base, task=task, max_shapes=2)[-1]
    assert generated.dims["Skv"] % 2 == 1  # deliberately partial 16-token page
    for dim in ("B", "H", "KV", "D"):
        assert generated.dims[dim] == base[0].dims[dim]


def test_self_attention_equality_fails_closed_without_coupled_declaration():
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
    assert not augmentation_policy(task).enabled
    assert augment_shapes(base, task=task, max_shapes=8) == base


def test_moe_freezes_topk_expert_and_group_constraints():
    base = [
        Shape("primary", {
            "M": 4096, "E": 256, "topk": 8,
            "n_groups": 8, "topk_group": 4,
        }),
    ]
    task = _task("moe_biased_grouped_topk", base)
    generated = augment_shapes(base, task=task, max_shapes=2)[-1]
    assert generated.dims["M"] % 2 == 1
    for dim in ("E", "topk", "n_groups", "topk_group"):
        assert generated.dims[dim] == base[0].dims[dim]
    assert generated.dims["E"] % generated.dims["n_groups"] == 0
    assert generated.dims["topk"] <= generated.dims["E"]


def test_reduction_and_quant_group_axes_remain_frozen():
    reduction = [Shape("primary", {"M": 4096, "N": 8192})]
    red_task = _task("red_topk256", reduction, raw={"generated": True})
    red_generated = augment_shapes(reduction, task=red_task, max_shapes=2)[-1]
    assert red_generated.dims["M"] % 2 == 1
    assert red_generated.dims["N"] == 8192

    quant = [Shape("primary", {"M": 4096, "N": 4096, "K": 4096})]
    quant_task = _task("gemm_int4_asym_group", quant, raw={"generated": True})
    quant_generated = augment_shapes(quant, task=quant_task, max_shapes=2)[-1]
    assert quant_generated.dims["M"] % 2 == 1
    assert quant_generated.dims["N"] == 4096
    assert quant_generated.dims["K"] == 4096


def test_block_quantization_fails_closed_when_outer_dim_is_divisible():
    base = [
        Shape("primary", {"M": 4096, "K": 4096}),
        Shape("validation_0", {"M": 4224, "K": 4096}),
    ]
    task = _task("qx_quant_fp8_block2d", base, raw={
        "generated": True,
        "op_family": "breadth_qx_quant_fp8_block2d",
    })
    assert not augmentation_policy(task).enabled
    assert augment_shapes(base, task=task, max_shapes=8) == base


def test_dim0_reduction_does_not_change_reduction_axis():
    base = [Shape("primary", {"M": 4096, "N": 8192})]
    task = _task("red_logsumexp_dim0", base, raw={"generated": True})
    generated = augment_shapes(base, task=task, max_shapes=2)[-1]
    assert generated.dims["M"] == 4096
    assert generated.dims["N"] % 2 == 1


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


def test_whole_registry_preserves_contracts_and_has_odd_coverage():
    report = audit_registry_shapes()
    assert report.task_count == 1334
    assert report.ok, report.failures[:10]
    assert report.generated_candidates > 0
    assert report.odd_candidates == report.generated_candidates
    assert report.hidden_shapes > 0
    assert all(task.originals_preserved for task in report.tasks)
    assert all(task.hidden_train_overlap == 0 for task in report.tasks)
    # Unsupported coupled families are deliberately explicit, not silently scaled.
    assert report.unsupported_tasks > 0
