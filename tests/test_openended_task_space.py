"""CPU-only tests for the parametric open-ended task space."""

from __future__ import annotations

import random

import pytest

from kore.openended import task_space as ts


def test_enumerate_is_deterministic_and_nonempty():
    a = ts.enumerate_descriptors()
    b = ts.enumerate_descriptors()
    assert a == b
    assert len(a) > 100
    # no duplicates
    assert len(set(a)) == len(a)


def test_families_cover_genops_and_vendor():
    fams = set(ts.families())
    assert {"unary", "binary", "reduce", "fusion", "gemm_fusion"} <= fams
    assert any(f.startswith("vendor_") for f in fams)


def test_include_vendor_toggle():
    with_v = ts.enumerate_descriptors(include_vendor=True)
    without_v = ts.enumerate_descriptors(include_vendor=False)
    assert all(d.source == "genops" for d in without_v)
    assert any(d.source == "vendor" for d in with_v)
    assert len(with_v) > len(without_v)


def test_task_id_matches_kore_convention():
    d = ts.TaskDescriptor("genops", "unary", "relu", "bf16", "primary")
    assert d.task_id == "gen_relu_bf16"
    v = ts.TaskDescriptor("vendor", "vendor_rmsnorm", "rmsnorm", "fp16", "primary")
    assert v.task_id == "genv_rmsnorm_fp16"


def test_descriptor_features_keys_and_niche():
    d = ts.TaskDescriptor("genops", "unary", "relu", "bf16", "minimal")
    feats = ts.descriptor_features(d)
    for k in ("family", "arithmetic_intensity", "fusion_depth",
              "dtype_precision", "shape_scale"):
        assert k in feats
    key = ts.descriptor_key(d)
    assert key == tuple(feats[f] for f in ts.NICHE_FIELDS)
    assert len(key) == len(ts.NICHE_FIELDS)


def test_arithmetic_intensity_gemm_is_compute_bound():
    gemm = next(d for d in ts.enumerate_descriptors() if d.family == "gemm_fusion")
    assert ts.arithmetic_intensity(gemm) == "compute-bound"
    unary = ts.TaskDescriptor("genops", "unary", "relu", "bf16", "primary")
    assert ts.arithmetic_intensity(unary) == "memory-bound"


def test_fusion_depth_ordering():
    unary = ts.TaskDescriptor("genops", "unary", "relu", "bf16", "primary")
    assert ts.fusion_depth(unary) == 1
    # a 3-input fusion should have depth 3, a 2-input depth 2
    d3 = next(d for d in ts.enumerate_descriptors()
              if d.family == "fusion" and ts.fusion_depth(d) == 3)
    d2 = next(d for d in ts.enumerate_descriptors()
              if d.family == "fusion" and ts.fusion_depth(d) == 2)
    assert ts.fusion_depth(d3) > ts.fusion_depth(d2)


def test_dtype_precision_class():
    assert ts.descriptor_features(
        ts.TaskDescriptor("genops", "unary", "relu", "bf16", "primary"))["dtype_precision"] == "16b"
    assert ts.descriptor_features(
        ts.TaskDescriptor("genops", "unary", "relu", "fp32", "primary"))["dtype_precision"] == "32b"


def test_shape_scale_grows_with_regime():
    minimal = ts.TaskDescriptor("genops", "unary", "relu", "bf16", "minimal")
    primary = ts.TaskDescriptor("genops", "unary", "relu", "bf16", "primary")
    assert ts.shape_scale(minimal) == "small"
    assert ts.shape_scale(primary) in ("medium", "large")
    # gemm work volume (M*N*K) should push to the large bucket at primary
    gemm = ts.TaskDescriptor("genops", "gemm_fusion", "gemm_bias", "bf16", "primary")
    assert ts.shape_scale(gemm) == "large"


def test_descriptor_shape_lookup():
    d = ts.TaskDescriptor("genops", "unary", "relu", "bf16", "minimal")
    dims = ts.descriptor_shape(d)
    assert set(dims) == {"M", "N"}
    gemm = ts.TaskDescriptor("genops", "gemm_fusion", "gemm_bias", "bf16", "primary")
    assert set(ts.descriptor_shape(gemm)) == {"M", "N", "K"}
    with pytest.raises(KeyError):
        ts.descriptor_shape(ts.TaskDescriptor("genops", "unary", "relu", "bf16", "nope"))


def test_static_difficulty_monotone_signals():
    easy = ts.TaskDescriptor("genops", "unary", "relu", "fp32", "minimal")
    hard = ts.TaskDescriptor("genops", "gemm_fusion", "gemm_bias_gelu", "bf16", "primary")
    assert 0.0 <= ts.static_difficulty(easy) <= 1.0
    assert 0.0 <= ts.static_difficulty(hard) <= 1.0
    assert ts.static_difficulty(hard) > ts.static_difficulty(easy)


def test_sample_descriptors_deterministic():
    a = ts.sample_descriptors(8, seed=7)
    b = ts.sample_descriptors(8, seed=7)
    c = ts.sample_descriptors(8, seed=8)
    assert a == b
    assert a != c  # extremely unlikely to collide across seeds


def test_mutate_changes_exactly_one_axis_and_stays_valid():
    d = ts.TaskDescriptor("genops", "fusion", "fma", "bf16", "minimal")
    valid = set(ts.enumerate_descriptors())
    rng = random.Random(0)
    for _ in range(50):
        m = ts.mutate(d, rng)
        assert m != d
        assert m in valid  # mutation stays inside the concrete parametric space
        # same family/source (mutation perturbs shape/dtype/op-within-family)
        assert m.family == d.family and m.source == d.source
        diffs = sum([m.op != d.op, m.dtype != d.dtype, m.shape_regime != d.shape_regime])
        assert diffs == 1


def test_mutate_deterministic_given_rng():
    d = ts.TaskDescriptor("genops", "fusion", "fma", "bf16", "minimal")
    m1 = ts.mutate(d, random.Random(3))
    m2 = ts.mutate(d, random.Random(3))
    assert m1 == m2


def test_mutate_shape_only():
    d = ts.TaskDescriptor("genops", "unary", "relu", "bf16", "minimal")
    rng = random.Random(1)
    m = ts.mutate(d, rng, kinds=("shape",))
    assert m.op == d.op and m.dtype == d.dtype and m.shape_regime != d.shape_regime


def test_describe_is_json_friendly():
    d = ts.TaskDescriptor("vendor", "vendor_rmsnorm", "rmsnorm", "bf16", "primary")
    info = ts.describe(d)
    assert info["task_id"] == "genv_rmsnorm_bf16"
    assert "shape" in info and "family" in info
