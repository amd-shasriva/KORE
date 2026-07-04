"""CPU-only tests for shape augmentation (data-scale coverage)."""

from __future__ import annotations

from kore.tasks.augment import augment_shapes
from kore.tasks.base import Shape


def test_expands_single_shape_to_multiple():
    base = [Shape("primary", {"M": 4096, "N": 4096, "K": 4096})]
    aug = augment_shapes(base, max_shapes=6)
    assert 4 <= len(aug) <= 6
    # deterministic small -> large ordering
    sizes = [sum(s.dims.values()) for s in aug]
    assert sizes == sorted(sizes)


def test_preserves_dim_keys_for_driver_compat():
    base = [Shape("primary", {"B": 8, "H": 32, "S": 1024, "D": 128})]
    aug = augment_shapes(base)
    for s in aug:
        assert set(s.dims) == {"B", "H", "S", "D"}


def test_alignment_rounding_keeps_multiples_of_8():
    base = [Shape("p", {"M": 100})]  # 100 -> scaled + rounded to /8
    aug = augment_shapes(base, factors=(1.0,), include_odd=False)
    assert all(s.dims["M"] % 8 == 0 for s in aug)


def test_includes_non_aligned_odd_stressor():
    base = [Shape("p", {"M": 4096, "N": 4096})]
    aug = augment_shapes(base, include_odd=True, max_shapes=10)
    assert any(any(v % 8 != 0 for v in s.dims.values()) for s in aug)


def test_deterministic_and_deduped():
    base = [Shape("p", {"M": 2048})]
    a = augment_shapes(base)
    b = augment_shapes(base)
    assert [(s.name, s.dims) for s in a] == [(s.name, s.dims) for s in b]
    keys = [tuple(sorted(s.dims.items())) for s in a]
    assert len(keys) == len(set(keys))  # no duplicate shapes


def test_empty_base_returns_empty():
    assert augment_shapes([]) == []
