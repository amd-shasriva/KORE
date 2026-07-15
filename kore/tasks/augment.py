"""Shape augmentation: multiply per-operator coverage for stronger generalization.

Data scale in kernel-RL is not just #operators - it is also the diversity of
SHAPES per operator. A policy tuned on one shape memorizes a tile config; a policy
graded on small/medium/large + non-aligned shapes must learn shape-robust code
(this is exactly how KernelBench / TritonBench stress generalization).

This module expands a task's base shapes into a diverse, deterministic set by
scaling each base shape's dims by several factors and adding one intentionally
NON-aligned ("odd") shape to punish kernels that only handle power-of-two tiles.
Scaling preserves the dim KEYS the driver already accepts (so no driver change is
needed) and rounds sizes to a hardware-friendly multiple (default 8) to stay on
the MFMA/tensor-core happy path, while the odd shape deliberately breaks alignment.

Pure and deterministic; opt-in via CONFIG.shape_augment (see KoreEnv._shapes).
"""

from __future__ import annotations

from typing import Iterable

from kore.tasks.base import Shape

DEFAULT_FACTORS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)


def _round_to(v: int, mult: int) -> int:
    if mult <= 1:
        return max(1, int(v))
    return max(mult, int(round(v / mult)) * mult)


def _scale_dims(dims: dict[str, int], factor: float, align: int) -> dict[str, int]:
    return {k: _round_to(int(v) * factor, align) if isinstance(v, (int, float)) else v
            for k, v in dims.items()}


def _odd_dims(dims: dict[str, int]) -> dict[str, int]:
    """A non-power-of-two, non-aligned variant to stress boundary handling."""
    out = {}
    for k, v in dims.items():
        if isinstance(v, (int, float)) and int(v) > 1:
            out[k] = int(v) + 1  # break alignment (e.g. 4096 -> 4097)
        else:
            out[k] = v
    return out


def augment_shapes(base_shapes: Iterable[Shape], *,
                   factors: tuple[float, ...] = DEFAULT_FACTORS,
                   align: int = 8, include_odd: bool = True,
                   max_shapes: int = 6) -> list[Shape]:
    """Expand ``base_shapes`` into a diverse, deduped, deterministic shape set.

    Returns at most ``max_shapes`` shapes: scaled variants of every base shape
    (rounded to ``align``) plus, if ``include_odd``, one non-aligned stressor
    derived from the largest base shape. Deterministic ordering (small -> large).
    """
    base = list(base_shapes)
    if not base:
        return []

    seen: set[tuple] = set()
    out: list[Shape] = []

    def _add(name: str, dims: dict[str, int]) -> None:
        key = tuple(sorted(dims.items()))
        if key in seen:
            return
        seen.add(key)
        out.append(Shape(name, dims))

    for bs in base:
        for f in factors:
            dims = _scale_dims(bs.dims, f, align)
            tag = f"{bs.name}_x{f:g}".replace(".", "p")
            _add(tag, dims)

    if include_odd:
        largest = max(base, key=lambda s: sum(int(v) for v in s.dims.values()
                                              if isinstance(v, (int, float))) or 0)
        _add(f"{largest.name}_odd", _odd_dims(_scale_dims(largest.dims, 1.0, align)))

    # deterministic small->large ordering; cap the count to bound eval cost.
    out.sort(key=lambda s: (sum(int(v) for v in s.dims.values()
                                if isinstance(v, (int, float))) or 0, s.name))
    return out[:max_shapes]
