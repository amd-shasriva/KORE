"""Metamorphic relations for the equivalence oracle.

A *metamorphic* check exploits algebraic identities the TRUE operator must satisfy for
any input, without needing the reference output. They are candidate-only
self-consistency checks and are deterministic, so a candidate that violates one is
rejected with certainty. Crucially they catch **structural** cheats that point-value
checks miss — e.g. a "pointwise" kernel that secretly mixes elements across a row, or
a reduction that is not actually order-invariant.

Relations by op class
---------------------
* ``elementwise``  (``f`` applied identically per element):
    - **row/column permutation equivariance**: ``f(P·x) == P·f(x)``.
    - **locality / block independence**: ``f([x_top; x_bot]) == [f(x_top); f(x_bot)]``
      (an element's output depends only on that element).
    - **reshape invariance**: ``f(reshape(x)) == reshape(f(x))``.
* ``reduction``  (per-row ``[M,N] -> [M]``, order-invariant reduce like sum/mean/max/l2):
    - **column-permutation invariance**: ``g(x[:, π]) == g(x)`` (order independence).
    - **row-permutation equivariance**: ``g(x[π, :]) == g(x)[π]``.
    - **row locality**: ``g([x_top; x_bot]) == [g(x_top); g(x_bot)]``.

Everything is duck-typed over numpy arrays and torch (CPU/GPU) tensors; torch is
imported lazily only when a torch tensor is actually handled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from kore.verify.equivalence import _to_f64

__all__ = ["MetamorphicRelation", "metamorphic_relations"]


@dataclass
class MetamorphicRelation:
    """One metamorphic identity.

    ``apply(candidate_fn, inputs) -> (actual, expected)`` returns two float64 numpy
    arrays that the relation requires to be equal (within the metamorphic tolerance).
    """

    name: str
    op_class: str
    apply: Callable

    def __call__(self, candidate_fn, inputs):
        return self.apply(candidate_fn, inputs)


# --------------------------------------------------------------------------- #
# framework-agnostic array ops (numpy / torch)
# --------------------------------------------------------------------------- #
def _is_torch(a) -> bool:
    return (type(a).__module__ or "").startswith("torch")


def _take(a, perm, axis: int):
    if _is_torch(a):
        import torch

        idx = torch.as_tensor(np.asarray(perm), device=a.device)
        return torch.index_select(a, axis, idx)
    return np.take(a, np.asarray(perm), axis=axis)


def _concat(parts, axis: int):
    if _is_torch(parts[0]):
        import torch

        return torch.cat(list(parts), dim=axis)
    return np.concatenate(list(parts), axis=axis)


def _reshape(a, newshape):
    return a.reshape(newshape)


def _shape(a):
    return tuple(a.shape)


# --------------------------------------------------------------------------- #
# elementwise relations
# --------------------------------------------------------------------------- #
def _rng(inputs):
    return np.random.default_rng(1234 + int(sum(_shape(inputs[0]))))


def _elem_row_perm(candidate_fn, inputs):
    a0 = inputs[0]
    m = _shape(a0)[0]
    perm = _rng(inputs).permutation(m)
    permuted = tuple(_take(x, perm, axis=0) for x in inputs)
    lhs = candidate_fn(*permuted)                       # f(P·x)
    rhs = _take(candidate_fn(*inputs), perm, axis=0)    # P·f(x)
    return _to_f64(lhs), _to_f64(rhs)


def _elem_col_perm(candidate_fn, inputs):
    a0 = inputs[0]
    shp = _shape(a0)
    if len(shp) < 2:
        # 1-D fallback: permute the single axis
        perm = _rng(inputs).permutation(shp[0])
        permuted = tuple(_take(x, perm, axis=0) for x in inputs)
        lhs = candidate_fn(*permuted)
        rhs = _take(candidate_fn(*inputs), perm, axis=0)
        return _to_f64(lhs), _to_f64(rhs)
    n = shp[1]
    perm = _rng(inputs).permutation(n)
    permuted = tuple(_take(x, perm, axis=1) for x in inputs)
    lhs = candidate_fn(*permuted)
    rhs = _take(candidate_fn(*inputs), perm, axis=1)
    return _to_f64(lhs), _to_f64(rhs)


def _elem_locality(candidate_fn, inputs):
    a0 = inputs[0]
    m = _shape(a0)[0]
    half = max(1, m // 2)
    tops = tuple(x[:half] for x in inputs)
    bots = tuple(x[half:] for x in inputs)
    lhs = _concat([candidate_fn(*tops), candidate_fn(*bots)], axis=0)  # [f(top); f(bot)]
    rhs = candidate_fn(*inputs)                                        # f([top; bot])
    return _to_f64(lhs), _to_f64(rhs)


def _elem_reshape(candidate_fn, inputs):
    a0 = inputs[0]
    shp = _shape(a0)
    if len(shp) < 2 or shp[1] % 2 != 0:
        # fall back to locality if we cannot reshape cleanly
        return _elem_locality(candidate_fn, inputs)
    m, n = shp[0], shp[1]
    new = (m * 2, n // 2)
    reshaped = tuple(_reshape(x, new) for x in inputs)
    lhs = candidate_fn(*reshaped)                                   # f(reshape(x))
    rhs = _reshape(candidate_fn(*inputs), new)                      # reshape(f(x))
    return _to_f64(lhs), _to_f64(rhs)


# --------------------------------------------------------------------------- #
# reduction relations (per-row [M,N] -> [M], order-invariant)
# --------------------------------------------------------------------------- #
def _red_col_perm(candidate_fn, inputs):
    a0 = inputs[0]
    n = _shape(a0)[1]
    perm = _rng(inputs).permutation(n)
    permuted = tuple(_take(x, perm, axis=1) for x in inputs)
    lhs = candidate_fn(*permuted)          # g(x[:, π])
    rhs = candidate_fn(*inputs)            # g(x)   (order-invariant)
    return _to_f64(lhs), _to_f64(rhs)


def _red_row_perm(candidate_fn, inputs):
    a0 = inputs[0]
    m = _shape(a0)[0]
    perm = _rng(inputs).permutation(m)
    permuted = tuple(_take(x, perm, axis=0) for x in inputs)
    lhs = candidate_fn(*permuted)                       # g(x[π, :])
    rhs = _take(candidate_fn(*inputs), perm, axis=0)    # g(x)[π]
    return _to_f64(lhs), _to_f64(rhs)


def _red_row_locality(candidate_fn, inputs):
    a0 = inputs[0]
    m = _shape(a0)[0]
    half = max(1, m // 2)
    tops = tuple(x[:half] for x in inputs)
    bots = tuple(x[half:] for x in inputs)
    lhs = _concat([candidate_fn(*tops), candidate_fn(*bots)], axis=0)
    rhs = candidate_fn(*inputs)
    return _to_f64(lhs), _to_f64(rhs)


# --------------------------------------------------------------------------- #
# public: relation registry
# --------------------------------------------------------------------------- #
def metamorphic_relations(op_class: str = "elementwise") -> list[MetamorphicRelation]:
    """Return the metamorphic relations for ``op_class``.

    ``"elementwise"`` -> permutation-equivariance (rows & cols), locality, reshape.
    ``"reduction"``   -> column-permutation invariance, row-permutation equivariance,
                         row locality. ``"generic"`` -> ``[]`` (no safe structural
                         identity assumed).
    """
    oc = (op_class or "").lower()
    if oc == "elementwise":
        return [
            MetamorphicRelation("elem_row_permutation", oc, _elem_row_perm),
            MetamorphicRelation("elem_col_permutation", oc, _elem_col_perm),
            MetamorphicRelation("elem_locality", oc, _elem_locality),
            MetamorphicRelation("elem_reshape", oc, _elem_reshape),
        ]
    if oc == "reduction":
        return [
            MetamorphicRelation("reduce_col_perm_invariance", oc, _red_col_perm),
            MetamorphicRelation("reduce_row_perm_equivariance", oc, _red_row_perm),
            MetamorphicRelation("reduce_row_locality", oc, _red_row_locality),
        ]
    return []
