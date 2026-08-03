"""GENERATED signature stub for a SPEC-SYNTHESIS task.
See kore/tasks/generate_spec.py. Do not hand-edit.

There is no implementation here on purpose: the task is to write one
from the prose contract in spec.md. This stub exists so the task keeps
the registry's 'every task has a declared seed artifact' invariant and
so the required entry-point signature is unambiguous.
"""
from __future__ import annotations

import torch  # noqa: F401 (the implementation will need it)


def gelu_tanh(x):
    raise NotImplementedError(
        "spec-synthesis task: implement gelu_tanh from spec.md"
    )
