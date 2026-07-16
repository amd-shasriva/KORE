"""Thin driver for a DRAFT training-side (BACKWARD) task: delegates to the shared
``kore/tasks/_drafts/training/_training_common.driver_main``.

Correctness (no --bench-mode): candidate gradients vs the fp32 AUTOGRAD oracle in
reference.py; prints SNR (worst over all gradients) / allclose / max_diff. Bench
(--bench-mode --impl {reference|candidate|torch}): cold-cache CUDA-event median
timing; ``reference`` is the perf-only framework fused autograd backward (NO AITER
backward kernel exists for these ops -- see VERIFICATION_CHECKLIST.md).

STAGED: not auto-discovered (nested under _drafts/training/). On promotion to
kore/tasks/<id>/, _training_common.py must be copied to kore/tasks/_training_common.py.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                     # task dir -> reference.py
sys.path.insert(0, os.path.dirname(_HERE))    # parent   -> _training_common.py

import reference as ref  # noqa: E402
from kore.tasks._training_common import driver_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(driver_main(ref, _HERE))
