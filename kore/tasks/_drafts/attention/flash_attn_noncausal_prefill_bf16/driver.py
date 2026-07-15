"""Thin driver for a DRAFT attention task: delegates to the shared
``kore/tasks/_drafts/attention/_attn_common.driver_main``.

Correctness (no --bench-mode): candidate vs the fp32 oracle in reference.py; prints
SNR / allclose / max_diff. Bench (--bench-mode --impl {reference|candidate|torch}):
cold-cache CUDA-event median timing; ``reference`` is the REAL AITER vendor op.

STAGED: not auto-discovered (nested under _drafts/). On promotion to
kore/tasks/<id>/, _attn_common.py must be copied to kore/tasks/_attn_common.py.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                     # task dir -> reference.py
sys.path.insert(0, os.path.dirname(_HERE))    # parent   -> _attn_common.py

import reference as ref  # noqa: E402
from _attn_common import driver_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(driver_main(ref, _HERE))
