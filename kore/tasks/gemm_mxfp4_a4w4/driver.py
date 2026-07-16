"""Thin driver for a DRAFT quantized-GEMM task: delegates to the shared
``kore/tasks/_drafts/quant/_quant_common.driver_main``.

Correctness (no --bench-mode): candidate vs the fp32 dequant-matmul oracle in
reference.py; prints SNR / allclose / max_diff. Bench (--bench-mode --impl
{reference|candidate|torch}): cold-cache CUDA-event median timing; ``reference`` is the
REAL vendor op (AITER scaled-GEMM / hipBLASLt).

STAGED: not auto-discovered (nested under _drafts/quant/). On promotion to
kore/tasks/<id>/, _quant_common.py must be copied to kore/tasks/_quant_common.py.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                     # task dir -> reference.py
sys.path.insert(0, os.path.dirname(_HERE))    # parent   -> _quant_common.py

import reference as ref  # noqa: E402
from kore.tasks._quant_common import driver_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(driver_main(ref, _HERE))
