"""GENERATED HIP driver shim for layernorm (fp16). See kore/tasks/_genops.py.
Do not hand-edit - regenerate via kore/tasks/generate_hip.py.

The candidate is staged as ``seed_hip.hip``-shaped HIP C++ in ``kernel.hip`` and
compiled by kore.env.hip_toolchain; _genops.driver_main is otherwise unchanged,
so this task gets the same paired cold-cache timing protocol, adversarial
battery and post-timing re-verification as every Triton task."""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
import reference as ref  # noqa: E402
from kore.tasks._genops import driver_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(driver_main(ref, _here))
