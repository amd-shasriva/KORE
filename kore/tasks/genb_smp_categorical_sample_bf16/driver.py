"""GENERATED breadth driver shim for smp_categorical_sample (bf16). See kore/tasks/_genops.py.
Do not hand-edit - regenerate via kore/tasks/generate_breadth.py."""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
import reference as ref  # noqa: E402
from kore.tasks._genops import driver_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(driver_main(ref, _here))
