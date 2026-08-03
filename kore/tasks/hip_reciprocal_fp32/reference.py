"""GENERATED HIP reference shim for reciprocal (fp32).
See kore/tasks/hip_ops.py. Do not hand-edit - regenerate via
kore/tasks/generate_hip.py."""
from kore.tasks.hip_ops import make_reference

globals().update(make_reference("reciprocal", "fp32"))
