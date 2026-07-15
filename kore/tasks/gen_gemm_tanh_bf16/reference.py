"""GENERATED reference shim for gemm_tanh (bf16). See kore/tasks/_genops.py.
Do not hand-edit - regenerate via kore/tasks/generate_ops.py."""
from kore.tasks._genops import make_reference

globals().update(make_reference("gemm_tanh", "gemm_fusion", "bf16"))
