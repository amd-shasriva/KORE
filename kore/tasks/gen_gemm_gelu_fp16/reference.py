"""GENERATED reference shim for gemm_gelu (fp16). See kore/tasks/_genops.py.
Do not hand-edit — regenerate via kore/tasks/generate_ops.py."""
from kore.tasks._genops import make_reference

globals().update(make_reference("gemm_gelu", "gemm_fusion", "fp16"))
