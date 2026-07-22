"""GENERATED breadth reference shim for gemm_fp8_grouped (fp8). See kore/tasks/breadth/gemm_ext.py.
Do not hand-edit - regenerate via kore/tasks/generate_breadth.py."""
from kore.tasks.breadth.gemm_ext import make_reference

globals().update(make_reference("gemm_fp8_grouped", "fp8"))
