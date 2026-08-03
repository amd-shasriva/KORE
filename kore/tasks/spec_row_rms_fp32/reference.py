"""GENERATED reference for a SPEC-SYNTHESIS task (row_rms, fp32).
See kore/tasks/generate_spec.py. Do not hand-edit.

The oracle, baseline and tolerance are _genops' proven ones, unchanged:
the spec shape changes the PROMPT, not the numerics."""
from kore.tasks._genops import make_reference

globals().update(make_reference('row_rms', 'reduce', 'fp32'))
