"""CPU-only test for the data-scale audit."""

from __future__ import annotations

from kore.tasks.audit import audit


def test_audit_reports_operators_families_and_shapes():
    rep = audit(shape_augment=False)
    assert rep.n_operators >= 12
    assert rep.n_train + rep.n_heldout == rep.n_operators
    assert len(rep.families) >= 5
    assert rep.total_base_shapes >= rep.n_operators  # >=1 shape/op
    assert "attention" in rep.heldout_families       # generalization family reserved


def test_shape_augmentation_increases_effective_shapes():
    base = audit(shape_augment=False)
    aug = audit(shape_augment=True, augment_max=6)
    assert aug.total_effective_shapes >= base.total_effective_shapes
    assert aug.shapes_per_op_max >= base.shapes_per_op_max
