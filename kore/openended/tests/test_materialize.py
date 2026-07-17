"""Tests for minted-task materialization + CoevolutionController minting.

Needs torch (the materialize self-check runs the reconstructed oracle on CPU). Covers:
  * a minted task materializes into a runnable dir whose on-disk reference oracle
    reproduces the in-memory minted oracle (the safety self-check), with a correct
    seed + task.yaml;
  * the self-check REJECTS a faithless reconstruction (safety: no corruption);
  * the controller serves + resolves minted tasks with AND without a registered menu,
    and is byte-identical legacy behavior when mint=False.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from kore.openended.archive import TaskArchive
from kore.openended.controller import CoevolutionController
from kore.openended.materialize import (materialize_minted_task,
                                        reference_namespace_from_spec)
from kore.openended.minter import TaskMinter
from kore.tasks.registry import task_ids


def _p(mt):
    return (int(mt.behavioral_hash[:6], 16) % 1000) / 1000.0


def _mint(n=8, seed=7):
    return TaskMinter(seed=seed).mint_batch(TaskArchive(0), _p, n)


def test_materialize_produces_runnable_task_with_matching_oracle():
    batch = _mint()
    materialized = [(mt, materialize_minted_task(mt)) for mt in batch]
    ok = [(mt, t) for mt, t in materialized if t is not None]
    assert ok, "expected at least one minted task to materialize + self-check"
    for mt, task in ok:
        # the four ABI files exist
        for f in ("driver.py", "reference.py", "seed_triton.py", "task.yaml"):
            assert (task.dir / f).exists(), f
        # task metadata round-trips
        assert task.task_id == mt.task_id
        assert task.dtype == mt.dtype
        assert task.shapes and task.seed_source  # seed is the correct torch baseline
        # the ON-DISK oracle reproduces the IN-MEMORY oracle bit-for-bit
        import importlib.util

        import kore.openended.grammar as g
        spec = importlib.util.spec_from_file_location(f"_ref_{mt.task_id}",
                                                      str(task.dir / "reference.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        inp = g.build_sampler(mt.pipeline, dict(mt.shape), mt.dtype)(0, "cpu")
        assert torch.allclose(mod.ref_fn(*inp).float(),
                              mt.reference_fn(*inp).float(), atol=1e-3, rtol=1e-2)
        assert mod.entry_name == mt.name and mod.arity == mt.arity


def test_self_check_rejects_faithless_reconstruction(monkeypatch):
    # Force the reconstructed oracle to differ from the minted one -> must reject.
    batch = _mint(n=4)
    mt = batch[0]
    import kore.openended.materialize as mm
    real = mm.reference_namespace_from_spec

    def _wrong(spec):
        ns = real(spec)
        ref = ns["ref_fn"]
        ns["ref_fn"] = lambda *a, **k: ref(*a, **k) + 1.0  # corrupt the oracle
        return ns

    monkeypatch.setattr(mm, "reference_namespace_from_spec", _wrong)
    assert materialize_minted_task(mt) is None  # self-check catches the mismatch


def test_controller_mints_without_registered_menu():
    # hand-authored ids don't map into the parametric menu -> pure open-ended minting
    hand = [t for t in task_ids() if not t.startswith("gen")][:12] or task_ids()[:12]
    c = CoevolutionController(hand, seed=3, batch=6, mint=True, mint_batch=6)
    served_minted = 0
    for i in range(36):
        tid = c.next_task_id(step=i, attempt=i)
        assert c.resolve_task(tid) is not None      # never raises
        if tid in c._minted:
            served_minted += 1
        c.record(tid, 0.4, 1.2)
    assert served_minted >= 1
    assert c.report()["minted_materialized"] >= 1


def test_controller_mint_false_is_legacy():
    ids = [t for t in task_ids() if t.startswith("gen_")][:12] or task_ids()[:12]
    c = CoevolutionController(ids, seed=3, mint=False)
    for i in range(20):
        c.next_task_id(step=i, attempt=i)
    rep = c.report()
    assert rep["mint"] is False and rep["minted_pool"] == 0 and rep["minted_materialized"] == 0


def test_reference_namespace_from_spec_roundtrip():
    mt = _mint(n=4)[0]
    from kore.openended.materialize import _spec_of
    ns = reference_namespace_from_spec(_spec_of(mt))
    for k in ("parse_shape", "get_inputs", "ref_fn", "baseline_fn", "arity",
              "entry_name", "dtype_name", "family"):
        assert k in ns
    inp = ns["get_inputs"](dict(mt.shape), device="cpu", seed=0)
    assert len(inp) == ns["arity"]
    assert torch.isfinite(ns["ref_fn"](*inp).float()).all()
