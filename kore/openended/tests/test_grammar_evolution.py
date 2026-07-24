"""CPU tests for the SELF-REFERENTIAL grammar-evolution extension of the minter.

The minter's fixed MOVES compose a FIXED grammar with bounded-depth templates - a
bounded task distribution (the POET/OMNI "bounded encoding" ceiling). The opt-in
``evolve_grammar`` mechanism EVOLVES the grammar itself (``grammar.Production``):
it grows new well-typed productions by composing existing ones (self-referential)
and mints tasks from them, reaching depths/structures the templates never
enumerate - WITHOUT weakening correct-by-construction.

These tests prove, all on CPU:
  * flag OFF (default) is BYTE-IDENTICAL to the pre-edit baseline (a golden snapshot
    captured from the unmodified minter) - enabling nothing changes current minting;
  * the grammar-evolution operators are well-typed by construction;
  * evolution reaches a STRICTLY larger valid-task space (deeper fusion, net-new
    niches) - more distinct valid tasks;
  * EVERY emitted evolved task still passes the FULL construction gate and the
    materialize self-check (correctness is preserved by construction, not assumption);
  * degenerate composed productions are rejected (fail-safe);
  * behavioral-hash dedup + QD niche placement still hold for the expanded space;
  * the grammar grows self-referentially (gate-passing productions are promoted +
    re-composed), and evolution is deterministic given the seed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from kore.openended import grammar as g
from kore.openended import minter as m
from kore.openended import task_space as ts
from kore.openended.archive import TaskArchive

_GOLDEN = Path(__file__).with_name("_golden_mint_baseline.json")


def _p(mt) -> float:
    """A deterministic, cross-process pseudo solve-rate for a minted task."""
    return (int(mt.behavioral_hash[:6], 16) % 1000) / 1000.0


def _row(t) -> list:
    """The identity tuple a golden entry pins (task_id + dedup + structure)."""
    return [t.task_id, list(t.dedup_key), t.pipeline.signature(), t.move,
            list(t.niche_key)]


# --------------------------------------------------------------------------- #
# SAFETY: flag OFF is byte-identical to the pre-edit baseline
# --------------------------------------------------------------------------- #
def test_flag_off_is_byte_identical_to_golden_baseline():
    """The default (evolution OFF) minter reproduces a golden snapshot captured
    from the UNMODIFIED code - so the live run's minting is unchanged."""
    golden = json.loads(_GOLDEN.read_text())
    assert golden, "golden fixture missing/empty"
    for seed_s, expected in golden.items():
        batch = m.mint_batch(TaskArchive(0), _p, 16, seed=int(seed_s),
                             evolve_grammar=False)
        assert [_row(t) for t in batch] == expected, f"seed {seed_s} drifted"


def test_flag_defaults_off_and_env_toggle(monkeypatch):
    """Default OFF; the env var flips the default; an explicit param wins over env."""
    monkeypatch.delenv(m._EVOLVE_GRAMMAR_ENV, raising=False)
    assert m.TaskMinter(seed=0).evolve_grammar is False
    assert m.TaskMinter(seed=0, evolve_grammar=True).evolve_grammar is True

    monkeypatch.setenv(m._EVOLVE_GRAMMAR_ENV, "1")
    assert m.TaskMinter(seed=0).evolve_grammar is True              # env flips default
    assert m.TaskMinter(seed=0, evolve_grammar=False).evolve_grammar is False  # param wins


def test_grammar_move_present_only_when_enabled():
    off = m.mint_batch(TaskArchive(0), _p, 24, seed=0, evolve_grammar=False)
    on = m.mint_batch(TaskArchive(0), _p, 24, seed=0, evolve_grammar=True)
    assert "grammar" not in {t.move for t in off}
    on_moves = {t.move for t in on}
    assert "grammar" in on_moves
    # the four base moves are still exercised alongside grammar evolution
    assert {"fusion", "extrapolate", "novel", "mutate_crossover"} <= on_moves


# --------------------------------------------------------------------------- #
# Grammar-evolution operators are well-typed BY CONSTRUCTION
# --------------------------------------------------------------------------- #
def test_base_productions_and_composition_are_well_typed():
    prods = g.base_productions()
    mats = [p for p in prods if p.in_type == g.MATRIX and p.out_type == g.MATRIX]
    terms = [p for p in prods if p.out_type == g.ROWVEC]
    assert mats and terms
    for p in prods:
        p.typecheck()                       # every axiom is sound
        assert p.depth == 1

    # sequential composition of two MATRIX blocks is a MATRIX->MATRIX block
    ab = g.compose_productions(mats[0], mats[1])
    assert ab is not None and ab.out_type == g.MATRIX and ab.depth == 2
    assert ab.stages == mats[0].stages + mats[1].stages
    ab.typecheck()

    # block then terminal is MATRIX->ROWVEC; a terminal can NEVER be extended
    bt = g.compose_productions(mats[0], terms[0])
    assert bt is not None and bt.out_type == g.ROWVEC
    assert g.compose_productions(terms[0], mats[0]) is None


def test_pipeline_from_production_typechecks_and_is_flat():
    src = g.source_prims()
    mid = g.middle_prims()
    body = g.compose_productions(
        g.Production("gelu", g.MATRIX, g.MATRIX, (mid["gelu"],)),
        g.Production("rmsnorm", g.MATRIX, g.MATRIX, (mid["rmsnorm"],)))
    pipe = g.pipeline_from_production(src["input"], body)
    assert pipe.signature() == "input->gelu->rmsnorm"
    # stages are the SAME named primitives -> materialize-safe / gate-ready
    assert all(isinstance(st, g.Primitive) for st in pipe.stages)
    with pytest.raises(g.GrammarTypeError):        # a non-source head is rejected
        g.pipeline_from_production(mid["gelu"], body)
    with pytest.raises(g.GrammarTypeError):        # a malformed production is rejected
        g.Production("bad", g.MATRIX, g.ROWVEC, (mid["gelu"],)).typecheck()


def test_evolved_pipeline_reference_matches_manual_composition():
    """A pipeline built from productions is correct-by-construction: its oracle
    equals an INDEPENDENT step-by-step torch re-derivation."""
    src, mid = g.source_prims(), g.middle_prims()
    body = g.compose_productions(
        g.compose_productions(g.Production("add", g.MATRIX, g.MATRIX, (mid["add"],)),
                              g.Production("gelu", g.MATRIX, g.MATRIX, (mid["gelu"],))),
        g.Production("rmsnorm", g.MATRIX, g.MATRIX, (mid["rmsnorm"],)))
    pipe = g.pipeline_from_production(src["input"], body).typecheck()
    ref = g.build_reference(pipe, "fp32")
    x, b, w = g.build_sampler(pipe, {"M": 16, "N": 24}, "fp32")(seed=1)
    fused = ref(x, b, w)
    y = torch.nn.functional.gelu(x.float() + b.float(), approximate="tanh")
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + g._NORM_EPS) * w.float()
    assert torch.allclose(fused, y, atol=1e-5)


# --------------------------------------------------------------------------- #
# Evolution reaches a STRICTLY larger valid-task space (more distinct tasks)
# --------------------------------------------------------------------------- #
def test_evolution_expands_reachable_space():
    off = m.mint_batch(TaskArchive(0), _p, 32, seed=0, evolve_grammar=False)
    on = m.mint_batch(TaskArchive(0), _p, 32, seed=0, evolve_grammar=True)

    fd_off = max(t.features["fusion_depth"] for t in off)
    fd_on = max(t.features["fusion_depth"] for t in on)
    # the fixed templates cap fusion depth at 6 (matmul + 4 middles + 1 reduce);
    # evolution composes deeper, reaching structures unreachable by any base move.
    assert fd_on > fd_off and fd_on >= 7

    # those deep tasks occupy NET-NEW behavior niches (the QD grid extends into
    # higher fusion-depth cells).
    fd_idx = ts.NICHE_FIELDS.index("fusion_depth")
    niches_off = {t.niche_key for t in off}
    niches_on = {t.niche_key for t in on}
    assert max(k[fd_idx] for k in niches_on) > max(k[fd_idx] for k in niches_off)

    # Any task deeper than the base cap is reachable ONLY because grammar evolution
    # composed it (directly via the "grammar" move, or re-cast by "extrapolate" of an
    # evolved pipeline) - so these are strictly-more, distinct, valid minted tasks.
    BASE_MAX_FUSION_DEPTH = 6
    assert not any(t.features["fusion_depth"] > BASE_MAX_FUSION_DEPTH for t in off)
    beyond = [t for t in on if t.features["fusion_depth"] > BASE_MAX_FUSION_DEPTH]
    assert beyond, "evolution should reach beyond the fixed-grammar depth cap"
    assert all(t.move in ("grammar", "extrapolate") for t in beyond)
    assert any(t.move == "grammar" for t in beyond)        # the grammar move itself goes deep
    assert len({t.dedup_key for t in beyond}) == len(beyond)   # all distinct


# --------------------------------------------------------------------------- #
# EVERY evolved task still passes the FULL construction gate + materialize check
# --------------------------------------------------------------------------- #
def test_every_evolved_task_passes_full_construction_gate():
    on = m.mint_batch(TaskArchive(0), _p, 30, seed=3, evolve_grammar=True)
    assert any(t.move == "grammar" for t in on)
    for t in on:
        # re-run the gate independently: type -> heldout -> executes -> finite ->
        # deterministic -> non-constant -> varies-per-axis -> sensitive-to-inputs
        res = m.construction_gate(t.pipeline, t.dtype, t.name, t.family)
        assert res.ok, f"{t.name}: {res.reason}"
        assert not m.is_heldout(t.name, t.family)
        out = t.reference_fn(*t.probe_inputs())
        assert torch.isfinite(out.float()).all()
        assert torch.equal(out, t.reference_fn(*t.probe_inputs()))   # deterministic


def test_evolved_tasks_materialize_with_matching_oracle():
    """Deep, grammar-evolved tasks reconstruct on disk bit-for-bit (the safety
    self-check that gates real training) -> correctness-by-construction end-to-end."""
    pytest.importorskip("kore.tasks.base")
    from kore.openended.materialize import materialize_minted_task

    on = m.mint_batch(TaskArchive(0), _p, 30, seed=3, evolve_grammar=True)
    grammar_tasks = [t for t in on if t.move == "grammar"]
    assert grammar_tasks, "expected grammar-evolved tasks in the batch"
    materialized = 0
    for t in grammar_tasks:
        task = materialize_minted_task(t)      # None iff self-check fails (fail-safe)
        assert task is not None, f"self-check rejected evolved task {t.name}"
        materialized += 1
        inp = g.build_sampler(t.pipeline, dict(t.shape), t.dtype)(0, "cpu")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            f"_ev_ref_{t.task_id}", str(task.dir / "reference.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert torch.allclose(mod.ref_fn(*inp).float(),
                              t.reference_fn(*inp).float(), atol=1e-3, rtol=1e-2)
        assert mod.entry_name == t.name and mod.arity == t.arity
    assert materialized == len(grammar_tasks)


# --------------------------------------------------------------------------- #
# Degenerate composed productions are rejected (fail-safe)
# --------------------------------------------------------------------------- #
def test_degenerate_evolved_production_rejected_by_gate():
    src, mid = g.source_prims(), g.middle_prims()
    # abs -> neg -> relu  ==  relu(-|x|) == 0 everywhere -> constant, must reject.
    deg = g.compose_productions(
        g.compose_productions(g.Production("abs", g.MATRIX, g.MATRIX, (mid["abs"],)),
                              g.Production("neg", g.MATRIX, g.MATRIX, (mid["neg"],))),
        g.Production("relu", g.MATRIX, g.MATRIX, (mid["relu"],)))
    pipe = g.pipeline_from_production(src["input"], deg)
    res = m.construction_gate(pipe, "fp32", pipe.signature().replace("->", "_"),
                             "fusion")
    assert not res.ok and res.reason == "constant_output"

    # and the minter NEVER emits a gate-failing task, even with evolution on.
    on = m.mint_batch(TaskArchive(0), _p, 24, seed=1, evolve_grammar=True)
    for t in on:
        assert m.construction_gate(t.pipeline, t.dtype, t.name, t.family).ok


# --------------------------------------------------------------------------- #
# Dedup + QD niching still hold for the expanded space
# --------------------------------------------------------------------------- #
def test_dedup_and_niche_preserved_under_evolution():
    arch = TaskArchive(0)
    on = m.mint_batch(arch, _p, 24, seed=7, evolve_grammar=True)
    keys = [t.dedup_key for t in on]
    assert len(keys) == len(set(keys))                     # behavioral-hash dedup holds
    # niche-placed into the archive as minted carriers, coverage == distinct niches
    assert arch.coverage() == len({t.niche_key for t in on})
    assert all(c.descriptor.source == "minted" for c in arch.cells_list())

    # re-registering the SAME evolved candidate is dropped as a duplicate
    minter = m.TaskMinter(seed=7, evolve_grammar=True)
    cand = next((c for c in (minter._make_candidate("grammar") for _ in range(300))
                 if c is not None), None)
    assert cand is not None and cand.move == "grammar"
    first = minter.register(cand, None, _p)
    second = minter.register(cand, None, _p)
    assert first is not None and second is None


# --------------------------------------------------------------------------- #
# Self-referential growth + determinism
# --------------------------------------------------------------------------- #
def test_grammar_grows_self_referentially():
    minter = m.TaskMinter(seed=5, evolve_grammar=True)
    base_sigs = {p.signature() for p in g.base_productions()}
    minter.mint_batch(TaskArchive(0), _p, 25)
    assert minter._grammar_promoted >= 1
    assert len(minter._productions) > len(base_sigs)
    # promoted productions are genuinely new (composed, depth >= 2) and reusable
    promoted = [p for p in minter._productions if p.signature() not in base_sigs]
    assert promoted and any(p.depth >= 2 for p in promoted)
    for p in promoted:
        p.typecheck()                                      # every evolved production sound


def test_evolution_is_deterministic():
    b1 = m.mint_batch(TaskArchive(0), _p, 20, seed=5, evolve_grammar=True)
    b2 = m.mint_batch(TaskArchive(0), _p, 20, seed=5, evolve_grammar=True)
    assert [t.task_id for t in b1] == [t.task_id for t in b2]
    assert [t.dedup_key for t in b1] == [t.dedup_key for t in b2]
