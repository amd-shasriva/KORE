"""Contract tests for the dead-code audit: what was REMOVED stays removed, what was
WIRED stays wired, and what was deliberately KEPT-but-unwired stays importable.

An audit found a systematic pattern in KORE: machinery built, tested, exported and
never called. That is worse than no machinery at all, because a reader (or a
reviewer, or a paper) reasonably assumes a module that exists and is tested is doing
something. These tests are the ratchet that stops the pattern from coming back:

  * REMOVED  - the symbol must not resolve. A test that only checks the *tests* were
               deleted would let the code quietly return.
  * WIRED    - a real, non-test caller must exist. Asserting only that a function is
               importable is exactly the mistake that produced the audit.
  * KEPT     - a documented, still-imported public surface stays importable, so a
               future cleanup does not remove something a live caller needs.

Deliberately source-level (``inspect.getsource`` / import probes) rather than
behavioral: the point is to pin the CALL GRAPH, which unit tests do not observe.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

# --------------------------------------------------------------------------- #
# REMOVED: confirmed-dead modules must not come back
# --------------------------------------------------------------------------- #
REMOVED_MODULES = [
    # TorchInductor torch->Triton capture. The midtrain corpus uses the pre-built
    # GPUMODE/KernelBook dataset, so the plan this implemented was never executed.
    "kore.data.synthetic",
    # Offline v1->v2 record upgrade, superseded by build-boundary canonicalization
    # in build_datasets._canonicalize_chat.
    "kore.data.upgrade_v1",
    # hipcc/clang register+diagnostic scraping. Superseded by the rocprofv3-counter
    # path (env -> reward.whitebox -> analysis.roofline.est_occupancy) against the
    # verified per-arch limits in verifier.pmc.
    "kore.verifier.parsers.compiler_output",
]


@pytest.mark.parametrize("modname", REMOVED_MODULES)
def test_removed_module_stays_removed(modname):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(modname)


REMOVED_SYMBOLS = [
    # shadowed the identically-named function in midtrain_corpus.py, which is what
    # everything actually imports
    ("kore.data.mixing", "build_midtrain_corpus"),
    # the standalone co-evolution loop; the live one is controller.CoevolutionController
    ("kore.openended.coevolve", "run_generation"),
    ("kore.openended.coevolve", "run_coevolution"),
    ("kore.openended.coevolve", "Outcome"),
    ("kore.openended.coevolve", "GenerationReport"),
    # online-refit path with no caller; replay_train.train_value_from_groups is the
    # live trainer and documents why Observation-derived rows are the wrong input
    ("kore.value.train_value", "refit_online"),
    ("kore.value.train_value", "row_from_observation"),
]


@pytest.mark.parametrize("modname,symbol", REMOVED_SYMBOLS)
def test_removed_symbol_stays_removed(modname, symbol):
    mod = importlib.import_module(modname)
    assert not hasattr(mod, symbol), (
        f"{modname}.{symbol} was removed as unreachable; if it is being revived, "
        f"add a real non-test caller and move it to the WIRED list"
    )


def test_no_dangling_entrypoints_for_removed_modules():
    """A dangling CLI entrypoint is worse than dead code. ``kore/cli.py`` and
    ``scripts/`` must not reference anything deleted above."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    targets = ["kore.data.synthetic", "kore.data.upgrade_v1",
               "verifier.parsers.compiler_output", "run_coevolution"]
    paths = [root / "kore" / "cli.py"]
    paths += [p for p in (root / "scripts").rglob("*.py")]
    paths += [p for p in (root / "scripts").rglob("*.json")]
    offenders = []
    for p in paths:
        if not p.is_file():
            continue
        text = p.read_text(errors="replace")
        for t in targets:
            if t in text:
                offenders.append(f"{p.relative_to(root)} -> {t}")
    assert not offenders, f"dangling references to removed code: {offenders}"


# --------------------------------------------------------------------------- #
# WIRED: each of these must have a real, non-test caller
# --------------------------------------------------------------------------- #
def test_pairwise_ranker_is_consulted_by_the_reranker():
    """``train_ranking`` ALWAYS fits a pairwise head and the campaign always runs it
    (replay_train.train_value_from_groups). The reranker must READ that head, or the
    ranker is trained on every run and thrown away."""
    from kore.value import rerank

    src = inspect.getsource(rerank._model_utility)
    assert "rank_scores" in src
    assert "_has_fitted_ranker" in src
    # and score_candidates must route through _model_utility (not the pointwise head)
    assert "_model_utility(" in inspect.getsource(rerank.score_candidates)


def test_rank_scores_probe_is_attribute_safe():
    """A model pickled before the ranking head existed has no ``.ranker`` at all;
    probing it with plain attribute access raises AttributeError."""
    from kore.value import rerank

    assert 'getattr(model, "ranker"' in inspect.getsource(rerank._has_fitted_ranker)


def test_top_k_recall_has_a_production_caller():
    """top-k recall is the metric that matches how the model is used (the prefilter
    benches only the top k), so it must be reported by the live trainer."""
    from kore.value import replay_train, train_value

    assert "top_k_recall" in inspect.getsource(train_value.groupwise_top_k_recall)
    src = inspect.getsource(replay_train.train_value_from_groups)
    assert "groupwise_top_k_recall" in src
    assert "heldout_group_top_k_recall" in src


def test_search_seed_is_actually_read():
    """``grpo`` passes ``seed=step`` into ``search_from_kernel``. The seed must reach
    the engine and be read, not accepted and dropped."""
    from kore.search import alphakernel, propose

    assert "seed=seed" in inspect.getsource(propose.search_from_kernel)
    assert "seed=seed" in inspect.getsource(alphakernel.search)
    assert "self.seed" in inspect.getsource(alphakernel._Search.__init__)
    # read at the one genuinely arbitrary decision in the search
    assert "self.seed" in inspect.getsource(alphakernel._Search._tiebreak)
    assert "_tiebreak" in inspect.getsource(alphakernel._Search._select)
    assert "seed" in inspect.getsource(alphakernel._Search.stats)


def test_search_seed_changes_tie_breaking_but_never_the_score():
    """The seed must only reorder EXACT ties; it can never perturb a real score."""
    from kore.search.alphakernel import Node, _Search

    def _engine(seed):
        eng = _Search.__new__(_Search)
        eng.seed = seed
        return eng

    a = Node(source="a", fingerprint="fp-a")
    b = Node(source="b", fingerprint="fp-b")
    # deterministic + reproducible for a fixed seed
    assert _engine(0)._tiebreak(a) == _engine(0)._tiebreak(a)
    # bounded to [0, 1) so it is a valid secondary sort key
    for seed in range(8):
        assert 0.0 <= _engine(seed)._tiebreak(a) < 1.0
    # distinct nodes and distinct seeds genuinely separate
    assert _engine(0)._tiebreak(a) != _engine(0)._tiebreak(b)
    assert any(_engine(s)._tiebreak(a) != _engine(0)._tiebreak(a) for s in range(1, 8))


def test_stale_value_artifact_cannot_be_installed_as_default():
    """The audited ``runs/value/value_model.pkl`` predates the schedule features and
    RAISES on predict. ``load_default_model`` must reject such an artifact instead of
    installing it and crashing the first real prediction."""
    from kore.value import rerank

    assert "_model_is_serviceable" in inspect.getsource(rerank.load_default_model)
    # and scoring itself must never propagate a model failure into a rollout
    assert "_heuristic_scores" in inspect.getsource(rerank.score_candidates)
    assert "except Exception" in inspect.getsource(rerank.score_candidates)


def test_headroom_regret_is_preserved_and_live():
    """The one live symbol in coevolve.py: the standalone loop around it was removed,
    this was not."""
    from kore.openended import controller
    from kore.openended.coevolve import _headroom_regret

    assert controller._headroom_regret is _headroom_regret
    assert "_headroom_regret(" in inspect.getsource(controller)


# --------------------------------------------------------------------------- #
# KEPT-but-unwired: documented public surface with live callers elsewhere
# --------------------------------------------------------------------------- #
KEPT_IMPORTABLE = [
    # Replay-validation metrics. benches_to_best + rank_correlation are read by
    # train_value.train_from_table; top_k_recall is now read by replay_train. All
    # three are the documented measurement-efficiency contract of the module.
    ("kore.value.rerank", "benches_to_best"),
    ("kore.value.rerank", "rank_correlation"),
    ("kore.value.rerank", "top_k_recall"),
    # The pairwise ranking head + its trainer (live: replay_train -> train_ranking).
    ("kore.value.model", "PairwiseRanker"),
    ("kore.value.model", "rank_scores"),
    ("kore.value.train_value", "train_ranking"),
    ("kore.value.train_value", "groupwise_rank_corr"),
    ("kore.value.train_value", "groupwise_top_k_recall"),
    # The P0 rerank contract consumed by alphakernel's PUCT prior, grpo's
    # search_value_prior and data/evolve.py.
    ("kore.value.rerank", "rank_candidates"),
    ("kore.value.rerank", "score_candidates"),
    ("kore.value.rerank", "load_default_model"),
    # Stage-1 SFT mixing (live: data/assemble.py).
    ("kore.data.mixing", "build_multicap_sft"),
    ("kore.data.mixing", "mixture_report"),
]


@pytest.mark.parametrize("modname,symbol", KEPT_IMPORTABLE)
def test_kept_surface_stays_importable(modname, symbol):
    mod = importlib.import_module(modname)
    if symbol == "rank_scores":  # a method, not a module attribute
        assert callable(getattr(mod.ValueModel, symbol, None))
        return
    assert getattr(mod, symbol, None) is not None


def test_synthesize_table_and_groups_kept_for_smoke_testing():
    """Fixture generators with no production caller BY DESIGN: they exist so the
    training path runs with zero real GPU data. Kept, and pinned here so their
    caller-less-ness is not mistaken for the audited pattern."""
    from kore.value.train_value import synthesize_groups, synthesize_table

    assert len(synthesize_table(4, seed=0)) == 4
    groups = synthesize_groups(3, group_size=2, seed=0)
    assert len(groups) == 3 and all(len(g) == 2 for g in groups)
    # they must carry the schedule-bearing `source`, or a model fit on them is blind
    assert all(r.get("source") for g in groups for r in g)
