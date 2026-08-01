"""CPU-only integrity tests for the five audited defects in the shipped corpus.

Each defect was measured against the on-disk artifacts, so each test locks the
measurable property rather than an implementation detail:

  1. ``baseline_wall_us`` was NEVER populated (0 of 218,732 group candidates), so
     ``build_dpo`` could not resolve an absolute baseline speedup and 45.5% of DPO
     pairs degraded from the margin-weighted ``beats_baseline`` anchor to a flat
     ``among_correct`` / ``correctness`` one.
  2. ``baseline_type`` was set on 7 of 5,605 wins, so a torch_add-relative win was
     indistinguishable from an aiter_flash_attn-relative one and no aggregate
     vendor-beating claim was supportable.
  3. Physically implausible speedups (p99 607x, max 9,381x) persisted into the
     corpus and scored identically to a 10x win under ``curate.quality_score``.
  4. Hard-negative labels were not persisted (0 of 218,732 candidates), so a
     reward-hack pair, a repair pair and an incomparable-wall pair all shipped as
     ``anchor="correctness"`` / ``weight=1.0``.
  5. Gold wins carry a parent-relative speedup that was schema-indistinguishable
     from a baseline-relative one, so the two scales could be pooled by accident.

Plus the backward-compatibility contract: records written BEFORE this change (v1,
v2, and the unversioned shape actually on disk) must still load, with every new
column defaulting to None.

No GPU, no teacher model: pure-python fake env + StubTeacher throughout.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from kore.config import CONFIG
from kore.data import curate as CU
from kore.data import reverify as RV
from kore.data.build_datasets import (
    NEGATIVE_KIND_INCOMPARABLE_WALL,
    NEGATIVE_KIND_REPAIR,
    NEGATIVE_KIND_REWARD_HACK,
    build_dpo,
    build_rft,
    build_sft,
    candidate_baseline_speedup,
)
from kore.data.evolve import EvolveConfig, evolve_task
from kore.data.gen_groups import (
    _evaluate,
    generate_groups,
    paired_baseline_wall_us,
    primary_timing_classification,
)
from kore.data.gen_wins import generate_wins
from kore.data.gold_wins import mint_gold_win
from kore.data.hard_negatives import build_hard_negative_group
from kore.data.schemas import (
    BASELINE_IDENTITY_DECLARED,
    BASELINE_IDENTITY_STATIC,
    BASELINE_KIND_TORCH,
    BASELINE_KIND_TORCH_COMPILE,
    BASELINE_KIND_UNKNOWN,
    BASELINE_KIND_VENDOR,
    CREDIBLE_SPEEDUP_MAX_DEFAULT,
    RECORD_SCHEMA_VERSION,
    SPEEDUP_BASIS_BASELINE,
    SPEEDUP_BASIS_SEED,
    SPEEDUP_BASIS_TRAJECTORY_INITIAL,
    RankedGroupRecord,
    RecordValidationError,
    WinRecord,
    baseline_relative_speedup,
    classify_baseline_kind,
    credible_speedup_max,
    is_credible_win,
    read_jsonl,
    read_jsonl_legacy,
    record_from_dict,
    resolve_baseline_identity,
    speedup_credibility,
    validate_record_dict,
    write_jsonl,
)
from kore.data.teacher import StubTeacher
from kore.policy.format import format_assistant_turn
from kore.reward.reward import Observation

# --------------------------------------------------------------------------- #
# shared stubs
# --------------------------------------------------------------------------- #
_SEED = "def k():\n    # wall=100 snr=999\n    x = 1\n    return x"


def _kernel(wall: float, tag: str, snr: float = 999.0) -> str:
    return f"def k():\n    # wall={wall} snr={snr}\n    x = {tag}\n    return x"


def _task(*, baseline: str = "aiter_rms_norm", operation: str = "rms_norm",
          source_family: str | None = None, dtype: str = "bf16"):
    task = SimpleNamespace(
        task_id=f"gen_{operation}_{dtype}",
        operation=operation,
        dtype=dtype,
        gpu_target="gfx950",
        comparison_baseline=baseline,
        seed_source=_SEED,
    )
    if source_family is not None:
        task.source_family = source_family
    return task


class _MarkerEnv:
    """Verifier stub: wall/snr read from markers in the source; fixed baseline.

    Emits the per-shape maps a real ``KoreEnv`` produces so the paired-baseline
    resolution and the baseline-relative classification are exercised.
    """

    def __init__(self, task=None, baseline_ms: float = 1.0):
        self.task = task
        self.baseline_ms = baseline_ms

    def step(self, source, full_validation=True, multi_shape=True):
        import re

        m = re.search(r"wall=([\d.]+)", source or "")
        wall_ms = float(m.group(1)) / 1000.0 if m else None
        s = re.search(r"snr=([\d.]+)", source or "")
        snr = float(s.group(1)) if s else 999.0
        obs = Observation(compiled=True, validation_passed=True, snr_db=snr,
                          snr_by_shape={"primary": snr}, wall_ms=wall_ms,
                          baseline_ms=self.baseline_ms, dtype="bf16")
        if wall_ms is not None:
            obs.wall_by_shape = {"primary": wall_ms}
            obs.baseline_by_shape = {"primary": self.baseline_ms}
            faster = wall_ms < self.baseline_ms
            obs.timing_classification_by_shape = {
                "primary": "faster" if faster else "tie"}
        obs.cv_pct = 1.0
        obs.baseline_cv_pct = 1.0
        obs.paired_ratio_cv_pct = 1.0
        obs.paired_ci_half_width_pct = 1.0
        return obs


def _teacher(*walls: float):
    """A StubTeacher emitting one improving kernel per call (cycling at the end)."""
    state = {"i": 0}

    def fn(messages):
        walls_list = list(walls) or [50.0]
        wall = walls_list[min(state["i"], len(walls_list) - 1)]
        tag = f"c{state['i']}"
        state["i"] += 1
        return format_assistant_turn(
            "Improve memory throughput.", "Rewrite the inner loop.",
            _kernel(wall, tag))

    return StubTeacher(fn=fn)


def _win(**overrides) -> WinRecord:
    fields = dict(
        task_id="gen_rms_norm_bf16",
        trajectory=[{"role": "user", "content": "opt"},
                    {"role": "assistant", "content": "FULL_KERNEL:\n```python\nk\n```"}],
        initial_wall_us=100.0,
        final_wall_us=50.0,
        speedup=2.0,
        final_source="def k(): pass",
        snr_db=80.0,
        operation="rms_norm",
        speedup_basis=SPEEDUP_BASIS_BASELINE,
    )
    fields.update(overrides)
    return WinRecord(**fields)


# =========================================================================== #
# DEFECT 1 - baseline_wall_us is populated and restores the DPO baseline anchor
# =========================================================================== #
def test_evaluate_carries_the_baseline_wall_time():
    task = _task()
    result = _evaluate(_MarkerEnv(task, baseline_ms=2.0), task,
                       _kernel(1000.0, "a"), CONFIG)
    assert result["wall_us"] == pytest.approx(1000.0)
    # THE defect: this key existed in the persisted candidate dict but _evaluate
    # never produced it, so it was always None.
    assert result["baseline_wall_us"] == pytest.approx(2000.0)
    assert result["speedup"] is not None and result["speedup"] > 1.0


def test_paired_baseline_wall_us_pairs_the_shape_that_reported_the_wall():
    # KoreEnv summarises wall_ms and baseline_ms as INDEPENDENT maxima, so the two
    # can come from different shapes; the anchor must be the baseline of the shape
    # the reported wall belongs to, or baseline/wall is not a measured ratio.
    obs = Observation(compiled=True, wall_ms=4.0, baseline_ms=9.0)
    obs.wall_by_shape = {"small": 1.0, "large": 4.0}
    obs.baseline_by_shape = {"small": 9.0, "large": 6.0}
    assert paired_baseline_wall_us(obs) == pytest.approx(6000.0)
    # No per-shape maps -> fall back to the summary baseline.
    assert paired_baseline_wall_us(
        Observation(compiled=True, wall_ms=1.0, baseline_ms=3.0)) == pytest.approx(3000.0)
    assert paired_baseline_wall_us(Observation(compiled=True)) is None


def test_generated_group_candidates_and_group_carry_the_anchor():
    task = _task()
    records = generate_groups(task, _teacher(500.0, 600.0, 700.0),
                              _MarkerEnv(task, baseline_ms=2.0),
                              n_parents=1, k=3, cfg=CONFIG)
    assert records
    group = records[0]
    assert group.candidates
    for cand in group.candidates:
        assert cand["baseline_wall_us"] == pytest.approx(2000.0)
        assert cand["speedup"] is not None
    # The group also records the single baseline every candidate was timed against,
    # so a candidate whose own timing came back unmeasurable is still anchored.
    assert group.baseline_wall_us == pytest.approx(2000.0)


def test_dpo_pairs_from_a_generated_group_are_baseline_anchored():
    task = _task(operation="gemm", baseline="hipblaslt_gemm")
    records = generate_groups(task, _teacher(200.0, 800.0),
                              _MarkerEnv(task, baseline_ms=2.0),
                              n_parents=1, k=2, cfg=CONFIG)
    rows = build_dpo(records)
    assert rows, "a faster-vs-slower correct pair must produce a DPO row"
    for row in rows:
        # Before the fix every one of these degraded to among_correct at weight 1.0.
        assert row["anchor"] == "beats_baseline"
        assert row["weight"] > 1.0
        assert row["_provenance"]["chosen_speedup"] is not None


def test_group_level_anchor_resolves_a_candidate_without_its_own_speedup():
    # A candidate whose reward-side speedup was unmeasurable still resolves an
    # absolute speedup from the group anchor.
    cand = {"source": "s", "wall_us": 100.0, "rank": 0}
    assert candidate_baseline_speedup(cand) is None
    assert candidate_baseline_speedup(cand, group_baseline_wall=400.0) == 4.0


def test_reverify_group_preserves_the_baseline_anchor():
    task = _task()
    env = _MarkerEnv(task, baseline_ms=2.0)
    group = {"task_id": task.task_id, "parent_id": "p", "type": "ranked_group",
             "candidates": [
                 {"source": _kernel(400.0, "fast"), "wall_us": 400.0,
                  "snr_db": 999.0, "rank": 0},
                 {"source": _kernel(900.0, "slow"), "wall_us": 900.0,
                  "snr_db": 999.0, "rank": 1}],
             "preferences": [[0, 1]]}
    out = RV.reverify_group(group, task, env, CONFIG)
    assert out["baseline_wall_us"] == pytest.approx(2000.0)
    for cand in out["candidates"]:
        assert cand["baseline_wall_us"] == pytest.approx(2000.0)


# =========================================================================== #
# DEFECT 2 - every win-producing path records WHICH baseline it beat
# =========================================================================== #
def test_classify_baseline_kind_separates_vendor_from_torch():
    assert classify_baseline_kind("aiter_flash_attn") == BASELINE_KIND_VENDOR
    assert classify_baseline_kind("hipblaslt_gemm") == BASELINE_KIND_VENDOR
    assert classify_baseline_kind("torch_add") == BASELINE_KIND_TORCH
    assert classify_baseline_kind("torch_compile_fusion") == "torch_compile"
    # Never promoted to a vendor claim on a guess.
    assert classify_baseline_kind(None) == BASELINE_KIND_UNKNOWN
    assert classify_baseline_kind("   ") == BASELINE_KIND_UNKNOWN
    assert classify_baseline_kind("mystery_bar") == BASELINE_KIND_UNKNOWN


def test_declared_identity_is_reported_as_declared():
    identity = resolve_baseline_identity(_task(baseline="aiter_rms_norm"))
    assert identity == {"baseline_type": "aiter_rms_norm",
                        "baseline_kind": BASELINE_KIND_VENDOR,
                        "baseline_identity_source": BASELINE_IDENTITY_DECLARED}
    # A task with no declaration at all stays honest rather than defaulting.
    assert resolve_baseline_identity(SimpleNamespace()) == {
        "baseline_type": None, "baseline_kind": BASELINE_KIND_UNKNOWN,
        "baseline_identity_source": BASELINE_IDENTITY_DECLARED}


def test_generated_op_identity_uses_the_real_baseline_resolver(monkeypatch):
    # gemm_fusion resolves to the hipBLASLt vendor GEMM even though the task
    # DECLARES a torch bar -> the resolved path wins and is labelled as such.
    monkeypatch.delenv("KORE_USE_VENDOR_BASELINE", raising=False)
    monkeypatch.delenv("KORE_COMPILE_BASELINE", raising=False)
    fused = _task(operation="gemm_bias_gelu", baseline="torch_gemm_bias_gelu",
                  source_family="gemm_fusion")
    identity = resolve_baseline_identity(fused)
    assert identity["baseline_kind"] == BASELINE_KIND_VENDOR
    assert identity["baseline_identity_source"] == BASELINE_IDENTITY_STATIC
    assert identity["baseline_type"] == "torch_gemm_bias_gelu"  # declaration kept

    # A fusion op AITER does not ship a fused kernel for resolves to a torch bar,
    # so it is never labelled vendor (the resolver only says vendor for gemm_fusion
    # and the two gated activations). WHICH torch bar depends on the compile
    # switch, which now defaults ON so a fusion task is graded against a fused
    # reference rather than an inflated unfused one -- assert the contract, not
    # whichever default happens to be current.
    plain = _task(operation="mul_tanh", baseline="torch_mul_tanh",
                  source_family="fusion")
    assert resolve_baseline_identity(plain)["baseline_kind"] == BASELINE_KIND_TORCH_COMPILE

    monkeypatch.setenv("KORE_COMPILE_BASELINE", "0")
    assert resolve_baseline_identity(plain)["baseline_kind"] == BASELINE_KIND_TORCH
    monkeypatch.delenv("KORE_COMPILE_BASELINE", raising=False)

    gated = _task(operation="silu_mul", baseline="torch_silu_mul",
                  source_family="fusion")
    assert resolve_baseline_identity(gated)["baseline_kind"] == BASELINE_KIND_VENDOR

    # The vendor path is env-gated, and the recorded kind follows the gate: with
    # the vendor bar off, a gemm_fusion task falls back to a torch bar and must
    # never still be labelled vendor. Which torch bar depends on the compile
    # switch, so pin both rather than whichever default is current.
    monkeypatch.setenv("KORE_USE_VENDOR_BASELINE", "0")
    assert resolve_baseline_identity(fused)["baseline_kind"] == BASELINE_KIND_TORCH_COMPILE

    monkeypatch.setenv("KORE_COMPILE_BASELINE", "0")
    assert resolve_baseline_identity(fused)["baseline_kind"] == BASELINE_KIND_TORCH


def test_generate_wins_records_baseline_identity_and_classification():
    task = _task(baseline="aiter_rms_norm")
    env = _MarkerEnv(task, baseline_ms=1.0)
    recs = generate_wins(task, _teacher(70.0, 50.0), env, gens=2)
    assert len(recs) == 1
    win = recs[0]
    assert win.baseline_type == "aiter_rms_norm"
    assert win.baseline_kind == BASELINE_KIND_VENDOR
    assert win.baseline_identity_source == BASELINE_IDENTITY_DECLARED
    assert win.timing_classification == "faster"
    assert win.baseline_wall_us == pytest.approx(1000.0)


def _evolve_cfg(**overrides) -> EvolveConfig:
    # Bench the generator's candidate every generation (the operator mutations of the
    # seed keep the seed's wall marker, so they never improve).
    fields = dict(seed=0, islands=1, candidates_per_gen=2, prefilter_k=2)
    fields.update(overrides)
    return EvolveConfig(**fields)


def test_evolve_win_records_baseline_identity_and_timing_provenance():
    task = _task(operation="gemm", baseline="hipblaslt_gemm")
    task.seed_source = _kernel(900.0, "seed")
    res = evolve_task(task, _teacher(400.0, 300.0, 200.0),
                      _MarkerEnv(task, baseline_ms=1.0), generations=3,
                      cfg=_evolve_cfg())
    assert res.wins, "a vendor-beating evolve win was expected"
    win = res.wins[0]
    assert win.baseline_type == "hipblaslt_gemm"
    assert win.baseline_kind == BASELINE_KIND_VENDOR
    assert win.baseline_wall_us == pytest.approx(1000.0)
    assert win.timing_classification == "faster"
    assert win.final_cv_pct == pytest.approx(1.0)


def test_reverify_win_records_baseline_identity():
    task = _task(baseline="torch_rms_norm")
    out = RV.reverify_win({"final_source": _kernel(500.0, "w"), "speedup": 9.9},
                          task, _MarkerEnv(task, baseline_ms=1.0), CONFIG)
    assert out is not None
    assert out["baseline_type"] == "torch_rms_norm"
    assert out["baseline_kind"] == BASELINE_KIND_TORCH
    assert out["baseline_identity_source"] == BASELINE_IDENTITY_DECLARED
    assert out["timing_classification"] == "faster"
    assert out["baseline_wall_us"] == pytest.approx(1000.0)
    assert out["final_cv_pct"] == pytest.approx(1.0)


def test_groups_record_which_baseline_their_candidates_were_timed_against():
    task = _task(baseline="aiter_rms_norm")
    env = _MarkerEnv(task, baseline_ms=2.0)
    groups = generate_groups(task, _teacher(500.0, 800.0), env,
                             n_parents=1, k=2, cfg=CONFIG)
    assert groups[0].baseline_type == "aiter_rms_norm"
    assert groups[0].baseline_kind == BASELINE_KIND_VENDOR
    # ...and it reaches the emitted DPO provenance, so a beats_baseline pair states
    # WHAT it beat instead of leaving the reader to guess.
    row = build_dpo(groups)[0]
    assert row["_provenance"]["baseline_kind"] == BASELINE_KIND_VENDOR
    assert row["_provenance"]["baseline_type"] == "aiter_rms_norm"

    # A re-verified group keeps the identity of the measurement that re-graded it.
    out = RV.reverify_group(groups[0].to_dict(), task, env, CONFIG)
    assert out["baseline_type"] == "aiter_rms_norm"
    assert out["baseline_kind"] == BASELINE_KIND_VENDOR


def test_gold_minted_win_inherits_the_groups_baseline_identity():
    # gold_wins reads baseline_type off the group; groups never carried it, which is
    # why 3,000 gold-minted wins shipped without one. It now flows through.
    group = {
        "task_id": "gen_rms_norm_bf16", "operation": "rms_norm", "arch": "gfx950",
        "baseline_type": "aiter_rms_norm", "baseline_kind": BASELINE_KIND_VENDOR,
        "candidates": [
            {"source": "def best(): pass", "wall_us": 100.0, "snr_db": 90.0, "rank": 0},
            {"source": "def parent(): pass", "wall_us": 150.0, "snr_db": 90.0, "rank": 1},
        ],
    }
    gold = mint_gold_win(group)
    assert gold.baseline_type == "aiter_rms_norm"


def test_win_provenance_exposes_baseline_identity_to_curation():
    prov = build_sft([_win(baseline_type="aiter_rms_norm",
                           baseline_kind=BASELINE_KIND_VENDOR)])[0]["_provenance"]
    assert prov["baseline_type"] == "aiter_rms_norm"
    assert prov["baseline_kind"] == BASELINE_KIND_VENDOR
    assert prov["speedup_basis"] == SPEEDUP_BASIS_BASELINE


def test_timing_classification_requires_every_shape_when_no_primary():
    obs = Observation(compiled=True)
    # A partial win must never be reported as faster, whichever shape comes first.
    obs.timing_classification_by_shape = {"a": "faster", "b": "tie"}
    assert primary_timing_classification(obs, SimpleNamespace()) == "tie"
    obs.timing_classification_by_shape = {"b": "tie", "a": "faster"}
    assert primary_timing_classification(obs, SimpleNamespace()) == "tie"
    obs.timing_classification_by_shape = {"a": "faster", "b": "faster"}
    assert primary_timing_classification(obs, SimpleNamespace()) == "faster"
    assert primary_timing_classification(Observation(compiled=True), None) is None
    # A resolvable primary shape keys directly off it.
    obs.timing_classification_by_shape = {"primary": "tie", "large": "faster"}
    task = SimpleNamespace(shapes=[SimpleNamespace(name="primary")])
    assert primary_timing_classification(obs, task) == "tie"


# =========================================================================== #
# DEFECT 3 - implausible speedups are flagged, kept, and excluded from exemplars
# =========================================================================== #
def test_speedup_credibility_flags_without_altering_the_value():
    assert speedup_credibility(9381.0) == {"speedup_exceeds_credible": True,
                                           "credible_speedup_max": 10.0}
    assert speedup_credibility(2.2)["speedup_exceeds_credible"] is False
    assert speedup_credibility(10.0)["speedup_exceeds_credible"] is False
    # Unmeasurable -> unknown, never a False "this is credible" claim.
    assert speedup_credibility(None)["speedup_exceeds_credible"] is None


def test_credible_ceiling_is_configurable(monkeypatch):
    monkeypatch.delenv("KORE_CREDIBLE_SPEEDUP_MAX", raising=False)
    assert credible_speedup_max() == CREDIBLE_SPEEDUP_MAX_DEFAULT
    # Shares the reward module's own excessive-speedup number by default.
    assert credible_speedup_max(CONFIG) == CONFIG.excessive_speedup_flag
    monkeypatch.setenv("KORE_CREDIBLE_SPEEDUP_MAX", "25")
    assert credible_speedup_max() == 25.0
    assert credible_speedup_max(threshold=3.0) == 3.0  # explicit arg wins


def test_reverify_flags_an_implausible_speedup_instead_of_deleting_it():
    # A sequence/SSM-style baseline: the candidate really is ~2000x the Python-loop
    # reference. The record must survive with the measured value AND the flag.
    task = _task(operation="ssm_scan", baseline="torch_ssm_scan")
    out = RV.reverify_win({"final_source": _kernel(500.0, "w"), "speedup": 1.0},
                          task, _MarkerEnv(task, baseline_ms=1000.0), CONFIG)
    assert out is not None, "an implausible win must not be silently dropped"
    assert out["speedup"] == pytest.approx(2000.0)
    assert out["speedup_exceeds_credible"] is True
    assert out["credible_speedup_max"] == 10.0


def test_quality_score_no_longer_ties_an_implausible_win_with_a_credible_one():
    implausible = build_sft([_win(speedup=9381.0,
                                  speedup_exceeds_credible=True,
                                  credible_speedup_max=10.0)])[0]
    at_ceiling = build_sft([_win(speedup=10.0)])[0]
    credible = build_sft([_win(speedup=3.0)])[0]
    # The old log(min(sp, 10)) clamp scored the 9,381x row EXACTLY like the 10x one.
    assert CU.quality_score(implausible) < CU.quality_score(credible)
    assert CU.quality_score(implausible) < CU.quality_score(at_ceiling)
    assert CU.is_implausible_win(implausible) is True
    assert CU.is_implausible_win(credible) is False


def test_shipped_unflagged_win_is_still_graded_from_its_speedup():
    # The already-shipped wins carry no flag; the verdict is re-derived so they are
    # not grandfathered in as credible.
    legacy = build_sft([_win(speedup=607.0, speedup_exceeds_credible=None,
                             credible_speedup_max=None)])[0]
    assert CU.is_implausible_win(legacy) is True


def test_curate_excludes_implausible_wins_from_the_sft_mixture():
    rows = build_sft([_win(speedup=3.0, final_source="a"),
                      _win(speedup=9381.0, speedup_exceeds_credible=True,
                           final_source="b")])
    kept, stats = CU.curate(rows, family_cap_frac=None)
    assert stats["dropped_implausible_wins"] == 1
    assert len(kept) == 1
    assert kept[0]["_provenance"]["speedup"] == 3.0
    # ...and it is a deliberate policy, not a hard-coded deletion.
    kept_all, stats_all = CU.curate(rows, family_cap_frac=None,
                                    drop_implausible_wins=False)
    assert len(kept_all) == 2 and stats_all["dropped_implausible_wins"] == 0
    # Repairs are never touched by the high-end gate.
    repair_rows = [{"messages": [{"role": "user", "content": "x"}],
                    "_provenance": {"kind": "repair", "speedup": 9381.0}}]
    assert CU.filter_implausible_wins(repair_rows)[0] == repair_rows


def test_build_rft_excludes_implausible_wins():
    # RFT rows carry no provenance, so curation cannot filter them downstream -
    # the exemplar gate has to be at this boundary.
    rows = build_rft([_win(speedup=3.0, final_source="a"),
                      _win(speedup=9381.0, speedup_exceeds_credible=True,
                           final_source="b")])
    assert len(rows) == 1


def test_dpo_does_not_up_weight_an_implausible_absolute_speedup():
    def group(chosen_speedup):
        return RankedGroupRecord(
            task_id="gemm_bf16", parent_id="p", operation="gemm",
            candidates=[{"source": "chosen", "wall_us": 100.0, "snr_db": 90.0,
                         "rank": 0, "speedup": chosen_speedup},
                        {"source": "rejected", "wall_us": 300.0, "snr_db": 90.0,
                         "rank": 1, "speedup": chosen_speedup / 3.0}],
            preferences=[[0, 1]])

    credible = build_dpo([group(5.0)])[0]
    implausible = build_dpo([group(9381.0)])[0]
    assert credible["anchor"] == implausible["anchor"] == "beats_baseline"
    assert credible["weight"] > 1.0
    assert implausible["weight"] == 1.0
    assert implausible["_provenance"]["speedup_exceeds_credible"] is True
    assert credible["_provenance"]["speedup_exceeds_credible"] is False


def test_win_speedup_stats_pools_only_baseline_relative_credible_wins():
    rows = build_sft([
        _win(speedup=2.0, final_source="a"),
        _win(speedup=3.0, final_source="b"),
        _win(speedup=1.5, final_source="c"),
        _win(speedup=9381.0, speedup_exceeds_credible=True, final_source="d"),
        _win(speedup=5.0, speedup_basis=SPEEDUP_BASIS_TRAJECTORY_INITIAL,
             final_source="e"),
        _win(speedup=331.0, speedup_basis=None, final_source="f"),
    ])
    stats = CU.win_speedup_stats(rows)
    assert stats["n_wins"] == 6
    assert stats["n_pooled"] == 3
    assert stats["n_excluded_implausible"] == 2      # 9381x and the 331x legacy row
    assert stats["n_excluded_not_baseline_relative"] == 1
    assert stats["median_speedup"] == 2.0
    assert stats["max_speedup"] == 3.0


# =========================================================================== #
# DEFECT 4 - negatives are labelled, so the three kinds stay distinguishable
# =========================================================================== #
def _neg_group(rejected_extra: dict) -> RankedGroupRecord:
    return RankedGroupRecord(
        task_id="gemm_bf16", parent_id="p", operation="gemm",
        candidates=[{"source": "chosen", "wall_us": 100.0, "snr_db": 90.0, "rank": 0},
                    {"source": "rejected", "snr_db": 10.0, "rank": 1,
                     **rejected_extra}],
        preferences=[[0, 1]])


def test_reward_hack_negatives_are_labelled_in_the_emitted_dpo_rows():
    rows = build_dpo([build_hard_negative_group("def m(a, b):\n    return a @ b\n")])
    assert len(rows) == 9
    labels = set()
    for row in rows:
        assert row["anchor"] == "correctness"      # still never a speed signal
        assert row["weight"] == 1.0                # still neutral, never dropped
        assert row["negative_kind"] == NEGATIVE_KIND_REWARD_HACK
        prov = row["_provenance"]
        assert prov["negative_kind"] == NEGATIVE_KIND_REWARD_HACK
        assert prov["negative_label"].startswith("reward_hack:")
        assert prov["hard_negative"] == prov["negative_label"]
        labels.add(prov["negative_label"])
    assert len(labels) == 9, "each of the nine hack kinds keeps its own label"


def test_the_three_correctness_negative_kinds_are_distinguishable():
    hack = build_dpo([_neg_group({"hard_negative": "reward_hack:vendor_call"})])[0]
    repair = build_dpo([_neg_group({"failure_class": "snr_fail"})])[0]
    incomparable = build_dpo([_neg_group({})])[0]
    # All three previously shipped as an identical anchor="correctness"/weight=1.0 row.
    assert hack["anchor"] == repair["anchor"] == incomparable["anchor"] == "correctness"
    kinds = [hack["negative_kind"], repair["negative_kind"],
             incomparable["negative_kind"]]
    assert kinds == [NEGATIVE_KIND_REWARD_HACK, NEGATIVE_KIND_REPAIR,
                     NEGATIVE_KIND_INCOMPARABLE_WALL]
    assert len(set(kinds)) == 3
    assert repair["_provenance"]["negative_label"] == "snr_fail"
    assert incomparable["_provenance"]["negative_label"] is None


def test_speed_pairs_carry_no_negative_kind():
    group = RankedGroupRecord(
        task_id="gemm_bf16", parent_id="p", operation="gemm",
        candidates=[{"source": "chosen", "wall_us": 100.0, "snr_db": 90.0, "rank": 0},
                    {"source": "rejected", "wall_us": 300.0, "snr_db": 90.0, "rank": 1}],
        preferences=[[0, 1]])
    row = build_dpo([group])[0]
    assert row["negative_kind"] is None
    assert row["_provenance"]["negative_kind"] is None


def test_hard_negative_label_survives_a_strict_shard_round_trip(tmp_path):
    group = build_hard_negative_group("def m(a, b):\n    return a @ b\n")
    group.task_id = "gemm_bf16"
    path = tmp_path / "hard.jsonl"
    write_jsonl(path, [group], validate_records=True,
                expected_task_id="gemm_bf16", expected_type="ranked_group")
    loaded = read_jsonl(path, mode="generic_training_row")[0]
    labels = [c.get("hard_negative") for c in loaded.candidates[1:]]
    assert all(isinstance(label, str) and label.startswith("reward_hack:")
               for label in labels)
    # The label is now a validated column, so a non-string can never ship as one.
    bad = group.to_dict()
    bad["candidates"][1]["hard_negative"] = 7
    with pytest.raises(RecordValidationError, match="hard_negative"):
        validate_record_dict(bad)


# =========================================================================== #
# DEFECT 5 - the speedup scale is explicit, so the two can never be pooled
# =========================================================================== #
def test_gen_wins_declares_a_trajectory_relative_basis():
    task = _task()
    recs = generate_wins(task, _teacher(70.0, 50.0), _MarkerEnv(task), gens=2)
    win = recs[0]
    # The stored footer is initial/final (self-relative) even though admission is
    # baseline-relative, so it must NOT be pooled as a production-relative number.
    assert win.speedup_basis == SPEEDUP_BASIS_TRAJECTORY_INITIAL
    assert win.speedup == pytest.approx(2.0)
    assert baseline_relative_speedup(win) is None


def test_evolve_declares_baseline_basis_under_the_vendor_gate_and_seed_basis_without():
    task = _task(operation="gemm", baseline="hipblaslt_gemm")
    task.seed_source = _kernel(900.0, "seed")
    env = _MarkerEnv(task, baseline_ms=1.0)
    gated = evolve_task(task, _teacher(400.0, 300.0), env, generations=2,
                        cfg=_evolve_cfg())
    assert gated.wins
    assert gated.wins[0].speedup_basis == SPEEDUP_BASIS_BASELINE
    assert baseline_relative_speedup(gated.wins[0]) == gated.wins[0].speedup

    seeded = evolve_task(task, _teacher(400.0, 300.0), env, generations=2,
                         cfg=_evolve_cfg(require_vendor_win=False))
    assert seeded.wins
    assert seeded.wins[0].speedup_basis == SPEEDUP_BASIS_SEED
    assert baseline_relative_speedup(seeded.wins[0]) is None


def test_reverified_win_declares_a_baseline_basis():
    task = _task()
    out = RV.reverify_win({"final_source": _kernel(500.0, "w"), "speedup": 9.9},
                          task, _MarkerEnv(task, baseline_ms=1.0), CONFIG)
    assert out["speedup_basis"] == SPEEDUP_BASIS_BASELINE
    assert baseline_relative_speedup(out) == out["speedup"]


def test_gold_minted_parent_relative_win_is_never_pooled_as_baseline_relative():
    group = {
        "task_id": "gen_rms_norm_bf16", "operation": "rms_norm", "arch": "gfx950",
        "candidates": [
            {"source": "def best(): pass", "wall_us": 100.0, "snr_db": 90.0, "rank": 0},
            {"source": "def parent(): pass", "wall_us": 150.0, "snr_db": 90.0, "rank": 1},
        ],
    }
    gold = mint_gold_win(group)
    assert gold is not None
    assert gold.speedup == pytest.approx(1.5)
    # gold_wins puts the PARENT's wall in baseline_wall_us and its ratio is
    # parent-relative; it declares no basis, so it can never be pooled with the
    # baseline-relative numbers.
    assert gold.baseline_wall_us == pytest.approx(150.0)
    assert gold.speedup_basis is None
    assert baseline_relative_speedup(gold) is None
    stats = CU.win_speedup_stats(build_sft([gold]))
    assert stats["n_wins"] == 1 and stats["n_pooled"] == 0
    assert stats["n_excluded_not_baseline_relative"] == 1


def test_a_typo_can_never_ship_as_a_new_basis_or_baseline_kind():
    for key, value in (("speedup_basis", "vendor_relative"),
                       ("baseline_kind", "aiter"),
                       ("baseline_identity_source", "runtime")):
        bad = _win().to_dict()
        bad[key] = value
        with pytest.raises(RecordValidationError, match=key):
            validate_record_dict(bad)
    flag = _win().to_dict()
    flag["speedup_exceeds_credible"] = "yes"
    with pytest.raises(RecordValidationError, match="speedup_exceeds_credible"):
        validate_record_dict(flag)


# =========================================================================== #
# BACKWARD COMPATIBILITY - records written before this change still load
# =========================================================================== #
# Byte-shape of a WinRecord written by the v1 writer (no timing block at all).
_V1_WIN = {
    "task_id": "gen_rms_norm_bf16",
    "trajectory": [{"role": "user", "content": "optimize"},
                   {"role": "assistant", "content": "FULL_KERNEL:\n```python\nk\n```"}],
    "initial_wall_us": 100.0, "final_wall_us": 50.0, "speedup": 2.0,
    "final_source": "def k(): pass", "snr_db": 80.0, "type": "win",
    "gpu": "gfx950", "operation": "rms_norm", "arch": "gfx950", "shape": None,
    "schema_version": 1,
}
# Byte-shape of a WinRecord written by the v2 (timing-rigor) writer, i.e. exactly
# what the shipped 5,605-win corpus contains - and nothing this change added.
_V2_WIN = {
    **_V1_WIN,
    "schema_version": 2,
    "baseline_type": "aiter_rms_norm", "baseline_wall_us": 90.0,
    "final_cv_pct": 1.2, "baseline_cv_pct": 0.9, "paired_ratio_cv_pct": 1.4,
    "paired_ci_half_width_pct": 2.1, "admit_cv_threshold_pct": 3.0,
    "timing_classification": "faster",
}
# The unversioned shape actually measured on disk in data/full14b/*: no
# schema_version, no timing block, candidates with only source/wall_us/snr_db/rank.
_UNVERSIONED_WIN = {k: v for k, v in _V1_WIN.items() if k != "schema_version"}
_UNVERSIONED_GROUP = {
    "task_id": "gen_sqrt_bf16", "parent_id": "p", "type": "ranked_group",
    "gpu": "gfx950", "operation": "sqrt", "arch": "gfx950", "shape": None,
    "candidates": [{"source": "a", "wall_us": 10.0, "snr_db": 90.0, "rank": 0},
                   {"source": "b", "wall_us": 20.0, "snr_db": 90.0, "rank": 1}],
    "preferences": [[0, 1]],
}

_NEW_WIN_COLUMNS = ("baseline_kind", "baseline_identity_source", "speedup_basis",
                    "speedup_exceeds_credible", "credible_speedup_max")


def test_record_schema_version_is_unchanged():
    # The new columns are OPTIONAL, so v2 stays v2 - a bump would invalidate every
    # existing shard receipt (parallel_datagen pins record_schema_version == this).
    assert RECORD_SCHEMA_VERSION == 2


@pytest.mark.parametrize("raw", [_V1_WIN, _V2_WIN])
def test_pre_change_win_records_still_validate_and_load(raw):
    assert validate_record_dict(dict(raw)) is not None
    win = record_from_dict(dict(raw))
    assert isinstance(win, WinRecord)
    assert win.speedup == 2.0
    # Every column this change added defaults to None; nothing is inferred.
    for column in _NEW_WIN_COLUMNS:
        assert getattr(win, column) is None
    assert baseline_relative_speedup(win) is None      # basis undeclared -> not pooled
    assert is_credible_win(win) is True                # 2.0x, re-derived


@pytest.mark.parametrize("version", [1, 2])
def test_pre_change_group_record_still_validates_and_loads(version):
    versioned = {**_UNVERSIONED_GROUP, "schema_version": version}
    assert validate_record_dict(dict(versioned)) is not None
    group = record_from_dict(versioned)
    assert isinstance(group, RankedGroupRecord)
    assert group.baseline_wall_us is None               # new column, absent -> None
    assert [c["rank"] for c in group.candidates] == [0, 1]
    assert all("hard_negative" not in c for c in group.candidates)


def test_unversioned_shipped_records_still_load_through_the_legacy_reader(tmp_path):
    path = tmp_path / "wins.jsonl"
    path.write_text(json.dumps(_UNVERSIONED_WIN) + "\n"
                    + json.dumps(_UNVERSIONED_GROUP) + "\n", encoding="utf-8")
    win, group = read_jsonl_legacy(path)
    assert isinstance(win, WinRecord) and win.speedup == 2.0
    assert isinstance(group, RankedGroupRecord)
    for column in _NEW_WIN_COLUMNS:
        assert getattr(win, column) is None


def test_new_columns_round_trip_through_a_strict_shard(tmp_path):
    win = _win(task_id="gen_rms_norm_bf16", baseline_type="aiter_rms_norm",
               baseline_kind=BASELINE_KIND_VENDOR,
               baseline_identity_source=BASELINE_IDENTITY_DECLARED,
               speedup_basis=SPEEDUP_BASIS_BASELINE,
               speedup_exceeds_credible=False, credible_speedup_max=10.0)
    path = tmp_path / "wins.jsonl"
    write_jsonl(path, [win], validate_records=True,
                expected_task_id="gen_rms_norm_bf16", expected_type="win")
    assert read_jsonl(path, mode="generic_training_row")[0] == win
    raw = json.loads(path.read_text().splitlines()[0])
    assert raw["schema_version"] == 2
    assert raw["speedup_basis"] == SPEEDUP_BASIS_BASELINE


def test_a_v2_reader_can_still_interpret_a_record_carrying_the_new_columns():
    # Forward compatibility: unknown keys are retained, not rejected, so a shard
    # written after this change is still readable by the pre-change validator path.
    forward = {**_V2_WIN, "speedup_basis": SPEEDUP_BASIS_BASELINE,
               "baseline_kind": BASELINE_KIND_VENDOR,
               "baseline_identity_source": BASELINE_IDENTITY_DECLARED,
               "speedup_exceeds_credible": False, "credible_speedup_max": 10.0}
    win = record_from_dict(forward)
    assert win.speedup_basis == SPEEDUP_BASIS_BASELINE
    assert baseline_relative_speedup(win) == 2.0
