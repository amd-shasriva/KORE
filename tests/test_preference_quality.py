"""CPU-only tests for baseline-anchored, margin-weighted DPO preference quality.

Guards the three audit fixes in the Stage-2 preference pipeline:

  (1) BASELINE ANCHORING - a speed preference is only a "good" (up-weighted) signal
      when ``chosen`` genuinely BEATS the production vendor baseline. A "win" that is
      faster than the rejected peer but STILL slower than production (e.g. a GEMM at
      0.598x hipBLASLt) is relabelled ``sub_baseline`` (never presented as good) or
      dropped - it is not learned as "this is the kernel to emit".
  (2) MARGIN PRESERVED - build_dpo carries a per-pair ``margin`` + ``weight`` +
      ``anchor`` (and mirrors them in ``_provenance``), and ``normalize`` must NOT
      strip them; near-tie speed pairs inside the measurement-noise band are dropped.
  (3) SIGNAL CONCENTRATION - high-margin compute-bound (gemm/attention/moe/quant)
      pairs are up-weighted over low-headroom elementwise near-ties, and per-family
      low-margin near-ties are capped.

Plus the invariant that the mined reward-hack HARD NEGATIVES (correctness pairs)
survive every anchoring/margin/cap pass untouched, at neutral weight.
"""

from __future__ import annotations

import json

from kore.data.build_datasets import (
    DPOPrefPolicy,
    build_dpo,
    candidate_baseline_speedup,
    resolve_pref_policy,
)
from kore.data.hard_negatives import build_hard_negative_group
from kore.data.normalize import normalize_dpo_row, normalize_file
from kore.data.schemas import RankedGroupRecord

_SRC = "import triton\nimport triton.language as tl\n\ndef {name}(a, b):\n    return a @ b  # {tag}\n"


def _cand(tag, wall_us=None, snr_db=85.0, speedup=None, rank=0, **extra):
    c = {"source": _SRC.format(name="matmul", tag=tag), "wall_us": wall_us,
         "snr_db": snr_db, "rank": rank}
    if speedup is not None:
        c["speedup"] = speedup
    c.update(extra)
    return c


def _speed_group(op, *, chosen_wall, rejected_wall, chosen_speedup=None,
                 rejected_speedup=None, task_id=None):
    """A 2-candidate ranked group emitting exactly the pref [chosen, rejected]."""
    return RankedGroupRecord(
        task_id=task_id or op,
        parent_id="p",
        candidates=[
            _cand("chosen", wall_us=chosen_wall, speedup=chosen_speedup, rank=0),
            _cand("rejected", wall_us=rejected_wall, speedup=rejected_speedup, rank=1),
        ],
        preferences=[[0, 1]],
        operation=op,
    )


def _only(rows):
    assert len(rows) == 1, f"expected exactly one pair, got {len(rows)}"
    return rows[0]


# --------------------------------------------------------------------------- #
# (1) baseline anchoring: drop / relabel sub-baseline "wins"
# --------------------------------------------------------------------------- #
def test_beats_baseline_pair_is_kept_and_labeled():
    # chosen beats production (3.0x > 1.0), faster than the rejected peer -> good signal.
    g = _speed_group("gemm_bf16", chosen_wall=100.0, rejected_wall=200.0,
                     chosen_speedup=3.0, rejected_speedup=1.5)
    row = _only(build_dpo([g]))
    assert row["anchor"] == "beats_baseline"
    assert row["margin"] == 2.0                       # rejected_wall / chosen_wall
    assert row["_provenance"]["chosen_speedup"] == 3.0
    assert row["weight"] > 1.0                         # up-weighted (real production win)


def test_subbaseline_win_is_relabelled_not_a_good_signal():
    # chosen (0.598x) is FASTER than the rejected peer (0.299x) but STILL slower than
    # production -> must not be a "this is good" speed signal.
    g = _speed_group("gemm_bf16", chosen_wall=200.0, rejected_wall=400.0,
                     chosen_speedup=0.598, rejected_speedup=0.299)
    row = _only(build_dpo([g]))                         # default = relabel
    assert row["anchor"] == "sub_baseline"
    assert row["margin"] == 2.0                         # margin still recorded
    assert row["weight"] < 1.0                          # de-emphasised, never up-weighted
    assert row["weight"] == DPOPrefPolicy().subbaseline_weight


def test_subbaseline_win_can_be_dropped():
    g = _speed_group("gemm_bf16", chosen_wall=200.0, rejected_wall=400.0,
                     chosen_speedup=0.598, rejected_speedup=0.299)
    assert build_dpo([g], subbaseline_mode="drop") == []   # dropped entirely


def test_anchoring_off_falls_back_to_among_correct():
    # With anchoring disabled we cannot claim a production win, so a real
    # faster-than-peer ordering is kept but neutral (never up-weighted).
    g = _speed_group("gemm_bf16", chosen_wall=200.0, rejected_wall=400.0,
                     chosen_speedup=0.598, rejected_speedup=0.299)
    row = _only(build_dpo([g], anchor_baseline=False))
    assert row["anchor"] == "among_correct"
    assert row["weight"] == 1.0
    assert row["margin"] == 2.0


def test_unknown_baseline_is_among_correct_not_beats_baseline():
    # No per-candidate speedup on disk (current groups store only wall_us) -> we
    # cannot assert it beats production, so it is among_correct, not beats_baseline.
    g = _speed_group("gemm_bf16", chosen_wall=100.0, rejected_wall=200.0)
    row = _only(build_dpo([g]))
    assert row["anchor"] == "among_correct"
    assert row["margin"] == 2.0
    assert row["weight"] == 1.0


def test_baseline_from_group_baseline_wall_us():
    # Anchoring also works from a group-level production baseline wall (no per-cand
    # speedup needed) -> chosen 300/100 = 3.0x beats baseline.
    g = _speed_group("gemm_bf16", chosen_wall=100.0, rejected_wall=200.0)
    g.baseline_wall_us = 300.0
    row = _only(build_dpo([g]))
    assert row["anchor"] == "beats_baseline"
    assert row["_provenance"]["chosen_speedup"] == 3.0


# --------------------------------------------------------------------------- #
# (2) margin preserved through normalize + near-tie noise filter
# --------------------------------------------------------------------------- #
def test_margin_equals_wall_ratio_and_mirrors_provenance():
    g = _speed_group("gemm_bf16", chosen_wall=100.0, rejected_wall=340.0,
                     chosen_speedup=3.4, rejected_speedup=1.0)
    row = _only(build_dpo([g]))
    assert row["margin"] == 3.4
    # legacy provenance.speedup (chosen-vs-rejected ratio) == margin (back-compat)
    assert row["_provenance"]["speedup"] == row["margin"]
    assert row["_provenance"]["margin"] == row["margin"]
    assert row["_provenance"]["weight"] == row["weight"]


def test_normalize_preserves_provenance_and_margin():
    g = _speed_group("gemm_bf16", chosen_wall=100.0, rejected_wall=340.0,
                     chosen_speedup=3.4, rejected_speedup=1.0)
    row = _only(build_dpo([g]))
    norm, _changed = normalize_dpo_row(row)
    assert norm.get("_provenance") is not None            # NOT stripped
    assert norm["margin"] == row["margin"]
    assert norm["weight"] == row["weight"]
    assert norm["anchor"] == row["anchor"]


def test_normalize_backfills_margin_from_legacy_provenance():
    # A shard written before the trainer-facing top-level fields existed: margin lives
    # only inside _provenance -> normalize lifts it to the top level (an upgrade path).
    legacy = {
        "prompt": [{"role": "user", "content": "opt"}],
        "chosen": [{"role": "assistant", "content": "FULL_KERNEL:\n```python\ndef a():\n    return 1\n```"}],
        "rejected": [{"role": "assistant", "content": "FULL_KERNEL:\n```python\ndef b():\n    return 2\n```"}],
        "_provenance": {"kind": "dpo_group", "speedup": 2.5, "weight": 4.0, "anchor": "beats_baseline"},
    }
    norm, changed = normalize_dpo_row(legacy)
    assert changed
    assert norm["margin"] == 2.5           # lifted from provenance.speedup
    assert norm["weight"] == 4.0
    assert norm["anchor"] == "beats_baseline"
    assert norm["_provenance"] == legacy["_provenance"]   # still present, untouched


def test_normalize_file_roundtrip_keeps_margin(tmp_path):
    g = _speed_group("gemm_bf16", chosen_wall=100.0, rejected_wall=340.0,
                     chosen_speedup=3.4, rejected_speedup=1.0)
    row = _only(build_dpo([g]))
    p = tmp_path / "pairs.jsonl"
    p.write_text(json.dumps(row) + "\n")
    stats = normalize_file(p, in_place=True, backup=False)
    assert stats["kind"] == "dpo"
    back = json.loads(p.read_text().splitlines()[0])
    assert back["_provenance"]["margin"] == 3.4           # margin survives the disk round-trip
    assert back["margin"] == 3.4
    assert back["weight"] == row["weight"]


def test_near_tie_within_noise_band_is_dropped():
    # 1% faster is inside the ~2% measurement-noise band -> not a real preference.
    g = _speed_group("gemm_bf16", chosen_wall=100.0, rejected_wall=101.0,
                     chosen_speedup=3.0, rejected_speedup=2.97)
    assert build_dpo([g]) == []


def test_margin_min_env_gate(monkeypatch):
    g = _speed_group("gemm_bf16", chosen_wall=100.0, rejected_wall=101.0,
                     chosen_speedup=3.0, rejected_speedup=2.97)
    # default 2% band drops it...
    assert build_dpo([g]) == []
    # ...loosening the band via env keeps it.
    monkeypatch.setenv("KORE_PREF_MARGIN_MIN", "0.005")
    row = _only(build_dpo([g]))
    assert abs(row["margin"] - 1.01) < 1e-9


def test_anchor_baseline_env_gate(monkeypatch):
    g = _speed_group("gemm_bf16", chosen_wall=200.0, rejected_wall=400.0,
                     chosen_speedup=0.598, rejected_speedup=0.299)
    monkeypatch.setenv("KORE_PREF_ANCHOR_BASELINE", "0")
    row = _only(build_dpo([g]))
    assert row["anchor"] == "among_correct"    # anchoring disabled -> no sub_baseline label


# --------------------------------------------------------------------------- #
# (3) signal concentration: up-weight compute-bound; cap near-ties
# --------------------------------------------------------------------------- #
def test_compute_bound_pair_outweighs_memory_bound_at_equal_margin():
    kw = dict(chosen_wall=100.0, rejected_wall=340.0, chosen_speedup=3.4, rejected_speedup=1.0)
    gemm = _only(build_dpo([_speed_group("gemm_bf16", **kw)]))
    silu = _only(build_dpo([_speed_group("silu_fp16", **kw)]))       # activation (elementwise)
    assert gemm["anchor"] == silu["anchor"] == "beats_baseline"
    assert gemm["margin"] == silu["margin"]
    assert gemm["weight"] > silu["weight"]      # compute-bound family up-weighted
    assert silu["weight"] > 1.0                 # still a real win, just less concentrated


def test_higher_margin_gets_higher_weight():
    low = _only(build_dpo([_speed_group("gemm_bf16", chosen_wall=100.0, rejected_wall=130.0,
                                        chosen_speedup=1.3, rejected_speedup=1.0)]))
    high = _only(build_dpo([_speed_group("gemm_bf16", chosen_wall=100.0, rejected_wall=500.0,
                                         chosen_speedup=5.0, rejected_speedup=1.0)]))
    assert high["weight"] > low["weight"] >= 1.0


def test_near_tie_pairs_are_capped_per_family():
    # Five low-margin (<1.10) among_correct near-ties in one family + one high-margin
    # substantive pair. A tight cap keeps only the 2 highest-margin near-ties + the
    # substantive one; beats_baseline / high-margin pairs are exempt.
    pol = DPOPrefPolicy(anchor_baseline=False, margin_min=0.0, weighting=True,
                        low_margin=1.10, neartie_cap_frac=0.0, neartie_min_keep=2)
    low = [_speed_group("silu_fp16", chosen_wall=100.0, rejected_wall=100.0 + d)
           for d in (3.0, 4.0, 5.0, 6.0, 7.0)]            # margins 1.03 .. 1.07
    high = _speed_group("silu_fp16", chosen_wall=100.0, rejected_wall=300.0)  # margin 3.0
    rows = build_dpo(low + [high], policy=pol)
    margins = sorted(r["margin"] for r in rows)
    assert len(rows) == 3                                 # 2 capped near-ties + 1 substantive
    assert margins == [1.06, 1.07, 3.0]                   # kept the two highest near-ties


def test_min_keep_floor_leaves_small_families_untouched():
    # The default min_keep (100) means small families / the CPU tests are never capped.
    low = [_speed_group("silu_fp16", chosen_wall=100.0, rejected_wall=100.0 + d,
                        chosen_speedup=0.5, rejected_speedup=0.4)   # sub-baseline, low margin
           for d in (3.0, 4.0, 5.0)]
    rows = build_dpo(low, margin_min=0.0)                 # keep near-ties (no noise drop)
    assert len(rows) == 3


# --------------------------------------------------------------------------- #
# hard negatives (reward-hack correctness pairs) MUST survive untouched
# --------------------------------------------------------------------------- #
def test_hard_negative_pairs_survive_and_are_neutral_weight():
    grp = build_hard_negative_group(_SRC.format(name="matmul", tag="ok"), task=None)
    rows = build_dpo([grp])
    assert len(rows) == 9                                  # all nine hacks preserved
    for r in rows:
        assert r["anchor"] == "correctness"                # never a speed signal
        assert r["weight"] == 1.0                          # neither up- nor down-weighted
        assert r["margin"] is None
        assert r["chosen"] != r["rejected"]
        assert "FULL_KERNEL" in r["chosen"][0]["content"]


def test_hard_negatives_survive_alongside_subbaseline_and_drop_modes():
    grp = build_hard_negative_group(_SRC.format(name="matmul", tag="ok"), task=None)
    sub = _speed_group("gemm_bf16", chosen_wall=200.0, rejected_wall=400.0,
                       chosen_speedup=0.598, rejected_speedup=0.299)
    for mode in ("relabel", "drop", "keep"):
        rows = build_dpo([grp, sub], subbaseline_mode=mode)
        hard = [r for r in rows if r["anchor"] == "correctness"]
        assert len(hard) == 9, f"hard negatives lost under subbaseline_mode={mode}"


# --------------------------------------------------------------------------- #
# helper-level + policy-resolution coverage
# --------------------------------------------------------------------------- #
def test_candidate_baseline_speedup_priority():
    assert candidate_baseline_speedup({"speedup": 2.0, "wall_us": 50.0}) == 2.0
    assert candidate_baseline_speedup({"speedup_vs_baseline": 1.7}) == 1.7
    assert candidate_baseline_speedup({"wall_us": 100.0}, group_baseline_wall=250.0) == 2.5
    assert candidate_baseline_speedup(
        {"wall_us": 100.0}, baseline_speedup_fn=lambda tid, c: 4.0) == 4.0
    assert candidate_baseline_speedup({"wall_us": 100.0}) is None      # unknown baseline


def test_resolve_pref_policy_env_and_overrides(monkeypatch):
    monkeypatch.setenv("KORE_PREF_ANCHOR_MIN", "1.25")
    monkeypatch.setenv("KORE_PREF_SUBBASELINE_MODE", "drop")
    pol = resolve_pref_policy()
    assert pol.anchor_min == 1.25 and pol.subbaseline_mode == "drop"
    # explicit kwargs win over env; None kwargs fall through to env/defaults.
    pol2 = resolve_pref_policy(anchor_min=2.0, subbaseline_mode=None)
    assert pol2.anchor_min == 2.0 and pol2.subbaseline_mode == "drop"


def test_anchor_min_threshold_relabels_marginal_wins():
    # A 1.1x-over-baseline "win" is below a 1.25x bar -> sub_baseline, not beats_baseline.
    g = _speed_group("gemm_bf16", chosen_wall=100.0, rejected_wall=150.0,
                     chosen_speedup=1.1, rejected_speedup=0.73)
    beats = _only(build_dpo([g], anchor_min=1.0))
    assert beats["anchor"] == "beats_baseline"
    marg = _only(build_dpo([g], anchor_min=1.25))
    assert marg["anchor"] == "sub_baseline"


def test_backward_compatible_call_shapes():
    # legacy positional / prompt_fn callers still work and produce valid trl rows.
    g = _speed_group("gemm_bf16", chosen_wall=100.0, rejected_wall=200.0,
                     chosen_speedup=3.0, rejected_speedup=1.5)
    assert build_dpo([g])                                  # (records)
    assert build_dpo([g], prompt_fn=lambda tid: None)      # (records, prompt_fn) + fallback
    row = _only(build_dpo([g]))
    assert {"prompt", "chosen", "rejected"} <= set(row)
    assert isinstance(row["chosen"], list) and row["chosen"][0]["role"] == "assistant"
