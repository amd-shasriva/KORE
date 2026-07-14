"""Turn raw KORE records into training-ready, HF-style chat datasets.

  - ``build_sft``: repair turns + winning trajectories -> {"messages": [...]}.
  - ``build_dpo``: ranked groups -> {"prompt", "chosen", "rejected"} preference
    pairs (chosen/rejected are assistant completions wrapping each candidate).
  - ``build_rft``: rejection-sampled SFT on the best candidate of each group and
    on winning trajectories -> {"messages": [...]}.

Plus corpus hygiene:
  - ``dedup_by_source_hash``: drop records with a duplicate representative source.
  - ``leakage_split``: split by a grouping key (default operation-family+arch) so
    the same op family never appears in more than one of train/val/test.

Everything here is PURE (no GPU / teacher) and unit-testable.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from typing import Any, Iterable, Optional

from kore.data.prompts import (
    SYSTEM_PROMPT,
    extract_kernel,
    normalize_assistant,
    wrap_full_kernel,
)
from kore.data.schemas import (
    RepairRecord,
    RankedGroupRecord,
    WinRecord,
    record_from_dict,
)
from kore.env.replay import kernel_hash
from kore.obs import get_logger

log = get_logger("data.build_datasets")


# --- coercion helpers ---
def _as_record(rec: Any):
    if isinstance(rec, (RepairRecord, RankedGroupRecord, WinRecord)):
        return rec
    if isinstance(rec, dict) and rec.get("type"):
        return record_from_dict(rec)
    return rec


# Canonical FULL_KERNEL completion wrapper (single source of truth: policy.format).
_wrap_full_kernel = wrap_full_kernel


def _generic_prompt(task_id: str, gpu: str = "gfx950") -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Optimize the Triton kernel for task '{task_id}' on {gpu}. "
                "Output the complete kernel under the FULL_KERNEL contract."
            ),
        },
    ]


# --- provenance (Pillar 5): auditable per-row metadata for curation ---
def _prov_common(rec: Any) -> dict:
    d = rec.to_dict() if hasattr(rec, "to_dict") else {}
    return {
        "task_id": getattr(rec, "task_id", None) or d.get("task_id"),
        "operation": getattr(rec, "operation", None) or d.get("operation"),
        "arch": (getattr(rec, "arch", None) or d.get("arch")
                 or getattr(rec, "gpu", None) or d.get("gpu")),
        "shape": getattr(rec, "shape", None) or d.get("shape"),
    }


def _prov_win(rec: Any) -> dict:
    p = _prov_common(rec)
    p.update({"kind": "win", "verified": True, "baseline": "measured",
              "speedup": getattr(rec, "speedup", None),
              "snr_db": getattr(rec, "snr_db", None)})
    return p


def _prov_repair(rec: Any) -> dict:
    p = _prov_common(rec)
    p.update({"kind": "repair", "verified": True,
              "failure_class": getattr(rec, "failure_class", None),
              "snr_db": getattr(rec, "child_snr_db", None)})
    return p


# --- SFT ---
def _canonicalize_chat(messages: list) -> list:
    """Re-render a chat's KORE system + assistant turns into the canonical contract.

    The build boundary is the single choke point for contract unification (Pillar 0):
    normalizing here canonicalizes fresh datagen wins AND reused v1 shards (legacy
    ``<think>/<answer>`` repairs, raw-teacher ``CHANGE:`` wins) with NO disk mutation,
    so training never sees a non-canonical kernel turn. No-op on already-canonical or
    non-kernel (retention) content.
    """
    out = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        role, content = m.get("role"), m.get("content", "")
        if role == "system" and isinstance(content, str) \
                and content.lstrip().startswith("You are KORE, an expert AMD GPU kernel engineer") \
                and content != SYSTEM_PROMPT:
            m = {**m, "content": SYSTEM_PROMPT}
        elif role == "assistant":
            nc = normalize_assistant(content)
            if nc != content:
                m = {**m, "content": nc}
        out.append(m)
    return out


def build_sft(records: Iterable[Any]) -> list[dict]:
    """Chat-SFT rows from repair turns and winning trajectories.

    Assistant turns are canonicalized at this boundary (:func:`_canonicalize_chat`)
    so both fresh datagen and reused v1 shards emit the single ANALYSIS/PROPOSED_CHANGE/
    FULL_KERNEL contract. Each row carries a ``_provenance`` block (kind / task / op /
    arch / measured speedup + snr / verified) — ignored by the trainer, consumed by
    curation, available for audit. ``mixing._tag`` shallow-copies rows, so it survives
    into the final multicap shard.
    """
    out: list[dict] = []
    n_repair = 0
    n_win = 0
    for raw in records:
        rec = _as_record(raw)
        if isinstance(rec, RepairRecord):
            if rec.messages:
                out.append({"messages": _canonicalize_chat(rec.messages),
                            "_provenance": _prov_repair(rec)})
                n_repair += 1
        elif isinstance(rec, WinRecord):
            if rec.trajectory:
                out.append({"messages": _canonicalize_chat(rec.trajectory),
                            "_provenance": _prov_win(rec)})
                n_win += 1
    log.metric("build_sft", rows=len(out), from_repair=n_repair, from_wins=n_win)
    return out


# --- DPO preference-quality policy (baseline-anchored, margin-weighted) ---
#
# Audit fix: v1 DPO "chosen" kernels were anchored to the SEED / seed-relative
# improvements that could STILL be slower than the production vendor baseline
# (hipBLASLt/AITER/torch) — e.g. a GEMM "win" at 0.598x hipBLASLt. DPO thus learned
# "prefer the less-bad kernel", never "prefer the kernel that beats production".
# The policy below re-anchors preferences to the production baseline, preserves the
# per-pair margin (so the trainer can margin-weight/filter), drops near-tie speed
# pairs inside the measurement-noise band, and up-weights high-margin compute-bound
# pairs so the signal concentrates where the speedups are real.
#
# Compute-bound op families whose speedups carry the most training signal (the ones
# the audit says are drowned out by low-headroom elementwise near-ties).
_COMPUTE_BOUND_FAMILIES = frozenset({"gemm", "attention", "moe", "quant"})


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _env_bool(name: str, default: bool = True) -> bool:
    v = os.environ.get(name)
    if v in (None, ""):
        return default
    return str(v).strip().lower() not in ("0", "false", "no", "off")


def _env_float(name: str, default: float) -> float:
    try:
        v = os.environ.get(name)
        return float(v) if v not in (None, "") else float(default)
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        v = os.environ.get(name)
        return int(v) if v not in (None, "") else int(default)
    except (TypeError, ValueError):
        return int(default)


@dataclass(frozen=True)
class DPOPrefPolicy:
    """Knobs governing baseline-anchoring, margin filtering, and weighting.

    Frontier defaults are ON (anchoring + weighting + a 2% noise floor). Every
    field is overridable per-call (build_dpo kwargs) or via ``KORE_PREF_*`` env.
    """

    anchor_baseline: bool = True      # KORE_PREF_ANCHOR_BASELINE
    anchor_min: float = 1.0           # KORE_PREF_ANCHOR_MIN  (speedup vs baseline to "win")
    margin_min: float = 0.02          # KORE_PREF_MARGIN_MIN  (noise band; ~noise_floor_pct/100)
    subbaseline_mode: str = "relabel"  # KORE_PREF_SUBBASELINE_MODE: relabel|drop|keep
    weighting: bool = True            # KORE_PREF_WEIGHTING
    low_margin: float = 1.10          # KORE_PREF_LOWMARGIN   (below => low-headroom near-tie)
    neartie_cap_frac: float = 1.0     # KORE_PREF_NEARTIE_CAP_FRAC (low-margin <= frac*substantive)
    neartie_min_keep: int = 100       # KORE_PREF_NEARTIE_MIN_KEEP (floor; small families untouched)
    compute_mult: float = 2.0         # KORE_PREF_COMPUTE_MULT (extra weight for compute-bound)
    weight_max: float = 8.0           # KORE_PREF_WEIGHT_MAX
    subbaseline_weight: float = 0.25  # KORE_PREF_SUBBASELINE_WEIGHT (relabelled sub-baseline)
    margin_gain: float = 1.0
    baseline_gain: float = 1.0
    margin_cap: float = 10.0


def resolve_pref_policy(**overrides) -> DPOPrefPolicy:
    """Build a :class:`DPOPrefPolicy` from ``KORE_PREF_*`` env, then apply non-None
    keyword overrides (call-site kwargs win over env, env wins over defaults)."""
    base = DPOPrefPolicy(
        anchor_baseline=_env_bool("KORE_PREF_ANCHOR_BASELINE", True),
        anchor_min=_env_float("KORE_PREF_ANCHOR_MIN", 1.0),
        margin_min=_env_float("KORE_PREF_MARGIN_MIN", 0.02),
        subbaseline_mode=_env_str("KORE_PREF_SUBBASELINE_MODE", "relabel"),
        weighting=_env_bool("KORE_PREF_WEIGHTING", True),
        low_margin=_env_float("KORE_PREF_LOWMARGIN", 1.10),
        neartie_cap_frac=_env_float("KORE_PREF_NEARTIE_CAP_FRAC", 1.0),
        neartie_min_keep=_env_int("KORE_PREF_NEARTIE_MIN_KEEP", 100),
        compute_mult=_env_float("KORE_PREF_COMPUTE_MULT", 2.0),
        weight_max=_env_float("KORE_PREF_WEIGHT_MAX", 8.0),
        subbaseline_weight=_env_float("KORE_PREF_SUBBASELINE_WEIGHT", 0.25),
    )
    clean = {k: v for k, v in overrides.items() if v is not None}
    return replace(base, **clean) if clean else base


def _num(x: Any) -> Optional[float]:
    """A finite float or None (rejecting bools, NaNs, and non-numerics)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _infer_family(rec: Any) -> str:
    from kore.data.mutate import infer_family
    op = getattr(rec, "operation", None) or getattr(rec, "task_id", None) or ""
    return infer_family(op)


def candidate_baseline_speedup(cand: dict, group_baseline_wall: Any = None,
                               baseline_speedup_fn=None, task_id: Any = None) -> Optional[float]:
    """A candidate's ABSOLUTE speedup vs the production baseline, or None if unknown.

    Priority: an explicit per-candidate ``speedup`` / ``speedup_vs_baseline`` (the
    reward's worst-shape baseline speedup, persisted by datagen) -> an injected
    ``baseline_speedup_fn(task_id, cand)`` -> ``baseline_wall_us / wall_us`` when a
    baseline wall is available on the candidate or the group. Returns None when no
    baseline reference exists (current on-disk groups store only ``wall_us``; see
    the gen_groups follow-up in the report), so anchoring degrades gracefully.
    """
    su = _num(cand.get("speedup"))
    if su is None:
        su = _num(cand.get("speedup_vs_baseline"))
    if su is not None and su > 0:
        return su
    if baseline_speedup_fn is not None:
        try:
            v = _num(baseline_speedup_fn(task_id, cand))
        except Exception:  # noqa: BLE001 - an injected fn must never abort the build
            v = None
        if v is not None and v > 0:
            return v
    bw = _num(cand.get("baseline_wall_us"))
    if bw is None:
        bw = _num(group_baseline_wall)
    w = _num(cand.get("wall_us"))
    if bw is not None and bw > 0 and w is not None and w > 0:
        return bw / w
    return None


def _is_correctness_pair(chosen_c: dict, rejected_c: dict) -> bool:
    """True when the pair is a correctness (not speed) preference.

    Reward-hack hard negatives (``hard_negative`` label), repair pairs
    (``failure_class`` on the broken side), and any pair missing a comparable wall
    time on either side are correctness pairs — kept at neutral weight, never
    dropped, never treated as a speed signal. This is what preserves the mined
    reward-hack negatives untouched.
    """
    if rejected_c.get("hard_negative"):
        return True
    if rejected_c.get("failure_class") is not None:
        return True
    cw, rw = _num(chosen_c.get("wall_us")), _num(rejected_c.get("wall_us"))
    return not (cw is not None and cw > 0 and rw is not None and rw > 0)


def _beats_baseline_weight(margin: Optional[float], chosen_speedup: Optional[float],
                           family: str, policy: DPOPrefPolicy) -> float:
    """Up-weight a beats-baseline pair by (log) margin + (log) absolute headroom,
    with an extra multiplier for compute-bound families. Clamped to [1, weight_max]."""
    m = margin if (margin is not None and margin > 1.0) else 1.0
    su = chosen_speedup if (chosen_speedup is not None and chosen_speedup > 1.0) else 1.0
    bonus = (policy.margin_gain * math.log(min(m, policy.margin_cap))
             + policy.baseline_gain * math.log(min(su, policy.margin_cap)))
    mult = policy.compute_mult if family in _COMPUTE_BOUND_FAMILIES else 1.0
    return round(min(max(1.0 + mult * bonus, 1.0), policy.weight_max), 4)


def _pair_meta(chosen_c: dict, rejected_c: dict, family: str, policy: DPOPrefPolicy,
               group_baseline_wall: Any, task_id: Any, baseline_speedup_fn) -> dict:
    """Classify one preference pair -> {anchor, margin, weight, speedups, drop}.

    anchor: ``beats_baseline`` (chosen beats the production baseline) | ``sub_baseline``
    (chosen faster than the rejected peer but still slower than production) |
    ``among_correct`` (faster-than-peer, baseline unknown/disabled) | ``near_tie``
    (speed gap inside the measurement-noise band -> dropped) | ``correctness``.
    """
    meta = {"anchor": "correctness", "margin": None, "weight": 1.0,
            "chosen_speedup": None, "rejected_speedup": None,
            "family": family, "drop": False, "reason": None}
    if _is_correctness_pair(chosen_c, rejected_c):
        return meta  # hard-negative / repair / incomparable -> neutral, always kept

    cw, rw = _num(chosen_c.get("wall_us")), _num(rejected_c.get("wall_us"))
    margin = rw / cw  # both > 0 here; == chosen_speedup / rejected_speedup (baseline cancels)
    meta["margin"] = round(margin, 6)
    cbs = candidate_baseline_speedup(chosen_c, group_baseline_wall, baseline_speedup_fn, task_id)
    rbs = candidate_baseline_speedup(rejected_c, group_baseline_wall, baseline_speedup_fn, task_id)
    meta["chosen_speedup"] = round(cbs, 6) if cbs is not None else None
    meta["rejected_speedup"] = round(rbs, 6) if rbs is not None else None

    # (2) Noise-band filter: a speed ordering within the measurement-noise band is
    # spurious (a 2% elementwise near-tie is not a preference). Drop it.
    if margin < (1.0 + policy.margin_min):
        meta["anchor"] = "near_tie"
        meta["drop"] = True
        meta["reason"] = f"margin {margin:.4f} within noise band +{policy.margin_min:g}"
        return meta

    # (1) Baseline anchoring: only a chosen that BEATS production is a "good" signal.
    if policy.anchor_baseline and cbs is not None:
        if cbs > policy.anchor_min:
            meta["anchor"] = "beats_baseline"
        else:
            meta["anchor"] = "sub_baseline"
            if policy.subbaseline_mode == "drop":
                meta["drop"] = True
                meta["reason"] = (f"chosen {cbs:.3f}x <= baseline anchor "
                                  f"{policy.anchor_min:g} (sub-baseline win)")
    else:
        # No baseline reference (or anchoring off): a real faster-than-peer ordering,
        # but we cannot claim it beats production, so it is never up-weighted.
        meta["anchor"] = "among_correct"

    if policy.weighting:
        if meta["anchor"] == "beats_baseline":
            meta["weight"] = _beats_baseline_weight(margin, cbs, family, policy)
        elif meta["anchor"] == "sub_baseline" and policy.subbaseline_mode == "relabel":
            meta["weight"] = round(float(policy.subbaseline_weight), 4)
        else:
            meta["weight"] = 1.0
    return meta


def _cap_near_tie_pairs(rows: list[dict], policy: DPOPrefPolicy) -> tuple[list[dict], dict]:
    """(3) Cap low-headroom near-tie pairs per op-family so they can't dominate.

    Only ``among_correct`` / ``sub_baseline`` pairs with margin < ``low_margin`` are
    cappable; ``beats_baseline`` and ``correctness`` (incl. hard negatives) are
    exempt. Per family, at most ``max(min_keep, cap_frac * n_substantive)`` low-margin
    pairs are kept (the highest-margin ones), the rest dropped. The ``min_keep`` floor
    keeps small families (and the CPU tests) untouched. Deterministic, order-stable.
    """
    def _is_low(r: dict) -> bool:
        anchor = r.get("anchor")
        m = r.get("margin")
        return (anchor in ("among_correct", "sub_baseline")
                and isinstance(m, (int, float)) and not isinstance(m, bool)
                and m < policy.low_margin)

    low_by_fam: dict[str, list[int]] = {}
    subst_by_fam: dict[str, int] = {}
    for i, r in enumerate(rows):
        fam = (r.get("_provenance") or {}).get("family") or "generic"
        if _is_low(r):
            low_by_fam.setdefault(fam, []).append(i)
        else:
            subst_by_fam[fam] = subst_by_fam.get(fam, 0) + 1
    drop_idx: set[int] = set()
    capped = 0
    for fam, idxs in low_by_fam.items():
        cap = max(policy.neartie_min_keep,
                  round(policy.neartie_cap_frac * subst_by_fam.get(fam, 0)))
        if len(idxs) <= cap:
            continue
        ranked = sorted(idxs, key=lambda i: ((rows[i].get("margin") or 0.0), -i), reverse=True)
        for i in ranked[cap:]:
            drop_idx.add(i)
            capped += 1
    if not drop_idx:
        return rows, {"neartie_capped": 0, "neartie_families": len(low_by_fam)}
    kept = [r for i, r in enumerate(rows) if i not in drop_idx]
    return kept, {"neartie_capped": capped, "neartie_families": len(low_by_fam)}


# --- DPO ---
def build_dpo(records: Iterable[Any], prompt_fn=None, *,
              anchor_baseline: Optional[bool] = None, anchor_min: Optional[float] = None,
              margin_min: Optional[float] = None, subbaseline_mode: Optional[str] = None,
              weighting: Optional[bool] = None, baseline_speedup_fn=None,
              policy: Optional[DPOPrefPolicy] = None) -> list[dict]:
    """Preference rows from ranked groups, in trl's *conversational* DPO shape.

    ``prompt_fn(task_id) -> messages`` supplies the DPO prompt. When given (the
    campaign passes an in-context builder = the GRPO turn-1 transcript with the seed
    kernel + contract), preferences are learned in the SAME context the policy sees
    at inference. Falls back to the generic one-shot prompt when ``prompt_fn`` is
    None or returns falsy (keeps CPU tests + legacy callers working).

    Each ``[chosen_idx, rejected_idx]`` preference becomes a DPO row whose
    ``prompt`` is a chat-message list and whose ``chosen``/``rejected`` are each a
    single-message assistant completion list wrapping the candidate source under
    the FULL_KERNEL contract — i.e. ``trl.DPOTrainer``'s conversational schema:

        {"prompt": [ ...chat... ],
         "chosen":   [{"role": "assistant", "content": "FULL_KERNEL:..."}],
         "rejected": [{"role": "assistant", "content": "FULL_KERNEL:..."}],
         "margin":  <float|None>, "weight": <float>, "anchor": <str>,
         "_provenance": {... speed-grounding metadata ...}}

    ``margin`` (chosen_speedup / rejected_speedup == rejected_wall / chosen_wall),
    ``weight`` (>=1 for beats-baseline pairs, up-weighted for compute-bound families;
    <1 for relabelled sub-baseline pairs; 1.0 for correctness), and ``anchor`` are
    the trainer-facing curation signals (see :class:`DPOPrefPolicy`). Behaviour is
    governed by ``policy`` (or the ``KORE_PREF_*`` env / these kwargs); the frontier
    defaults baseline-anchor + margin-weight + drop noise-band near-ties.

    Degenerate pairs where the chosen and rejected sources are identical are skipped
    (no learnable preference signal)."""
    pol = policy or resolve_pref_policy(
        anchor_baseline=anchor_baseline, anchor_min=anchor_min, margin_min=margin_min,
        subbaseline_mode=subbaseline_mode, weighting=weighting)
    out: list[dict] = []
    n_groups = 0
    n_prefs = 0
    n_degenerate = 0
    n_noise = 0
    n_subbaseline_dropped = 0
    n_anchor = {"beats_baseline": 0, "sub_baseline": 0, "among_correct": 0, "correctness": 0}
    for raw in records:
        rec = _as_record(raw)
        if not isinstance(rec, RankedGroupRecord):
            continue
        n_groups += 1
        cands = rec.candidates
        family = _infer_family(rec)
        group_baseline_wall = getattr(rec, "baseline_wall_us", None)
        prompt = (prompt_fn(rec.task_id) if prompt_fn else None) or _generic_prompt(rec.task_id, rec.gpu)
        for pair in rec.preferences:
            if len(pair) != 2:
                continue
            ci, ri = pair
            if not (0 <= ci < len(cands) and 0 <= ri < len(cands)):
                continue
            n_prefs += 1
            chosen_c, rejected_c = cands[ci], cands[ri]
            chosen_src = chosen_c.get("source", "")
            rejected_src = rejected_c.get("source", "")
            if chosen_src == rejected_src:
                n_degenerate += 1
                continue  # degenerate: identical sources carry no preference
            meta = _pair_meta(chosen_c, rejected_c, family, pol,
                              group_baseline_wall, rec.task_id, baseline_speedup_fn)
            if meta["drop"]:
                if meta["anchor"] == "near_tie":
                    n_noise += 1
                else:
                    n_subbaseline_dropped += 1
                continue
            n_anchor[meta["anchor"]] = n_anchor.get(meta["anchor"], 0) + 1
            cw, rw = chosen_c.get("wall_us"), rejected_c.get("wall_us")
            out.append(
                {
                    "prompt": prompt,
                    "chosen": [
                        {"role": "assistant", "content": _wrap_full_kernel(chosen_src)}
                    ],
                    "rejected": [
                        {"role": "assistant", "content": _wrap_full_kernel(rejected_src)}
                    ],
                    # Trainer-facing curation signals (consumed by a margin-aware DPO
                    # loss / filter; see kore/policy/dpo.py follow-up in the report).
                    "margin": meta["margin"],
                    "weight": meta["weight"],
                    "anchor": meta["anchor"],
                    # Speed-grounding metadata (Pillar 5/3): auditable, baseline-anchored.
                    "_provenance": {
                        "kind": "dpo_group", "task_id": rec.task_id,
                        "operation": getattr(rec, "operation", None),
                        "family": family,
                        "arch": getattr(rec, "gpu", None), "verified": True,
                        "chosen_wall_us": cw, "rejected_wall_us": rw,
                        "chosen_snr_db": chosen_c.get("snr_db"),
                        "rejected_snr_db": rejected_c.get("snr_db"),
                        "chosen_speedup": meta["chosen_speedup"],
                        "rejected_speedup": meta["rejected_speedup"],
                        "anchor": meta["anchor"],
                        "margin": meta["margin"],
                        "weight": meta["weight"],
                        # Legacy field (chosen-vs-rejected speedup ratio == margin).
                        "speedup": meta["margin"],
                    },
                }
            )
    out, cap_stats = _cap_near_tie_pairs(out, pol)
    log.metric("build_dpo", groups=n_groups, pairs_considered=n_prefs,
               degenerate_dropped=n_degenerate, noise_tie_dropped=n_noise,
               subbaseline_dropped=n_subbaseline_dropped,
               beats_baseline=n_anchor["beats_baseline"],
               sub_baseline=n_anchor["sub_baseline"],
               among_correct=n_anchor["among_correct"],
               correctness=n_anchor["correctness"],
               anchor_baseline=pol.anchor_baseline, margin_min=pol.margin_min,
               pairs=len(out), **cap_stats)
    return out


# --- RFT (rejection-sampled SFT) ---
def build_rft(records: Iterable[Any]) -> list[dict]:
    """Chat-SFT rows on the single best candidate per group + win trajectories."""
    out: list[dict] = []
    n_group = 0
    n_win = 0
    for raw in records:
        rec = _as_record(raw)
        if isinstance(rec, RankedGroupRecord):
            n_group += 1
            best = None
            for c in rec.candidates:
                if c.get("rank") == 0:
                    best = c
                    break
            if best is None and rec.candidates:
                best = min(rec.candidates, key=lambda c: c.get("rank", 1 << 30))
            if best is not None:
                out.append(
                    {
                        "messages": _generic_prompt(rec.task_id, rec.gpu)
                        + [
                            {
                                "role": "assistant",
                                "content": _wrap_full_kernel(best.get("source", "")),
                            }
                        ]
                    }
                )
        elif isinstance(rec, WinRecord):
            if rec.trajectory:
                out.append({"messages": _canonicalize_chat(rec.trajectory)})  # parity with build_sft
                n_win += 1
    log.metric("build_rft", rows=len(out), from_groups=n_group, from_wins=n_win)
    return out


# --- hygiene: dedup ---
def _record_source(rec: Any) -> str:
    """A representative source string for a record, for dedup hashing."""
    rec = _as_record(rec)
    if isinstance(rec, RepairRecord):
        for m in reversed(rec.messages):
            if m.get("role") == "assistant":
                k = extract_kernel(m.get("content", ""))
                if k:
                    return k
        return rec.parent_hash
    if isinstance(rec, WinRecord):
        return rec.final_source or ""
    if isinstance(rec, RankedGroupRecord):
        return "||".join(c.get("source", "") for c in rec.candidates)
    return repr(rec)


def dedup_by_source_hash(records: Iterable[Any]) -> list:
    """Keep the first record for each distinct representative-source hash."""
    seen: set[str] = set()
    out: list = []
    n_in = 0
    for rec in records:
        n_in += 1
        h = kernel_hash(_record_source(rec))
        if h in seen:
            continue
        seen.add(h)
        out.append(rec)
    log.metric("dedup_by_source_hash", n_in=n_in, kept=len(out),
               dropped=n_in - len(out))
    return out


def _record_score(rec: Any) -> float:
    """Preference score for near-dup dedup: higher = keep. Fastest win wins."""
    rec = _as_record(rec)
    if isinstance(rec, WinRecord):
        try:
            return float(getattr(rec, "speedup", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def dedup_near_source(records: Iterable[Any], per_fingerprint_cap: int = 1,
                      fuzzy_threshold: float = 0.0) -> list:
    """Near-duplicate dedup on the representative kernel source (STRUCTURAL).

    Complements :func:`dedup_by_source_hash` (exact) by collapsing kernels that
    differ only by variable renaming, whitespace, or comments (see
    ``kore.data.dedup``). Keeps the highest-scoring record per structural cluster
    (fastest :class:`WinRecord`), up to ``per_fingerprint_cap``.

    Apply this to WIN/gold records — NOT to repair records: each broken->fixed
    transition is a distinct lesson even when the fixed kernels converge, so
    collapsing repairs by fixed-kernel structure would delete real signal.
    """
    from kore.data.dedup import dedup_near

    recs = list(records)
    items = [{"_rec": r, "source": _record_source(r), "_score": _record_score(r)}
             for r in recs]
    kept, stats = dedup_near(items, source_key="source",
                             scorer=lambda d: d["_score"],
                             per_fingerprint_cap=per_fingerprint_cap,
                             fuzzy_threshold=fuzzy_threshold)
    log.metric("dedup_near_source", **stats)
    return [it["_rec"] for it in kept]


# --- hygiene: leakage-aware split ---
def _group_key(rec: Any, by: tuple = ("operation", "arch")) -> str:
    """Build a grouping key from ``by`` fields, tolerating missing fields.

    Fields are looked up on the record's dict. The ``operation`` field is
    normalized to its op *family* via ``mutate.infer_family`` (so gemm_bf16 and
    gemm_fp8_a8w8 group together as 'gemm'), falling back to ``task_id`` when the
    provenance field is absent. This replaces the brittle leading-``_`` split so
    the same op family never leaks across train/val/test."""
    from kore.data.mutate import infer_family
    from kore.tasks.registry import TRAIN_ARCH, TRAIN_ARCHS

    rec = _as_record(rec)
    d = rec.to_dict() if hasattr(rec, "to_dict") else dict(rec)
    parts: list[str] = []
    for field in by:
        val = d.get(field)
        if field == "operation":
            val = infer_family(val or d.get("task_id", ""))
        elif field == "arch":
            # Resolve arch<-gpu, and CANONICALIZE the CDNA3/CDNA4 lineage: a record
            # tagged gfx942 (previous gen) and its gfx950 sibling are the same op
            # family on the same hardware lineage, so they must land in the SAME
            # split — otherwise gemm|gfx942 and gemm|gfx950 fracture into different
            # groups and the op family leaks across train/val (audit C1/C5). A
            # genuinely foreign arch (e.g. gfx1100) keeps its own key.
            val = val or d.get("gpu")
            if val in TRAIN_ARCHS:
                val = TRAIN_ARCH
        parts.append(str(val) if val is not None else "")
    key = "|".join(parts)
    return key or str(d.get("task_id", ""))


def leakage_split(
    records: Iterable[Any],
    by: tuple = ("operation", "arch"),
    ratios: tuple = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> tuple[list, list, list]:
    """Split records into (train, val, test) so no ``by``-group crosses splits.

    Whole groups are assigned to a single split; deterministic given ``seed``."""
    records = list(records)
    # bucket records by group key
    groups: dict[str, list] = {}
    for rec in records:
        groups.setdefault(_group_key(rec, by), []).append(rec)

    keys = sorted(groups.keys())
    # deterministic shuffle by seed
    import random

    random.Random(seed).shuffle(keys)

    n = len(keys)
    tr, va, _te = ratios
    n_train = int(round(n * tr))
    n_val = int(round(n * va))
    # guard rounding so all keys are assigned
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train : n_train + n_val])
    test_keys = set(keys[n_train + n_val :])

    def collect(kset):
        out: list = []
        for k in kset:
            out.extend(groups[k])
        return out

    train, val, test = collect(train_keys), collect(val_keys), collect(test_keys)
    log.metric(
        "leakage_split", by=list(by), n_records=len(records), n_groups=n,
        train_groups=len(train_keys), val_groups=len(val_keys),
        test_groups=len(test_keys),
        train=len(train), val=len(val), test=len(test),
    )
    return train, val, test
