"""Generate ranked candidate groups (KORE Stage 2: RFT + DPO).

For a parent kernel, sample ``k`` candidate rewrites from the teacher (varying
the mode for diversity), verify each through the environment + reward, then rank
them with the PURE ``rank_candidates`` function and emit a ``RankedGroupRecord``
carrying every preference pair implied by the ranking:

    faster-correct  >  slower-correct  >  incorrect  >  non-compiling

``rank_candidates`` and ``build_preferences`` are pure and unit-testable; they
operate on lightweight result dicts, not on the GPU.
"""

from __future__ import annotations

import random
import time
from typing import Optional

from kore.config import CONFIG
from kore.data.prompts import SYSTEM_PROMPT, build_turn_prompt, extract_kernel
from kore.data.schemas import RankedGroupRecord
from kore.data.teacher import TeacherClient
from kore.env.replay import kernel_hash
from kore.obs import get_logger
from kore.reward.reward import compute_reward

log = get_logger("data.gen_groups")


def _quality_key(c: dict) -> tuple:
    """Total-order key (higher is better) for a candidate result dict.

    Tiers: correct (2) > compiled-but-incorrect (1) > non-compiling (0). Within
    the correct tier, higher speedup wins, then higher SNR."""
    correct = bool(c.get("correct"))
    compiled = bool(c.get("compiled", False))
    level = 2 if correct else (1 if compiled else 0)
    speedup = c.get("speedup")
    speed = float(speedup) if (correct and speedup is not None) else 0.0
    snr = c.get("snr_db")
    snr_v = float(snr) if snr is not None else float("-inf")
    return (level, speed, snr_v)


def rank_candidates(results: list[dict]) -> list[int]:
    """Return candidate indices ordered best-first.

    Ordering: valid (correct) first, then higher speedup, then higher SNR."""
    return sorted(
        range(len(results)), key=lambda i: _quality_key(results[i]), reverse=True
    )


def _speed_tie(a: dict, b: dict, band: float) -> bool:
    """True if two candidates' speedups are indistinguishable within ``band``.

    ``band`` is a fractional tolerance (e.g. 0.03 == 3%). Missing speedups only
    tie one another (a benched-vs-unbenched pair is a real difference)."""
    sa, sb = a.get("speedup"), b.get("speedup")
    if sa is not None and sb is not None and sa > 0 and sb > 0:
        hi, lo = (sa, sb) if sa >= sb else (sb, sa)
        return (hi / lo - 1.0) <= band
    if sa is None and sb is None:
        return True
    return False


def _snr_tie(a: dict, b: dict, band_db: float) -> bool:
    """True if two candidates' SNRs are within ``band_db`` dB of each other."""
    na, nb = a.get("snr_db"), b.get("snr_db")
    if na is not None and nb is not None:
        return abs(float(na) - float(nb)) <= band_db
    if na is None and nb is None:
        return True
    return False


def _is_noise_tie(a: dict, b: dict, speedup_noise_band: float,
                  snr_noise_band_db: float) -> bool:
    """A measurement-noise tie between two CORRECT candidates.

    The ranking is lexicographic (speedup first, then SNR). Two correct
    candidates whose speedups agree within ``speedup_noise_band`` *and* whose
    SNRs agree within ``snr_noise_band_db`` differ only by measurement noise, so
    the ordering between them is spurious and must not become a DPO preference.
    Only correct-vs-correct pairs can be noise ties: a correct-vs-incorrect or
    compiled-vs-noncompiling ordering is always a real preference."""
    if not (a.get("correct") and b.get("correct")):
        return False
    return (_speed_tie(a, b, speedup_noise_band)
            and _snr_tie(a, b, snr_noise_band_db))


def build_preferences(
    results: list[dict],
    speedup_noise_band: float = 0.0,
    snr_noise_band_db: float = 0.0,
) -> list[list[int]]:
    """All [chosen_idx, rejected_idx] pairs where chosen is strictly better.

    MARGIN GATE: when ``speedup_noise_band`` / ``snr_noise_band_db`` are > 0, a
    strict ordering between two correct candidates that is within the noise band
    on BOTH speedup and SNR is dropped (a measurement-noise tie is not a real
    preference). With the default 0.0 bands nothing extra is dropped, so the pure
    ranking behaviour is preserved exactly."""
    prefs: list[list[int]] = []
    n = len(results)
    for i in range(n):
        ki = _quality_key(results[i])
        for j in range(n):
            if i == j:
                continue
            if ki > _quality_key(results[j]):
                if _is_noise_tie(results[i], results[j],
                                 speedup_noise_band, snr_noise_band_db):
                    continue  # margin gate: within-noise tie, not a preference
                prefs.append([i, j])
    return prefs


def _evaluate(env, task, source: str, cfg) -> dict:
    """Run one candidate through the verifier + reward into a result dict."""
    try:
        obs = env.step(source, full_validation=True, multi_shape=True)
    except Exception as e:  # keep the group intact even if one candidate explodes
        return {
            "source": source,
            "compiled": False,
            "correct": False,
            "speedup": None,
            "snr_db": None,
            "wall_us": None,
            "error": str(e)[:200],
        }
    rr = compute_reward(obs, source, dtype=task.dtype, cfg=cfg)
    wall_us = obs.wall_ms * 1000.0 if obs.wall_ms is not None else None
    return {
        "source": source,
        "compiled": bool(obs.compiled),
        "correct": bool(rr.correct),
        "speedup": rr.speedup,
        "snr_db": obs.snr_db,
        "wall_us": wall_us,
    }


def resolve_noise_bands(cfg) -> tuple[float, float]:
    """(speedup_noise_band, snr_noise_band_db) for the preference margin gate.

    Reads optional ``cfg`` overrides, else derives the speed band from the
    verifier's timing ``noise_floor_pct`` (so a "faster" candidate must clear the
    measured timing noise to earn a preference). The SNR band defaults to 0.0
    (SNR differences above the correctness gate are treated as real unless a
    band is configured)."""
    speed_band = getattr(cfg, "preference_speedup_noise_band", None)
    if speed_band is None:
        speed_band = float(getattr(cfg, "noise_floor_pct", 0.0)) / 100.0
    snr_band = float(getattr(cfg, "preference_snr_noise_band_db", 0.0))
    return float(speed_band), snr_band


def generate_groups(
    task,
    teacher: TeacherClient,
    env,
    n_parents: int,
    k: int,
    seed: int = 0,
    cfg=CONFIG,
) -> list[RankedGroupRecord]:
    """Produce ranked groups: ``n_parents`` groups of ``k`` candidates each."""
    with log.stage("generate_groups", task=task.task_id, n_parents=n_parents, k=k):
        rng = random.Random(seed)
        speed_band, snr_band = resolve_noise_bands(cfg)
        modes = ["exploit", "explore", "repair"]
        records: list[RankedGroupRecord] = []
        parent_src = task.seed_source
        t_start = time.time()
        tot_candidates = 0
        tot_correct = 0
        tot_pairs = 0

        for p in range(n_parents):
            results: list[dict] = []
            for c in range(k):
                mode = modes[c % len(modes)] if k >= 3 else rng.choice(modes)
                prompt = build_turn_prompt(parent_source=parent_src, mode=mode)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
                response = teacher.generate(messages)
                cand_src = extract_kernel(response)
                if not cand_src:
                    # Extraction failed: skip this candidate rather than injecting the
                    # parent as a fake candidate (which would pollute the ranking with
                    # a duplicate and manufacture spurious preference pairs).
                    log.debug("group candidate had no kernel; skipping",
                              task=task.task_id, parent=p, cand=c, mode=mode)
                    continue
                res = _evaluate(env, task, cand_src, cfg)
                log.event(
                    "group_candidate", task=task.task_id, parent=p, cand=c,
                    mode=mode, compiled=res["compiled"], correct=res["correct"],
                    snr_db=res["snr_db"], speedup=res["speedup"],
                    wall_us=res["wall_us"],
                )
                results.append(res)

            order = rank_candidates(results)
            rank_of = {idx: pos for pos, idx in enumerate(order)}
            candidates = [
                {
                    "source": r["source"],
                    "wall_us": r["wall_us"],
                    "snr_db": r["snr_db"],
                    "rank": rank_of[i],
                }
                for i, r in enumerate(results)
            ]
            prefs = build_preferences(results, speed_band, snr_band)
            records.append(
                RankedGroupRecord(
                    task_id=task.task_id,
                    parent_id=kernel_hash(parent_src),
                    candidates=candidates,
                    preferences=prefs,
                    gpu=task.gpu_target,
                    operation=getattr(task, "operation", None),
                    arch=getattr(task, "gpu_target", None),
                )
            )
            tot_candidates += len(results)
            n_correct = sum(1 for r in results if r.get("correct"))
            tot_correct += n_correct
            tot_pairs += len(prefs)
            log.progress(p + 1, n_parents, "groups", t_start=t_start,
                         candidates=len(results), correct=n_correct, pairs=len(prefs))
            # advance the parent to the best correct candidate, if any, for diversity
            best_idx = order[0] if order else None
            if best_idx is not None and results[best_idx].get("correct"):
                parent_src = results[best_idx]["source"]

        log.metric(
            "groups_summary", task=task.task_id, parents=len(records),
            candidates=tot_candidates, correct=tot_correct, pairs=tot_pairs,
            speedup_noise_band=speed_band, snr_noise_band_db=snr_band,
        )
        return records
