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


def build_preferences(results: list[dict]) -> list[list[int]]:
    """All [chosen_idx, rejected_idx] pairs where chosen is strictly better."""
    prefs: list[list[int]] = []
    n = len(results)
    for i in range(n):
        ki = _quality_key(results[i])
        for j in range(n):
            if i == j:
                continue
            if ki > _quality_key(results[j]):
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
            prefs = build_preferences(results)
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
        )
        return records
