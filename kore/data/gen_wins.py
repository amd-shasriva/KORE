"""Generate winning trajectories (KORE Stage 3 seed / SFT-on-wins).

A short greedy evolve loop: the parent is the best-so-far kernel, the teacher
proposes a rewrite conditioned on the last verifier feedback, we verify it, and
keep it as the new best only if it is correct AND meaningfully faster
(wall < best_wall * 0.98). The full multi-turn chat is stored as a ``WinRecord``
if the loop achieved any net speedup over the initial kernel.
"""

from __future__ import annotations

from typing import Optional

from kore.config import CONFIG
from kore.data.prompts import SYSTEM_PROMPT, build_turn_prompt, extract_kernel
from kore.data.schemas import WinRecord
from kore.reward.reward import compute_reward

_IMPROVE_FACTOR = 0.98  # a kept step must beat best wall by >= 2%


def _feedback(obs, rr) -> str:
    if not obs.compiled:
        return f"FAILED to compile: {obs.error_text[:400]}"
    if not rr.correct:
        return (
            f"Correct? NO. snr_db={obs.snr_db}. {obs.error_text[:200]}\n"
            "Fix correctness before optimizing further."
        )
    wall_us = obs.wall_ms * 1000.0 if obs.wall_ms is not None else None
    return (
        f"Correct? YES. wall={wall_us:.1f}us speedup={rr.speedup:.3f}x. "
        "Now make it faster with one more change."
    )


def generate_wins(
    task,
    teacher,
    env,
    gens: int,
    cfg=CONFIG,
) -> list[WinRecord]:
    """Run a single evolve trajectory of ``gens`` turns; return [WinRecord] if it
    produced a net speedup, else []."""
    seed_src = task.seed_source
    best_src = seed_src

    # Measure the seed as the starting point.
    obs = env.step(seed_src, full_validation=True, multi_shape=True)
    rr = compute_reward(obs, seed_src, dtype=task.dtype, cfg=cfg)
    initial_wall = obs.wall_ms * 1000.0 if obs.wall_ms is not None else None
    best_wall = initial_wall
    best_snr = obs.snr_db

    trajectory: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    feedback = _feedback(obs, rr)
    mode = "exploit"

    for turn in range(gens):
        prompt = build_turn_prompt(parent_source=best_src, feedback=feedback, mode=mode)
        trajectory.append({"role": "user", "content": prompt})
        response = teacher.generate(trajectory)
        trajectory.append({"role": "assistant", "content": response})

        cand_src = extract_kernel(response)
        if not cand_src:
            feedback = "No kernel found in your response. Output a full FULL_KERNEL block."
            mode = "repair"
            continue

        try:
            c_obs = env.step(cand_src, full_validation=True, multi_shape=True)
        except Exception as e:
            feedback = f"Verifier crashed: {str(e)[:200]}"
            mode = "repair"
            continue

        c_rr = compute_reward(c_obs, cand_src, dtype=task.dtype, cfg=cfg)
        feedback = _feedback(c_obs, c_rr)

        if not c_rr.correct:
            mode = "repair"
            continue

        cand_wall = c_obs.wall_ms * 1000.0 if c_obs.wall_ms is not None else None
        improved = (
            cand_wall is not None
            and best_wall is not None
            and cand_wall < best_wall * _IMPROVE_FACTOR
        )
        if improved:
            best_src = cand_src
            best_wall = cand_wall
            best_snr = c_obs.snr_db
            mode = "exploit"
        else:
            mode = "explore"  # plateau -> try a structural change next

    speedup = None
    if initial_wall and best_wall and best_wall > 0:
        speedup = initial_wall / best_wall

    is_win = (
        best_src != seed_src
        and speedup is not None
        and speedup > 1.0
    )
    if not is_win:
        return []

    return [
        WinRecord(
            task_id=task.task_id,
            trajectory=trajectory,
            initial_wall_us=initial_wall,
            final_wall_us=best_wall,
            speedup=speedup,
            final_source=best_src,
            snr_db=best_snr,
            gpu=task.gpu_target,
            operation=getattr(task, "operation", None),
            arch=getattr(task, "gpu_target", None),
        )
    ]
