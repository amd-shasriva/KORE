"""Generate winning trajectories (KORE Stage 3 seed / SFT-on-wins).

A short greedy evolve loop: the parent is the best-so-far kernel, the teacher
proposes a rewrite conditioned on the last verifier feedback, we verify it, and
keep it as the new best only if it is correct AND meaningfully faster
(wall < best_wall * 0.98). The full multi-turn chat is stored as a ``WinRecord``
if the loop achieved any net speedup over the initial kernel.
"""

from __future__ import annotations


from kore.config import CONFIG
from kore.data.prompts import (
    SYSTEM_PROMPT,
    build_turn_prompt,
    extract_kernel,
    normalize_assistant,
)
from kore.data.schemas import WinRecord
from kore.obs import get_logger
from kore.reward.reward import compute_reward

log = get_logger("data.gen_wins")

_IMPROVE_FACTOR = 0.98  # a kept step must beat best wall by >= 2%


def _feedback(obs, rr) -> str:
    # error_text is Optional[str]: it is None for a compiled-but-incorrect kernel
    # (an SNR failure carries no error string), so guard before slicing — otherwise
    # a correctness miss (common on the tighter fp16 SNR thresholds) crashes the
    # whole wins shard with 'NoneType' is not subscriptable.
    err = obs.error_text or ""
    if not obs.compiled:
        return f"FAILED to compile: {err[:400]}"
    if not rr.correct:
        return (
            f"Correct? NO. snr_db={obs.snr_db}. {err[:200]}\n"
            "Fix correctness before optimizing further."
        )
    wall_us = obs.wall_ms * 1000.0 if obs.wall_ms is not None else None
    # wall_ms/speedup can be None when timing is unmeasurable on this stack
    # (e.g. fp8 on ROCm) — format defensively so the wins shard isn't lost.
    wall_s = f"{wall_us:.1f}us" if wall_us is not None else "n/a"
    speedup_s = f"{rr.speedup:.3f}x" if rr.speedup is not None else "n/a"
    return (
        f"Correct? YES. wall={wall_s} speedup={speedup_s}. "
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
    with log.stage("generate_wins", task=task.task_id, gens=gens):
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

        def _emit_turn(turn: int, turn_mode: str, improved: bool) -> None:
            sp = (initial_wall / best_wall
                  if (initial_wall and best_wall and best_wall > 0) else None)
            log.event(
                "win_turn", task=task.task_id, turn=turn, mode=turn_mode,
                improved=improved, best_wall_us=best_wall, best_snr=best_snr,
                speedup=sp,
            )
            log.progress(turn + 1, gens, "wins", best_wall_us=best_wall,
                         best_snr=best_snr, speedup=sp)

        for turn in range(gens):
            turn_mode = mode
            improved = False
            prompt = build_turn_prompt(parent_source=best_src, feedback=feedback, mode=mode)
            trajectory.append({"role": "user", "content": prompt})
            response = teacher.generate(trajectory)
            # Store the assistant turn in the CANONICAL contract (Pillar 0): the raw
            # teacher text may be loosely shaped; normalize_assistant re-renders it to
            # ANALYSIS/PROPOSED_CHANGE/FULL_KERNEL (no-op if it carries no kernel) so
            # the win trajectory that feeds SFT never leaks a non-canonical shape.
            trajectory.append({"role": "assistant", "content": normalize_assistant(response)})

            cand_src = extract_kernel(response)
            if not cand_src:
                feedback = "No kernel found in your response. Output a full FULL_KERNEL block."
                mode = "repair"
                _emit_turn(turn, turn_mode, improved)
                continue

            try:
                c_obs = env.step(cand_src, full_validation=True, multi_shape=True)
            except Exception as e:
                feedback = f"Verifier crashed: {str(e)[:200]}"
                mode = "repair"
                _emit_turn(turn, turn_mode, improved)
                continue

            c_rr = compute_reward(c_obs, cand_src, dtype=task.dtype, cfg=cfg)
            feedback = _feedback(c_obs, c_rr)

            if not c_rr.correct:
                mode = "repair"
                _emit_turn(turn, turn_mode, improved)
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
            _emit_turn(turn, turn_mode, improved)

        speedup = None
        if initial_wall and best_wall and best_wall > 0:
            speedup = initial_wall / best_wall

        is_win = (
            best_src != seed_src
            and speedup is not None
            and speedup > 1.0
        )
        log.metric(
            "wins_summary", task=task.task_id, turns=gens, is_win=is_win,
            speedup=speedup, initial_wall_us=initial_wall, final_wall_us=best_wall,
            best_snr=best_snr,
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
