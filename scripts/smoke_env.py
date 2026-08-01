"""GPU smoke: run every registered task's seed kernel through KoreEnv.step and
print the verified Observation + lexicographic reward. Proves build + 5-stage
correctness + AITER-baseline benchmarking + reward are wired on real hardware.

    PYTHONPATH=. python scripts/smoke_env.py [task_id ...]

Exits non-zero if any task crashes or any seed fails to compile, so it can be used
as a gate (scripts/spur_gpu_smoke.sbatch). Correctness/reward are reported but not
gated: a seed that compiles yet misses its SNR gate is a task-authoring problem,
not the env wiring this smoke is proving.
"""

from __future__ import annotations

import sys

from kore.env.kore_env import KoreEnv
from kore.reward.reward import compute_reward
from kore.tasks.registry import all_tasks, get_task


def main(argv: list[str]) -> int:
    tasks = [get_task(t) for t in argv] if argv else all_tasks()
    ok = 0
    ncorrect = 0
    crashed: list[str] = []
    uncompiled: list[str] = []
    for task in tasks:
        env = KoreEnv(task, use_replay=False)
        print(f"\n=== {task.task_id} ({task.dtype}, baseline={task.comparison_baseline}) ===")
        try:
            obs = env.step(task.seed_source, full_validation=True, multi_shape=True)
        except Exception as e:  # noqa: BLE001
            print(f"  CRASH: {e}")
            crashed.append(task.task_id)
            continue
        if not obs.compiled:
            uncompiled.append(task.task_id)
        rr = compute_reward(obs, task.seed_source, dtype=task.dtype)
        ncorrect += int(bool(rr.correct))
        print(f"  compiled={obs.compiled} correct={rr.correct} tier={rr.tier}")
        print(f"  snr_by_shape={obs.snr_by_shape}")
        print(f"  cand_ms={obs.wall_by_shape}")
        print(f"  base_ms={obs.baseline_by_shape}")
        print(f"  speedup(worst)={rr.speedup} reward={rr.reward:.3f} flags={rr.flags}")
        if obs.error_text:
            print(f"  error={obs.error_text[:300]}")
        ok += int(obs.compiled)
    print(f"\n[smoke] {ok}/{len(tasks)} seeds compiled, {ncorrect}/{len(tasks)} correct")
    if not tasks:
        print("[smoke] FAILED: no tasks selected")
        return 1
    if crashed:
        print(f"[smoke] FAILED: {len(crashed)} task(s) crashed: {', '.join(crashed)}")
        return 1
    if uncompiled:
        print(f"[smoke] FAILED: {len(uncompiled)} seed(s) did not compile: "
              f"{', '.join(uncompiled)}")
        return 1
    print("[smoke] OK: every seed built and benchmarked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
