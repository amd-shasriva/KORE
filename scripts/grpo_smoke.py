"""Tiny real GRPO (fallback loop) run: a few steps on one task to prove the
multi-turn rollout -> verified reward -> policy-gradient path works.

    PYTHONPATH=. python scripts/grpo_smoke.py --task rmsnorm_aiter
"""

from __future__ import annotations

import argparse

from kore.policy.configs import GRPOConfig
from kore.policy.grpo import train_grpo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="rmsnorm_aiter",
                    help="comma-separated task_ids (co-evolution ranges over these)")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--out", default="runs/grpo_smoke")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--trajectories", type=int, default=2)
    ap.add_argument("--turns", type=int, default=2)
    ap.add_argument("--max-response", type=int, default=1024)
    ap.add_argument("--tasks-per-step", type=int, default=None)
    ap.add_argument("--coevolve", action="store_true",
                    help="drive task selection with the open-ended frontier proposer")
    args = ap.parse_args()

    tasks = [t.strip() for t in args.task.split(",") if t.strip()]
    cfg = GRPOConfig(model_id=args.model, output_dir=args.out)
    cfg.total_steps = args.steps
    cfg.num_trajectories = args.trajectories
    cfg.num_turns = args.turns
    cfg.max_response_length = args.max_response
    cfg.coevolve = args.coevolve
    if args.tasks_per_step is not None:
        cfg.tasks_per_step = args.tasks_per_step
        cfg.target_groups = args.tasks_per_step
    out = train_grpo(cfg, tasks=tasks, backend="fallback")
    print(f"[grpo_smoke] OK -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
