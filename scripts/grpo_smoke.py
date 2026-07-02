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
    ap.add_argument("--task", default="rmsnorm_aiter")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--out", default="runs/grpo_smoke")
    ap.add_argument("--steps", type=int, default=2)
    args = ap.parse_args()

    cfg = GRPOConfig(model_id=args.model, output_dir=args.out)
    cfg.total_steps = args.steps
    cfg.num_trajectories = 2
    cfg.num_turns = 2
    cfg.max_response_length = 1024
    out = train_grpo(cfg, tasks=[args.task], backend="fallback")
    print(f"[grpo_smoke] OK -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
