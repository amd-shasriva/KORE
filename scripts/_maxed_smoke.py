"""Maxed-levers GRPO smoke: load the flagship template (ALL paradigm-v3 levers ON),
shrink it to a couple of steps on one GPU, and force search/mint every step so the
deep-search + B&B + value-prior + transform-discover + live-rho + hack-ceiling +
grammar-evolution + regret-vs-Opus paths all actually execute on real MI350X.

    HIP_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/_maxed_smoke.py
"""
from __future__ import annotations

import json
import sys

from kore.policy.grpo import grpo_config_from_dict, train_grpo


def main() -> int:
    d = json.load(open("configs/grpo_14b_full.json"))
    d.update(dict(
        model_id="Qwen/Qwen3-14B",
        output_dir="runs/grpo_maxed_smoke",
        distributed=False, use_lora=False,
        total_steps=2, num_trajectories=2, num_turns=2,
        tasks_per_step=1, target_groups=1,
        max_prompt_length=2048, max_response_length=384,
        search_every=1, search_budget=8, search_k_expand=3, search_max_depth=3,
        coevolve_mint_batch=2, save_steps=0, adaptive_steps=False,
        ref_anchor_coef=0.0,  # skip the extra ref-model load for the smoke
    ))
    cfg = grpo_config_from_dict(d)
    print("[maxed_smoke] levers:",
          "bnb", cfg.search_bnb, "value_prior", cfg.search_value_prior,
          "discover", cfg.transform_discover, "live_rho", cfg.physics_live_counters,
          "gate", cfg.roofline_gate, "grammar", cfg.coevolve_evolve_grammar,
          "regret", cfg.coevolve_regret_vs_opus, flush=True)
    out = train_grpo(cfg, tasks=["rmsnorm_aiter", "gen_add_bf16"])
    print(f"[maxed_smoke] OK -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
