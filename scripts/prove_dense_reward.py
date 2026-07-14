"""Prove the dense hardware-counter reward is LIVE on gfx942 (not silently inert):
exercises the exact GRPO path (env.collect_counters -> roofline_dense_score ->
dense bonus + counter feedback) on a real compute-bound kernel via rocprofv3.

Run pinned:  HIP_VISIBLE_DEVICES=3 python scripts/prove_dense_reward.py
"""
from __future__ import annotations

import os
import sys

os.environ.pop("ROCR_VISIBLE_DEVICES", None)
os.environ.pop("CUDA_VISIBLE_DEVICES", None)
os.environ["KORE_PROFILE_REWARD_WEIGHT"] = "0.1"   # activate before CONFIG import

from kore.config import CONFIG  # noqa: E402
from kore.data.verify_rigor import set_rigorous_verification  # noqa: E402
from kore.env.kore_env import KoreEnv  # noqa: E402
from kore.policy.grpo import _dense_profile_bonus, _dense_profile_weight  # noqa: E402
from kore.tasks.registry import get_task  # noqa: E402

set_rigorous_verification(True)

print(f"[dense] gate weight = {_dense_profile_weight(CONFIG)} (want 0.1)")

# Compute-bound op with an analytical roofline model.
for tid in ("gemm_bf16", "genv_rmsnorm_bf16"):
    task = get_task(tid)
    env = KoreEnv(task, use_replay=False)
    obs = env.step(task.seed_source, full_validation=True, multi_shape=True)
    print(f"\n[dense] {tid}: correct={obs.validation_passed} wall_ms={getattr(obs,'wall_ms',None)}")
    # direct counter collection (the only new GPU work in the rollout)
    counters = env.collect_counters(task.seed_source) if hasattr(env, "collect_counters") else None
    n = len(counters) if counters else 0
    nonzero = sum(1 for v in (counters or {}).values() if isinstance(v, (int, float)) and v)
    print(f"[dense] {tid}: rocprofv3 counters collected = {n} (nonzero={nonzero})")
    dense, feedback = _dense_profile_bonus(env, task, task.seed_source, obs, CONFIG)
    print(f"[dense] {tid}: dense_term = {dense:.5f}")
    print(f"[dense] {tid}: feedback = {feedback[:240]}")
    live = (n > 0 and nonzero > 0)
    print(f"[dense] {tid}: LIVE={'YES' if live else 'NO'}  (counters flowing + score computed)")

print("\nDENSE_REWARD_PROOF_DONE")
sys.exit(0)
