"""GPU-prove the v3 new-op seeds on gfx942: compile + rigorous correctness
(5-seed multi-trial + adversarial no-lucky-pass) + speedup vs the real baseline.

Run pinned:  HIP_VISIBLE_DEVICES=3 python scripts/prove_new_ops.py
Exits non-zero if ANY op fails to compile or fails correctness (poison gate).
"""
from __future__ import annotations

import os
import sys

os.environ.pop("ROCR_VISIBLE_DEVICES", None)
os.environ.pop("CUDA_VISIBLE_DEVICES", None)
os.environ.setdefault("KORE_VERIFIED_CORRECTNESS", "1")  # adversarial battery
os.environ.setdefault("KORE_BENCH_COLD", "1")

from kore.data.verify_rigor import set_rigorous_verification  # noqa: E402
from kore.env.kore_env import KoreEnv  # noqa: E402
from kore.tasks.registry import get_task  # noqa: E402

set_rigorous_verification(True)

NEW_OPS = [
    "genv_rope_gptj_bf16",
    "genv_rope_partial_bf16",
    "genv_embedding_gather_bf16",
    "fused_rmsnorm_quant_fp8",
    "fused_silu_mul_quant_fp8",
    "gemm_w4a16",
    "rmsnorm_backward",
]

rows = []
ok_all = True
for tid in NEW_OPS:
    try:
        task = get_task(tid)
        env = KoreEnv(task, use_replay=False)
        obs = env.step(task.seed_source, full_validation=True, multi_shape=True)
        sp = getattr(obs, "worst_speedup", None)
        sp = sp if sp is not None else getattr(obs, "speedup", None)
        ok = bool(obs.compiled) and bool(obs.validation_passed)
        ok_all = ok_all and ok
        rows.append((tid, obs.compiled, obs.validation_passed, obs.snr_db, sp, ok))
        print(f"[prove] {tid:32s} compiled={obs.compiled} correct={obs.validation_passed} "
              f"snr={obs.snr_db} speedup={sp} -> {'OK' if ok else 'FAIL'}", flush=True)
    except Exception as e:  # noqa: BLE001
        ok_all = False
        rows.append((tid, False, False, None, None, False))
        print(f"[prove] {tid:32s} EXCEPTION {type(e).__name__}: {e}", flush=True)

print("\n==== SUMMARY ====")
for tid, comp, corr, snr, sp, ok in rows:
    print(f"  {tid:32s} {'OK  ' if ok else 'FAIL'} compiled={comp} correct={corr} snr={snr} speedup={sp}")
print(f"\nALL_PASS={ok_all}")
sys.exit(0 if ok_all else 1)
