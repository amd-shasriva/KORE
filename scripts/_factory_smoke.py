"""Data-factory smoke: prove teacher -> GPU-verify datagen works on one GPU.

On SPUR, request the full ``gpu:mi355x:8`` node: partial-GPU GRES allocations do
not expose a usable ROCm device. Generates one win record for one simple task
pinned to a chosen physical GPU.

    GPU=3 python scripts/_factory_smoke.py
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    gpu = int(os.environ.get("GPU", "3"))
    # Full-node SPUR jobs inherit a physical ROCR list. KORE pins the verifier
    # with HIP_VISIBLE_DEVICES; retaining both masks can hide every child device.
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    from kore.data.gen_wins import generate_wins
    from kore.data.teacher import make_teacher
    from kore.env.kore_env import KoreEnv
    from kore.tasks.registry import all_tasks

    t = next(x for x in all_tasks() if x.task_id == "gen_add_bf16")
    print(f"[smoke] task={t.task_id} dtype={t.dtype} pinned physical GPU={gpu}", flush=True)
    env = KoreEnv(t, gpu=gpu)
    teacher = make_teacher("claude", resilient=True)
    print("[smoke] teacher built; generating 1 win (teacher call + GPU verify)...", flush=True)
    recs = generate_wins(task=t, teacher=teacher, env=env, gens=1)
    if not recs:
        print("[smoke] RESULT: 0 records (teacher returned nothing or verify failed) "
              "-- check gateway reachability / GPU", flush=True)
        return 2
    r = recs[0]
    print(f"[smoke] RESULT: records={len(recs)} correct={getattr(r,'correct',None)} "
          f"speedup={getattr(r,'speedup',None)}", flush=True)
    print("[smoke] OK -- teacher + GPU verify both work", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
