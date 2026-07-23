"""Data-factory smoke for b05-2: prove teacher -> GPU-verify datagen works on an
idle GPU, co-existing with other containers. Generates ONE win record for one
simple task pinned to a chosen physical GPU.

    GPU=3 python scripts/_factory_smoke.py
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    gpu = int(os.environ.get("GPU", "3"))
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
    print("[smoke] OK -- teacher + GPU verify both work on b05-2", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
