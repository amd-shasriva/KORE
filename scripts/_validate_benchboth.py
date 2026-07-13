"""A/B validation: legacy per-impl timing vs new --bench-both batched timing.

Runs the SAME verified kernel through both paths (fresh measurement, full rigor) N
times each and compares the measured speedup distributions. The batched path is only
safe to ship if its speedups match the legacy path within run-to-run variance.
"""
import json
import os
import statistics as st
import sys

os.environ.pop("ROCR_VISIBLE_DEVICES", None)
os.environ.pop("CUDA_VISIBLE_DEVICES", None)
os.environ.setdefault("HIP_VISIBLE_DEVICES", "3")

from kore.data.verify_rigor import set_rigorous_verification
set_rigorous_verification(True)

from kore.env.kore_env import KoreEnv
from kore.reward.reward import _worst_speedup
from kore.tasks.registry import get_task


def _source_for(task_id: str) -> str:
    root = "data/full14b"
    for sub, key in (("wins", "final_source"), ("groups", None)):
        p = f"{root}/{sub}/{task_id}.jsonl"
        if not os.path.exists(p):
            continue
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            if key and r.get(key):
                return r[key]
            for c in (r.get("candidates") or []):
                if c.get("source"):
                    return c["source"]
    raise SystemExit(f"no source found for {task_id}")


def measure(task, src, bench_both: bool, n: int) -> list:
    if bench_both:
        os.environ.pop("KORE_NO_BENCH_BOTH", None)
    else:
        os.environ["KORE_NO_BENCH_BOTH"] = "1"
    out = []
    for _ in range(n):
        env = KoreEnv(task, use_replay=False, gpu=None)
        obs = env.step(src, full_validation=True, multi_shape=True)
        out.append((obs.validation_passed, _worst_speedup(obs)))
    return out


def main():
    task_id = sys.argv[1] if len(sys.argv) > 1 else "gen_add_gelu_bf16"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    task = get_task(task_id)
    src = _source_for(task_id)
    print(f"task={task_id}  source_len={len(src)}  n={n} each")

    legacy = measure(task, src, bench_both=False, n=n)
    batched = measure(task, src, bench_both=True, n=n)

    def summ(name, rows):
        sp = [s for ok, s in rows if ok and s]
        print(f"{name}: correct={[ok for ok,_ in rows]}  speedups={[round(x,4) for x in sp]}")
        if sp:
            print(f"   mean={st.mean(sp):.4f}  spread=[{min(sp):.4f},{max(sp):.4f}]"
                  f"  cv={100*st.pstdev(sp)/st.mean(sp):.2f}%")
        return sp

    ls = summ("LEGACY  (per-impl processes)", legacy)
    bs = summ("BATCHED (--bench-both)      ", batched)
    if ls and bs:
        rel = abs(st.mean(bs) - st.mean(ls)) / st.mean(ls) * 100
        print(f"\n=> mean speedup delta: {rel:.2f}%  "
              f"({'PASS' if rel < 8 else 'REVIEW'}: batched vs legacy)")


if __name__ == "__main__":
    main()
