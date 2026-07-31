import os, sys, json, time, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutTimeout
sys.path.insert(0, "/home/shasriva/kore_offline"); sys.path.insert(0, "/home/shasriva/Kore-RL/KORE")
import kernel_math as km
from kore.data.teacher import make_teacher

UNIQ = int(os.environ.get("T1_UNIQ", "10000"))      # target UNIQUE problems to teacher-call
WORKERS = int(os.environ.get("T1_WORKERS", "128"))
BUDGET = int(os.environ.get("T1_BUDGET", "3000"))
OUT = "/home/shasriva/kore_offline/gen/kernel_math.jsonl"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
teacher = make_teacher("claude", resilient=True)

# 1) pre-dedup problems so every teacher call is a UNIQUE question
probs = []; seen = set(); seed = 0
while len(probs) < UNIQ and seed < UNIQ * 20:
    msgs, gt, chk, name = km.build_problem(seed); seed += 1
    h = hashlib.sha256(msgs[1]["content"].encode()).hexdigest()
    if h in seen:
        continue
    seen.add(h); probs.append((msgs, gt, chk, name))
print(f"unique problems built={len(probs)} (scanned {seed} seeds)", flush=True)


def one(spec):
    msgs, gt, chk, name = spec
    try:
        r = teacher.generate(msgs)
    except Exception:
        return None
    if not r or not km.verify(r, chk, gt):
        return None
    return {"messages": msgs + [{"role": "assistant", "content": r}],
            "_source": "kernel_math", "_qa_type": name, "_arch": "gfx950",
            "_verified": True, "_gt": gt}


t0 = time.time(); kept = 0; done = 0; deadline = t0 + BUDGET
ex = ThreadPoolExecutor(max_workers=WORKERS)
futs = [ex.submit(one, s) for s in probs]
try:
    with open(OUT, "w") as f:
        it = as_completed(futs, timeout=BUDGET)
        while True:
            if time.time() >= deadline:
                print(f"DEADLINE kept={kept}", flush=True); break
            try:
                fut = next(it)
            except (StopIteration, FutTimeout):
                break
            done += 1
            row = fut.result()
            if row:
                f.write(json.dumps(row) + "\n"); kept += 1
            if done % 500 == 0:
                print(f"done={done}/{len(probs)} kept={kept} verify={kept/max(1,done):.2f} {time.time()-t0:.0f}s", flush=True)
finally:
    ex.shutdown(wait=False, cancel_futures=True)
print(f"TIER1_DONE kept={kept} done={done} {time.time()-t0:.0f}s -> {OUT}", flush=True)
