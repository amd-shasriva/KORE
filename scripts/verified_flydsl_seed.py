#!/usr/bin/env python3
"""Seed FlyDSL twins with the verifier inside the generation loop.

One-shot seeding does not work for FlyDSL and the numbers say so plainly. The
last gate judged 438 twins written by claude-opus-5 and passed 44 -- 10%,
against 8.9% for opus-4.8, so a better model bought nothing. In the arena the
same model scores 98% on triton2flydsl, and the only structural difference is
that the arena lets it see the failure and try again.

That is what this does. Generate, compile and check on the spot, hand the exact
error back, regenerate. The existing pipeline does the same thing eventually --
seed, wait for a gate job, repair from the verdict -- but a round trip takes a
gate cycle and the repair loop was measured rescuing zero FlyDSL kernels,
because by then the error is a summary rather than the traceback.

Why the errors are worth handing back: of 394 failures in that gate, 124 were
"too many positional arguments", 28 "module has no attribute", 21 "object has
no attribute". Those are API mistakes with an obvious correction, not wrong
algorithms -- exactly the class of error a second attempt fixes.

Runs on a GPU node because verification compiles and executes the kernel. A
task that passes is left in the output root as an ordinary seed, so the gate,
harvest and mining stages downstream need no changes and will simply find it
already correct.

    python scripts/verified_flydsl_seed.py --task-list runs/flydsl_retry.txt \\
        --out data/registry_flydsl_frontier --attempts 4 --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

FEEDBACK = """Your previous FlyDSL port of this kernel was compiled and run against the \
reference on gfx950. It did not pass. Here is exactly what happened:

{error}

Rewrite the kernel so this specific failure cannot happen. Notes on reading the \
error:

* "too many positional arguments" or "missing a required argument" means you \
called a FlyDSL function with the wrong arity. Find that call in the API listing \
above and pass exactly the arguments its signature shows.
* "module has no attribute" means the symbol is not on that module. It may be on \
the other one -- `from_torch_tensor` is on flyc, not fx -- or it may not exist at \
all, in which case build the behaviour from symbols that do.
* "object has no attribute" on a FlyDSL value usually means torch semantics were \
assumed. A FlyDSL value has no `.shape`, `.reshape`, `.device` or `.dtype`.
* "requires a Context" means IR was built outside the compiler's scope. All IR \
construction belongs inside the jitted function.
* An SNR far below zero with no Python error means the kernel ran and computed \
the wrong values. Re-derive the indexing and the accumulation order.

Return the complete corrected file. Do not explain."""


def _load_ids(path: str) -> list[str]:
    return [ln.split("#", 1)[0].strip()
            for ln in Path(path).read_text().splitlines()
            if ln.split("#", 1)[0].strip()]


def seed_with_feedback(task_id: str, spec: dict, source: str, teacher,
                       out_root: Path, attempts: int, timeout: int,
                       gpu: int) -> dict:
    """Generate, verify, and retry on the error until it passes or runs out."""
    from materialize_pool_flydsl import (_build_prompt, _extract_code,
                                         materialize)
    from verify_pool_hip_seeds import verify_one

    prompt, entry = _build_prompt(spec, source)
    history: list[dict] = [{"role": "user", "content": prompt}]
    last = ""
    for attempt in range(1, attempts + 1):
        try:
            reply = teacher.generate(history)
            seed = _extract_code(reply)
            if "flyc.jit" not in seed:
                raise ValueError("no @flyc.jit launch wrapper")
            if f"def {entry}" not in seed:
                raise ValueError(f"does not define {entry!r}")
        except Exception as exc:  # noqa: BLE001 - retry structural misses too
            last = f"{type(exc).__name__}: {exc}"
            history += [{"role": "assistant", "content": "(unusable reply)"},
                        {"role": "user", "content": FEEDBACK.format(error=last)}]
            continue

        task_dir = materialize(task_id, seed, out_root)
        verdict = verify_one(task_dir, timeout=timeout, gpu=gpu)
        if verdict.get("status") == "pass":
            return {"task_id": task_id, "status": "pass", "attempt": attempt,
                    "snr_db": verdict.get("snr_db")}

        # Keep the error verbatim. The summary the gate stores loses the
        # traceback line naming the offending call, which is the only part of
        # the message a fix can be derived from.
        detail = " ".join(verdict.get("diagnostics") or []) or verdict.get("error", "")
        last = (detail or "no diagnostic")[:4000]
        history += [{"role": "assistant", "content": seed},
                    {"role": "user", "content": FEEDBACK.format(error=last)}]
        # A failed candidate must not be left where the gate would count it as
        # a judged seed; the next attempt overwrites it, and the final failure
        # is removed below.
    shutil.rmtree(out_root / "tasks" / task_id, ignore_errors=True)
    return {"task_id": task_id, "status": "failed", "attempt": attempts,
            "error": last[:300]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-list", required=True)
    ap.add_argument("--out", default="data/registry_flydsl_frontier")
    ap.add_argument("--source-root", default="kore/tasks")
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from kore.data.teacher import load_env_local, make_teacher
    from kore.data.twins import spec_of
    load_env_local()

    # materialize() copies driver.py and reference.py from a module-level POOL
    # that defaults to the external task pool, and main() in that module is
    # what normally repoints it. Importing the function without setting it sent
    # every task looking for its driver under data/task_pool and failing with
    # FileNotFoundError before a single kernel was verified.
    import materialize_pool_flydsl as mpf
    mpf.POOL = Path(args.source_root).resolve()
    if not mpf.POOL.is_dir():
        print(f"source root does not exist: {mpf.POOL}", file=sys.stderr)
        return 2
    print(f"source root: {mpf.POOL}")

    out_root = Path(args.out)
    ids = _load_ids(args.task_list)
    if args.limit:
        ids = ids[: args.limit]
    gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""]

    work = []
    for tid in ids:
        d = Path(args.source_root) / tid
        if not (d / "task.yaml").is_file():
            continue
        try:
            spec = spec_of(d)
        except Exception:  # noqa: BLE001 - a task we cannot read is not a task
            continue
        src = (d / "reference.py").read_text(errors="ignore")
        work.append((tid, spec, src))

    print(f"verified FlyDSL seeding: {len(work)} tasks, up to {args.attempts} "
          f"attempts each, {args.workers} workers over GPUs {gpus}", flush=True)

    teacher = make_teacher("claude", resilient=True)
    done = {"pass": 0, "failed": 0}
    started = time.time()
    ledger = out_root / "verified_seed_attempts.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(seed_with_feedback, tid, spec, src, teacher, out_root,
                        args.attempts, args.timeout, gpus[i % len(gpus)]): tid
            for i, (tid, spec, src) in enumerate(work)
        }
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                row = fut.result()
            except Exception as exc:  # noqa: BLE001
                row = {"task_id": futs[fut], "status": "failed",
                       "error": f"{type(exc).__name__}: {exc}"[:200]}
            done[row["status"]] = done.get(row["status"], 0) + 1
            with ledger.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            if n % 10 == 0 or row["status"] == "pass":
                rate = 100 * done["pass"] / max(n, 1)
                print(f"  [{n}/{len(work)}] {row['task_id']}: {row['status']}"
                      f" (attempt {row.get('attempt')})  running pass rate "
                      f"{rate:.0f}%", flush=True)

    mins = (time.time() - started) / 60
    total = sum(done.values()) or 1
    print(f"\nverified seeding done in {mins:.0f} min: "
          f"{done['pass']}/{total} passed ({100*done['pass']/total:.0f}%) "
          f"-- one-shot seeding scored 10%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
