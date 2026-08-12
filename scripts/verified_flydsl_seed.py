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
import re
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


RETRY_PROMPT = """You are fixing a FlyDSL kernel that compiled and ran on gfx950 but did \
not match the reference. Return the complete corrected file defining `{entry}`, \
nothing else.

These are the only symbols available, with exact signatures. Pass exactly the \
arguments each signature shows; take each symbol from the module it is listed \
under.

flyc (flydsl.compiler): {flyc_api}

fx (flydsl.expr): {fx_api}

The kernel that failed:

```python
{kernel}
```

What the verifier reported:

{error}

How to read it: "too many positional arguments" or "missing a required \
argument" means the arity is wrong -- find the call above in the signature list. \
"module has no attribute" means the symbol is on the other module or does not \
exist. "object has no attribute" on a FlyDSL value means torch semantics were \
assumed; a FlyDSL value has no .shape, .reshape, .device or .dtype. "requires a \
Context" means IR was built outside the jitted function. An SNR far below zero \
with no Python error means it ran and computed wrong values -- re-derive the \
indexing and accumulation order."""


NO_OUTPUT_NOTE = """Your previous answer to this exact request came back with no \
usable file ({error}). Answer with the file itself and nothing around it: open \
with ```python on its own line, close the fence, and keep the file compact -- a \
working port of a kernel this size is 150 to 350 lines. Do not restate the \
task, do not describe your plan, do not add commentary after the fence."""


#: How a driver reaches into the candidate module. Generated tasks go through
#: kore.tasks._genops, which does getattr(mod, ref.entry_name); the hand-written
#: ones pull the attribute themselves, and those are the ones that disagree.
_DRIVER_ENTRY_PATTERNS = (
    r"return\s+mod\.([A-Za-z_]\w*)",
    r"getattr\(\s*mod\s*,\s*['\"]([A-Za-z_]\w*)['\"]",
    r"\bmod\.([A-Za-z_]\w*)\s*\(",
)


def driver_entry_name(task_dir: Path) -> str | None:
    """The function the driver will actually look for, read from the driver.

    spec_of derives the entry from task.yaml's ``operation``, which is the name
    of the task and not always the name of the function. For 11 of the 441
    frontier tasks these differ, and the difference is fatal on its own: the
    prompt orders a kernel called fused_rmsnorm_quant_fp8, the driver asks for
    quant, and the verifier answers "module 'candidate_kernel' has no attribute
    'quant'" no matter how good the kernel is. Those 11 are attention, MoE and
    quantization -- the tasks most worth having.
    """
    try:
        text = (task_dir / "driver.py").read_text(errors="ignore")
    except OSError:
        return None
    for pattern in _DRIVER_ENTRY_PATTERNS:
        m = re.search(pattern, text)
        if m:
            return m.group(1)
    return None


def publish_gate_rows(out_root: Path, passes: list[dict]) -> int:
    """Record passes where the harvester looks for them.

    Promotion does not read the seed root; it reads runs/<root>_gate.json and
    copies the rows that say "pass". A twin verified here but missing from that
    report is finished work that no later stage can see -- the same shape of
    failure as the gated twins that sat unharvested for hours, one step earlier.

    Re-gating on a GPU would produce this file, and would also re-run the exact
    verify_one() that already answered. Writing the answer down instead costs a
    file.
    """
    report = Path("runs") / f"{out_root.name}_gate.json"
    rows: dict[str, dict] = {}
    if report.is_file():
        try:
            for row in json.loads(report.read_text()).get("rows", []):
                if row.get("task_id"):
                    rows[row["task_id"]] = row
        except Exception:  # noqa: BLE001 - a corrupt report is not a reason to
            rows = {}       # discard verified work; it is rebuilt from here
    for p in passes:
        tid = p.get("twin_id") or f"{p['task_id']}__flydsl"
        rows[tid] = {"task_id": tid, "hip_twin_of": None,
                     "family": p.get("family"), "seconds": p.get("seconds"),
                     "status": "pass", "snr_db": p.get("snr_db")}
    counts: dict[str, int] = {}
    for row in rows.values():
        counts[row.get("status", "?")] = counts.get(row.get("status", "?"), 0) + 1
    payload = {"counts": counts, "rows": list(rows.values())}
    # The pipeline reads this file on its own schedule, so it must never observe
    # a half-written one.
    tmp = report.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(report)
    return len(passes)


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

    from materialize_pool_flydsl import _api_surface

    prompt, entry = _build_prompt(spec, source)
    last = ""
    last_seed = ""

    def message(attempt: int) -> list[dict]:
        """One user turn, fixed size, whatever the attempt number.

        Accumulating the conversation looked natural and was the thing that
        broke this: each rejected kernel was appended verbatim, so prompts ran
        19k, 25k, 35k, 40k, 49k characters and the gateway answered the later
        ones with APITimeout -- 55 timeouts against 7 completions in 95
        minutes. A retry needs the API, the kernel it is fixing and the error,
        and nothing it already saw.

        Every retry must differ from the prompt that just failed. Selecting on
        ``last_seed`` alone did not: a call that produced no kernel left it
        empty, so the retry re-sent the opening prompt byte for byte and drew
        the same failure. That is the whole of the 96-call run -- 24 tasks,
        four identical attempts each, one distinct request among them.
        """
        if attempt == 1 or not last:
            return [{"role": "user", "content": prompt}]
        if not last_seed:
            return [{"role": "user", "content":
                     f"{prompt}\n\n{NO_OUTPUT_NOTE.format(error=last[:500])}"}]
        return [{"role": "user", "content": RETRY_PROMPT.format(
            flyc_api=_api_surface("flydsl.compiler"),
            fx_api=_api_surface("flydsl.expr"),
            entry=entry,
            kernel=last_seed[:8000],
            error=last[:2000],
        )}]
    empty = 0
    for attempt in range(1, attempts + 1):
        try:
            reply = teacher.generate(message(attempt))
            # An empty reply is not a model that had nothing to say. The teacher
            # returns "" when Anthropic stopped at max_tokens, which here meant
            # the budget went entirely to thinking tokens, so saying "no launch
            # wrapper" pointed the retry at a kernel that was never written.
            if not reply.strip():
                empty += 1
                raise ValueError("the reply was empty -- the output limit was "
                                 "reached before any file was written")
            seed = _extract_code(reply)
            if "flyc.jit" not in seed:
                raise ValueError("no @flyc.jit launch wrapper")
            if f"def {entry}" not in seed:
                raise ValueError(f"does not define {entry!r}")
            if "from kernels" in seed or "import kernels" in seed:
                raise ValueError("imports kernels.* which is not on the path "
                                 "here; define any helper in the file itself")
        except Exception as exc:  # noqa: BLE001 - retry structural misses too
            last = f"{type(exc).__name__}: {exc}"
            continue

        task_dir = materialize(task_id, seed, out_root)
        verdict = verify_one(task_dir, timeout=timeout, gpu=gpu)
        if verdict.get("status") == "pass":
            return {"task_id": task_id, "status": "pass", "attempt": attempt,
                    "twin_id": task_dir.name, "snr_db": verdict.get("snr_db"),
                    "seconds": verdict.get("seconds")}

        # Keep the error verbatim. The summary the gate stores loses the
        # traceback line naming the offending call, which is the only part of
        # the message a fix can be derived from.
        detail = " ".join(verdict.get("diagnostics") or []) or verdict.get("error", "")
        last = (detail or "no diagnostic")[:4000]
        last_seed = seed
        # A failed candidate must not be left where the gate would count it as
        # a judged seed; the next attempt overwrites it, and the final failure
        # is removed below.
    # materialize() writes tasks/<id>__flydsl, and this removed tasks/<id>, so
    # it removed nothing: 453 directories had accumulated in a root holding 438
    # judged twins, every rejected candidate still sitting where the harvester
    # looks.
    shutil.rmtree(out_root / "tasks" / f"{task_id}__flydsl", ignore_errors=True)
    return {"task_id": task_id, "status": "failed", "attempt": attempts,
            "empty_replies": empty, "error": last[:300]}


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

    # Set before the teacher is built, and set here rather than in the job
    # script so a stale environment cannot reproduce the 96-call empty run.
    # Thinking off is what makes this work at all; 16k is then generous, since
    # the answer that used to be unreachable arrives in about 5k tokens.
    os.environ.setdefault("KORE_TEACHER_THINKING", "off")
    # 16k truncated 3 of the first 53 calls, and a truncated call costs a whole
    # attempt. The ceiling only matters for the kernels that run long, since a
    # typical answer here is about 5k tokens.
    os.environ["KORE_TEACHER_MAX_TOKENS"] = os.environ.get(
        "KORE_FLYDSL_SEED_MAX_TOKENS", "24576")

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

    # A task that already passed is finished work, and this job is preemptible
    # on burst capacity, so it has to be able to start again without paying for
    # what it already bought. Failures are not skipped: they were judged by a
    # verifier the model never got to see an answer from, and are exactly the
    # ones worth retrying.
    done_ids = set()
    ledger_path = out_root / "verified_seed_attempts.jsonl"
    if ledger_path.is_file():
        for line in ledger_path.read_text(errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001 - a torn last line is not a verdict
                continue
            if row.get("status") == "pass" and row.get("task_id"):
                done_ids.add(row["task_id"])
    if done_ids:
        print(f"resuming: {len(done_ids)} task(s) already passed, skipping them",
              flush=True)

    work = []
    renamed: list[tuple[str, str | None, str]] = []
    for tid in ids:
        if tid in done_ids:
            continue
        d = Path(args.source_root) / tid
        if not (d / "task.yaml").is_file():
            continue
        try:
            spec = spec_of(d)
        except Exception:  # noqa: BLE001 - a task we cannot read is not a task
            continue
        real = driver_entry_name(d)
        if real and real != spec.get("entry_name"):
            renamed.append((tid, spec.get("entry_name"), real))
            spec["entry_name"] = real
        src = (d / "reference.py").read_text(errors="ignore")
        work.append((tid, spec, src))

    if renamed:
        print(f"entry name taken from the driver for {len(renamed)} task(s) "
              f"whose task.yaml disagreed:", flush=True)
        for tid, was, now in renamed:
            print(f"    {tid}: {was} -> {now}", flush=True)

    print(f"verified FlyDSL seeding: {len(work)} tasks, up to {args.attempts} "
          f"attempts each, {args.workers} workers over GPUs {gpus}", flush=True)
    if not work:
        print("nothing left to seed from this list", flush=True)
        return 0

    teacher = make_teacher("claude", resilient=True)

    # One call before committing the node. The previous run held eight GPUs for
    # 44 minutes and made 96 requests that every one of them came back empty
    # from, because nothing checked that the teacher could answer this prompt
    # at all. That is a four-minute question, so ask it first.
    probe_id, probe_spec, probe_src = work[0]
    probe_prompt, probe_entry = mpf._build_prompt(probe_spec, probe_src)
    t_probe = time.time()
    probe_reply = teacher.generate([{"role": "user", "content": probe_prompt}])
    probe_code = mpf._extract_code(probe_reply)
    print(f"preflight on {probe_id}: {time.time() - t_probe:.0f}s, "
          f"{len(probe_reply)} chars, {len(probe_code)} extracted, "
          f"jit={'flyc.jit' in probe_code}", flush=True)
    if not probe_reply.strip():
        print("preflight: the teacher returned nothing. The output budget is "
              "being spent before any text is written -- check that thinking "
              "is disabled and KORE_TEACHER_MAX_TOKENS is not tiny.",
              file=sys.stderr)
        return 3
    if "flyc.jit" not in probe_code:
        print(f"preflight: reply carried no @flyc.jit launch wrapper; refusing "
              f"to spend the node. First 400 chars:\n{probe_reply[:400]}",
              file=sys.stderr)
        return 3

    done = {"pass": 0, "failed": 0}
    passes: list[dict] = []
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
            # Publish on every pass rather than at the end. This job runs on
            # preemptible capacity, and a twin verified at hour one should be
            # promotable even if the allocation is lost at hour two.
            if row["status"] == "pass":
                passes.append(row)
                publish_gate_rows(out_root, [row])
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
    print(f"published {len(passes)} pass row(s) to "
          f"runs/{out_root.name}_gate.json for the harvester", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
