#!/usr/bin/env python
"""Show a failing twin its own error and let the teacher fix it.

Seeding is one-shot: the teacher writes a kernel, the gate runs it on real
gfx950, and a kernel that fails is discarded. For HIP that is affordable --
96.5% of pool twins pass. For FlyDSL it is most of the work: of 3,974 gated
ports, 173 passed. And the failures are overwhelmingly not bad reasoning about
the kernel. 3,109 of them crash before producing a number at all, on things
like ``fx.from_torch_tensor`` (it is on ``flyc``), ``as_numeric()`` given two
arguments, or ``Vector.load()`` missing three -- each reported by the gate in
one precise line that the model never sees.

So the gate's verdict becomes the next prompt. The kernel and the exact error
go back to the teacher, which returns a revised file, and the stale verdict is
dropped so the next gate pass re-runs it. This is the same loop a person would
run by hand and the reason it was not happening is that nothing connected the
two halves.

Bounded on purpose:

* ``--max-attempts`` per task, ledgered, so a kernel that cannot be fixed stops
  costing teacher calls instead of looping forever.
* Only failures carrying a diagnosable error are retried. A verdict with no
  message gives the teacher nothing to work with, and asking it to "try again"
  is just re-rolling the dice at full price.

    python scripts/repair_twin_seeds.py --root data/pool_flydsl \\
        --gate runs/pool_flydsl_gate.json --limit 200
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

#: Seed filename and the structural marker a repaired kernel must still carry,
#: per twin dialect. The marker is what the materializer checks at write time;
#: a "fix" that drops the entry point is not a fix.
DIALECTS = {
    "__flydsl": ("seed_flydsl.py", "flyc.jit", "python"),
    "__hip": ("seed_hip.hip", "PYBIND11_MODULE", "cpp"),
    "__hipf": ("seed_hip.hip", "PYBIND11_MODULE", "cpp"),
}

REPAIR_PROMPT = """The kernel below was written for this task and then run on a \
real AMD MI355X (gfx950). It failed. Your job is to return a corrected version \
of the whole file.

This is the failure the hardware and harness reported:

```
{error}
```

This is the current kernel:

```{lang}
{kernel}
```

{api}

Rules:

1. Fix the reported failure. Do not rewrite the kernel from scratch, and do not
   change the approach unless the error shows the approach itself cannot work.
2. Keep the entry point and its name exactly as they are. The harness looks up
   that symbol and nothing else.
3. Read every extent from the tensors at runtime. The harness re-runs at other
   shapes, so a size baked in passes one case and fails the rest.
4. If the error is an out-of-bounds or illegal memory access, the cause is
   almost always a tile that crosses the end of a tensor: predicate the copy
   instead of shrinking the problem.

Return ONLY the complete corrected file, in a single ```{lang} code block.
"""


def _suffix(task_id: str):
    for suffix, meta in DIALECTS.items():
        if task_id.endswith(suffix):
            return suffix, meta
    return None, None


def _error_text(row: dict, limit: int = 4000) -> str:
    """The most specific failure text available, tail-first.

    The tail is where the exception is: a traceback's first lines are the
    harness calling into the candidate, which is identical for every task and
    tells the teacher nothing.
    """
    parts = []
    diag = row.get("diagnostics")
    if diag:
        parts.append(str(diag))
    err = row.get("error") or ""
    if err:
        parts.append(err[-limit:])
    return "\n".join(parts)[-limit:].strip()


#: What a verdict has to contain before it is worth a teacher call. Presence of
#: a named failure, not length: "SNR: -999.00 dB" is 15 characters and is the
#: single most common thing the gate says about a FlyDSL port.
_DIAGNOSABLE = re.compile(r"Error|error:|Exception|SNR|allclose|failed")


def _diagnosable(text: str) -> bool:
    """Whether the verdict says anything a fix could be based on."""
    return bool(text) and bool(_DIAGNOSABLE.search(text))


def load_attempts(path: Path) -> dict:
    counts: dict = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001 - torn line after a kill
                continue
            tid = rec.get("task_id")
            if tid:
                counts[tid] = counts.get(tid, 0) + 1
    return counts


def failing_rows(gate: Path) -> list:
    if not gate.is_file():
        return []
    try:
        rows = json.loads(gate.read_text()).get("rows", [])
    except Exception:  # noqa: BLE001 - a torn verdict file is a real condition
        return []
    return [r for r in rows if r.get("status") != "pass" and r.get("task_id")]


def drop_verdicts(gate: Path, task_ids: set) -> int:
    """Forget the verdicts for repaired tasks so the gate re-runs them.

    The gate resumes from this file. A repaired kernel whose old verdict is
    still recorded is never looked at again, which would make the whole pass a
    no-op that reports success.
    """
    if not task_ids or not gate.is_file():
        return 0
    import collections

    data = json.loads(gate.read_text())
    keep = [r for r in data.get("rows", []) if r.get("task_id") not in task_ids]
    dropped = len(data.get("rows", [])) - len(keep)
    data["rows"] = keep
    data["counts"] = dict(collections.Counter(r.get("status") for r in keep))
    tmp = gate.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(gate)
    return dropped


#: Built once. Introspecting the package costs an import and ~15k characters,
#: and this is called on every repair across eight workers -- and the sys.path
#: insert it needs would stack one duplicate entry per call.
_API_BLOCK: dict = {}


def _api_block(dialect: str) -> str:
    if dialect != "__flydsl":
        return ""
    if "flydsl" not in _API_BLOCK:
        scripts = str(REPO / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        from materialize_pool_flydsl import _api_surface

        _API_BLOCK["flydsl"] = (
            "These are the ONLY symbols the FlyDSL modules export, with exact "
            "signatures. Take each from the module it is listed under and pass "
            "exactly the arguments shown.\n\n"
            f"flyc: {_api_surface('flydsl.compiler')}\n\n"
            f"fx: {_api_surface('flydsl.expr')}\n")
    return _API_BLOCK["flydsl"]


def repair_one(item, teacher, root: Path) -> dict:
    task_id, error = item
    suffix, meta = _suffix(task_id)
    if meta is None:
        return {"task_id": task_id, "status": "skipped", "error": "unknown dialect"}
    seed_name, marker, lang = meta
    seed_path = root / "tasks" / task_id / seed_name
    if not seed_path.is_file():
        return {"task_id": task_id, "status": "skipped", "error": "no seed on disk"}

    kernel = seed_path.read_text(errors="ignore")
    prompt = REPAIR_PROMPT.format(error=error, kernel=kernel[:20000], lang=lang,
                                  api=_api_block(suffix))
    try:
        from kore.data.twins import extract_code

        reply = teacher.generate([{"role": "user", "content": prompt}])
        fixed = extract_code(reply, must_contain=marker)
        if marker not in fixed:
            raise ValueError(f"repair dropped {marker}")
        if len(fixed) < 200:
            raise ValueError("repair suspiciously short")
        if fixed.strip() == kernel.strip():
            raise ValueError("repair changed nothing")
        seed_path.write_text(fixed)
        return {"task_id": task_id, "status": "repaired", "chars": len(fixed)}
    except Exception as exc:  # noqa: BLE001 - one bad repair must not end the pass
        return {"task_id": task_id, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}"[:200]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="twin root, e.g. data/pool_flydsl")
    ap.add_argument("--gate", default="", help="verdict json (default: runs/<root>_gate.json)")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-attempts", type=int, default=2,
                    help="repairs per task before giving up on it")
    ap.add_argument("--teacher", default="claude")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = REPO / args.root
    gate = Path(args.gate) if args.gate else REPO / "runs" / f"{root.name}_gate.json"
    ledger = root / "repair_attempts.jsonl"

    attempts = load_attempts(ledger)
    selected = []
    skipped_spent = skipped_blank = 0
    for row in failing_rows(gate):
        tid = row["task_id"]
        if attempts.get(tid, 0) >= args.max_attempts:
            skipped_spent += 1
            continue
        text = _error_text(row)
        if not _diagnosable(text):
            skipped_blank += 1
            continue
        selected.append((tid, text))
        if len(selected) >= args.limit:
            break

    print(f"repairable: {len(selected)}"
          + (f"; {skipped_spent} already at --max-attempts" if skipped_spent else "")
          + (f"; {skipped_blank} with no diagnosable error" if skipped_blank else ""))
    if args.dry_run:
        for tid, text in selected[:5]:
            first = next((l for l in text.splitlines() if l.strip()), "")
            print(f"  {tid}\n      {first[:150]}")
        return 0
    if not selected:
        return 0

    from kore.data.teacher import load_env_local, make_teacher

    load_env_local()
    teacher = make_teacher(args.teacher, resilient=True)

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    lock = threading.Lock()
    ok = fail = 0
    repaired: set = set()
    with ledger.open("a") as handle, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(repair_one, it, teacher, root) for it in selected]
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            if rec["status"] == "repaired":
                ok += 1
                repaired.add(rec["task_id"])
            else:
                fail += 1
            with lock:
                handle.write(json.dumps(rec) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if i % 25 == 0:
                print(f"  [{i}/{len(selected)}] repaired={ok} failed={fail}", flush=True)

    dropped = drop_verdicts(gate, repaired)
    print(f"\nrepaired {ok}, failed {fail}; dropped {dropped} stale verdict(s) "
          f"so the gate re-runs them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
