#!/usr/bin/env python
"""Run our model over AgentKernelArena tasks and score it against the Opus bar.

Three subcommands, deliberately separable so a failure in one does not cost the
others' GPU time:

  discover   list gfx950-runnable tasks by type (no GPU, no model)
  baseline   time the UNMODIFIED kernel of each task, establishing the reference
  run        generate an optimized kernel per task, then compile/check/time it

Each task runs in its own copied workspace. That is not tidiness: the tasks are
checked into a git tree we do not own, and the harnesses write artefacts next to
the source, so editing in place would corrupt the benchmark for every subsequent
run and silently change what later numbers mean.

Scoring lives in kore/eval/agent_kernel_arena.py and is AKA's own formula
unchanged, so the output is directly comparable to the published results:

  PyTorch-to-HIP 6.89x | HIP-to-HIP 6.69x | Triton-to-Triton 2.13x  (Opus)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kore.eval.agent_kernel_arena import (  # noqa: E402
    ArenaResult, discover_tasks, evaluate_task, summarize)

DEFAULT_ARENA = Path.home() / "third_party" / "AgentKernelArena"

PROMPT = """You are optimizing a GPU kernel for an AMD MI355X (gfx950).

{instructions}

Rewrite the file `{filename}` so the kernel is FASTER while producing
numerically identical results. Keep the function name(s) {targets} and the
module's public interface exactly as they are -- the test harness imports them
by name.

Return ONLY the complete new contents of `{filename}` in a single ```python
code block, with no commentary before or after.

Current implementation:
```python
{source}
```"""


def _extract_code(text: str) -> str:
    """Pull the first fenced python block; fall back to the whole reply.

    A model that ignores the fence still usually emits valid source, and
    refusing it would score a formatting slip as a compile failure.
    """
    import re

    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def _workspace(task, out_root: Path) -> Path:
    ws = out_root / "workspaces" / task.task_id.replace("/", "__")
    if ws.exists():
        shutil.rmtree(ws)
    ws.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(task.root, ws)
    return ws


def cmd_discover(args) -> int:
    tasks = discover_tasks(args.arena_root, task_types=args.types or None,
                           gpu_arch=args.gpu_arch)
    by = {}
    for t in tasks:
        by.setdefault(t.task_type, []).append(t)
    print(f"{len(tasks)} tasks runnable on {args.gpu_arch}")
    for k, v in sorted(by.items()):
        print(f"  {len(v):>4}  {k}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"n": len(tasks), "by_type": {k: len(v) for k, v in by.items()},
             "task_ids": [t.task_id for t in tasks]}, indent=2) + "\n")
    return 0


def cmd_baseline(args) -> int:
    """Time each task's shipped kernel, unmodified.

    This is the control. Without it a low score is ambiguous between "our model
    wrote a slow kernel" and "this task does not run on our stack at all", and
    those call for completely different responses.
    """
    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    tasks = discover_tasks(args.arena_root, task_types=args.types or None,
                           gpu_arch=args.gpu_arch)[: args.limit or None]
    results = []
    for i, task in enumerate(tasks, 1):
        ws = _workspace(task, out_root)
        r = evaluate_task(task, ws, timeout=args.timeout)
        results.append(r)
        print(f"[{i}/{len(tasks)}] {task.task_id}: compiled={r.compiled} "
              f"correct={r.correct} speedup={r.speedup} score={r.score:.0f}",
              flush=True)
        if not args.keep_workspaces:
            shutil.rmtree(ws, ignore_errors=True)
    _write(out_root / "baseline_results.json", results, args)
    return 0


def cmd_run(args) -> int:
    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    tasks = discover_tasks(args.arena_root, task_types=args.types or None,
                           gpu_arch=args.gpu_arch)[: args.limit or None]
    print(f"{len(tasks)} tasks; model={args.model}", flush=True)

    from kore.eval.policies import model_policy  # local: heavy import

    # model_policy takes the checkpoint positionally and has no `revision` or
    # `model_id` parameter; the previous call passed both by keyword and raised
    # TypeError before a single task ran. `revision` belongs to the hub-pinning
    # path and is meaningless for a local checkpoint directory, which is what an
    # evaluated run always points at.
    policy = model_policy(args.model,
                          max_tokens=args.max_tokens,
                          temperature=args.temperature)

    # Durable, append-as-you-go ledger. The previous version accumulated results
    # in memory and wrote once after the loop, so a preempted run lost every task
    # it had scored -- and on this cluster preemption is the normal way a long job
    # ends, not an exception. A full sweep is 254 tasks at 6-11 minutes each, so
    # that was throwing away many hours at a time.
    ledger = out_root / f"results_{args.arm}.partial.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    results = []
    done: set[str] = set()
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001 - a torn last line is expected after a kill
                continue
            if row.get("task_id"):
                done.add(row["task_id"])
                results.append(_result_from_dict(row))
        print(f"resume: {len(done)} task(s) already scored in {ledger.name}",
              flush=True)

    t0 = time.time()
    for i, task in enumerate(tasks, 1):
        if task.task_id in done:
            continue
        ws = _workspace(task, out_root)
        try:
            src_rel = task.source_files[0] if task.source_files else "kernel.py"
            source = (ws / src_rel).read_text() if (ws / src_rel).exists() else ""
            prompt = PROMPT.format(
                instructions=task.instructions or "Optimize this kernel.",
                filename=src_rel, targets=", ".join(task.target_functions) or "all",
                source=source)
            reply = policy(prompt)
            (ws / src_rel).write_text(_extract_code(reply))
            r = evaluate_task(task, ws, timeout=args.timeout)
        except Exception as exc:  # noqa: BLE001 - one bad task must not end the run
            r = ArenaResult(task_id=task.task_id, task_type=task.task_type,
                            error=f"{type(exc).__name__}: {exc}")
        results.append(r)
        # Append + flush + fsync before moving on. A score that is only in memory
        # is a score we will pay for twice.
        with ledger.open("a") as fh:
            fh.write(json.dumps(r.to_dict()) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        print(f"[{i}/{len(tasks)}] {task.task_id}: compiled={r.compiled} "
              f"correct={r.correct} speedup={r.speedup} score={r.score:.0f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if not args.keep_workspaces:
            shutil.rmtree(ws, ignore_errors=True)
    _write(out_root / f"results_{args.arm}.json", results, args)
    return 0



def _result_from_dict(row: dict) -> "ArenaResult":
    """Rebuild an ArenaResult from a ledger line.

    Only the fields ArenaResult actually declares are passed through: the ledger
    is written by to_dict() and may gain keys later, and a resumed run must not
    die because it was written by a newer build than the one reading it.
    """
    import dataclasses
    allowed = {f.name for f in dataclasses.fields(ArenaResult)}
    return ArenaResult(**{k: v for k, v in row.items() if k in allowed})


def _write(path: Path, results, args) -> None:
    summary = summarize(results)
    path.write_text(json.dumps(
        {"summary": summary, "results": [r.to_dict() for r in results]},
        indent=2) + "\n")
    print("\n==== SUMMARY ====")
    for ttype, row in summary["by_type"].items():
        bar = row.get("opus_published_mean_speedup")
        verdict = ""
        if bar is not None:
            verdict = ("  BEATS Opus" if row.get("beats_opus")
                       else f"  (Opus {bar}x)")
        ms = row["mean_speedup"]
        print(f"  {ttype:<18} n={row['n']:<4} correct={row['correct']:<4} "
              f"mean_speedup={ms if ms is None else round(ms, 3)}{verdict}")
    print(f"\nwrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("discover", "baseline", "run"))
    ap.add_argument("--arena-root", default=str(DEFAULT_ARENA))
    ap.add_argument("--gpu-arch", default="gfx950")
    ap.add_argument("--types", nargs="*", default=[],
                    help="task types, e.g. triton2triton hip2hip torch2hip")
    ap.add_argument("--out", default="runs/aka")
    ap.add_argument("--arm", default="kore")
    ap.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--keep-workspaces", action="store_true")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()
    return {"discover": cmd_discover, "baseline": cmd_baseline,
            "run": cmd_run}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
