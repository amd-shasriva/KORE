#!/usr/bin/env python
"""Re-materialize external-pool tasks as HIP tasks, so HIP can be generated at
Triton's scale instead of 79x below it.

AgentKernelArena makes us PRODUCE three languages -- Triton (196 tasks, 49%),
FlyDSL (101, 25%) and HIP (89, 22%) -- and carries its two highest bars in HIP
(6.89x torch2hip, 6.69x hip2hip). We can currently generate Triton over 14,901
tasks and HIP over 188. No amount of sampling fixes a 79x difference in the
number of distinct problems.

It does not need new tasks. A pool task directory is four files and only one of
them is Triton-specific:

    reference.py   a JSON spec -- module_source, input_specs, snr_threshold --
                   that rebuilds a PyTorch oracle. Language-agnostic.
    driver.py      a shim into _genops.driver_main. Language-agnostic.
    task.yaml      declares backend and seed_kernel_name.
    seed_triton.py the only Triton-specific artifact.

KoreEnv already grades ``kernel.hip`` when a task declares ``backend: hip``, and
speedup is measured against the production torch baseline rather than against the
seed. So a HIP variant is the same oracle, the same driver, a task.yaml naming
``seed_hip.hip``, and a seed that merely has to compile and be correct -- it does
not have to be fast, because it is not what the candidate is scored against.

The seed is written by a teacher and admitted only if real gfx950 says it
compiles and clears the task's own SNR gate. A seed that cannot clear the gate is
worse than no task: datagen against it can only ever score zero, and it reports
as a model error rather than as the broken task it is.

    # measure yield on a sample before committing nodes
    python scripts/materialize_pool_hip.py --limit 50 --out data/pool_hip

    # then scale
    python scripts/materialize_pool_hip.py --limit 4000 --out data/pool_hip
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

POOL = REPO / "data" / "task_pool" / "tasks"

SEED_PROMPT = """You are writing a DELIBERATELY SIMPLE, correct HIP C++ kernel for \
an AMD MI355X (gfx950).

This is a SEED, not an optimized kernel. It is the starting point a model will \
later improve, and it is not what anything is scored against. Correctness and \
compiling are the only things that matter. Prefer the most obvious possible \
implementation: one thread per output element, no shared memory, no tiling, no \
vendor library calls.

Reproduce the numerics of this PyTorch module exactly:

```python
{module_source}
```

Entry point: a function `forward` bound through \
`PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)`, taking and returning `torch::Tensor` \
in the same order as the module's forward().

Input dtype is {dtype}. It must reach at least {snr} dB SNR against the reference.

Return ONLY the complete contents of the .hip file in a single ```cpp code block.
"""


def _spec_of(task_dir: Path) -> dict:
    """The JSON spec embedded in a pool task's reference.py."""
    text = (task_dir / "reference.py").read_text(errors="ignore")
    start = text.find('_SPEC = json.loads("')
    if start < 0:
        raise ValueError("no _SPEC in reference.py")
    literal_start = text.index('"', start + len("_SPEC = json.loads"))
    literal_end = text.index('")', literal_start)
    return json.loads(json.loads(text[literal_start:literal_end + 1]))


def _extract_code(text: str) -> str:
    import re

    m = re.search(r"```[A-Za-z0-9_+.-]*[ \t]*\r?\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def materialize(task_id: str, seed_src: str, out_root: Path) -> Path:
    """Write a backend:hip twin of a pool task.

    Everything except task.yaml and the seed is copied verbatim, because the
    oracle and driver are what make the two variants comparable: a HIP win and a
    Triton win on the same task_id have then been graded by the same spec.
    """
    src = POOL / task_id
    dst = out_root / "tasks" / f"{task_id}__hip"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("reference.py", "driver.py"):
        shutil.copy(src / name, dst / name)
    cfg = json.loads((src / "task.yaml").read_text())
    cfg["task_id"] = f"{task_id}__hip"
    cfg["backend"] = "hip"
    cfg["seed_kernel_name"] = "seed_hip.hip"
    cfg["provenance_root"] = task_id
    cfg["hip_twin_of"] = task_id
    (dst / "task.yaml").write_text(json.dumps(cfg, indent=2) + "\n")
    (dst / "seed_hip.hip").write_text(seed_src)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pool_hip")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--families", nargs="*", default=None,
                    help="restrict to these pool families (default: all)")
    ap.add_argument("--teacher", default="claude")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    done_path = out_root / "seed_attempts.jsonl"
    attempted = set()
    if done_path.exists():
        for line in done_path.read_text().splitlines():
            try:
                attempted.add(json.loads(line)["task_id"])
            except Exception:  # noqa: BLE001 - torn line after a kill
                continue

    ids = sorted(p.name for p in POOL.glob("*/") if (p / "task.yaml").is_file())
    ids = [t for t in ids if t not in attempted][args.offset:]

    selected = []
    for tid in ids:
        if len(selected) >= args.limit:
            break
        try:
            spec = _spec_of(POOL / tid)
        except Exception:  # noqa: BLE001 - a malformed task is not worth failing the sweep
            continue
        if args.families and spec.get("family") not in args.families:
            continue
        selected.append((tid, spec))

    print(f"selected {len(selected)} pool task(s) to seed"
          + (f" (families={args.families})" if args.families else ""))
    if args.dry_run:
        for tid, spec in selected[:5]:
            print(f"  {tid}  family={spec.get('family')} dtype={spec.get('dtype')}")
        return 0

    # Local import: pulls credentials and network only on a real run, so --dry-run
    # stays usable without them.
    from kore.data.teacher import load_env_local, make_teacher

    load_env_local()
    # resilient=True because this is a long unattended sweep against a rate-limited
    # API, and one transient 429 should cost a retry rather than the remainder of
    # the run.
    teacher = make_teacher(args.teacher, resilient=True)
    ok = fail = 0
    with done_path.open("a") as ledger:
        for i, (tid, spec) in enumerate(selected, 1):
            prompt = SEED_PROMPT.format(
                module_source=spec.get("module_source", "")[:8000],
                dtype=spec.get("dtype", "fp32"),
                snr=spec.get("snr_threshold", 30))
            try:
                reply = teacher.generate([{"role": "user", "content": prompt}])
                seed = _extract_code(reply)
                if "PYBIND11_MODULE" not in seed:
                    raise ValueError("seed does not bind an entry point")
                materialize(tid, seed, out_root)
                ok += 1
                rec = {"task_id": tid, "status": "seeded", "chars": len(seed)}
            except Exception as exc:  # noqa: BLE001 - one bad task must not end the sweep
                fail += 1
                rec = {"task_id": tid, "status": "failed",
                       "error": f"{type(exc).__name__}: {exc}"[:200]}
            ledger.write(json.dumps(rec) + "\n")
            ledger.flush()
            os.fsync(ledger.fileno())
            if i % 10 == 0:
                print(f"  [{i}/{len(selected)}] seeded={ok} failed={fail}", flush=True)

    print(f"\nseeded {ok}, failed {fail} -> {out_root}")
    print("NEXT: gate these on real gfx950 before generating against them:")
    print("  PYTHONPATH=. python scripts/verify_hip_seeds.py --json seed_gate.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
