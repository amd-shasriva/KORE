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
import re
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

The harness calls your entry point as `{entry_name}({arg_list})` with {arity}
tensor argument(s) of dtype {dtype}, and compares the result against the module
above.

HARD REQUIREMENTS -- a seed that misses any of these is discarded:

0. SHAPES ARE NOT FIXED. The example shape {example_shape} is only one case; the
   harness re-runs at other sizes (it scales to {primary_scale} elements and
   validates at further scales). Read every extent from the tensor at runtime
   with `.size(i)` / `.numel()`. A kernel with a compile-time shape baked in
   passes the example and fails everything else.


1. Bind the entry point under EXACTLY this name, which is the task's own and is \
NOT "forward":

       PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
         m.def("{entry_name}", &{entry_name}, "seed");
       }}

   Its arguments and return must match the module's forward() in order and type \
(`torch::Tensor`).

2. Do NOT call any stream API. In particular \
`at::cuda::getCurrentHIPStreamMasqueradingAsCUDA` does not exist in this ROCm \
build. Launch on the default stream by passing 0 as the stream argument.

Input dtype is {dtype}. It must reach at least {snr} dB SNR against the reference.

Return ONLY the complete contents of the .hip file in a single ```cpp code block.
"""


#: Layers that carry learned state. A module using any of them cannot be
#: expressed as a bare .hip function: the harness calls the candidate with only
#: the declared input tensors, so the weights are invisible to it and the kernel
#: can never reproduce e.g. conv2d(x, W) for a W it cannot see. A Triton
#: candidate is a Python file and can just instantiate the module to read them --
#: that asymmetry is why the pool materializes cleanly to Triton and not to HIP.
#:
#: Measured on the pool: 3,607 of 13,570 modules (26.6%) are parameter-free,
#: concentrated in reduction (79%), activation (85%) and data_movement (82%) --
#: which are also the families where a naive HIP kernel is tractable to write.
_PARAM_LAYERS = re.compile(
    r"nn\.(Conv\d ?d?|Linear|BatchNorm\d ?d?|LayerNorm|Embedding|GroupNorm|"
    r"InstanceNorm\dd|LSTM|GRU|RNN|MultiheadAttention|Bilinear|"
    r"ConvTranspose\dd|PReLU|Parameter)", re.I)
_EXPLICIT_PARAM = re.compile(r"nn\.Parameter|register_parameter|register_buffer")


def is_parameter_free(spec: dict) -> bool:
    src = spec.get("module_source", "")
    return not _PARAM_LAYERS.search(src) and not _EXPLICIT_PARAM.search(src)


def _build_prompt(spec: dict) -> tuple[str, str]:
    """The seed prompt for one task, and the entry symbol it must export."""
    entry = spec.get("entry_name") or "forward"
    specs = spec.get("input_specs") or []
    arity = len(specs) or 1
    return SEED_PROMPT.format(
        module_source=spec.get("module_source", "")[:8000],
        dtype=spec.get("dtype", "fp32"),
        snr=spec.get("snr_threshold", 30),
        entry_name=entry,
        arity=arity,
        arg_list=", ".join(f"t{i}" for i in range(arity)),
        example_shape=(specs[0].get("shape") if specs else "[N]"),
        primary_scale=spec.get("primary_scale", "a larger size")), entry


def _seed_one(item, teacher, out_root: Path) -> dict:
    tid, spec = item
    prompt, entry = _build_prompt(spec)
    try:
        reply = teacher.generate([{"role": "user", "content": prompt}])
        seed = _extract_code(reply)
        if "PYBIND11_MODULE" not in seed:
            raise ValueError("seed does not bind an entry point")
        if f'"{entry}"' not in seed:
            raise ValueError(f"seed does not export {entry!r}")
        materialize(tid, seed, out_root)
        return {"task_id": tid, "status": "seeded", "chars": len(seed)}
    except Exception as exc:  # noqa: BLE001 - one bad task must not end the sweep
        return {"task_id": tid, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}"[:200]}


def _seed_parallel(selected, teacher, done_path: Path, out_root: Path, args) -> int:
    """Seed many tasks at once. The work is remote latency, not local compute."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    lock = threading.Lock()
    ok = fail = 0
    with done_path.open("a") as ledger, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_seed_one, it, teacher, out_root): it[0]
                for it in selected}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            if rec["status"] == "seeded":
                ok += 1
            else:
                fail += 1
            with lock:
                ledger.write(json.dumps(rec) + "\n")
                ledger.flush()
                os.fsync(ledger.fileno())
            if i % 50 == 0:
                print(f"  [{i}/{len(selected)}] seeded={ok} failed={fail}",
                      flush=True)
    print(f"\nseeded {ok}, failed {fail} -> {out_root}")
    return 0


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


#: reference.py for a functionalized twin. The oracle is rebuilt from the same
#: spec the Triton task uses, so a HIP win and a Triton win on one task_id are
#: still graded against the same semantics -- only the calling convention differs.
_FUNCTIONAL_REFERENCE = '''"""GENERATED functionalized reference for a HIP pool twin.
The module's parameters are passed as explicit inputs so a .hip candidate can see
them. See kore/tasks/functionalized.py. Do not hand-edit."""
import json

from kore.tasks.functionalized import functional_namespace_from_spec

_SPEC = json.loads({spec!r})
globals().update(functional_namespace_from_spec(_SPEC))
'''


def materialize_functional(task_id: str, spec: dict, seed_src: str,
                           out_root: Path) -> Path:
    """A HIP twin whose oracle takes (activation, *parameters)."""
    src = POOL / task_id
    dst = out_root / "tasks" / f"{task_id}__hipf"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "driver.py", dst / "driver.py")
    (dst / "reference.py").write_text(
        _FUNCTIONAL_REFERENCE.format(spec=json.dumps(spec)))
    cfg = json.loads((src / "task.yaml").read_text())
    cfg.update({"task_id": f"{task_id}__hipf", "backend": "hip",
                "seed_kernel_name": "seed_hip.hip",
                "provenance_root": task_id, "hip_twin_of": task_id,
                "functionalized": True})
    (dst / "task.yaml").write_text(json.dumps(cfg, indent=2) + "\n")
    (dst / "seed_hip.hip").write_text(seed_src)
    return dst


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
    ap.add_argument("--allow-parameterized", action="store_true",
                    help="also seed modules with learned weights (they cannot "
                         "pass: a .hip function never sees the parameters)")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel teacher calls; the work is remote latency, "
                         "so this is the difference between 19h and ~2h")
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
    skipped_param = 0
    for tid in ids:
        if len(selected) >= args.limit:
            break
        try:
            spec = _spec_of(POOL / tid)
        except Exception:  # noqa: BLE001 - a malformed task is not worth failing the sweep
            continue
        if args.families and spec.get("family") not in args.families:
            continue
        if not args.allow_parameterized and not is_parameter_free(spec):
            skipped_param += 1
            continue
        selected.append((tid, spec))

    print(f"selected {len(selected)} pool task(s) to seed"
          + (f" (families={args.families})" if args.families else "")
          + (f"; skipped {skipped_param} with learned parameters"
             if skipped_param else ""))
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
    # Seeding is teacher-bound at ~19s a seed and was running serially, which is
    # 19 hours for 3,607 tasks -- the long pole in the whole HIP pipeline, and
    # spent entirely waiting on a remote API rather than on anything local. The
    # ledger append is under a lock and each task is independent, so this
    # parallelises cleanly.
    if args.workers > 1:
        return _seed_parallel(selected, teacher, done_path, out_root, args)

    ok = fail = 0
    with done_path.open("a") as ledger:
        for i, (tid, spec) in enumerate(selected, 1):
            # The pool builds its oracle from this spec, and entry_name is the
            # operation -- not "forward", which is what the hand-authored HIP
            # tasks happen to use. Getting this wrong is not a soft failure: the
            # extension compiles and then the loader rejects it for exporting no
            # such symbol, which is how the first 12 seeds all died.
            entry = spec.get("entry_name") or "forward"
            specs = spec.get("input_specs") or []
            arity = len(specs) or 1
            example = specs[0].get("shape") if specs else "[N]"
            prompt = SEED_PROMPT.format(
                module_source=spec.get("module_source", "")[:8000],
                dtype=spec.get("dtype", "fp32"),
                snr=spec.get("snr_threshold", 30),
                entry_name=entry,
                arity=arity,
                arg_list=", ".join(f"t{i}" for i in range(arity)),
                example_shape=example,
                primary_scale=spec.get("primary_scale", "a larger size"))
            try:
                reply = teacher.generate([{"role": "user", "content": prompt}])
                seed = _extract_code(reply)
                if "PYBIND11_MODULE" not in seed:
                    raise ValueError("seed does not bind an entry point")
                # Check the symbol here rather than paying a GPU compile to find
                # out: the loader looks up this exact name.
                if f'"{entry}"' not in seed:
                    raise ValueError(f"seed does not export {entry!r}")
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
