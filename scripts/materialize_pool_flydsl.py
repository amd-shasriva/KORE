#!/usr/bin/env python
"""Materialize pool tasks as FlyDSL twins, by translating their Triton kernel.

Why translate rather than synthesize from the PyTorch module, as the HIP path
does: FlyDSL is not a kernel language but an MLIR builder API. The smallest
kernel in the arena is a 255-line softmax, against roughly 30 lines of Triton,
and authoring one means writing explicit tiling, copy atoms, predication and
shared-memory layout. Asking a teacher to invent that from ``torch.softmax`` is a
much longer reach than asking it to port a working Triton kernel that already
expresses the same tiling decisions.

Why pool tasks and not the arena's own FlyDSL tasks: the arena ships 60 working
FlyDSL kernels and training on them would be training on the benchmark. Pool
tasks come from KernelBook -- real GitHub repositories, disjoint from the arena --
so a verified FlyDSL kernel here is uncontaminated data that should transfer to
any benchmark rather than to this one.

The API is taught from FlyDSL's own ``examples/`` and ``CLAUDE.md``, not from
arena task sources, for the same reason.

    python scripts/materialize_pool_flydsl.py --limit 24 --workers 8 \
        --out data/pool_flydsl
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import pathlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DEFAULT_SOURCE_ROOT = REPO / "data" / "task_pool" / "tasks"
#: The directory whose task dirs get a twin in the target language. Defaults to
#: the external pool, which is what this was built for, but the transformation is
#: not pool-specific: a task dir is reference.py + driver.py + task.yaml + a
#: language-specific seed, and the registry's hand-authored frontier tasks --
#: flash attention, fused MoE, fp8 GEMM -- have exactly that shape.
#:
#: That matters because the pool is where the easy work is. Its median baseline
#: is 17us and 86% of it is under 100us, so a HIP or FlyDSL twin of a pool task
#: is a twin of a launch-bound kernel. Pointing --source-root at kore/tasks
#: produces the same twin for a kernel that actually has headroom.
POOL = DEFAULT_SOURCE_ROOT
FLYDSL_REPO = Path.home() / "third_party" / "flydsl"

#: A worked example and the project's own authoring rules. Both come from the
#: FlyDSL repository, so nothing here is derived from benchmark task sources.
_EXAMPLE = FLYDSL_REPO / "examples" / "01-vectorAdd.py"
_GUIDE = FLYDSL_REPO / "CLAUDE.md"


SEED_PROMPT = """You are porting a working GPU kernel to FlyDSL for an AMD MI355X \
(gfx950).

FlyDSL is a Python-embedded MLIR builder with a CuTe-style layout API. It is not \
Triton: there is no `tl.load`/`tl.store` and no automatic masking. You describe \
tiling explicitly with layouts, copy atoms and predicates.

Here are FlyDSL's own authoring conventions:

{conventions}

Here is a complete, working FlyDSL kernel (vectorized 2D elementwise add) showing \
the required structure -- a `@flyc.kernel` device kernel plus a `@flyc.jit` launch \
wrapper:

```python
{example}
```

TASK. Reproduce the numerics of this PyTorch module:

```python
{module_source}
```

A working Triton implementation of the same computation is given below. It \
already encodes correct tiling and indexing for this problem; port it, do not \
redesign it:

```python
{triton_source}
```

HARD REQUIREMENTS -- a seed that misses any of these is discarded:

1. Export a `@flyc.jit` launch wrapper named EXACTLY `{entry_name}`, called as
   `{entry_name}({arg_list})` with {arity} torch tensor argument(s) of dtype \
{dtype}, returning a torch tensor.

2. SHAPES ARE NOT FIXED. Read every extent from the tensors at runtime. The \
harness re-runs at other sizes, so a compile-time shape baked in passes one case \
and fails the rest.

3. Use only `flydsl.compiler` (`flyc`) and `flydsl.expr` (`fx`). Do NOT import \
from `kernels.*` -- those helper modules are not on the path here. If you need a \
helper, define it in the file.

3a. These are the ONLY names those two modules export. If what you want is not \
in this list it does not exist -- build it out of what is here rather than \
guessing a name, and match the capitalisation exactly.

flyc: {flyc_api}

fx: {fx_api}

3b. A FlyDSL value is not a torch tensor. It has no `.device`, `.reshape`, \
`.shape`, `.dtype` or any other torch method; shapes come from the layout you \
declare and from extents you read explicitly. Only the `@flyc.jit` wrapper sees \
real torch tensors.

4. Predicate every copy whose tile can exceed the tensor bounds, as the example \
does with `fx.elem_less`. An unpredicated border block reads out of bounds.

5. Keep a single exit path in traced functions: no early `return`, no \
branch-local `return`.

It must reach at least {snr} dB SNR against the reference.

Return ONLY the complete contents of the .py file in a single ```python code block.
"""


def _spec_of(task_dir: Path) -> dict:
    """The spec for a task, pool or registry. See kore.data.twins.

    This used to raise on anything without an embedded ``_SPEC``, which is
    every hand-authored registry task -- so while HIP twinned flash attention,
    fused MoE and fp8 GEMM, FlyDSL could only ever port the pool, whose median
    baseline is 17us. 480 of the 482 frontier registry tasks ship a working
    seed_triton.py, which is exactly what this path needs to port from.
    """
    from kore.data.twins import spec_of

    return spec_of(task_dir)


def _extract_code(reply: str) -> str:
    """The kernel file, which is not always the first fenced block."""
    from kore.data.twins import extract_code

    return extract_code(reply, must_contain="flyc.jit")


def _conventions() -> str:
    """The authoring rules, trimmed to what governs writing one kernel.

    The full guide is 217 lines of repository routing and module-placement policy
    that would only crowd the prompt; the section that constrains kernel code is
    what the teacher needs.
    """
    if not _GUIDE.is_file():
        return "(FlyDSL guide unavailable)"
    text = _GUIDE.read_text(errors="ignore")
    start = text.find("## Kernel Authoring Conventions")
    if start < 0:
        return "(conventions section not found)"
    end = text.find("\n## ", start + 10)
    body = text[start:end if end > 0 else len(text)]
    # Drop the repository-organisation bullets: they govern where a helper lives
    # in the FlyDSL tree, which has no bearing on a standalone kernel file.
    keep = [ln for ln in body.splitlines()
            if not ln.lstrip().startswith("- **Helper placement")
            and not ln.lstrip().startswith("- **`expr/")]
    return "\n".join(keep)[:4000]


def _api_surface(module: str) -> str:
    """Every public name a FlyDSL module exports, as a flat list.

    The teacher was inventing the API. Measured across 2,005 gated ports, 242
    failures were a name that does not exist -- ``fx.constexpr`` for
    ``fx.Constexpr`` 137 times on its own, then ``fx.empty`` and ``fx.maximum``
    -- and another 104 called torch methods on FlyDSL values. None of that is a
    reasoning failure; it is a model writing against an API it has never seen,
    from a guide that documents conventions rather than symbols.

    Listing the names costs a few thousand characters of prompt and removes the
    entire class. It is read from the installed package, so it cannot drift from
    the FlyDSL the gate actually compiles against.
    """
    try:
        import importlib  # noqa: PLC0415 - only needed when a prompt is built
        import sys

        if str(FLYDSL_REPO / "python") not in sys.path:
            sys.path.insert(0, str(FLYDSL_REPO / "python"))
        mod = importlib.import_module(module)
    except Exception:  # noqa: BLE001 - a dry-run without FlyDSL still builds
        return "(unavailable -- follow the example and the guide above)"
    return ", ".join(sorted(n for n in dir(mod) if not n.startswith("_")))


def _build_prompt(spec: dict, triton_source: str) -> tuple[str, str]:
    entry = spec.get("entry_name") or "forward"
    specs = spec.get("input_specs") or []
    arity = len(specs) or 1
    return SEED_PROMPT.format(
        flyc_api=_api_surface("flydsl.compiler"),
        fx_api=_api_surface("flydsl.expr"),
        conventions=_conventions(),
        example=_EXAMPLE.read_text(errors="ignore")[:6000]
        if _EXAMPLE.is_file() else "(example unavailable)",
        module_source=spec.get("module_source", "")[:6000],
        triton_source=triton_source[:8000],
        entry_name=entry,
        arity=arity,
        arg_list=", ".join(f"t{i}" for i in range(arity)),
        dtype=spec.get("dtype", "fp32"),
        snr=spec.get("snr_threshold", 30)), entry


#: The twin reuses the pool task's own reference.py and driver.py, so a FlyDSL
#: candidate is graded against exactly the oracle its Triton counterpart is.
def materialize(task_id: str, seed_src: str, out_root: Path) -> Path:
    src = POOL / task_id
    dst = out_root / "tasks" / f"{task_id}__flydsl"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("driver.py", "reference.py"):
        shutil.copy(src / name, dst / name)
    from kore.data.twins import read_task_cfg

    cfg = read_task_cfg(src)
    cfg.update({"task_id": f"{task_id}__flydsl", "backend": "flydsl",
                "seed_kernel_name": "seed_flydsl.py",
                "provenance_root": task_id, "flydsl_twin_of": task_id})
    (dst / "task.yaml").write_text(json.dumps(cfg, indent=2) + "\n")
    (dst / "seed_flydsl.py").write_text(seed_src)
    return dst


def _seed_one(item, teacher, out_root: Path) -> dict:
    tid, spec, triton_src = item
    prompt, entry = _build_prompt(spec, triton_src)
    try:
        reply = teacher.generate([{"role": "user", "content": prompt}])
        seed = _extract_code(reply)
        # Cheap structural checks. They cost nothing and keep a reply that
        # ignored the contract from consuming a gate slot on a GPU node.
        if "flyc.jit" not in seed:
            raise ValueError("no @flyc.jit launch wrapper")
        if f"def {entry}" not in seed:
            raise ValueError(f"does not define {entry!r}")
        if "from kernels" in seed or "import kernels" in seed:
            raise ValueError("imports kernels.* which is not on the path")
        materialize(tid, seed, out_root)
        return {"task_id": tid, "status": "seeded", "chars": len(seed)}
    except Exception as exc:  # noqa: BLE001 - one bad task must not end the sweep
        return {"task_id": tid, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}"[:200]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pool_flydsl")
    ap.add_argument("--source-root", default=None,
                    help="task dirs to twin (default: the external pool; point at kore/tasks for the frontier registry set)")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--task-list", default=None,
                    help="file of task ids to port, one per line. A source root "
                         "is not a work list: kore/tasks is 1,549 dirs of which "
                         "482 are frontier, and the rest are taken first in "
                         "name order")
    ap.add_argument("--teacher", default="claude")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reseed-existing", action="store_true",
                    help="also port tasks that already have a FlyDSL twin in "
                         "another output root. Off by default: re-porting one "
                         "spends a teacher call to rewrite a file that exists")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.source_root:
        global POOL
        POOL = pathlib.Path(args.source_root).resolve()
        if not POOL.is_dir():
            print(f"source root does not exist: {POOL}", file=sys.stderr)
            return 2
        print(f"source root: {POOL}")

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

    if not args.reseed_existing:
        from kore.data.twins import TWIN_SUFFIXES, existing_twins

        cross = existing_twins(TWIN_SUFFIXES["flydsl"], REPO / "data")
        fresh = cross - attempted
        if fresh:
            print(f"skipping {len(fresh)} task(s) already twinned in another "
                  f"output root")
        attempted |= cross

    ids = sorted(p.name for p in POOL.glob("*/") if (p / "task.yaml").is_file())
    if args.task_list:
        from kore.data.twins import read_task_list

        wanted = read_task_list(Path(args.task_list))
        ids = [t for t in ids if t in wanted]
        print(f"restricted to {len(ids)} task(s) from {args.task_list}")
    ids = [t for t in ids if t not in attempted][args.offset:]

    selected = []
    no_triton = 0
    for tid in ids:
        if len(selected) >= args.limit:
            break
        td = POOL / tid
        seed = td / "seed_triton.py"
        if not seed.is_file():
            # Nothing to port from. Synthesizing FlyDSL without a working
            # reference is a far harder ask, so skip rather than spend a call.
            no_triton += 1
            continue
        try:
            spec = _spec_of(td)
        except Exception:  # noqa: BLE001 - a malformed task is not worth failing on
            continue
        if args.families and spec.get("family") not in args.families:
            continue
        selected.append((tid, spec, seed.read_text(errors="ignore")))

    print(f"selected {len(selected)} pool task(s) to port to FlyDSL"
          + (f"; skipped {no_triton} without a Triton kernel to port"
             if no_triton else ""))
    if args.dry_run:
        for tid, spec, _ in selected[:5]:
            print(f"  {tid}  family={spec.get('family')} entry={spec.get('entry_name')}")
        return 0

    from kore.data.twins import mark_exhausted

    mark_exhausted(out_root, len(selected), len(ids))
    if not selected:
        print("nothing left to port for this root")
        return 0

    from kore.data.teacher import load_env_local, make_teacher

    load_env_local()
    teacher = make_teacher(args.teacher, resilient=True)

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
            ok += rec["status"] == "seeded"
            fail += rec["status"] != "seeded"
            with lock:
                ledger.write(json.dumps(rec) + "\n")
                ledger.flush()
                os.fsync(ledger.fileno())
            if i % 10 == 0:
                print(f"  [{i}/{len(selected)}] seeded={ok} failed={fail}",
                      flush=True)
    print(f"\nseeded {ok}, failed {fail} -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
