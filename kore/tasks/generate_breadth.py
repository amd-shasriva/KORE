"""Generate breadth-family KORE task dirs (torch-baselined op-class expansion).

Materializes the self-contained breadth authoring engines under
``kore/tasks/breadth/`` (scan/SSM + conv1d, conv2d/pooling/resize, sort/select +
sparse, losses + fused optimizers) into ``genb_<op>_<dtype>/`` task dirs, each with

    task.yaml            + thin reference shim (-> breadth.<module>.make_reference)
    seed_triton.py       (the module's naive-but-correct Triton seed)
    driver.py            + thin driver shim (-> _genops.driver_main)

Idempotent; the ``genb_`` prefix avoids collision with ``gen_``/``genv_``. Registry
discovery picks them up automatically, maps each source module through the versioned
taxonomy, and keeps the explicit stratified near-probe IDs eval-only.

    python -m kore.tasks.generate_breadth [--list]

SAFETY: this generator is NOT imported by the running campaign (the registry
discovers task DIRS, not this module), so authoring/committing it is inert. But
*running* it writes ``genb_*`` dirs into the live registry of THIS checkout - only
run it on the node whose task suite you intend to widen (e.g. the datagen factory /
the 32B run), never on a node whose in-flight run must keep a frozen task set.
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
from pathlib import Path

import kore.tasks.breadth as _breadth_pkg
from kore.tasks._genops import DTYPES

TASKS_DIR = Path(__file__).resolve().parent


def _discover_modules() -> tuple:
    """Auto-discover every conformant breadth authoring engine under
    ``kore/tasks/breadth/`` (any module exposing OPS + make_reference +
    seed_source). New op-family modules are picked up with zero edits here; the
    ``tests`` subpkg and private ``_*`` modules are skipped. Deterministic order."""
    mods = []
    for m in sorted(pkgutil.iter_modules(_breadth_pkg.__path__), key=lambda x: x.name):
        if m.name == "tests" or m.name.startswith("_"):
            continue
        mod = importlib.import_module(f"kore.tasks.breadth.{m.name}")
        if all(hasattr(mod, a) for a in ("OPS", "SHAPES", "make_reference", "seed_source")):
            mods.append(mod)
    return tuple(mods)


# breadth authoring engines (all expose the shared ABI:
# OPS / OP_DTYPES / SHAPES / make_reference(op,dtype) / seed_source(op,dtype)).
_MODULES = _discover_modules()

_REF_SHIM = '''"""GENERATED breadth reference shim for {op} ({dtype}). See kore/tasks/breadth/{mod}.py.
Do not hand-edit - regenerate via kore/tasks/generate_breadth.py."""
from kore.tasks.breadth.{mod} import make_reference

globals().update(make_reference("{op}", "{dtype}"))
'''

_DRIVER_SHIM = '''"""GENERATED breadth driver shim for {op} ({dtype}). See kore/tasks/_genops.py.
Do not hand-edit - regenerate via kore/tasks/generate_breadth.py."""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
import reference as ref  # noqa: E402
from kore.tasks._genops import driver_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(driver_main(ref, _here))
'''


def _op_module_map() -> dict[str, tuple[str, object]]:
    """op_name -> (module_basename, module); raises on cross-module op collisions."""
    m: dict[str, tuple[str, object]] = {}
    for mod in _MODULES:
        modname = mod.__name__.rsplit(".", 1)[-1]
        for op in mod.OPS:
            if op in m:
                raise ValueError(
                    f"duplicate breadth op '{op}' in {modname} and {m[op][0]}")
            m[op] = (modname, mod)
    return m


def _shape_str(s: dict) -> str:
    return "{" + ", ".join(f"{k}: {v}" for k, v in s.items()) + "}"


def _dtypes_for(mod, op: str) -> list[str]:
    fn = getattr(mod, "op_dtypes", None)
    if callable(fn):
        return list(fn(op))
    return list(getattr(mod, "OP_DTYPES", {}).get(op, ["bf16", "fp16"]))


def _yaml(mod, op: str, dtype: str, snr: float) -> str:
    shp = mod.SHAPES[op]
    lines = [
        f"task_id: genb_{op}_{dtype}",
        f"operation: {op}",
        f"dtype: {dtype}",
        "backend: triton",
        "gpu_target: gfx950",
        "seed_kernel_name: seed_triton.py",
        f"snr_threshold: {snr}",
        f"op_family: breadth_{op}",
        "baseline_tier: breadth",
        "generated: true",
        "shapes:",
        f"  minimal: {_shape_str(shp['minimal'])}",
        f"  primary: {_shape_str(shp['primary'])}",
        "  validation:",
    ]
    for s in shp["validation"]:
        lines.append(f"    - {_shape_str(s)}")
    lines += ["targets:", f"  snr_db: {snr}", f"  comparison_baseline: torch_{op}"]
    return "\n".join(lines) + "\n"


def generate(dry: bool = False) -> list[str]:
    written: list[str] = []
    opmap = _op_module_map()
    for op, (modname, mod) in sorted(opmap.items()):
        for dtype in _dtypes_for(mod, op):
            if dtype not in DTYPES:
                raise ValueError(f"breadth op {op}: unknown dtype {dtype!r}")
            snr = DTYPES[dtype][2]
            tid = f"genb_{op}_{dtype}"
            written.append(tid)
            if dry:
                continue
            d = TASKS_DIR / tid
            d.mkdir(exist_ok=True)
            (d / "task.yaml").write_text(_yaml(mod, op, dtype, snr))
            (d / "reference.py").write_text(
                _REF_SHIM.format(op=op, dtype=dtype, mod=modname))
            (d / "seed_triton.py").write_text(mod.seed_source(op, dtype))
            (d / "driver.py").write_text(_DRIVER_SHIM.format(op=op, dtype=dtype))
    return written


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true",
                   help="dry-run: list task ids that would be generated")
    a = p.parse_args(argv)
    written = generate(dry=a.list)
    print(f"{'would generate' if a.list else 'generated'} {len(written)} breadth tasks:")
    for t in written:
        print(f"  {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
