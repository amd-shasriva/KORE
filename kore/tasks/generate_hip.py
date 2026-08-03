"""Materialize HIP C++ task directories from :mod:`kore.tasks.hip_ops`.

Writes ``hip_<op>_<dtype>/`` into the registry, each with

    task.yaml       backend: hip, seed_kernel_name: seed_hip.hip
    reference.py    thin shim -> hip_ops.make_reference
    seed_hip.hip    the compiling HIP C++ seed
    driver.py       thin shim -> _genops.driver_main

Idempotent.  Registry discovery picks the directories up automatically; the
operations are registered in :data:`kore.tasks.taxonomy.HAND_OPERATION_FAMILIES`,
so a task whose operation was never reviewed fails registry validation instead of
being silently classified.

    python -m kore.tasks.generate_hip [--list]

Every seed is gated here before it can reach the registry: it must be non-empty,
must bind the ``forward`` entry point through ``PYBIND11_MODULE``, must not
``#include`` a vendor math library, and must pass the same anti-hack scanner the
environment applies to a model's candidate.  That last one matters -- a seed the
scanner would reject is a task no correct kernel can ever score on.

Compiling is NOT checked here, because it needs a GPU.  Run
``scripts/verify_hip_seeds.py`` for that; it is the gate that decides which
(op, dtype) pairs are allowed to become tasks at all.

SAFETY: importing this module is inert (the registry discovers task DIRS, not
this file).  *Running* it writes into the live registry of this checkout, so only
run it where you intend to widen the task suite.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from kore.reward.reward import scan_for_hacks
from kore.tasks import taxonomy
from kore.tasks.hip_ops import GPU_TARGET, HIP_OPS, seed_source
from kore.tasks.shape_policy import shape_policy_yaml_lines

TASKS_DIR = Path(__file__).resolve().parent
TASK_PREFIX = "hip_"
SEED_FILENAME = "seed_hip.hip"

#: Vendor math libraries a seed may never include.  A HIP kernel that calls into
#: hipBLASLt/rocBLAS/MIOpen is delegating the operator, which is the same reward
#: hack as calling torch.matmul from Triton -- and here it would also be
#: delegating to the very library the task's baseline is measured against.
_FORBIDDEN_INCLUDES = re.compile(
    r"#\s*include\s*[<\"](?:hipblas|hipblaslt|rocblas|miopen|rocsolver|"
    r"hipsparse|rocsparse|composable_kernel|ck/)",
    re.IGNORECASE,
)

_REF_SHIM = '''"""GENERATED HIP reference shim for {op} ({dtype}).
See kore/tasks/hip_ops.py. Do not hand-edit - regenerate via
kore/tasks/generate_hip.py."""
from kore.tasks.hip_ops import make_reference

globals().update(make_reference("{op}", "{dtype}"))
'''

_DRIVER_SHIM = '''"""GENERATED HIP driver shim for {op} ({dtype}). See kore/tasks/_genops.py.
Do not hand-edit - regenerate via kore/tasks/generate_hip.py.

The candidate is staged as ``{seed}``-shaped HIP C++ in ``kernel.hip`` and
compiled by kore.env.hip_toolchain; _genops.driver_main is otherwise unchanged,
so this task gets the same paired cold-cache timing protocol, adversarial
battery and post-timing re-verification as every Triton task."""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
import reference as ref  # noqa: E402
from kore.tasks._genops import driver_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(driver_main(ref, _here))
'''


class HipTaskError(ValueError):
    """A HIP op cannot be materialized into a valid task."""


def task_id(op_id: str, dtype_id: str) -> str:
    return f"{TASK_PREFIX}{op_id}_{dtype_id}"


def operation_id(op_id: str) -> str:
    return f"{TASK_PREFIX}{op_id}"


def _shape_str(dims) -> str:
    return "{" + ", ".join(f"{k}: {v}" for k, v in dims.items()) + "}"


def _validated_seed(op_id: str, dtype_id: str) -> str:
    """Admit a seed only if it is a real, scanner-clean HIP candidate."""
    source = seed_source(op_id)
    if not isinstance(source, str) or not source.strip():
        raise HipTaskError(f"{op_id}/{dtype_id}: seed_source returned no text")
    if "PYBIND11_MODULE(TORCH_EXTENSION_NAME" not in source:
        raise HipTaskError(
            f"{op_id}/{dtype_id}: seed does not bind a module through "
            "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)")
    if not re.search(r'm\s*\.\s*def\s*\(\s*"forward"', source):
        raise HipTaskError(f"{op_id}/{dtype_id}: seed does not export a 'forward' entry")
    forbidden = _FORBIDDEN_INCLUDES.search(source)
    if forbidden:
        raise HipTaskError(
            f"{op_id}/{dtype_id}: seed includes a vendor math library "
            f"({forbidden.group(0)!r}), which delegates the operator")
    reason = scan_for_hacks(source, "cpp")
    if reason is not None:
        raise HipTaskError(f"{op_id}/{dtype_id}: seed rejected by the scanner: {reason}")
    return source


def _declared_shapes(spec) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for lane, value in spec.shapes.items():
        if isinstance(value, dict):
            out.append((lane, value))
        elif isinstance(value, list):
            out.extend((f"{lane}_{i}", dims) for i, dims in enumerate(value))
    return out


def _check_shape_constraints(op_id: str, spec) -> None:
    """Reject a lane the op's own representation cannot express.

    MXFP4 packs 32 elements per shared exponent, so a row length that is not a
    multiple of 32 has no legal packing.  A lane like that compiles and passes a
    minimal/primary spot-check, then throws at datagen on a validation shape --
    which is exactly what happened before this gate existed.
    """
    if not spec.dim_multiples:
        return
    for lane, dims in _declared_shapes(spec):
        for dim, multiple in spec.dim_multiples.items():
            value = dims.get(dim)
            if value is None:
                raise HipTaskError(
                    f"{op_id}: shape lane {lane!r} declares no {dim!r}, but the op "
                    f"requires {dim} % {multiple} == 0")
            if int(value) % int(multiple) != 0:
                raise HipTaskError(
                    f"{op_id}: shape lane {lane!r} has {dim}={value}, which is not a "
                    f"multiple of {multiple} required by this op's representation")


def _yaml(op_id: str, dtype_id: str) -> str:
    spec = HIP_OPS[op_id]
    _check_shape_constraints(op_id, spec)
    operation = operation_id(op_id)
    family = taxonomy.HAND_OPERATION_FAMILIES.get(operation)
    if family is None:
        raise HipTaskError(
            f"operation {operation!r} is absent from "
            "kore.tasks.taxonomy.HAND_OPERATION_FAMILIES; a task whose operation "
            "was never reviewed must not enter the registry")
    if family != spec.product_family:
        raise HipTaskError(
            f"operation {operation!r}: taxonomy says {family!r} but hip_ops "
            f"declares {spec.product_family!r}")

    lines = [
        f"task_id: {task_id(op_id, dtype_id)}",
        f"operation: {operation}",
        f"dtype: {dtype_id}",
        "backend: hip",
        f"gpu_target: {GPU_TARGET}",
        f"seed_kernel_name: {SEED_FILENAME}",
        f"snr_threshold: {spec.snr_db}",
        f"op_family: hip_{spec.source_family}",
        "baseline_tier: hip",
        f"# {spec.description}, in HIP C++ ({spec.product_family} family).",
        f"# Candidate ABI: a .hip file binding `forward` through",
        f"# PYBIND11_MODULE(TORCH_EXTENSION_NAME, m), taking/returning torch::Tensor.",
        f"# Correctness oracle: fp32 torch. Baseline (--impl reference):",
        f"# {spec.baseline_note}.",
        "# The seed is correct but deliberately naive, so the headroom HipKittens",
        "# measures for C++ tile primitives (1.3-3.0x over Triton on BF16 GEMM) is",
        "# what the model has to find.",
    ]
    lines += shape_policy_yaml_lines(
        operation, spec.shapes, source=f"generator:hip_{op_id}")
    lines += [
        "shapes:",
        f"  minimal: {_shape_str(spec.shapes['minimal'])}",
        f"  primary: {_shape_str(spec.shapes['primary'])}",
        "  validation:",
    ]
    for dims in spec.shapes["validation"]:
        lines.append(f"    - {_shape_str(dims)}")
    lines += [
        "targets:",
        f"  snr_db: {spec.snr_db}",
        f"  comparison_baseline: {spec.baseline_kind}",
    ]
    return "\n".join(lines) + "\n"


def generate(dry: bool = False, output_dir: Path | None = None,
             ops: tuple[str, ...] = ()) -> list[str]:
    written: list[str] = []
    root = TASKS_DIR if output_dir is None else Path(output_dir)
    if not dry:
        root.mkdir(parents=True, exist_ok=True)
    for op_id in sorted(ops or tuple(HIP_OPS)):
        spec = HIP_OPS[op_id]
        # An op that cannot be TIMED is not a task: it would consume datagen GPU
        # time and return a performance-ineligible episode every single run. The
        # op stays defined, with its measured reason, rather than being deleted.
        if not spec.timing_admissible and not ops:
            continue
        for dtype_id in spec.dtypes:
            if dtype_id not in taxonomy.TRAIN_DTYPES:
                raise HipTaskError(
                    f"{op_id}: dtype {dtype_id!r} is not in taxonomy.TRAIN_DTYPES, so "
                    "the task would silently be eval-only")
            tid = task_id(op_id, dtype_id)
            written.append(tid)
            if dry:
                continue
            source = _validated_seed(op_id, dtype_id)
            body = _yaml(op_id, dtype_id)
            directory = root / tid
            directory.mkdir(exist_ok=True)
            (directory / "task.yaml").write_text(body)
            (directory / "reference.py").write_text(
                _REF_SHIM.format(op=op_id, dtype=dtype_id))
            (directory / SEED_FILENAME).write_text(source)
            (directory / "driver.py").write_text(
                _DRIVER_SHIM.format(op=op_id, dtype=dtype_id, seed=SEED_FILENAME))
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true",
                        help="dry-run: list the task ids that would be generated")
    parser.add_argument("--ops", default="",
                        help="comma-separated op ids (default: all)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="materialize under this directory instead of kore/tasks")
    args = parser.parse_args(argv)
    ops = tuple(o for o in args.ops.split(",") if o)
    written = generate(dry=args.list, output_dir=args.output_dir, ops=ops)
    print(f"{'would generate' if args.list else 'generated'} {len(written)} HIP tasks:")
    for tid in written:
        print(f"  {tid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
