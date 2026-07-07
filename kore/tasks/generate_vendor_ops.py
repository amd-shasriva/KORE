"""Generate vendor-baselined KORE task dirs (graded vs real AITER kernels).

Writes ``genv_<op>_<dtype>/`` with task.yaml + thin reference shim
(-> vendor_ops.make_vendor_reference) + thin driver shim (-> _genops.driver_main)
+ a REAL Triton seed. Idempotent; ``genv_`` prefix avoids collision. Registry
discovery picks them up automatically.

    python -m kore.tasks.generate_vendor_ops [--list]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kore.tasks import vendor_ops as V

TASKS_DIR = Path(__file__).resolve().parent

_REF_SHIM = '''"""GENERATED vendor reference shim for {op} ({dtype}). See kore/tasks/vendor_ops.py.
Do not hand-edit — regenerate via kore/tasks/generate_vendor_ops.py."""
from kore.tasks.vendor_ops import make_vendor_reference

globals().update(make_vendor_reference("{op}", "{dtype}"))
'''

_DRIVER_SHIM = '''"""GENERATED vendor driver shim for {op} ({dtype}). See kore/tasks/_genops.py.
Do not hand-edit — regenerate via kore/tasks/generate_vendor_ops.py."""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
import reference as ref  # noqa: E402
from kore.tasks._genops import driver_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(driver_main(ref, _here))
'''


def _shape_str(s: dict) -> str:
    return "{" + ", ".join(f"{k}: {v}" for k, v in s.items()) + "}"


def _yaml(op: str, dtype: str, snr: float) -> str:
    shp = V.VENDOR_SHAPES[op]
    lines = [
        f"task_id: genv_{op}_{dtype}",
        f"operation: {op}",
        f"dtype: {dtype}",
        "backend: triton",
        "gpu_target: gfx942",
        "seed_kernel_name: seed_triton.py",
        f"snr_threshold: {snr}",
        f"op_family: vendor_{op}",
        "baseline_tier: vendor",
        "generated: true",
        "shapes:",
        f"  minimal: {_shape_str(shp['minimal'])}",
        f"  primary: {_shape_str(shp['primary'])}",
        "  validation:",
    ]
    for s in shp["validation"]:
        lines.append(f"    - {_shape_str(s)}")
    lines += ["targets:", f"  snr_db: {snr}", f"  comparison_baseline: aiter_{op}"]
    return "\n".join(lines) + "\n"


def generate(dry: bool = False) -> list[str]:
    written: list[str] = []
    for op in V.VENDOR_OPS:
        for dtype in V.vendor_op_dtypes(op):
            snr = V.DTYPES[dtype][2]
            tid = f"genv_{op}_{dtype}"
            written.append(tid)
            if dry:
                continue
            d = TASKS_DIR / tid
            d.mkdir(exist_ok=True)
            (d / "task.yaml").write_text(_yaml(op, dtype, snr))
            (d / "reference.py").write_text(_REF_SHIM.format(op=op, dtype=dtype))
            (d / "seed_triton.py").write_text(V.vendor_seed_source(op, dtype))
            (d / "driver.py").write_text(_DRIVER_SHIM.format(op=op, dtype=dtype))
    return written


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true")
    a = p.parse_args(argv)
    written = generate(dry=a.list)
    print(f"{'would generate' if a.list else 'generated'} {len(written)} vendor-baselined tasks:")
    for t in written:
        print(f"  {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
