"""Generate KORE task directories at scale from the _genops op registry.

Writes, per (op, dtype), a task dir ``gen_<op>_<dtype>/`` containing:
  * task.yaml         — metadata + shapes (family-appropriate)
  * reference.py      — thin shim -> _genops.make_reference (torch oracle + inputs)
  * seed_triton.py    — a REAL compiling Triton starter kernel (policy edits this)
  * driver.py         — thin shim -> _genops.driver_main (the verifier contract)

Idempotent: re-running overwrites the generated files. Use a ``gen_`` prefix so
generated tasks never collide with the 15 hand-authored ones. Registry discovery
(kore.tasks.registry) then picks them up automatically.

    python -m kore.tasks.generate_ops            # generate all
    python -m kore.tasks.generate_ops --list     # list what would be generated
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kore.tasks import _genops

TASKS_DIR = Path(__file__).resolve().parent

# dtypes to emit per family (kept modest to bound eval cost while staying wide).
FAMILY_DTYPES = {
    "unary": ("bf16", "fp16", "fp32"),
    "binary": ("bf16", "fp16"),
    "reduce": ("bf16", "fp16"),
    "fusion": ("bf16", "fp16", "fp32"),   # the high-headroom class -> all dtypes
}

# Honest headroom tier per family: fusions beat torch-eager multi-kernel (real
# speedup headroom); single elementwise/reduction are near-roofline (correctness-
# training value, low speedup headroom). Recorded in task.yaml for the audit/reward.
FAMILY_TIER = {
    "unary": "elementwise", "binary": "elementwise", "reduce": "elementwise",
    "fusion": "fusion",
}

# Family-appropriate shape sweeps (small/medium/large + a non-power-of-two tail).
SHAPES = {
    "minimal": {"M": 64, "N": 512},
    "primary": {"M": 4096, "N": 8192},
    "validation": [
        {"M": 8192, "N": 4096},      # tall
        {"M": 2048, "N": 11008},     # Llama MLP inter dim
        {"M": 4096, "N": 8191},      # non-pow2 N tail (masking edge)
    ],
}

_REF_SHIM = '''"""GENERATED reference shim for {op} ({dtype}). See kore/tasks/_genops.py.
Do not hand-edit — regenerate via kore/tasks/generate_ops.py."""
from kore.tasks._genops import make_reference

globals().update(make_reference("{op}", "{family}", "{dtype}"))
'''

_DRIVER_SHIM = '''"""GENERATED driver shim for {op} ({dtype}). See kore/tasks/_genops.py.
Do not hand-edit — regenerate via kore/tasks/generate_ops.py."""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
import reference as ref  # noqa: E402
from kore.tasks._genops import driver_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(driver_main(ref, _here))
'''


def _yaml(op: str, family: str, dtype: str, snr: float) -> str:
    lines = [
        f"task_id: gen_{op}_{dtype}",
        f"operation: {op}",
        f"dtype: {dtype}",
        "backend: triton",
        "gpu_target: gfx942",
        "seed_kernel_name: seed_triton.py",
        f"snr_threshold: {snr}",
        f"op_family: {family}",
        f"baseline_tier: {FAMILY_TIER[family]}",
        "generated: true",
        "shapes:",
        f"  minimal: {{M: {SHAPES['minimal']['M']}, N: {SHAPES['minimal']['N']}}}",
        f"  primary: {{M: {SHAPES['primary']['M']}, N: {SHAPES['primary']['N']}}}",
        "  validation:",
    ]
    for s in SHAPES["validation"]:
        lines.append(f"    - {{M: {s['M']}, N: {s['N']}}}")
    lines += [
        "targets:",
        f"  snr_db: {snr}",
        f"  comparison_baseline: torch_{op}",
    ]
    return "\n".join(lines) + "\n"


def _plan() -> list[tuple[str, str, str, float]]:
    """(op, family, dtype, snr) for everything to generate."""
    reg = _genops._registry()
    plan = []
    for op in sorted(reg):
        family, _ = reg[op]
        for dtype in FAMILY_DTYPES[family]:
            snr = _genops.DTYPES[dtype][2]
            plan.append((op, family, dtype, snr))
    return plan


def generate(dry: bool = False) -> list[str]:
    written: list[str] = []
    for op, family, dtype, snr in _plan():
        tid = f"gen_{op}_{dtype}"
        d = TASKS_DIR / tid
        written.append(tid)
        if dry:
            continue
        d.mkdir(exist_ok=True)
        (d / "task.yaml").write_text(_yaml(op, family, dtype, snr))
        (d / "reference.py").write_text(_REF_SHIM.format(op=op, family=family, dtype=dtype))
        (d / "seed_triton.py").write_text(_genops.seed_source(op, family, dtype))
        (d / "driver.py").write_text(_DRIVER_SHIM.format(op=op, dtype=dtype))
    return written


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true", help="list tasks without writing")
    a = p.parse_args(argv)
    written = generate(dry=a.list)
    verb = "would generate" if a.list else "generated"
    print(f"{verb} {len(written)} operator tasks:")
    for t in written:
        print(f"  {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
