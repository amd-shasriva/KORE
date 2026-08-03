"""Generate SPEC-SYNTHESIS tasks: a prose specification and no kernel to edit.

Why this shape exists as its own thing
--------------------------------------
Measured with ``scripts/seed_provenance_partition.py``, the trainable corpus is
14,924 tasks and **90.9% of them already require synthesis**: the 13,570
external-pool seeds alias an eager-torch baseline, and eager torch cannot be
edited into a faster kernel -- it has to be replaced by one written from
scratch.  So "the model only ever learns local search" was not true, and this
module is not here to fix that.

What is genuinely absent is the *specification medium*.  In every existing task
the specification is **executable torch** the model can read.  AgentKernelArena's
``instruction2triton`` shape instead states the contract in **English** -- what
the kernel computes, what each parameter means, what the layout is -- and hands
over a required entry point with nothing behind it.  28 of its 31 tasks give no
implementation at all.  Reading a specification is a different act from reading
a reference, and no task in the corpus asked for it.

So a spec task is defined by exactly three departures from a generated task:

* ``spec.md`` carries the contract in prose, and it is what the prompt leads with;
* the seed is a **signature-only stub**, so there is nothing to locally improve
  and no reference implementation to paraphrase;
* ``task_kind: spec_synthesis`` marks both facts, so a prompt builder or a
  verification policy can tell this apart from "optimize this kernel" instead of
  inferring it from the seed's contents.

Everything else is deliberately unchanged.  The oracle, the SNR gate, the
adversarial battery, the timing protocol and the production baseline all come
from :mod:`kore.tasks._genops` exactly as they do for a ``gen_`` task, because
the point under test is the task shape, not new numerics.  Reusing a proven
oracle is also what makes these tasks provable: the reference is not new code.

On operator reuse: these operations already exist as ``gen_*`` tasks, whose
seeds are working Triton kernels for the same math.  That is not a leak -- the
policy's tool set is build/test/bench/pmc/keep/revert with no filesystem read
(see :mod:`kore.agent.tools`), so a sibling task's seed is not reachable from an
episode.  The capability itself is scored on AKA's ``instruction2triton``, which
stays untrained; these tasks teach the shape on our own operators.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

TASKS_DIR = Path(__file__).resolve().parent

TASK_KIND = "spec_synthesis"
SPEC_FILENAME = "spec.md"
SEED_FILENAME = "seed_triton.py"

#: ``minted`` routes classification through
#: ``taxonomy.product_family_for_source("minted", ...)``, so each task declares a
#: canonical product family directly and no taxonomy rule has to change.  The
#: values mirror ``taxonomy.GENOPS_SOURCE_FAMILIES`` (plus its ``gelu_mul``
#: override) so a spec task and its ``gen_`` sibling land in the SAME family --
#: otherwise the split would treat the same operator as two different families.
_FAMILY_TO_PRODUCT = {
    "unary": "activation",
    "binary": "elementwise",
    "reduce": "reduction",
    "fusion": "fusion",
    "gemm_fusion": "gemm",
}
_PRODUCT_OVERRIDES = {"gelu_mul": "activation"}

#: SNR gates copied from the corresponding ``gen_`` tasks rather than chosen
#: here.  A spec task must not be easier to pass than the optimize task over the
#: same operator and dtype, or the shape would be trained at a discount.
_SNR_BY_DTYPE = {"fp32": 40.0, "bf16": 30.0, "fp16": 30.0}

_MN_SHAPES = {
    "minimal": {"M": 64, "N": 512},
    "primary": {"M": 4096, "N": 8192},
    "validation": [
        {"M": 8192, "N": 4096},
        {"M": 2048, "N": 11008},
        {"M": 4096, "N": 8191},
    ],
}
_MNK_SHAPES = {
    "minimal": {"M": 64, "N": 256, "K": 256},
    "primary": {"M": 512, "N": 4096, "K": 4096},
    "validation": [
        {"M": 1024, "N": 2048, "K": 2048},
        {"M": 256, "N": 14336, "K": 4096},
        {"M": 512, "N": 4096, "K": 4095},
    ],
}


@dataclass(frozen=True)
class SpecOp:
    """One operation's prose contract.

    ``summary`` and ``math`` are the specification proper: they must pin the
    computation exactly, because the model has nothing else to go on.  They are
    hand-written per operation -- a templated paraphrase of the torch expression
    would just be the reference in worse notation.
    """

    op: str
    family: str
    summary: str
    math: str
    notes: tuple[str, ...] = ()


#: The specifications.  Each one is written against the oracle in
#: ``_genops._registry()[op]`` and must state the same function, including the
#: details that are easy to get wrong (which tanh-GELU constant, whether the
#: reduction divides by N, which operand the activation applies to).
SPEC_OPS: tuple[SpecOp, ...] = (
    SpecOp(
        "gelu_tanh", "unary",
        "Apply the tanh approximation of GELU elementwise.",
        "y[i] = 0.5 * x[i] * (1 + tanh(sqrt(2/pi) * (x[i] + 0.044715 * x[i]**3)))",
        (
            "Use the tanh approximation, NOT the exact erf form: the oracle is the "
            "tanh variant and the two differ by more than the SNR gate allows.",
            "sqrt(2/pi) is 0.7978845608028654.",
        ),
    ),
    SpecOp(
        "mish", "unary",
        "Apply the Mish activation elementwise.",
        "y[i] = x[i] * tanh(softplus(x[i])),  softplus(x) = log(1 + exp(x))",
        (
            "softplus overflows in fp32 for large x. Guard it: for x above roughly "
            "20, softplus(x) == x to within fp32 resolution, so return x there "
            "instead of evaluating log(1 + exp(x)).",
        ),
    ),
    SpecOp(
        "hardswish", "unary",
        "Apply the Hard-Swish activation elementwise.",
        "y[i] = x[i] * clamp(x[i] + 3, 0, 6) / 6",
    ),
    SpecOp(
        "add", "binary",
        "Add two tensors elementwise.",
        "y[i] = a[i] + b[i]",
        (
            "This is bandwidth-bound: three tensor traversals and one add. The "
            "whole problem is issuing wide, coalesced, well-occupied loads.",
        ),
    ),
    SpecOp(
        "mul_sig", "binary",
        "Gate the first tensor by the sigmoid of the second, elementwise.",
        "y[i] = a[i] * sigmoid(b[i])",
        (
            "The sigmoid applies to the SECOND operand only; the first passes "
            "through ungated.",
        ),
    ),
    SpecOp(
        "add_silu", "fusion",
        "Add two tensors, then apply SiLU to the sum, in one pass.",
        "s[i] = a[i] + b[i];  y[i] = s[i] * sigmoid(s[i])",
        (
            "Compute the sum ONCE and reuse it for both factors. The baseline is a "
            "compiler-fused torch chain, so a two-pass implementation that "
            "materializes the sum to memory will not beat it.",
        ),
    ),
    SpecOp(
        "gelu_mul", "fusion",
        "Apply tanh-GELU to the first tensor and multiply by the second.",
        "y[i] = gelu_tanh(a[i]) * b[i]",
        (
            "This is the GeGLU feed-forward gate: `a` is the gate projection and "
            "`b` the up projection.",
            "gelu_tanh is the tanh approximation defined above, with "
            "sqrt(2/pi) = 0.7978845608028654.",
        ),
    ),
    SpecOp(
        "reglu", "fusion",
        "Apply ReLU to the first tensor and multiply by the second.",
        "y[i] = max(a[i], 0) * b[i]",
        ("This is the ReGLU feed-forward gate.",),
    ),
    SpecOp(
        "row_sum", "reduce",
        "Sum each row of a 2-D tensor.",
        "y[m] = sum over n of x[m, n]",
        (
            "Accumulate in fp32 even when the input is bf16/fp16: an N of 8192 "
            "summed in bf16 loses the gate on its own.",
        ),
    ),
    SpecOp(
        "row_max", "reduce",
        "Take the maximum of each row of a 2-D tensor.",
        "y[m] = max over n of x[m, n]",
        (
            "N is not always a multiple of the block size. Mask the tail with "
            "-inf (or the dtype's most negative value), NOT with 0.0, or a row "
            "that is entirely negative returns 0.",
        ),
    ),
    SpecOp(
        "row_rms", "reduce",
        "Compute the root-mean-square of each row of a 2-D tensor.",
        "y[m] = sqrt( (1/N) * sum over n of x[m, n]**2 )",
        (
            "Divide the sum of squares by N before the square root. This is the "
            "normalizer inside RMSNorm.",
            "Accumulate the squares in fp32.",
            "Mask the tail of a partial block with 0.0, which is the identity for "
            "this accumulator.",
        ),
    ),
    SpecOp(
        "gemm_bias_silu", "gemm_fusion",
        "Multiply two matrices, add a per-column bias, then apply SiLU.",
        "y[m, n] = silu( sum over k of a[m, k] * b[k, n] + bias[n] ),  "
        "silu(t) = t * sigmoid(t)",
        (
            "`a` is [M, K], `b` is [K, N], `bias` is [N], output is [M, N].",
            "Use tl.dot so Triton emits MFMA; do not hand-roll the inner product.",
            "Accumulate in fp32 and apply the bias and activation to the fp32 "
            "accumulator, before the cast back to the output dtype.",
            "The baseline is a compiler-fused epilogue GEMM, so the bias and the "
            "activation must stay in the same kernel as the accumulation.",
        ),
    ),
)

SPEC_OP_BY_NAME = {s.op: s for s in SPEC_OPS}

#: bf16 is the serving dtype for the product target; fp32 is kept because a
#: synthesis task the model can actually solve early is what makes the shape
#: learnable rather than a wall of zero reward.
DTYPES: tuple[str, ...] = ("bf16", "fp32")


def _entry_signature(spec: SpecOp) -> str:
    if spec.family == "unary":
        return f"def {spec.op}(x: torch.Tensor) -> torch.Tensor"
    if spec.family == "binary":
        return f"def {spec.op}(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor"
    if spec.family == "fusion":
        return f"def {spec.op}(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor"
    if spec.family == "reduce":
        return f"def {spec.op}(x: torch.Tensor) -> torch.Tensor"
    if spec.family == "gemm_fusion":
        return (f"def {spec.op}(a: torch.Tensor, b: torch.Tensor, "
                f"bias: torch.Tensor) -> torch.Tensor")
    raise ValueError(f"unknown family {spec.family!r}")


def _arg_names(spec: SpecOp) -> tuple[str, ...]:
    return {
        "unary": ("x",),
        "reduce": ("x",),
        "binary": ("a", "b"),
        "fusion": ("a", "b"),
        "gemm_fusion": ("a", "b", "bias"),
    }[spec.family]


def _io_section(spec: SpecOp, dtype: str) -> str:
    if spec.family == "gemm_fusion":
        return (
            f"## Inputs\n"
            f"- `a`: 2-D, shape `[M, K]`, dtype {dtype}, contiguous.\n"
            f"- `b`: 2-D, shape `[K, N]`, dtype {dtype}, contiguous.\n"
            f"- `bias`: 1-D, shape `[N]`, dtype {dtype}, contiguous.\n\n"
            f"## Output\n"
            f"- 2-D, shape `[M, N]`, dtype {dtype}. Allocate it yourself and "
            f"return it.\n"
        )
    if spec.family == "reduce":
        return (
            f"## Inputs\n"
            f"- `x`: 2-D, shape `[M, N]`, dtype {dtype}, contiguous, row-major. "
            f"The reduction is over the LAST axis.\n\n"
            f"## Output\n"
            f"- 1-D, shape `[M]`, dtype {dtype} (one value per row). Allocate it "
            f"yourself and return it.\n"
        )
    args = _arg_names(spec)
    lines = "".join(
        f"- `{a}`: 2-D, shape `[M, N]`, dtype {dtype}, contiguous.\n" for a in args
    )
    return (
        f"## Inputs\n{lines}\n"
        f"## Output\n"
        f"- 2-D, shape `[M, N]`, dtype {dtype}. Allocate it yourself and return it.\n"
    )


def spec_markdown(spec: SpecOp, dtype: str, snr_db: float) -> str:
    """The natural-language contract the model is given INSTEAD of a kernel."""
    notes = "".join(f"- {n}\n" for n in spec.notes)
    notes_block = f"\n## Requirements and pitfalls\n{notes}" if notes else ""
    return (
        f"# Specification: `{spec.op}` ({dtype})\n\n"
        f"{spec.summary}\n\n"
        f"## Definition\n\n"
        f"```\n{spec.math}\n```\n\n"
        f"{_io_section(spec, dtype)}"
        f"{notes_block}\n"
        f"## Required entry point\n\n"
        f"Your module MUST define this exact top-level function, with this name "
        f"and this parameter order:\n\n"
        f"```python\n{_entry_signature(spec)}\n```\n\n"
        f"It is called directly by the verifier. A module that does not define "
        f"`{spec.op}` fails to load, which scores zero regardless of what else "
        f"it contains.\n\n"
        f"## How you are graded\n\n"
        f"1. The module must import and expose `{spec.op}`.\n"
        f"2. `{spec.op}` must reach at least **{snr_db:.0f} dB SNR** against the "
        f"fp32 oracle on EVERY validation shape, including the one whose N is not "
        f"a multiple of a power of two.\n"
        f"3. Only then is it timed, against the production baseline.\n\n"
        f"Write a real Triton kernel. Calling the framework op "
        f"(`torch.nn.functional.*`, `torch.matmul`) as the implementation "
        f"satisfies neither the intent nor the speed target.\n"
    )


def seed_stub(spec: SpecOp) -> str:
    """The 'seed': the required signature and nothing behind it.

    This is what makes the task synthesis rather than editing.  It is deliberately
    NOT a working implementation and deliberately NOT the torch one-liner -- a
    torch body would be a reference to paraphrase, which is the shape we already
    have 13,570 of.
    """
    args = ", ".join(_arg_names(spec))
    return (
        f'"""GENERATED signature stub for a SPEC-SYNTHESIS task.\n'
        f"See kore/tasks/generate_spec.py. Do not hand-edit.\n\n"
        f"There is no implementation here on purpose: the task is to write one\n"
        f"from the prose contract in spec.md. This stub exists so the task keeps\n"
        f"the registry's 'every task has a declared seed artifact' invariant and\n"
        f'so the required entry-point signature is unambiguous.\n"""\n'
        f"from __future__ import annotations\n\n"
        f"import torch  # noqa: F401 (the implementation will need it)\n\n\n"
        f"def {spec.op}({args}):\n"
        f'    raise NotImplementedError(\n'
        f'        "spec-synthesis task: implement {spec.op} from spec.md"\n'
        f"    )\n"
    )


def reference_source(spec: SpecOp, dtype: str) -> str:
    return (
        f'"""GENERATED reference for a SPEC-SYNTHESIS task ({spec.op}, {dtype}).\n'
        f"See kore/tasks/generate_spec.py. Do not hand-edit.\n\n"
        f"The oracle, baseline and tolerance are _genops' proven ones, unchanged:\n"
        f'the spec shape changes the PROMPT, not the numerics."""\n'
        f"from kore.tasks._genops import make_reference\n\n"
        f"globals().update(make_reference({spec.op!r}, {spec.family!r}, {dtype!r}))\n"
    )


def driver_source(spec: SpecOp, dtype: str) -> str:
    return (
        f'"""GENERATED driver shim for a SPEC-SYNTHESIS task ({spec.op}, {dtype}).\n'
        f'See kore/tasks/generate_spec.py. Do not hand-edit."""\n'
        f"import os\n"
        f"import sys\n\n"
        f"_here = os.path.dirname(os.path.abspath(__file__))\n"
        f"sys.path.insert(0, _here)\n"
        f"import reference as ref  # noqa: E402\n"
        f"from kore.tasks._genops import driver_main  # noqa: E402\n\n"
        f'if __name__ == "__main__":\n'
        f"    raise SystemExit(driver_main(ref, _here))\n"
    )


def task_id_for(op: str, dtype: str) -> str:
    return f"spec_{op}_{dtype}"


def task_yaml(spec: SpecOp, dtype: str) -> str:
    snr = _SNR_BY_DTYPE[dtype]
    shapes = _MNK_SHAPES if spec.family == "gemm_fusion" else _MN_SHAPES
    product = _PRODUCT_OVERRIDES.get(spec.op, _FAMILY_TO_PRODUCT[spec.family])
    meta = {
        "task_id": task_id_for(spec.op, dtype),
        "operation": spec.op,
        "dtype": dtype,
        "backend": "triton",
        "gpu_target": "gfx950",
        "seed_kernel_name": SEED_FILENAME,
        "snr_threshold": snr,
        "op_family": spec.family,
        "baseline_tier": (
            "gemm_fusion" if spec.family == "gemm_fusion" else "elementwise"),
        # The two fields that make this a spec task rather than an optimize task.
        "task_kind": TASK_KIND,
        "spec_file": SPEC_FILENAME,
        "minted": True,
        "taxonomy_family": product,
        "provenance_root": task_id_for(spec.op, dtype),
        "shapes": shapes,
        "targets": {
            "snr_db": snr,
            "comparison_baseline": f"torch_{spec.op}",
        },
    }
    return json.dumps(meta, indent=2, sort_keys=True) + "\n"


def write_task(spec: SpecOp, dtype: str, root: Optional[Path] = None) -> Path:
    root = Path(root) if root is not None else TASKS_DIR
    d = root / task_id_for(spec.op, dtype)
    d.mkdir(parents=True, exist_ok=True)
    (d / "task.yaml").write_text(task_yaml(spec, dtype), encoding="utf-8")
    (d / SPEC_FILENAME).write_text(
        spec_markdown(spec, dtype, _SNR_BY_DTYPE[dtype]), encoding="utf-8")
    (d / "reference.py").write_text(reference_source(spec, dtype), encoding="utf-8")
    (d / "driver.py").write_text(driver_source(spec, dtype), encoding="utf-8")
    (d / SEED_FILENAME).write_text(seed_stub(spec), encoding="utf-8")
    return d


def generate(root: Optional[Path] = None,
             dtypes: tuple[str, ...] = DTYPES) -> list[str]:
    written: list[str] = []
    for spec in SPEC_OPS:
        for dtype in dtypes:
            write_task(spec, dtype, root)
            written.append(task_id_for(spec.op, dtype))
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--dtypes", nargs="*", default=list(DTYPES))
    args = ap.parse_args()
    ids = generate(args.root, tuple(args.dtypes))
    print(f"wrote {len(ids)} spec-synthesis tasks")
    for t in ids:
        print(f"  {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
