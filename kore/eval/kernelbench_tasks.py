"""Make REAL KernelBench problems executable through KORE's verified pipeline.

:mod:`kore.eval.kernelbench_amd` can already turn a KernelBench-style *spec* into
a KORE :class:`~kore.tasks.base.Task` and render a ``fast_p`` report, but a task
minted that way has no directory on disk, and :class:`~kore.env.kore_env.KoreEnv`
executes a candidate by copying ``task.dir/*.py`` into a workdir and running
``driver.py`` there. So the minted tasks were never runnable: the adapter could
describe a KernelBench problem but nothing could measure one.

This module closes that gap. :func:`materialize` reads a KernelBench checkout and
writes, per problem, the four files a KORE task directory is defined by
(``task.yaml`` / ``reference.py`` / ``driver.py`` / seed), where:

  * ``driver.py`` is the SAME one-line shim every generated KORE task uses, so the
    whole publication-grade protocol - paired balanced AB/BA timing with raw
    per-pair samples, fresh inputs per pair, the post-timing anti-hack
    re-verification, the versioned capability handshake - is inherited from
    :mod:`kore.tasks._genops` rather than re-implemented here;
  * ``reference.py`` exposes that driver's reference ABI (:func:`make_reference`)
    on top of the KernelBench problem, with the PyTorch module's forward as BOTH
    the correctness oracle and the timing baseline, which is exactly KernelBench's
    torch-eager baseline;
  * the seed file is a SPECIFICATION, not a runnable kernel (see below).

FUNCTIONALIZATION.  A KernelBench problem is an ``nn.Module`` with its own
randomly-initialized parameters, while a KORE candidate is a plain function. The
module is therefore functionalized: the module is built ONCE under a fixed seed
and its parameters/buffers become trailing ARGUMENTS of the candidate entry
point, so ``Model(x)`` with a ``nn.Linear`` becomes ``kb_forward(x, weight,
bias)``. The oracle evaluates the untouched ``Model.forward`` on those same
tensors through ``torch.func.functional_call``, so the function being graded is
bit-for-bit the KernelBench problem - only its calling convention changed.

NO RUNNABLE SEED, ON PURPOSE.  KernelBench hands a model the reference PyTorch
module and asks for a rewrite; there is no starter kernel. Emitting a torch-eager
"seed" here would be worse than useless: :func:`kore.reward.reward.scan_for_hacks`
rejects torch delegation as a reward hack, so such a seed would score as a hack,
and any candidate could ``import`` it to obtain the oracle. The seed file is
therefore a fully commented specification - the required entry signature, every
argument's shape/dtype, and the verbatim KernelBench reference source - which is
what the prompt shows the model and which yields nothing when imported. For the
same reason the problem source is EMBEDDED as a string literal inside
``reference.py`` (already unimportable by contract) instead of being shipped as a
second module a candidate could import.

Import-safe / offline: torch is imported lazily inside the produced closures and
inside :func:`materialize`, so importing this module needs no GPU and no torch.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

# The candidate entry point every materialized KernelBench task declares. One
# uniform name (rather than a per-problem one) keeps the contract unambiguous:
# the seed states it verbatim, so a model has one thing to reproduce.
ENTRY_NAME = "kb_forward"

# Seed used to initialize the problem module's parameters. Fixed, so the graded
# function is the same one on every process, node and rerun.
PARAM_SEED = 0

# The single nominal shape a materialized task declares. KernelBench fixes its
# sizes inside ``get_inputs``, so there is exactly one; ``elems`` carries the
# largest input tensor's element count, which is real information (it is what
# :func:`kore.eval.kernelbench_amd.shape_regime` classifies on) rather than a
# placeholder.
SHAPE_NAME = "kernelbench"

# Family tag. Deliberately NOT one of ``kore.tasks._genops._GENERIC_ADV_FAMILIES``:
# those generic adversarial fills (all-zeros / all-1e3 / alternating-sign over
# EVERY float input, including convolution weights) are calibrated for the
# generated elementwise corpus and are not part of the KernelBench contract, so
# claiming them here would reject kernels for regimes the benchmark never asks
# about. The oracle that does run is unchanged: five reseeded random trials, the
# dtype-step elementwise bound, the SNR gate, the determinism re-check and the
# post-timing re-verification.
FAMILY = "kernelbench"

# KernelBench grades against torch-eager, never a vendor library. Recorded on
# every timing pair so an artifact says which bar produced its speedups.
BASELINE_KIND = "torch_eager"

DEFAULT_DTYPE = "fp32"
DEFAULT_SNR_DB = 40.0
DEFAULT_GPU_TARGET = "gfx950"


# --------------------------------------------------------------------------- #
# Reference ABI: one KernelBench problem as a kore.tasks._genops reference.
# --------------------------------------------------------------------------- #
def _exec_problem(source: str, key: str) -> dict:
    """Execute a KernelBench problem's source in a private namespace."""
    namespace: dict = {"__name__": f"kernelbench_{key}", "__file__": f"<kernelbench:{key}>"}
    exec(compile(source, f"<kernelbench:{key}>", "exec"), namespace)  # noqa: S102
    return namespace


def problem_entities(source: str, key: str, device: str, *,
                     param_seed: int = PARAM_SEED) -> dict:
    """Build the graded objects for one problem: module, params, input factory.

    The module is constructed ONCE, under ``param_seed``, with the default device
    set so its parameters are allocated straight onto ``device`` (KernelBench's
    own ``get_inputs``/``__init__`` hard-code CPU allocation, and materializing a
    multi-gigabyte tensor on the host only to copy it across PCIe would dominate
    every measurement).
    """
    import torch

    namespace = _exec_problem(source, key)
    model_cls = namespace.get("Model")
    get_inputs = namespace.get("get_inputs")
    if model_cls is None or get_inputs is None:
        raise ValueError("problem does not define both Model and get_inputs")
    get_init_inputs = namespace.get("get_init_inputs") or (lambda: [])

    torch.manual_seed(int(param_seed))
    with torch.device(device):
        init_args = list(get_init_inputs() or [])
        model = model_cls(*init_args)

    param_names: list[str] = []
    param_tensors: list[Any] = []
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        param_names.append(name)
        param_tensors.append(tensor.detach())
    return {
        "model": model,
        "get_inputs": get_inputs,
        "param_names": param_names,
        "param_tensors": param_tensors,
        "device": device,
    }


def parse_shape(spec: Optional[str]) -> dict:
    """Parse a ``k=v,k=v`` shape spec. KernelBench fixes its own sizes.

    The dims are carried for provenance only - ``get_inputs`` below ignores them,
    because changing them would change the problem rather than the shape.
    """
    if not spec or spec == "default":
        return {}
    dims: dict[str, int] = {}
    for item in str(spec).split(","):
        if "=" not in item:
            continue
        key, _, value = item.partition("=")
        try:
            dims[key.strip()] = int(value)
        except ValueError:
            continue
    return dims


def make_reference(problem_source: str, *, key: str, n_model_inputs: int,
                   mutates_input: bool, entry_name: str = ENTRY_NAME,
                   dtype: str = DEFAULT_DTYPE,
                   param_seed: int = PARAM_SEED) -> dict:
    """Return the ``kore.tasks._genops`` reference namespace for one problem.

    ``ref_fn`` and ``baseline_fn`` are the SAME callable on purpose: KernelBench's
    baseline IS its reference ``Model.forward``, so the oracle and the timing bar
    are one object and cannot drift apart.

    Nothing is built at call time - the module and its parameters are constructed
    on first use and cached per device - so ``reference.py`` stays importable on a
    machine with no GPU.
    """
    state: dict = {}

    def _state():
        device = os.environ.get("KORE_KB_DEVICE", "cuda")
        cached = state.get(device)
        if cached is None:
            cached = problem_entities(problem_source, key, device,
                                      param_seed=param_seed)
            state[device] = cached
        return cached

    def get_inputs(shape=None, device: str = "cuda", seed: int = 0) -> tuple:
        """Model inputs for ``seed``, followed by the module's fixed parameters."""
        import torch
        entities = _state()
        torch.manual_seed(int(seed))
        with torch.device(entities["device"]):
            model_inputs = list(entities["get_inputs"]())
        return tuple(model_inputs) + tuple(entities["param_tensors"])

    def ref_fn(*args):
        """The untouched KernelBench ``Model.forward``, evaluated functionally."""
        from torch.func import functional_call
        entities = _state()
        model_inputs = tuple(args[:n_model_inputs])
        parameters = dict(zip(entities["param_names"], args[n_model_inputs:]))
        return functional_call(entities["model"], parameters, model_inputs)

    def baseline_fn(*args):
        from kore.tasks.aiter_ref import _mark_baseline
        _mark_baseline(BASELINE_KIND)
        return ref_fn(*args)

    return {
        "parse_shape": parse_shape,
        "get_inputs": get_inputs,
        "ref_fn": ref_fn,
        "baseline_fn": baseline_fn,
        "entry_name": entry_name,
        "dtype_name": dtype,
        "family": FAMILY,
        "baseline_kind": BASELINE_KIND,
        "mutates_input": bool(mutates_input),
        "n_model_inputs": int(n_model_inputs),
    }


# --------------------------------------------------------------------------- #
# Materialization: a KernelBench checkout -> runnable KORE task directories.
# --------------------------------------------------------------------------- #
_DRIVER_SHIM = '''"""GENERATED driver shim for KernelBench {level}/{name}.

The verifier contract itself lives in kore/tasks/_genops.py - this task inherits
it unchanged. Regenerate via kore/eval/kernelbench_tasks.py; do not hand-edit.
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
import reference as ref  # noqa: E402
from kore.tasks._genops import driver_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(driver_main(ref, _here))
'''

_REFERENCE_SHIM = '''"""GENERATED reference for KernelBench {level}/{name}.

The problem source is EMBEDDED rather than shipped as an importable sibling
module: kore.reward.reward.scan_for_hacks already rejects a candidate that
imports ``reference``, and a second module carrying the same oracle would be an
unguarded way to reach it. Regenerate via kore/eval/kernelbench_tasks.py.
"""
from kore.eval.kernelbench_tasks import make_reference

PROBLEM_SOURCE = {source!r}

globals().update(make_reference(
    PROBLEM_SOURCE,
    key={key!r},
    n_model_inputs={n_model_inputs},
    mutates_input={mutates_input},
    entry_name={entry_name!r},
    dtype={dtype!r},
))
'''


def _sanitize(name: str) -> str:
    return "".join(c if (c.isalnum() or c == "_") else "_" for c in name.lower())


def task_id_for(name: str, level: int) -> str:
    """``kb_l{level}_{sanitized name}`` - matches ``KernelBenchSpec.task_id``."""
    return f"kb_l{int(level)}_{_sanitize(name)}"


def operation_for(name: str) -> str:
    """A readable operator label for the prompt (``12_Gemm_Multiply_LeakyReLU``
    -> ``gemm_multiply_leakyrelu``)."""
    stripped = re.sub(r"^\d+_", "", name)
    return _sanitize(stripped).strip("_") or "kernelbench_problem"


def _describe(value, index: int, label: str) -> str:
    import torch
    if torch.is_tensor(value):
        dims = "x".join(str(int(d)) for d in value.shape) or "scalar"
        return (f"#  arg{index:<2} {label:<28} {str(value.dtype).replace('torch.', ''):<10}"
                f" [{dims}]")
    return f"#  arg{index:<2} {label:<28} {type(value).__name__:<10} (non-tensor)"


def _seed_document(name: str, level: int, entry_name: str, arg_labels: Sequence[str],
                   arg_values: Sequence[Any], output_desc: str,
                   problem_source: str) -> str:
    """The seed file: a contract specification, deliberately not runnable code.

    KernelBench provides no starter kernel, so neither do we. Everything here is
    a comment, which means importing this file from a candidate yields nothing.
    """
    signature = f"def {entry_name}(" + ", ".join(arg_labels) + "):"
    lines = [
        f'# KernelBench level {level} / {name}',
        "#",
        "# There is no seed kernel for a KernelBench problem: the benchmark asks you",
        "# to REPLACE the PyTorch reference below with a real GPU kernel. This file is",
        "# the contract, not runnable code.",
        "#",
        "# REQUIRED ENTRY POINT - kernel.py must define exactly:",
        "#",
        f"#     {signature}",
        "#",
        "# ARGUMENTS (the module's parameters/buffers are passed in, so the kernel is",
        "# a pure function of its inputs):",
        "#",
    ]
    for index, (label, value) in enumerate(zip(arg_labels, arg_values)):
        lines.append(_describe(value, index, label))
    lines += [
        "#",
        f"#  returns  {output_desc}",
        "#",
        "# The baseline you must beat is this exact PyTorch module running eagerly.",
        "# Computing the result with torch ops (or importing this file) is a reward",
        "# hack and scores zero - write the kernel.",
        "#",
        "# ---- KernelBench reference (verbatim) ------------------------------------",
        "#",
    ]
    lines += ["# " + line if line else "#" for line in problem_source.splitlines()]
    lines.append("")
    return "\n".join(lines)


def _mutation_detected(before: Sequence[Any], after: Sequence[Any]) -> bool:
    import torch
    for a, b in zip(before, after):
        if torch.is_tensor(a) and torch.is_tensor(b) and not torch.equal(a, b):
            return True
    return False


def _probe(problem_source: str, key: str, device: str, param_seed: int) -> dict:
    """Run one problem once to learn its ABI: arg names, shapes, and mutation.

    Everything the task directory declares about a problem is MEASURED here
    rather than guessed: how many model inputs there are, what each argument is,
    what the reference returns, and - the one fact no signature reveals - whether
    a forward pass mutates its own parameters/buffers (a ``BatchNorm`` left in
    KernelBench's default train mode updates its running statistics in place, and
    a timing loop that reuses one buffer for every invocation would then be
    timing a drifting function).
    """
    import torch
    from torch.func import functional_call

    entities = problem_entities(problem_source, key, device, param_seed=param_seed)
    model, param_names = entities["model"], entities["param_names"]
    param_tensors = entities["param_tensors"]

    torch.manual_seed(0)
    with torch.device(device):
        model_inputs = list(entities["get_inputs"]())

    forward_names = _forward_arg_names(model, len(model_inputs))
    arg_labels = list(forward_names) + [_sanitize(n) for n in param_names]
    arg_values = list(model_inputs) + list(param_tensors)

    before = [t.detach().clone() for t in param_tensors]
    parameters = dict(zip(param_names, [t.detach().clone() for t in param_tensors]))
    output = functional_call(model, parameters, tuple(model_inputs))
    torch.cuda.synchronize() if device.startswith("cuda") else None
    mutates = _mutation_detected(before, list(parameters.values()))

    outputs = output if isinstance(output, (tuple, list)) else (output,)
    output_desc = ", ".join(
        f"{str(o.dtype).replace('torch.', '')}[{'x'.join(str(int(d)) for d in o.shape)}]"
        if torch.is_tensor(o) else type(o).__name__
        for o in outputs)
    largest = max((int(t.numel()) for t in arg_values if torch.is_tensor(t)), default=1)

    return {
        "n_model_inputs": len(model_inputs),
        "arg_labels": arg_labels,
        "arg_values": arg_values,
        "output_desc": output_desc,
        "mutates_input": bool(mutates),
        "largest_input_elems": max(1, largest),
        "n_parameters": len(param_names),
    }


def _forward_arg_names(model, count: int) -> list[str]:
    """Parameter names of ``Model.forward``, falling back to ``x0, x1, ...``."""
    import inspect
    try:
        params = list(inspect.signature(type(model).forward).parameters)[1:]
    except (TypeError, ValueError):
        params = []
    names = [_sanitize(p) for p in params[:count]]
    while len(names) < count:
        names.append(f"x{len(names)}")
    return names


def _task_yaml(*, task_id: str, operation: str, level: int, name: str, dtype: str,
               gpu_target: str, snr_db: float, elems: int, revision: str,
               problem_sha: str, entry_name: str, n_model_inputs: int,
               n_parameters: int, mutates_input: bool) -> str:
    payload = {
        "task_id": task_id,
        "operation": operation,
        "dtype": dtype,
        "backend": "triton",
        "gpu_target": gpu_target,
        "seed_kernel_name": "seed_reference.py",
        "snr_threshold": float(snr_db),
        # NOT a kore.tasks._genops task: the metamorphic prong keys off
        # ``generated`` + a ``gen_`` id and must stay not-applicable here, because
        # no structural identity is proven for an arbitrary KernelBench module.
        "generated": False,
        "source": "kernelbench",
        "kernelbench": {
            "name": name,
            "level": int(level),
            "revision": revision,
            "problem_sha256": problem_sha,
            "entry_name": entry_name,
            "n_model_inputs": int(n_model_inputs),
            "n_parameters": int(n_parameters),
            "mutates_input": bool(mutates_input),
        },
        "shapes": {SHAPE_NAME: {"elems": int(elems)}},
        "targets": {"snr_db": float(snr_db), "comparison_baseline": BASELINE_KIND},
    }
    import yaml
    return yaml.safe_dump(payload, sort_keys=False)


def level_dir(kb_root, level: int) -> Path:
    """The directory holding level ``level``'s problem files in a checkout."""
    base = Path(kb_root)
    nested = base / "KernelBench" / f"level{int(level)}"
    return nested if nested.exists() else base / f"level{int(level)}"


def checkout_revision(kb_root) -> str:
    """The KernelBench checkout's git revision, or ``"unknown"``."""
    head = Path(kb_root) / ".git" / "HEAD"
    try:
        text = head.read_text().strip()
    except OSError:
        return "unknown"
    if text.startswith("ref:"):
        ref_path = Path(kb_root) / ".git" / text.split(" ", 1)[1].strip()
        try:
            return ref_path.read_text().strip()
        except OSError:
            return "unknown"
    return text


def materialize(kb_root, out_dir, *, levels: Sequence[int] = (1, 2),
                gpu_target: str = DEFAULT_GPU_TARGET, dtype: str = DEFAULT_DTYPE,
                snr_db: float = DEFAULT_SNR_DB, device: str = "cuda",
                entry_name: str = ENTRY_NAME, param_seed: int = PARAM_SEED,
                only: Optional[Sequence[str]] = None, limit: Optional[int] = None,
                log: Optional[Callable[[str], None]] = None) -> dict:
    """Write a runnable KORE task directory for every KernelBench problem.

    Returns a manifest ``{kernelbench_root, revision, levels, tasks, skipped}``.
    A problem that cannot be probed (it needs an unavailable op, its module will
    not build, it exhausts memory) is SKIPPED with its reason recorded rather
    than silently dropped, so the manifest states exactly which slice of the
    benchmark the run covers.
    """
    root = Path(kb_root)
    if not root.exists():
        raise FileNotFoundError(
            f"KernelBench checkout not found at {str(root)!r}. Clone it "
            "(github.com/ScalingIntelligence/KernelBench) and point --kernelbench-root "
            "at the repo so level{1,2}/*.py can be loaded.")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    revision = checkout_revision(root)
    wanted = {str(s) for s in (only or [])}

    tasks: list[dict] = []
    skipped: list[dict] = []
    for level in levels:
        directory = level_dir(root, level)
        if not directory.exists():
            skipped.append({"name": f"level{level}", "level": int(level),
                            "reason": f"level directory {directory} does not exist"})
            continue
        paths = sorted(directory.glob("*.py"))
        if wanted:
            paths = [p for p in paths if p.stem in wanted]
        if limit is not None:
            paths = paths[:int(limit)]
        for path in paths:
            name = path.stem
            try:
                record = _materialize_one(
                    path, name, level, out, revision=revision, gpu_target=gpu_target,
                    dtype=dtype, snr_db=snr_db, device=device, entry_name=entry_name,
                    param_seed=param_seed)
            except BaseException as exc:  # noqa: BLE001 - one bad problem must not lose 199
                skipped.append({"name": name, "level": int(level),
                                "reason": f"{type(exc).__name__}: {str(exc)[:300]}"})
                if log:
                    log(f"[materialize] SKIP l{level}/{name}: "
                        f"{type(exc).__name__}: {str(exc)[:160]}")
                continue
            tasks.append(record)
            if log:
                log(f"[materialize] l{level}/{name} -> {record['task_id']} "
                    f"({record['n_model_inputs']} inputs + {record['n_parameters']} params"
                    f"{', mutating' if record['mutates_input'] else ''})")

    manifest = {
        "schema": "kore.kernelbench-manifest/v1",
        "kernelbench_root": str(root),
        "revision": revision,
        "levels": [int(v) for v in levels],
        "entry_name": entry_name,
        "param_seed": int(param_seed),
        "gpu_target": gpu_target,
        "dtype": dtype,
        "baseline": BASELINE_KIND,
        "n_tasks": len(tasks),
        "n_skipped": len(skipped),
        "tasks": tasks,
        "skipped": skipped,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _materialize_one(path: Path, name: str, level: int, out: Path, *, revision: str,
                     gpu_target: str, dtype: str, snr_db: float, device: str,
                     entry_name: str, param_seed: int) -> dict:
    problem_source = path.read_text()
    problem_sha = hashlib.sha256(problem_source.encode("utf-8")).hexdigest()
    task_id = task_id_for(name, level)
    probe = _probe(problem_source, task_id, device, param_seed)

    task_dir = out / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "driver.py").write_text(_DRIVER_SHIM.format(level=level, name=name))
    (task_dir / "reference.py").write_text(_REFERENCE_SHIM.format(
        level=level, name=name, source=problem_source, key=task_id,
        n_model_inputs=probe["n_model_inputs"], mutates_input=probe["mutates_input"],
        entry_name=entry_name, dtype=dtype))
    (task_dir / "seed_reference.py").write_text(_seed_document(
        name, level, entry_name, probe["arg_labels"], probe["arg_values"],
        probe["output_desc"], problem_source))
    (task_dir / "task.yaml").write_text(_task_yaml(
        task_id=task_id, operation=operation_for(name), level=level, name=name,
        dtype=dtype, gpu_target=gpu_target, snr_db=snr_db,
        elems=probe["largest_input_elems"], revision=revision,
        problem_sha=problem_sha, entry_name=entry_name,
        n_model_inputs=probe["n_model_inputs"], n_parameters=probe["n_parameters"],
        mutates_input=probe["mutates_input"]))

    return {
        "task_id": task_id,
        "name": name,
        "level": int(level),
        "operation": operation_for(name),
        "dir": str(task_dir),
        "problem_sha256": problem_sha,
        "n_model_inputs": probe["n_model_inputs"],
        "n_parameters": probe["n_parameters"],
        "mutates_input": probe["mutates_input"],
        "largest_input_elems": probe["largest_input_elems"],
        "output": probe["output_desc"],
        "signature": f"{entry_name}(" + ", ".join(probe["arg_labels"]) + ")",
    }


# --------------------------------------------------------------------------- #
# Loading materialized tasks back for an eval run.
# --------------------------------------------------------------------------- #
def read_manifest(out_dir) -> dict:
    return json.loads((Path(out_dir) / "manifest.json").read_text())


def load_tasks(out_dir, *, levels: Optional[Sequence[int]] = None,
               task_ids: Optional[Sequence[str]] = None) -> list:
    """Load the materialized task directories as KORE :class:`Task` objects."""
    from kore.tasks.base import Task

    manifest = read_manifest(out_dir)
    wanted_levels = {int(v) for v in levels} if levels else None
    wanted_ids = {str(v) for v in task_ids} if task_ids else None
    tasks = []
    for record in manifest["tasks"]:
        if wanted_levels is not None and int(record["level"]) not in wanted_levels:
            continue
        if wanted_ids is not None and record["task_id"] not in wanted_ids:
            continue
        tasks.append(Task.from_dir(Path(record["dir"])))
    return tasks


def specs_for_report(out_dir, tasks: Optional[Sequence] = None) -> list:
    """Level-carrying :class:`KernelBenchSpec` stubs for the per-level breakdown.

    :func:`kore.eval.kernelbench_amd.to_kernelbench_report` segments its report by
    level and needs only ``task_id`` and ``level`` to do it; the executable parts
    of a spec live in the materialized task directory, not here.
    """
    from kore.eval.kernelbench_amd import KernelBenchSpec

    manifest = read_manifest(out_dir)
    keep = {getattr(t, "task_id", None) for t in tasks} if tasks is not None else None
    return [
        KernelBenchSpec(
            name=record["name"], level=int(record["level"]),
            family=FAMILY, operation=record["operation"],
            reference=None, make_inputs=None, input_shapes=[],
            dtype=manifest.get("dtype", DEFAULT_DTYPE),
            snr_threshold=DEFAULT_SNR_DB, entry_name=manifest.get("entry_name", ENTRY_NAME),
        )
        for record in manifest["tasks"]
        if keep is None or record["task_id"] in keep
    ]


__all__ = [
    "BASELINE_KIND",
    "ENTRY_NAME",
    "FAMILY",
    "PARAM_SEED",
    "SHAPE_NAME",
    "checkout_revision",
    "level_dir",
    "load_tasks",
    "make_reference",
    "materialize",
    "operation_for",
    "parse_shape",
    "problem_entities",
    "read_manifest",
    "specs_for_report",
    "task_id_for",
]
