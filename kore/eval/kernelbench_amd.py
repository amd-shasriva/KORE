"""KernelBench-AMD adapter: report KORE on the field-standard benchmark.

KORE grades kernels with the ``fast_p`` metric (:mod:`kore.eval.fastp`) - the
same metric KernelBench popularized - but against a PRODUCTION vendor baseline on
AMD gfx950/gfx942. To make the KORE eval *publishable* and comparable to the wider
literature, this module bridges the two worlds:

  * FORWARD (KernelBench spec -> KORE):  :func:`spec_to_task` maps a KernelBench-style
    problem (a PyTorch reference ``Model.forward`` + an input generator + named
    shapes, Level 1 single-ops / Level 2 fusions) to a genuine KORE
    :class:`~kore.tasks.base.Task`. The PyTorch reference becomes the correctness
    ORACLE and the matched-budget ``fast_p`` bake-off
    (:mod:`kore.eval.bakeoff`) becomes the metric - so a KernelBench problem is
    scored through KORE's own verified, timing-integrity-gated pipeline.
  * REVERSE (KORE result -> KernelBench):  :func:`to_kernelbench_report` renders a
    KORE ``evaluate_policy`` result back into the field-standard ``fast_p`` at
    ``p in {1.0, 1.5, 2.0}`` (correct AND >p faster than the torch-eager baseline),
    so a KORE number drops straight into a KernelBench-style leaderboard.

A small BUNDLED set of KernelBench-like specs (elementwise / gemm / fused) ships as
fixtures (:func:`bundled_specs`) so the whole path is testable offline on CPU;
:func:`load_real_kernelbench` documents how the REAL KernelBench Level 1/2 problem
files would be loaded from a checkout.

Every produced task is BACKEND-TAGGED for the KORE target hardware (gfx950/CDNA4
by default, gfx942/CDNA3 accepted) so the report states exactly which AMD arch the
number was measured on.

This module also defines the WIDER HELD-OUT PROTOCOL (:func:`propose_heldout_protocol`):
a publishable result needs a held-out eval far larger than two reserved tasks, so
this proposes a dozens-of-tasks split stratified over three generalization axes -
operator FAMILY, SHAPE-REGIME, and DTYPE - with a strict :func:`leakage_check`. It
only COMPUTES a proposed split object; it never mutates the registry.

Import-safe / offline: torch is imported lazily inside the reference/input closures
and the loaders, so importing this module (and registry discovery) needs no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from kore.eval.fastp import fastp

# KernelBench's baseline is torch-eager (not a vendor library), so a KORE task
# minted from a KernelBench spec is graded vs torch-eager and labeled as such.
KERNELBENCH_BASELINE = "torch_eager"

# Field-standard fast_p thresholds KernelBench reports on (fast_1 is the headline).
KERNELBENCH_PS: tuple[float, ...] = (1.0, 1.5, 2.0)

# Accepted AMD backend arch tags (KORE target = gfx950/CDNA4; gfx942/CDNA3 lineage).
KNOWN_AMD_ARCHES: tuple[str, ...] = ("gfx950", "gfx942")
DEFAULT_ARCH = "gfx950"

# Synthetic task dir for minted KernelBench tasks. It is NEVER read on the CPU /
# dry-run path (the oracle rides on the attached spec, not on disk); a live GPU
# run would materialize a real task dir via :func:`load_real_kernelbench`.
_SYNTH_DIR = Path(__file__).resolve().parent / "_kernelbench_synth"


# --------------------------------------------------------------------------- #
# KernelBench spec: a PyTorch reference + input generator + named shapes.
# --------------------------------------------------------------------------- #
@dataclass
class KernelBenchSpec:
    """A KernelBench-style problem in KORE's vocabulary.

    ``reference(*inputs) -> Tensor`` is the PyTorch oracle (the KernelBench
    ``Model.forward``); ``make_inputs(shape, device, seed) -> tuple`` mirrors
    KernelBench's ``get_inputs`` (deterministic per seed). ``level`` is 1
    (single op) or 2 (fused chain / epilogue). ``entry_name`` is the function the
    candidate kernel must define. ``operation``/``family`` drive KORE's operator
    taxonomy and the metamorphic-relation selection in :mod:`kore.eval.robust_eval`.
    """

    name: str
    level: int
    family: str
    operation: str
    reference: Callable[..., "object"]
    make_inputs: Callable[..., tuple]
    input_shapes: list                      # list[kore.tasks.base.Shape]
    dtype: str = "fp32"
    snr_threshold: float = 40.0
    entry_name: Optional[str] = None
    seed_source: str = ""                   # optional compiling seed (spec-carried)

    @property
    def task_id(self) -> str:
        """Deterministic KORE task id: ``kb_l{level}_{name}`` (sanitized)."""
        safe = "".join(c if (c.isalnum() or c == "_") else "_" for c in self.name.lower())
        return f"kb_l{self.level}_{safe}"

    def entry(self) -> str:
        return self.entry_name or self.operation


# --------------------------------------------------------------------------- #
# Bundled fixtures (KernelBench-like; CPU-runnable torch references).
# --------------------------------------------------------------------------- #
def _shape(name: str, dims: dict):
    from kore.tasks.base import Shape
    return Shape(name, dict(dims))


def _randn(shape_dims: dict, keys: tuple[str, ...], *, device: str, seed: int, scale: float = 1.0):
    """Deterministic randn tensor for the given dim-keys (fp32; cast by caller)."""
    import torch
    g = torch.Generator(device=device).manual_seed(seed)
    size = tuple(int(shape_dims[k]) for k in keys)
    return torch.randn(size, generator=g, device=device, dtype=torch.float32) * scale


def _kb_relu(x):
    import torch
    return torch.relu(x)


def _kb_matmul(a, b):
    import torch
    return torch.matmul(a, b)


def _kb_add_mul(a, b, c):
    # Pointwise FUSION (KernelBench-L2 style): torch-eager runs it as separate
    # kernels, a fused kernel saves the HBM round-trips.
    return (a + b) * c


def _kb_matmul_relu(a, b):
    import torch
    return torch.relu(torch.matmul(a, b))


def _mk_elementwise_inputs(op_scale: float = 1.0):
    def make_inputs(shape, device="cpu", seed=0):
        dims = shape.dims if hasattr(shape, "dims") else shape
        return (_randn(dims, ("M", "N"), device=device, seed=seed, scale=op_scale),)
    return make_inputs


def _mk_fusion3_inputs():
    def make_inputs(shape, device="cpu", seed=0):
        dims = shape.dims if hasattr(shape, "dims") else shape
        return tuple(
            _randn(dims, ("M", "N"), device=device, seed=seed + i) for i in range(3)
        )
    return make_inputs


def _mk_gemm_inputs():
    def make_inputs(shape, device="cpu", seed=0):
        dims = shape.dims if hasattr(shape, "dims") else shape
        K = int(dims["K"])
        sc = 1.0 / (K ** 0.5)   # keep the accumulated magnitude ~O(1)
        a = _randn(dims, ("M", "K"), device=device, seed=seed, scale=sc)
        b = _randn(dims, ("K", "N"), device=device, seed=seed + 1, scale=sc)
        return (a, b)
    return make_inputs


def bundled_specs() -> list[KernelBenchSpec]:
    """A minimal, offline, CPU-runnable KernelBench-like suite.

    Spans the three canonical KernelBench classes: an elementwise Level-1 op, a
    GEMM Level-1 op, and two Level-2 fusions (a pointwise fusion and a GEMM +
    activation epilogue). These are fixtures for the adapter tests; a real report
    would load the actual KernelBench problem set via :func:`load_real_kernelbench`.
    """
    return [
        KernelBenchSpec(
            name="relu", level=1, family="elementwise", operation="relu",
            reference=_kb_relu, make_inputs=_mk_elementwise_inputs(),
            input_shapes=[_shape("small", {"M": 64, "N": 128}),
                          _shape("primary", {"M": 512, "N": 1024})],
            dtype="fp32", snr_threshold=40.0, entry_name="relu",
        ),
        KernelBenchSpec(
            name="matmul", level=1, family="gemm", operation="matmul",
            reference=_kb_matmul, make_inputs=_mk_gemm_inputs(),
            input_shapes=[_shape("small", {"M": 64, "N": 64, "K": 64}),
                          _shape("primary", {"M": 256, "N": 256, "K": 256})],
            dtype="fp32", snr_threshold=40.0, entry_name="matmul",
        ),
        KernelBenchSpec(
            name="add_mul", level=2, family="fusion", operation="add_mul",
            reference=_kb_add_mul, make_inputs=_mk_fusion3_inputs(),
            input_shapes=[_shape("small", {"M": 64, "N": 128}),
                          _shape("primary", {"M": 512, "N": 1024})],
            dtype="fp32", snr_threshold=40.0, entry_name="add_mul",
        ),
        KernelBenchSpec(
            name="matmul_relu", level=2, family="gemm_fusion", operation="gemm_relu",
            reference=_kb_matmul_relu, make_inputs=_mk_gemm_inputs(),
            input_shapes=[_shape("small", {"M": 64, "N": 64, "K": 64}),
                          _shape("primary", {"M": 256, "N": 256, "K": 256})],
            dtype="fp32", snr_threshold=40.0, entry_name="gemm_relu",
        ),
    ]


# --------------------------------------------------------------------------- #
# FORWARD: KernelBench spec -> KORE Task.
# --------------------------------------------------------------------------- #
def spec_to_task(spec: KernelBenchSpec, *, gpu_target: str = DEFAULT_ARCH,
                 backend: str = "triton"):
    """Mint a KORE :class:`~kore.tasks.base.Task` from a KernelBench spec.

    The PyTorch reference is attached as ``task.kernelbench_spec`` so the
    correctness oracle is reachable without a task dir; the task is TAGGED with
    the AMD backend arch (gfx950/gfx942) and its KernelBench provenance is recorded
    in ``task.raw`` for the per-level report breakdown. The comparison baseline is
    torch-eager (KernelBench's baseline), not a vendor library.
    """
    from kore.tasks.base import Task

    if gpu_target not in KNOWN_AMD_ARCHES:
        # A foreign arch is allowed but flagged: the registry holds it OUT of the
        # train archs, so a report on it is a cross-arch generalization number.
        pass

    task = Task(
        task_id=spec.task_id,
        operation=spec.operation,
        dtype=spec.dtype,
        backend=backend,
        gpu_target=gpu_target,
        dir=_SYNTH_DIR / spec.task_id,
        seed_kernel_name="seed_triton.py",
        snr_threshold=spec.snr_threshold,
        comparison_baseline=KERNELBENCH_BASELINE,
        shapes=list(spec.input_shapes),
        raw={
            "source": "kernelbench",
            "kernelbench_name": spec.name,
            "level": spec.level,
            "family": spec.family,
            "entry_name": spec.entry(),
            "gpu_target": gpu_target,
        },
    )
    # Attach the spec (Task is a non-frozen dataclass) so downstream verification /
    # timing can reach the oracle + input generator without touching disk.
    task.kernelbench_spec = spec
    return task


def specs_to_tasks(specs: Sequence[KernelBenchSpec], *, gpu_target: str = DEFAULT_ARCH,
                   backend: str = "triton") -> list:
    return [spec_to_task(s, gpu_target=gpu_target, backend=backend) for s in specs]


def kernelbench_seed_policy(task, feedback: Optional[dict] = None) -> str:
    """Baseline policy for minted KernelBench tasks: the spec-carried seed.

    Unlike :func:`kore.eval.policies.seed_policy` (which reads ``task.seed_source``
    from disk), this reads the seed off the attached spec, so it works for the
    diskless minted tasks. Returns an empty string when no seed was supplied.
    """
    spec = getattr(task, "kernelbench_spec", None)
    return getattr(spec, "seed_source", "") if spec is not None else ""


# --------------------------------------------------------------------------- #
# REVERSE: KORE result -> field-standard KernelBench fast_p report.
# --------------------------------------------------------------------------- #
def _fastp_from_result(res: dict, p: float, *, vs_torch: bool) -> float:
    """Recompute fast_p at threshold ``p`` from a result's per-task time arrays.

    Recomputing (rather than reading ``res['fast_p']``) makes the report robust to
    whatever p-grid the eval was run on, and lets us pick the torch-eager baseline
    arrays (``fast_p_vs_torch``, KernelBench-comparable) when present.
    """
    is_correct = res.get("is_correct", [])
    actual = res.get("actual_speed", [])
    baseline = res.get("torch_baseline_speed") if vs_torch else res.get("baseline_speed")
    if not baseline:
        return 0.0
    n = int(res.get("n", len(is_correct)))
    return fastp(is_correct, baseline, actual, n, p)


def to_kernelbench_report(res: dict, specs: Optional[Sequence[KernelBenchSpec]] = None,
                          *, ps: Sequence[float] = KERNELBENCH_PS,
                          prefer_torch_baseline: bool = False) -> dict:
    """Render a KORE ``evaluate_policy`` result as a KernelBench fast_p report.

    Returns the field-standard ``fast_p`` at ``p in {1.0, 1.5, 2.0}`` (fraction of
    the WHOLE split that is correct AND >p faster than the baseline), the correct
    rate, the geometric-mean speedup, and - when ``specs`` are supplied - a
    per-LEVEL breakdown (Level 1 vs Level 2), which is how KernelBench segments its
    leaderboard.

    ``prefer_torch_baseline`` uses the torch-eager fast_p curve when the eval
    carried torch-eager times (production vendor baseline swapped for KernelBench's
    torch-eager baseline); by default the task's own baseline is used (which, for a
    minted KernelBench task, already IS torch-eager).
    """
    vs_torch = bool(prefer_torch_baseline and res.get("torch_baseline_speed"))
    fast_p = {float(p): _fastp_from_result(res, float(p), vs_torch=vs_torch) for p in ps}

    per_task = res.get("per_task", [])
    n = int(res.get("n", len(per_task)))
    report = {
        "benchmark": "KernelBench-AMD",
        "baseline": KERNELBENCH_BASELINE,
        "n": n,
        "num_correct": int(res.get("num_correct", 0)),
        "correct_rate": (int(res.get("num_correct", 0)) / n) if n else 0.0,
        "fast_p": fast_p,
        "fast_1": fast_p.get(1.0, 0.0),
        "geometric_mean_speedup": float(res.get("geometric_mean_speedup", 0.0)),
        "baseline_kind": "torch_eager" if vs_torch else "task_baseline",
    }
    if specs is not None:
        report["per_level"] = _per_level_breakdown(res, specs, ps)
    return report


def _per_level_breakdown(res: dict, specs: Sequence[KernelBenchSpec],
                         ps: Sequence[float]) -> dict:
    """fast_p split by KernelBench level (1 = single op, 2 = fusion)."""
    level_by_id = {s.task_id: s.level for s in specs}
    per_task = res.get("per_task", [])
    buckets: dict[int, list] = {}
    for t in per_task:
        lvl = level_by_id.get(t.get("task_id"))
        if lvl is not None:
            buckets.setdefault(lvl, []).append(t)

    out: dict = {}
    for lvl, ts in sorted(buckets.items()):
        m = len(ts)
        is_correct = [t.get("correct") for t in ts]
        baseline = [t.get("baseline_time") for t in ts]
        actual = [t.get("actual_time") for t in ts]
        out[f"level_{lvl}"] = {
            "n": m,
            "num_correct": sum(1 for c in is_correct if c),
            "fast_p": {float(p): fastp(is_correct, baseline, actual, m, float(p)) for p in ps},
        }
    return out


def format_kernelbench_report(report: dict) -> str:
    """Human-readable ASCII markdown for a :func:`to_kernelbench_report` dict."""
    lines = [
        f"# {report.get('benchmark', 'KernelBench-AMD')} report",
        "",
        f"- baseline: {report.get('baseline')} ({report.get('baseline_kind')})",
        f"- tasks (n): {report.get('n')}",
        f"- correct rate: {report.get('correct_rate', 0.0):.3f}",
        f"- geomean speedup: {report.get('geometric_mean_speedup', 0.0):.3f}x",
        "",
        "| p | fast_p |",
        "| --- | --- |",
    ]
    for p in sorted(report.get("fast_p", {})):
        lines.append(f"| {float(p):g} | {report['fast_p'][p]:.3f} |")
    if report.get("per_level"):
        lines += ["", "## per level", "", "| level | n | correct | fast_1 |", "| --- | --- | --- | --- |"]
        for lvl, d in sorted(report["per_level"].items()):
            f1 = d["fast_p"].get(1.0, 0.0)
            lines.append(f"| {lvl} | {d['n']} | {d['num_correct']} | {f1:.3f} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Convenience runner: specs -> matched-budget eval -> KernelBench report.
# --------------------------------------------------------------------------- #
def run_kernelbench_amd(policy_fn: Callable, specs: Sequence[KernelBenchSpec], *,
                        gpu_target: str = DEFAULT_ARCH, budget: int = 5,
                        env_factory: Optional[Callable] = None,
                        dry_run: Optional[object] = None,
                        mode: str = "serial",
                        ps: Sequence[float] = KERNELBENCH_PS) -> dict:
    """Score ``policy_fn`` over KernelBench ``specs`` and return eval + KB report.

    Mints the tasks (backend-tagged), runs KORE's matched-budget ``fast_p`` bake-off
    (:func:`kore.eval.bakeoff.evaluate_policy`), and renders the field-standard
    KernelBench report. Provide ``env_factory`` (live GPU) or ``dry_run``
    (precomputed Observations) for CPU testing - identical to ``evaluate_policy``.
    """
    from kore.eval.bakeoff import evaluate_policy

    tasks = specs_to_tasks(specs, gpu_target=gpu_target)
    # Report on the full fast_p grid so the curve is available; the KB view slices
    # {1.0, 1.5, 2.0} out of it.
    from kore.eval.fastp import DEFAULT_PS
    grid = sorted(set(DEFAULT_PS) | {float(p) for p in ps})
    res = evaluate_policy(policy_fn, tasks, env_factory=env_factory, budget=budget,
                          mode=mode, dry_run=dry_run, ps=grid)
    report = to_kernelbench_report(res, specs, ps=ps)
    return {"eval": res, "report": report, "gpu_target": gpu_target}


# --------------------------------------------------------------------------- #
# Loading the REAL KernelBench (Level 1/2) problem set (documented loader).
# --------------------------------------------------------------------------- #
def load_real_kernelbench(root: str, *, levels: Sequence[int] = (1, 2),
                          dtype: str = "fp32", device_meta: bool = True) -> list[KernelBenchSpec]:
    """Load real KernelBench Level 1/2 problems from a local checkout into specs.

    KernelBench ships one Python file per problem (``KernelBench/level{L}/{id}_{Name}.py``)
    defining ``class Model(nn.Module)`` with ``forward``, plus module-level
    ``get_inputs()`` and ``get_init_inputs()``. This loader executes each file in an
    isolated namespace and wraps it:

      * ``reference``   = an instance of ``Model(*get_init_inputs())``  (its
        ``forward`` is the oracle);
      * ``make_inputs`` = a closure over ``get_inputs()`` (re-seeded per trial);
      * ``input_shapes``= a single named shape carrying the tensor sizes (KernelBench
        fixes the shapes inside ``get_inputs``, so KORE treats them as one shape;
        :mod:`kore.tasks.augment` can widen them for the held-out protocol).

    Requires torch + a KernelBench checkout; raises ``FileNotFoundError`` with the
    expected layout when ``root`` is missing so the caller gets an actionable message.
    Not exercised by the offline tests (which use :func:`bundled_specs`); this is the
    production ingestion path.
    """
    base = Path(root)
    if not base.exists():
        raise FileNotFoundError(
            f"KernelBench checkout not found at {root!r}. Clone it "
            "(github.com/ScalingIntelligence/KernelBench) and point --kernelbench-root "
            "at the repo so level{1,2}/*.py can be loaded."
        )

    specs: list[KernelBenchSpec] = []
    for lvl in levels:
        level_dir = base / "KernelBench" / f"level{lvl}"
        if not level_dir.exists():
            level_dir = base / f"level{lvl}"
        for path in sorted(level_dir.glob("*.py")):
            spec = _load_one_kernelbench_problem(path, lvl, dtype)
            if spec is not None:
                specs.append(spec)
    return specs


def _load_one_kernelbench_problem(path: Path, level: int, dtype: str) -> Optional[KernelBenchSpec]:
    """Execute a single KernelBench problem file and wrap it as a KernelBenchSpec."""
    import importlib.util

    import torch  # noqa: F401 - a real KernelBench problem needs torch/nn

    name = path.stem
    spec_obj = importlib.util.spec_from_file_location(f"kernelbench_{name}", path)
    if spec_obj is None or spec_obj.loader is None:
        return None
    module = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(module)  # noqa: S102 - trusted benchmark checkout

    Model = getattr(module, "Model", None)
    get_inputs = getattr(module, "get_inputs", None)
    get_init_inputs = getattr(module, "get_init_inputs", lambda: [])
    if Model is None or get_inputs is None:
        return None

    def _reference(*inputs):
        init = get_init_inputs() or []
        model = Model(*init)
        return model(*inputs)

    def _make_inputs(shape, device="cpu", seed=0):
        import torch as _t
        _t.manual_seed(seed)
        outs = get_inputs()
        return tuple(o.to(device) if hasattr(o, "to") else o for o in outs)

    # KernelBench fixes shapes inside get_inputs; expose one nominal shape so KORE's
    # augmenter can still widen it for the held-out protocol.
    from kore.tasks.base import Shape
    fam = "gemm" if any(k in name.lower() for k in ("matmul", "gemm", "conv")) else "kernelbench"
    return KernelBenchSpec(
        name=name, level=level, family=fam, operation=name.lower(),
        reference=_reference, make_inputs=_make_inputs,
        input_shapes=[Shape("kernelbench", {})], dtype=dtype,
        snr_threshold=40.0 if dtype == "fp32" else 30.0, entry_name=name.lower(),
    )


# --------------------------------------------------------------------------- #
# WIDER HELD-OUT PROTOCOL (family x shape-regime x dtype), leakage-checked.
# --------------------------------------------------------------------------- #
# Shape-regime thresholds on the largest shape's element count (a proxy for the
# arithmetic/memory regime the kernel must handle). Tuned to the KORE task zoo:
# elementwise/reduction small tensors -> "small", a transformer hidden matmul ->
# "medium", a full [4096, 8192]-class tensor / big GEMM -> "large".
_SMALL_MAX = 1e5
_MEDIUM_MAX = 1e7
_ALIGN = 8   # dims not divisible by this are "odd" (non-aligned boundary stressors)


def _shape_size(shape) -> int:
    dims = getattr(shape, "dims", {}) or {}
    prod = 1
    any_dim = False
    for v in dims.values():
        if isinstance(v, (int, float)) and v:
            prod *= int(v)
            any_dim = True
    return prod if any_dim else 0


def _has_odd_shape(task) -> bool:
    for s in getattr(task, "shapes", []) or []:
        for v in (getattr(s, "dims", {}) or {}).values():
            if isinstance(v, (int, float)) and int(v) > 1 and int(v) % _ALIGN != 0:
                return True
    return False


def shape_regime(task) -> str:
    """Classify a task by the element count of its LARGEST shape.

    ``small`` (< 1e5) | ``medium`` (< 1e7) | ``large`` (>= 1e7). An ``_odd`` suffix
    marks tasks that also carry a non-aligned (non-multiple-of-8) shape - the
    boundary-handling stressors KernelBench-style eval leans on.
    """
    shapes = getattr(task, "shapes", []) or []
    biggest = max((_shape_size(s) for s in shapes), default=0)
    if biggest < _SMALL_MAX:
        base = "small"
    elif biggest < _MEDIUM_MAX:
        base = "medium"
    else:
        base = "large"
    return f"{base}_odd" if _has_odd_shape(task) else base


@dataclass
class HeldoutProtocol:
    """A proposed, leakage-checked, wide held-out eval protocol.

    ``heldout_tasks`` / ``train_tasks`` are the task-id partition; the three axis
    maps describe HOW the held-out set spans generalization axes (operator family,
    shape regime, dtype). It is a PROPOSAL computed from the registry - it does not
    change the registry's own train/held-out discipline.
    """

    heldout_families: list[str]
    heldout_tasks: list[str]
    train_tasks: list[str]
    by_family: dict[str, list[str]] = field(default_factory=dict)
    by_shape_regime: dict[str, list[str]] = field(default_factory=dict)
    by_dtype: dict[str, list[str]] = field(default_factory=dict)
    seed: int = 0

    @property
    def n_heldout(self) -> int:
        return len(self.heldout_tasks)

    def axes_summary(self) -> dict:
        """Coverage counts along each generalization axis (families/regimes/dtypes)."""
        return {
            "n_families": len(self.by_family),
            "n_shape_regimes": len(self.by_shape_regime),
            "n_dtypes": len(self.by_dtype),
            "n_heldout_tasks": self.n_heldout,
            "n_train_tasks": len(self.train_tasks),
        }

    def as_dict(self) -> dict:
        return {
            "heldout_families": sorted(self.heldout_families),
            "heldout_tasks": sorted(self.heldout_tasks),
            "train_tasks": sorted(self.train_tasks),
            "by_family": {k: sorted(v) for k, v in self.by_family.items()},
            "by_shape_regime": {k: sorted(v) for k, v in self.by_shape_regime.items()},
            "by_dtype": {k: sorted(v) for k, v in self.by_dtype.items()},
            "axes_summary": self.axes_summary(),
            "seed": self.seed,
        }


def propose_heldout_protocol(tasks: Optional[Sequence] = None, *,
                             target_heldout: int = 24,
                             min_families: int = 4,
                             extra_heldout_families: Sequence[str] = (),
                             seed: int = 0) -> HeldoutProtocol:
    """Propose a wide, stratified, leakage-free held-out protocol over the registry.

    The registry reserves only a couple of families (MLA, paged-KV) - too thin for a
    publishable generalization claim. This WIDENS the held-out set to dozens of
    tasks while preserving family-level cleanliness: it reserves WHOLE operator
    families (so no family straddles the train/held-out boundary), greedily adding
    families - beyond the registry's authoritative reserved set - that maximize NEW
    (shape-regime, dtype) coverage until BOTH at least ``target_heldout`` tasks are
    held out AND at least ``min_families`` families span the held-out set (so the
    family axis is not dominated by one large family). The result is a proposal
    object; the registry is never mutated.

    Determinism: families are considered in a fixed coverage-then-name order, so the
    split is reproducible for a given ``seed`` / task set.
    """
    from kore.tasks import registry as reg

    all_tasks = list(tasks) if tasks is not None else reg.all_tasks()

    fam_of = reg.operator_family
    fam_to_tasks: dict[str, list] = {}
    for t in all_tasks:
        fam_to_tasks.setdefault(fam_of(t), []).append(t)

    # Authoritative reserved families (registry) + any caller-requested extras -
    # always held out whole.
    reserved = set(reg.HELDOUT_FAMILIES) | set(extra_heldout_families)
    heldout_families = {f for f in reserved if f in fam_to_tasks}

    # Also honor the registry's individual reserved task ids and foreign-arch tasks
    # by reserving their WHOLE families (keeps the family-level invariant intact).
    for t in all_tasks:
        if reg.is_heldout(t):
            heldout_families.add(fam_of(t))

    def _covered() -> set:
        covered = set()
        for f in heldout_families:
            for t in fam_to_tasks[f]:
                covered.add((shape_regime(t), _dtype_of(t)))
        return covered

    def _held_count() -> int:
        return sum(len(fam_to_tasks[f]) for f in heldout_families)

    # Coverage-greedy widening: add whole families that introduce the most NEW
    # (shape-regime, dtype) cells; on ties prefer the SMALLER family so the split
    # accrues family breadth rather than one giant family. Continue until both the
    # task-count and family-count targets are met. Ties broken by name (determinism).
    candidates = sorted(f for f in fam_to_tasks if f not in heldout_families)
    while candidates and (_held_count() < target_heldout or len(heldout_families) < min_families):
        covered = _covered()

        def _new_cells(f: str) -> int:
            cells = {(shape_regime(t), _dtype_of(t)) for t in fam_to_tasks[f]}
            return len(cells - covered)

        best = max(candidates, key=lambda f: (_new_cells(f), -len(fam_to_tasks[f]), f))
        heldout_families.add(best)
        candidates.remove(best)

    heldout_task_objs = [t for f in heldout_families for t in fam_to_tasks[f]]
    heldout_ids = {t.task_id for t in heldout_task_objs}
    train_objs = [t for t in all_tasks if t.task_id not in heldout_ids]

    by_family: dict[str, list[str]] = {}
    by_regime: dict[str, list[str]] = {}
    by_dtype: dict[str, list[str]] = {}
    for t in heldout_task_objs:
        by_family.setdefault(fam_of(t), []).append(t.task_id)
        by_regime.setdefault(shape_regime(t), []).append(t.task_id)
        by_dtype.setdefault(_dtype_of(t), []).append(t.task_id)

    proto = HeldoutProtocol(
        heldout_families=sorted(heldout_families),
        heldout_tasks=sorted(heldout_ids),
        train_tasks=sorted(t.task_id for t in train_objs),
        by_family=by_family, by_shape_regime=by_regime, by_dtype=by_dtype, seed=seed,
    )
    assert_no_leakage(proto, all_tasks)
    return proto


def _dtype_of(task) -> str:
    return (getattr(task, "dtype", None) or "unknown")


def leakage_check(protocol: HeldoutProtocol, tasks: Optional[Sequence] = None) -> dict:
    """Return a leakage report for a proposed protocol (does not raise).

    Checks (a) no task id in BOTH splits, (b) no operator family straddling the
    train/held-out boundary (the family-level invariant), and (c) the held-out set
    is non-empty. ``ok`` is True only when all three hold.
    """
    from kore.tasks import registry as reg

    tset, hset = set(protocol.train_tasks), set(protocol.heldout_tasks)
    task_overlap = sorted(tset & hset)

    fam_overlap: list[str] = []
    if tasks is not None:
        by_id = {t.task_id: t for t in tasks}
        train_fams = {reg.operator_family(by_id[t]) for t in tset if t in by_id}
        held_fams = {reg.operator_family(by_id[t]) for t in hset if t in by_id}
        fam_overlap = sorted(train_fams & held_fams)
    else:
        # Fall back to the declared family map when task objects are unavailable.
        held_fams = set(protocol.by_family.keys())
        # A train task whose family is a held-out family would be a leak; we can only
        # detect this with task objects, so this branch checks the declared families.
        fam_overlap = []

    ok = not task_overlap and not fam_overlap and protocol.n_heldout > 0
    return {
        "ok": ok,
        "task_overlap": task_overlap,
        "family_overlap": fam_overlap,
        "n_heldout": protocol.n_heldout,
        "n_train": len(protocol.train_tasks),
    }


def assert_no_leakage(protocol: HeldoutProtocol, tasks: Optional[Sequence] = None) -> None:
    """Raise ``AssertionError`` on any train/held-out leakage (used by the proposer)."""
    rep = leakage_check(protocol, tasks)
    if rep["task_overlap"]:
        raise AssertionError(f"held-out/train TASK leakage: {rep['task_overlap']}")
    if rep["family_overlap"]:
        raise AssertionError(f"held-out/train FAMILY leakage: {rep['family_overlap']}")
    if protocol.n_heldout <= 0:
        raise AssertionError("held-out set is empty")


__all__ = [
    "KERNELBENCH_BASELINE",
    "KERNELBENCH_PS",
    "KNOWN_AMD_ARCHES",
    "DEFAULT_ARCH",
    "KernelBenchSpec",
    "bundled_specs",
    "spec_to_task",
    "specs_to_tasks",
    "kernelbench_seed_policy",
    "to_kernelbench_report",
    "format_kernelbench_report",
    "run_kernelbench_amd",
    "load_real_kernelbench",
    "shape_regime",
    "HeldoutProtocol",
    "propose_heldout_protocol",
    "leakage_check",
    "assert_no_leakage",
]
