"""Turn mined PyTorch modules into decontaminated, deduplicated KORE pool tasks.

The registry's ~1,052 trainable tasks are the binding constraint on datagen: many
episodes over few tasks produce near-duplicates.  Upstream corpora
(``GPUMODE/KernelBook`` and the Popcorn synthetic pipeline) publish tens of
thousands of standalone ``torch.nn.Module`` definitions in the KernelBench
format, and a module *is* a task specification -- it defines an oracle and an
input distribution.  Their Inductor-generated Triton is NVIDIA-targeted and is
deliberately not used; only the PyTorch reference side is ingested.

Admission has five gates, in this order, and every drop records which one fired:

``safety``        the module must import only from an allowlist and must not call
                  out of the process.  It is executed by the verifier, so
                  "probably harmless" is not a usable standard.
``classification``the module must land in exactly one canonical product family,
                  decided from the torch operators it calls rather than from its
                  class name (upstream names are ``Net``, ``Actor``, ``Block``).
``decontamination``family, literal held-out task id, and the full
                  ``kore.data.decontam`` content analysis against the held-out
                  KORE task sources *and* the KernelBench eval problems.
``execution``     the module must run, be deterministic under a fixed seed, and
                  still run at a shape large enough to be an optimization target
                  rather than a launch-overhead measurement.
``dedup``         structural (alpha-renamed AST) and semantic-graph fingerprints,
                  both within the new set and against every registry task source.

The order matters for cost, not for safety: the cheap static gates run before the
expensive execution probe, and every gate is independently sufficient to drop.
"""

from __future__ import annotations

import ast
import os
import signal
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from kore.tasks.external import (
    MAX_INPUT_ELEMENTS,
    MIN_PRIMARY_ELEMENTS,
    SCALE_LADDER,
    SNR_BY_DTYPE,
    TARGET_PRIMARY_ELEMENTS,
    ExternalTaskError,
    ExternalTaskSpec,
    InputSpec,
    build_inputs,
    classify_module,
    exec_module_source,
    make_identity,
    module_safety_reason,
    nondeterminism_reason,
    reserved_family_marker,
    split_decision_for_spec,
)

#: Wall-clock budget for one probed forward pass on CPU.  A module that cannot
#: complete a forward inside this is not a task, it is a hang risk in datagen.
FORWARD_TIMEOUT_SECONDS = 8

#: How many scales the probe may try before giving up.  Walking the whole ladder
#: costs a full forward per rung, and a module that fails its analytic target and
#: two large step-downs is not going to succeed at a third.
MAX_SCALE_ATTEMPTS = 3

#: Bit-exact repeat required of the oracle.  A module whose output changes
#: between two identical calls cannot grade a kernel at any tolerance.
DETERMINISM_TRIALS = 2

#: Weight-init seeds used to decide whether the oracle is a function of its
#: DECLARED INPUTS.  The task ships ``input_specs`` (the forward arguments) and
#: nothing else, so a candidate kernel receives only those tensors; if
#: re-constructing the module under a different seed changes the answer for
#: identical inputs, the module's parameters are a hidden variable no kernel can
#: read and the task is unanswerable.  Two seeds are enough: the property is
#: "does the output depend on the init RNG at all", not an estimate of how much.
HIDDEN_STATE_SEEDS = (1234, 60817)

#: Relative tolerance for that comparison.  Deliberately loose: the question is
#: whether the output MOVED with the weights, not whether it moved by a
#: numerically interesting amount, and a loose threshold errs toward admitting a
#: task rather than dropping a good one.
HIDDEN_STATE_RTOL = 1e-3
HIDDEN_STATE_ATOL = 1e-5


class ProbeTimeout(RuntimeError):
    """A probed module exceeded its wall-clock budget."""


@contextmanager
def time_limit(seconds: int):
    """Bound a block by wall clock, on platforms that have ``SIGALRM``.

    Used around module import and forward passes.  On a platform without
    ``SIGALRM`` this is a no-op, which is correct for the CPU test suite and
    irrelevant on the Linux compute nodes where the ingest actually runs.
    """
    if not hasattr(signal, "SIGALRM") or seconds <= 0:
        yield
        return

    def _raise(signum, frame):  # noqa: ARG001
        raise ProbeTimeout(f"exceeded {seconds}s")

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


# --------------------------------------------------------------------------- #
# Candidates and outcomes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Candidate:
    """One mined module, before any gate has run."""

    source_id: str
    row_id: str
    module_name: str
    entry_class: str
    module_source: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class Outcome:
    """What happened to one candidate."""

    candidate: Candidate
    accepted: bool
    reason: str = ""
    detail: str = ""
    spec: Optional[ExternalTaskSpec] = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    @property
    def task_id(self) -> str:
        return self.spec.task_id if self.spec is not None else ""


@dataclass
class MiningReport:
    """Counts and per-reason drops for one ingest run."""

    considered: int = 0
    accepted: int = 0
    drops: Counter = field(default_factory=Counter)
    families: Counter = field(default_factory=Counter)
    sources: Counter = field(default_factory=Counter)
    contamination_evidence: list[dict] = field(default_factory=list)

    def record(self, outcome: Outcome) -> None:
        self.considered += 1
        if outcome.accepted and outcome.spec is not None:
            self.accepted += 1
            self.families[outcome.spec.family] += 1
            self.sources[outcome.candidate.source_id] += 1
            return
        self.drops[outcome.reason or "unknown"] += 1
        if outcome.reason.startswith("decontam:") and len(
            self.contamination_evidence
        ) < 200:
            self.contamination_evidence.append({
                "row_id": outcome.candidate.row_id,
                "source_id": outcome.candidate.source_id,
                "reason": outcome.reason,
                **dict(outcome.evidence),
            })

    def as_dict(self) -> dict[str, Any]:
        return {
            "considered": self.considered,
            "accepted": self.accepted,
            "dropped": self.considered - self.accepted,
            "drops_by_reason": dict(sorted(self.drops.items())),
            "accepted_by_family": dict(sorted(self.families.items())),
            "accepted_by_source": dict(sorted(self.sources.items())),
            "contamination_evidence": self.contamination_evidence,
        }


# --------------------------------------------------------------------------- #
# Execution probe
# --------------------------------------------------------------------------- #
def _normalize_init(value: Any) -> tuple[list, dict]:
    """Normalize KernelBench's several ``get_init_inputs()`` return shapes."""
    if value is None:
        return [], {}
    if isinstance(value, dict):
        return [], dict(value)
    if isinstance(value, (list, tuple)):
        if len(value) == 2 and isinstance(value[1], dict):
            first = value[0]
            args = list(first) if isinstance(first, (list, tuple)) else [first]
            return args, dict(value[1])
        if len(value) == 1 and isinstance(value[0], (list, tuple)):
            return list(value[0]), {}
        return list(value), {}
    return [value], {}


def _json_round_trips(value: Any) -> bool:
    """Whether ``value`` survives the JSON round trip the pool index requires.

    Some mined modules take a tensor or a live object as a constructor argument.
    Such a task could not be persisted -- the pool index is JSONL -- and the
    argument cannot even cross a process boundary intact, because torch reduces
    a tensor through shared memory that the screening worker's address-space
    limit refuses. Both failures have the same fix: the constructor arguments
    must be plain data.
    """
    import json

    try:
        json.loads(json.dumps(value))
    except (TypeError, ValueError):
        return False
    return True


def describe_tensor(tensor) -> InputSpec:
    """Summarize an observed input tensor as a regenerable distribution.

    The upstream ``get_inputs()`` is a fixed-shape constructor; keeping it would
    freeze every task at whatever toy shape the source repository used.  What is
    kept instead is the shape, dtype, and enough of the distribution to
    regenerate statistically similar inputs at any leading dimension.
    """
    import torch

    shape = tuple(int(x) for x in tensor.shape)
    if tensor.dtype == torch.bool:
        return InputSpec(shape, "bool", "bool", mean=float(tensor.float().mean()))
    if not tensor.is_floating_point():
        return InputSpec(
            shape,
            str(tensor.dtype).replace("torch.", ""),
            "integer",
            low=float(tensor.min()),
            high=float(tensor.max()),
        )
    values = tensor.detach().float()
    low, high = float(values.min()), float(values.max())
    mean = float(values.mean())
    std = float(values.std()) if values.numel() > 1 else 1.0
    # A tensor that sits inside the unit interval came from ``rand``; anything
    # wider is better reproduced by a matched normal than by a clipped uniform.
    if 0.0 <= low and high <= 1.0:
        return InputSpec(shape, "float32", "uniform", low=low, high=max(high, low + 1e-6))
    return InputSpec(shape, "float32", "normal", mean=mean, std=max(std, 1e-6))


def _outputs_ok(value) -> bool:
    import torch

    items = value if isinstance(value, (tuple, list)) else (value,)
    if not items:
        return False
    saw_float = False
    for item in items:
        if not torch.is_tensor(item):
            return False
        if item.numel() == 0:
            return False
        if item.is_floating_point():
            saw_float = True
            if not torch.isfinite(item).all():
                return False
    return saw_float


def _same(a, b) -> bool:
    import torch

    xs = a if isinstance(a, (tuple, list)) else (a,)
    ys = b if isinstance(b, (tuple, list)) else (b,)
    if len(xs) != len(ys):
        return False
    return all(
        torch.is_tensor(x) and torch.is_tensor(y) and x.shape == y.shape
        and torch.equal(x, y)
        for x, y in zip(xs, ys)
    )


def _close(a, b) -> bool:
    """Approximate equality over a tensor or a tuple of tensors."""
    import torch

    xs = a if isinstance(a, (tuple, list)) else (a,)
    ys = b if isinstance(b, (tuple, list)) else (b,)
    if len(xs) != len(ys):
        return False
    for x, y in zip(xs, ys):
        if not (torch.is_tensor(x) and torch.is_tensor(y)):
            return False
        if x.shape != y.shape:
            return False
        if not x.is_floating_point():
            if not torch.equal(x, y):
                return False
            continue
        if not torch.allclose(
            x.float(), y.float(), rtol=HIDDEN_STATE_RTOL, atol=HIDDEN_STATE_ATOL,
            equal_nan=True,
        ):
            return False
    return True


def _elements(specs: Sequence[InputSpec], scale: int) -> int:
    total = 0
    for spec in specs:
        count = 1
        for dim in spec.sized(scale):
            count *= int(dim)
        total += count
    return total


def hidden_state_reason(
    namespace: Mapping[str, Any],
    entry_class: str,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    specs: Sequence[InputSpec],
    dtype: str,
    timeout: int = FORWARD_TIMEOUT_SECONDS,
) -> Optional[str]:
    """Why this module's oracle is not a function of its declared inputs, else None.

    A pool task ships ``input_specs`` -- the forward arguments -- and the driver
    hands exactly those tensors to the candidate kernel.  A mined ``nn.Module``,
    however, usually owns parameters: ``nn.Conv2d`` and ``nn.Linear`` weights are
    created inside ``__init__`` from the torch RNG and are NEVER passed to the
    kernel.  The oracle's answer therefore depends on a variable the candidate
    cannot read, and no honest kernel can match it at any tolerance.  The only
    channel to those weights is re-executing the torch module, which is precisely
    what :func:`kore.reward.reward.scan_for_hacks` rejects as delegation, so the
    task is unanswerable rather than merely hard.

    This is measured, not pattern-matched.  Re-constructing under a second seed
    and comparing outputs catches hand-written ``nn.Parameter``s, randomly
    initialized buffers and third-party layers that no list of layer names would
    contain -- and, just as importantly, ADMITS a module that happens to own
    parameters its forward never reads.  A static scan for parameter-owning layer
    names over the shipped 13,570-task pool flagged 9,860 tasks, but 809 of those
    did reach a correct kernel in the overnight campaign; the behavioural check
    keeps exactly that kind of task.
    """
    import torch

    cls = namespace.get(entry_class)
    if cls is None:
        return f"no class {entry_class!r} in module"

    outputs = []
    for seed in HIDDEN_STATE_SEEDS:
        with time_limit(timeout):
            torch.manual_seed(seed)
            model = cls(*args, **dict(kwargs))
        if not isinstance(model, torch.nn.Module):
            return "entry class is not an nn.Module"
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        inputs = build_inputs(specs, 1, device="cpu", seed=0, dtype=dtype)
        with time_limit(timeout), torch.no_grad():
            outputs.append(model(*inputs))

    if _close(outputs[0], outputs[1]):
        return None
    n_params = sum(p.numel() for p in model.parameters())
    return (
        "oracle depends on hidden module state: re-initializing the module under "
        f"a different weight seed changes the output for identical inputs "
        f"({n_params} parameter elements are not among the "
        f"{len(specs)} declared inputs), so no kernel that receives only the "
        "declared inputs can reproduce it"
    )


@dataclass
class ProbeResult:
    input_specs: tuple[InputSpec, ...]
    init_args: tuple[Any, ...]
    init_kwargs: Mapping[str, Any]
    primary_scale: int
    validation_scales: tuple[int, ...]
    primary_elements: int


def probe_module(
    source: str,
    entry_class: str,
    dtype: str = "fp32",
    timeout: int = FORWARD_TIMEOUT_SECONDS,
) -> ProbeResult:
    """Execute a mined module on CPU and derive its runnable task shape.

    Raises :class:`ExternalTaskError` with a specific reason on any failure, so
    the caller records *why* a module is not a task instead of silently skipping.
    """
    import torch

    namespace = exec_module_source(source)
    cls = namespace.get(entry_class)
    if cls is None:
        raise ExternalTaskError(f"no class {entry_class!r} in module")
    if not callable(namespace.get("get_inputs")):
        raise ExternalTaskError("module has no get_inputs()")

    args, kwargs = _normalize_init(
        namespace["get_init_inputs"]() if callable(namespace.get("get_init_inputs"))
        else None
    )
    if not _json_round_trips([args, kwargs]):
        raise ExternalTaskError("constructor arguments are not plain JSON data")
    with time_limit(timeout):
        model = cls(*args, **kwargs)
    if not isinstance(model, torch.nn.Module):
        raise ExternalTaskError("entry class is not an nn.Module")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    with time_limit(timeout):
        torch.manual_seed(0)
        observed = namespace["get_inputs"]()
    if not isinstance(observed, (list, tuple)) or not observed:
        raise ExternalTaskError("get_inputs() returned no tensors")
    if not all(torch.is_tensor(item) for item in observed):
        raise ExternalTaskError("get_inputs() returned a non-tensor")
    specs = tuple(describe_tensor(item) for item in observed)
    if any(not spec.shape for spec in specs):
        raise ExternalTaskError("get_inputs() returned a 0-d tensor")

    def forward_with(candidate_specs, scale: int):
        inputs = build_inputs(candidate_specs, scale, device="cpu", seed=0, dtype=dtype)
        with time_limit(timeout), torch.no_grad():
            return model(*inputs)

    def forward(scale: int):
        return forward_with(specs, scale)

    # Which inputs scale with the leading dimension is MEASURED, not assumed.
    # ``describe_tensor`` marks every input scalable, which is right for a module
    # whose arguments are all activations and wrong for one that also takes a
    # weight: scaling a conv weight's out-channels breaks the forward at every
    # scale above 1, and the module was then dropped as "no runnable scale" -- a
    # good task lost to a bad guess. Two hypotheses cover the real cases: all
    # inputs share the batch dimension, or only the first does.
    if len(specs) > 1:
        try:
            probe_all = forward_with(specs, 2)
        except Exception:  # noqa: BLE001 - the hypothesis under test
            probe_all = None
        if not _outputs_ok(probe_all):
            fixed = (specs[0],) + tuple(
                replace(spec, scalable=False) for spec in specs[1:]
            )
            try:
                if _outputs_ok(forward_with(fixed, 2)):
                    specs = fixed
            except Exception:  # noqa: BLE001 - neither hypothesis holds; leave as-is
                pass

    try:
        baseline = forward(1)
    except ProbeTimeout:
        raise ExternalTaskError("forward timed out at scale 1")
    except Exception as exc:  # noqa: BLE001
        raise ExternalTaskError(f"forward failed: {type(exc).__name__}: {exc}")
    if not _outputs_ok(baseline):
        raise ExternalTaskError("forward produced no finite float tensor")

    for _ in range(DETERMINISM_TRIALS - 1):
        try:
            repeat = forward(1)
        except Exception as exc:  # noqa: BLE001
            raise ExternalTaskError(f"repeat forward failed: {type(exc).__name__}")
        if not _same(baseline, repeat):
            raise ExternalTaskError("oracle is not deterministic under a fixed seed")

    # Determinism above proves the oracle repeats. It does NOT prove the oracle is
    # a function of the tensors the kernel will receive: a module holding conv or
    # linear weights repeats perfectly and is still unanswerable, because the
    # weights are not among ``specs``. That is a separate property and it is the
    # one that broke the campaign, so it is checked separately.
    try:
        hidden = hidden_state_reason(
            namespace, entry_class, args, kwargs, specs, dtype, timeout
        )
    except ProbeTimeout:
        raise ExternalTaskError("hidden-state check timed out")
    except Exception as exc:  # noqa: BLE001 - an upstream module may do anything
        raise ExternalTaskError(
            f"hidden-state check failed: {type(exc).__name__}: {exc}"
        )
    if hidden is not None:
        raise ExternalTaskError(hidden)

    # Pick the scale analytically and then verify it, rather than walking the
    # whole ladder: an 18k-module ingest cannot afford ten CPU forward passes per
    # candidate, and the element count is known without running anything.
    reachable = [
        scale for scale in SCALE_LADDER
        if _elements(specs, scale) <= max(TARGET_PRIMARY_ELEMENTS, _elements(specs, 1))
        and _elements(specs, scale) <= MAX_INPUT_ELEMENTS
    ]
    if not reachable:
        raise ExternalTaskError("no scale fits the input element budget")
    if _elements(specs, max(reachable)) < MIN_PRIMARY_ELEMENTS:
        raise ExternalTaskError(
            f"largest affordable shape is {_elements(specs, max(reachable))} "
            f"elements, below the {MIN_PRIMARY_ELEMENTS}-element "
            "optimization-target floor"
        )

    # Try the analytic target, then two sharp step-downs.  Stepping one rung at a
    # time would cost a full CPU forward per rung across tens of thousands of
    # modules for no extra information: a module that cannot take its target
    # shape has a structural reason, not a marginal one.
    ordered = sorted(reachable, reverse=True)
    attempts = [
        scale for scale in (ordered[0], ordered[0] // 8, ordered[0] // 64)
        if scale >= 1 and _elements(specs, scale) >= MIN_PRIMARY_ELEMENTS
    ][:MAX_SCALE_ATTEMPTS]

    primary = 0
    for scale in attempts:
        try:
            out = forward(scale)
        except ProbeTimeout:
            continue
        except Exception:  # noqa: BLE001 - a scale this module cannot take
            continue
        if _outputs_ok(out):
            primary = scale
            break
    if primary == 0:
        raise ExternalTaskError("no runnable scale above the optimization floor")

    # Validation shapes are strictly smaller instances of a shape that already
    # ran, so they are recorded without re-running the module.
    validations = tuple(
        scale for scale in (primary // 2, primary // 8) if scale >= 1
    )
    elements = _elements(specs, primary)
    return ProbeResult(
        input_specs=specs,
        init_args=tuple(args),
        init_kwargs=dict(kwargs),
        primary_scale=primary,
        validation_scales=validations,
        primary_elements=elements,
    )


# --------------------------------------------------------------------------- #
# Decontamination
# --------------------------------------------------------------------------- #
class Decontaminator:
    """The held-out gate, assembled from the repo's own decontamination module.

    Three independent checks, mirroring ``scripts/audit_decontamination.py``,
    because each misses what the others catch: family attribution, a literal
    held-out task id, and content analysis (exact / normalized-AST / semantic
    graph / directional containment / MinHash) against the held-out reference
    documents.  ``extra_references`` carries the evaluation benchmarks, so a
    mined module that *is* a KernelBench problem is refused even though it comes
    from a different upstream corpus.
    """

    def __init__(
        self,
        extra_references: Optional[Iterable[Mapping[str, Any]]] = None,
        ngram: int = 8,
    ) -> None:
        from kore.data.decontam import (
            build_heldout_ngrams,
            heldout_families,
            heldout_task_ids,
        )

        self.heldout_task_ids = frozenset(t for t in heldout_task_ids() if t)
        self.heldout_families = frozenset(f for f in heldout_families() if f)
        if not self.heldout_task_ids or not self.heldout_families:
            # A silently empty gate would admit everything.  The registry loader
            # already fails closed; this makes the same guarantee local.
            raise RuntimeError("held-out gate is empty; refusing to decontaminate")
        self.index = build_heldout_ngrams(ngram, extra_sources=list(extra_references or ()))
        self.n_references = len(self.index.references)

    def check(self, candidate: Candidate) -> Optional[tuple[str, dict]]:
        """Return ``(reason, evidence)`` when ``candidate`` is contaminated."""
        from kore.data.decontam import analyze_text_contamination, record_family

        text = candidate.module_source or ""
        marker = reserved_family_marker(text, candidate.module_name)
        if marker:
            return "decontam:reserved_family_marker", {"marker": marker}

        record = {
            "task_id": "",
            "operation": candidate.module_name,
            "source_metadata": {
                "source_id": candidate.source_id,
                "row_id": candidate.row_id,
                "path": candidate.metadata.get("repo_link", ""),
            },
        }
        family = record_family(record)
        if family in self.heldout_families:
            return "decontam:heldout_family", {"family": family}

        for task_id in self.heldout_task_ids:
            if task_id in text:
                return "decontam:heldout_task_id_literal", {"task_id": task_id}

        match = analyze_text_contamination(
            text,
            self.index,
            metadata=dict(record["source_metadata"]),
            family=family if family != "unclassified" else "",
        )
        if match is not None:
            return f"decontam:{match.reason}", {
                "reference_id": match.reference_id,
                "score": round(float(match.score), 6),
                **{k: v for k, v in dict(match.evidence).items() if k != "text"},
            }
        return None


def benchmark_references(problems: Iterable[Mapping[str, Any]]) -> list[dict]:
    """Wrap evaluation-benchmark problems as decontamination references."""
    out: list[dict] = []
    for index, problem in enumerate(problems):
        text = str(problem.get("code") or problem.get("text") or "")
        if not text.strip():
            continue
        name = str(problem.get("name") or problem.get("problem_id") or index)
        level = str(problem.get("level") or "")
        out.append({
            "reference_id": f"kernelbench:{level}:{name}",
            "text": text,
            "source_id": "kernelbench",
            "lineage_id": f"kernelbench:{level}",
        })
    return out


# --------------------------------------------------------------------------- #
# Dedup
# --------------------------------------------------------------------------- #
class _SubmoduleRenamer(ast.NodeTransformer):
    """Canonicalize ``self.<attr>`` to ``self.a0``, ``self.a1``, ... by first use."""

    def __init__(self) -> None:
        self._map: dict[str, str] = {}

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            node.attr = self._map.setdefault(node.attr, f"a{len(self._map)}")
        return node


def canonical_module_source(source: str) -> str:
    """Rewrite a module so submodule attribute names stop being distinguishing.

    ``kore.data.dedup`` deliberately preserves attribute names, because for a
    Triton kernel ``tl.dot`` and ``tl.load`` are different operations and must
    never collapse.  For an ``nn.Module`` the same rule misfires: ``self.fc`` and
    ``self.dense`` are the same computation under a different author's naming,
    and leaving them distinct would report copies of one module as many distinct
    tasks.  Only attributes hung off ``self`` are renamed, so the property that
    matters for kernels is untouched.
    """
    try:
        tree = ast.parse(source or "")
    except SyntaxError:
        return source or ""
    tree = _SubmoduleRenamer().visit(tree)
    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree)
    except Exception:  # noqa: BLE001 - unparse is best effort
        return source or ""


class Deduplicator:
    """Structural and semantic-graph dedup, seeded with the existing registry.

    Exact-string dedup is useless here: the same module reappears across
    repositories with a renamed variable or a reflowed comment.  Both
    fingerprints come from ``kore.data.dedup`` -- the alpha-renamed AST catches
    type-1/type-2 clones, and the semantic-graph hash additionally collapses
    modules that differ only by a tuned constant -- applied to the
    submodule-canonicalized source so an author's attribute naming does not read
    as a new task.
    """

    def __init__(self, seed_sources: Iterable[str] = ()) -> None:
        from kore.data.dedup import graph_fingerprint, structural_fingerprint

        self._structural = structural_fingerprint
        self._graph = graph_fingerprint
        self.structural_seen: dict[str, str] = {}
        self.graph_seen: dict[str, str] = {}
        self.n_registry_fingerprints = 0
        for source in seed_sources:
            canonical = canonical_module_source(source)
            key = self._structural(canonical)
            if key:
                self.structural_seen.setdefault(key, "registry")
                self.n_registry_fingerprints += 1
            graph = self._graph(canonical)
            if graph:
                self.graph_seen.setdefault(graph, "registry")

    def check(self, source: str, task_id: str) -> Optional[tuple[str, dict]]:
        """Return ``(reason, evidence)`` when ``source`` duplicates a known one."""
        canonical = canonical_module_source(source)
        key = self._structural(canonical)
        if key in self.structural_seen:
            return "dedup:structural", {"duplicate_of": self.structural_seen[key]}
        graph = self._graph(canonical)
        if graph and graph in self.graph_seen:
            return "dedup:semantic_graph", {"duplicate_of": self.graph_seen[graph]}
        if key:
            self.structural_seen[key] = task_id
        if graph:
            self.graph_seen[graph] = task_id
        return None


def registry_task_sources(limit_files: int = 4) -> list[str]:
    """Python source of every registry task, as dedup seed material."""
    from kore.tasks.registry import TASKS_DIR

    sources: list[str] = []
    for yml in sorted(Path(TASKS_DIR).glob("*/task.yaml")):
        for index, path in enumerate(sorted(yml.parent.glob("*.py"))):
            if index >= limit_files:
                break
            try:
                sources.append(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError):
                continue
    return sources


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #
def build_spec(
    candidate: Candidate,
    family: str,
    probe: ProbeResult,
    dtype: str,
) -> ExternalTaskSpec:
    operation, task_id = make_identity(
        candidate.source_id, candidate.module_name, candidate.module_source, dtype
    )
    from kore.data.dedup import content_hash

    provenance = {
        "source_id": candidate.source_id,
        "row_id": candidate.row_id,
        "module_name": candidate.module_name,
        "entry_class": candidate.entry_class,
        "content_hash": content_hash(candidate.module_source),
        "primary_elements": probe.primary_elements,
        **{
            key: candidate.metadata[key]
            for key in ("repo_name", "repo_link", "sha", "licenses", "dataset",
                        "revision", "stars")
            if key in candidate.metadata
        },
    }
    return ExternalTaskSpec(
        task_id=task_id,
        operation=operation,
        dtype=dtype,
        family=family,
        entry_class=candidate.entry_class,
        entry_name=operation,
        module_source=candidate.module_source,
        init_args=probe.init_args,
        init_kwargs=probe.init_kwargs,
        input_specs=probe.input_specs,
        primary_scale=probe.primary_scale,
        validation_scales=probe.validation_scales,
        snr_threshold=SNR_BY_DTYPE.get(dtype, 40.0),
        provenance=provenance,
    )


def screen_candidate(
    candidate: Candidate,
    dtype: str = "fp32",
    timeout: int = FORWARD_TIMEOUT_SECONDS,
) -> Outcome:
    """Run the static, classification, and execution gates for one candidate.

    Deliberately excludes decontamination and dedup: those need shared state
    (the held-out index, the fingerprints seen so far) and must run in one
    process, while this half is pure and parallelizes across workers.
    """
    reason = module_safety_reason(candidate.module_source)
    if reason is not None:
        return Outcome(candidate, False, "safety", reason)

    reason = nondeterminism_reason(candidate.module_source)
    if reason is not None:
        return Outcome(candidate, False, "nondeterministic_oracle", reason)

    family = classify_module(candidate.module_source, candidate.module_name)
    if family is None:
        return Outcome(candidate, False, "unclassifiable_operation")

    try:
        probe = probe_module(
            candidate.module_source, candidate.entry_class, dtype, timeout
        )
    except ExternalTaskError as exc:
        text = str(exc)
        bucket = (
            "shape_too_small" if "optimization-target floor" in text
            else "nondeterministic_oracle" if "not deterministic" in text
            else "hidden_state_oracle" if "hidden module state" in text
            else "no_runnable_scale" if "no runnable scale" in text
            else "unserializable_init_args" if "plain JSON data" in text
            else "execution_failed"
        )
        return Outcome(candidate, False, bucket, text)
    except ProbeTimeout as exc:
        return Outcome(candidate, False, "execution_timeout", str(exc))
    except Exception as exc:  # noqa: BLE001 - an upstream module may do anything
        return Outcome(candidate, False, "execution_failed",
                       f"{type(exc).__name__}: {exc}")

    spec = build_spec(candidate, family, probe, dtype)
    decision = split_decision_for_spec(spec)
    if decision.split != "train":
        return Outcome(candidate, False, f"split:{decision.reason}")
    return Outcome(candidate, True, spec=spec)


def admit(
    screened: Iterable[Outcome],
    decontaminator: "Decontaminator",
    deduplicator: "Deduplicator",
    report: Optional[MiningReport] = None,
) -> tuple[list[ExternalTaskSpec], MiningReport]:
    """Apply the shared-state gates to already-screened candidates."""
    report = report if report is not None else MiningReport()
    accepted: list[ExternalTaskSpec] = []
    seen_ids: set[str] = set()
    for outcome in screened:
        if not outcome.accepted or outcome.spec is None:
            report.record(outcome)
            continue
        contaminated = decontaminator.check(outcome.candidate)
        if contaminated is not None:
            reason, evidence = contaminated
            report.record(Outcome(outcome.candidate, False, reason, evidence=evidence))
            continue
        duplicate = deduplicator.check(
            outcome.candidate.module_source, outcome.spec.task_id
        )
        if duplicate is not None:
            reason, evidence = duplicate
            report.record(Outcome(outcome.candidate, False, reason, evidence=evidence))
            continue
        if outcome.spec.task_id in seen_ids:
            report.record(Outcome(outcome.candidate, False, "dedup:task_id"))
            continue
        seen_ids.add(outcome.spec.task_id)
        accepted.append(outcome.spec)
        report.record(outcome)
    return accepted, report


__all__ = [
    "Candidate",
    "DETERMINISM_TRIALS",
    "Decontaminator",
    "Deduplicator",
    "FORWARD_TIMEOUT_SECONDS",
    "HIDDEN_STATE_ATOL",
    "HIDDEN_STATE_RTOL",
    "HIDDEN_STATE_SEEDS",
    "MAX_SCALE_ATTEMPTS",
    "MiningReport",
    "Outcome",
    "ProbeResult",
    "ProbeTimeout",
    "admit",
    "benchmark_references",
    "build_spec",
    "canonical_module_source",
    "describe_tensor",
    "hidden_state_reason",
    "probe_module",
    "registry_task_sources",
    "screen_candidate",
    "time_limit",
]
