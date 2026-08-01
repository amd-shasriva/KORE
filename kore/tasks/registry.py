"""Strict task discovery and the authoritative immutable train/eval split."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional

from kore.tasks.base import Task
from kore.tasks import taxonomy, verification

TASKS_DIR = Path(__file__).resolve().parent

# Compatibility exports.  Definitions live in taxonomy.py and are not
# environment-overridable, so a manifest means the same thing in every process.
TRAIN_ARCH = taxonomy.PRIMARY_TRAIN_ARCHITECTURE
TRAIN_ARCHS = taxonomy.TRAIN_ARCHITECTURES
TRAIN_DTYPES = taxonomy.TRAIN_DTYPES
HELDOUT_FAMILIES: tuple[str, ...] = tuple(sorted(taxonomy.WHOLE_FAMILY_HOLDOUTS))
NEAR_GENERALIZATION_TASKS = taxonomy.NEAR_GENERALIZATION_TASK_IDS
HELDOUT_TASKS = frozenset(
    set(NEAR_GENERALIZATION_TASKS)
    | {"mla_decode_bf16", "paged_attn_decode_bf16"}
)
# Held out AND already seen by the pretrained model: still never trainable, and
# additionally not scoreable as zero-shot generalization.
CONTAMINATED_TASKS = taxonomy.CONTAMINATED_TASK_IDS
TAXONOMY_VERSION = taxonomy.TAXONOMY_VERSION


class TaskRegistryError(RuntimeError):
    """The on-disk task registry is malformed or ambiguous."""


class SplitManifestError(ValueError):
    """A split manifest violates the current split contract."""


class StaleSplitManifestError(SplitManifestError):
    """A manifest was authored under a different taxonomy/task inventory."""


class ContaminatedGeneralizationError(ValueError):
    """A generalization claim included a task whose source leaked into training."""


@dataclass(frozen=True)
class SplitManifest:
    """Immutable IDs and lineage roots under one taxonomy digest."""

    taxonomy_version: str
    taxonomy_digest: str
    train_ids: tuple[str, ...]
    eval_ids: tuple[str, ...]
    train_provenance_roots: tuple[tuple[str, str], ...]
    eval_provenance_roots: tuple[tuple[str, str], ...]
    whole_family_holdouts: tuple[str, ...]
    near_generalization_ids: tuple[str, ...]
    contaminated_eval_ids: tuple[str, ...] = ()

    @property
    def generalization_eval_ids(self) -> tuple[str, ...]:
        """Eval IDs that may back a zero-shot claim (held out and never seen)."""
        contaminated = set(self.contaminated_eval_ids)
        return tuple(
            task_id for task_id in self.eval_ids if task_id not in contaminated
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "taxonomy": {
                "version": self.taxonomy_version,
                "digest": self.taxonomy_digest,
            },
            "train_ids": list(self.train_ids),
            "eval_ids": list(self.eval_ids),
            "provenance_roots": {
                "train": dict(self.train_provenance_roots),
                "eval": dict(self.eval_provenance_roots),
            },
            "policy": {
                "whole_family_holdouts": list(self.whole_family_holdouts),
                "near_generalization_ids": list(self.near_generalization_ids),
                "train_architectures": sorted(TRAIN_ARCHS),
                "train_dtypes": sorted(TRAIN_DTYPES),
                # Serialized so a manifest authored before the leak was found
                # fails validation instead of silently scoring 45 eval tasks.
                "contaminated_task_ids": sorted(CONTAMINATED_TASKS),
                "contaminated_eval_ids": list(self.contaminated_eval_ids),
                "generalization_eval_ids": list(self.generalization_eval_ids),
            },
        }


def operator_family(task: Task) -> str:
    """Canonical product-family leaf (back-compatible public name)."""
    family = taxonomy.product_family_for_task(task, strict=True)
    assert family is not None
    return family


def analysis_family(task: Task) -> str:
    """Canonical reporting/LOFO parent for ``task``."""
    return taxonomy.analysis_family_for_task(task, strict=True)


def split_decision(task: Task) -> taxonomy.SplitDecision:
    return taxonomy.split_decision(task, strict=True)


def is_heldout(task: Task) -> bool:
    return split_decision(task).heldout


def is_contaminated(task: Task) -> bool:
    """True when this task's source reached the pretraining corpus."""
    return split_decision(task).contaminated


def contamination_record(task: Task) -> Optional[taxonomy.ContaminationRecord]:
    return split_decision(task).contamination


@lru_cache(maxsize=1)
def _discover() -> dict[str, Task]:
    tasks: dict[str, Task] = {}
    errors: list[str] = []
    for yml in sorted(TASKS_DIR.glob("*/task.yaml")):
        try:
            t = Task.from_dir(yml.parent)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{yml.parent.name}: {e}")
            continue
        if t.task_id in tasks:
            errors.append(
                f"duplicate task_id {t.task_id!r} in "
                f"{tasks[t.task_id].dir} and {t.dir}"
            )
            continue
        tasks[t.task_id] = t
    if not tasks:
        errors.append(f"no task.yaml files discovered under {TASKS_DIR}")
    if not errors:
        try:
            taxonomy.validate_task_assignments(tasks.values())
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
        # A contamination exclusion naming a task the registry no longer has is
        # unenforceable, so it aborts the registry like any other malformed entry.
        missing = taxonomy.validate_contamination_coverage(tasks)
        if missing:
            errors.append(
                "contaminated tasks are absent from the registry: " + ", ".join(missing)
            )
    if errors:
        raise TaskRegistryError(
            "task registry validation failed:\n  - " + "\n  - ".join(errors)
        )
    return tasks


def all_tasks() -> list[Task]:
    return list(_discover().values())


def task_ids() -> list[str]:
    return list(_discover().keys())


def get_task(task_id: str) -> Task:
    tasks = _discover()
    if task_id not in tasks:
        raise KeyError(f"unknown task '{task_id}'; known: {sorted(tasks)}")
    return tasks[task_id]


def find_task(task_id: str) -> Optional[Task]:
    """Return a registered task or ``None`` without weakening registry validation."""
    return _discover().get(task_id)


def operation_family_map() -> Mapping[str, str]:
    """Exact live operation -> product-family assignments."""
    return taxonomy.validate_task_assignments(all_tasks())


def taxonomy_digest() -> str:
    """Digest of taxonomy rules plus all live task assignments."""
    return taxonomy.taxonomy_digest(all_tasks())


def taxonomy_description() -> dict[str, Any]:
    """Machine-derived registry/taxonomy summary."""
    return taxonomy.describe(all_tasks())


def _record_mapping(record: Any) -> Mapping[str, Any]:
    if isinstance(record, Mapping):
        return record
    values = getattr(record, "__dict__", None)
    return values if isinstance(values, Mapping) else {}


def record_split_decision(
    record: Any,
    extra_eval_ids: Iterable[str] = (),
) -> taxonomy.SplitDecision:
    """Classify a persisted record with the same authority as registry tasks.

    Unknown operations, foreign architectures/dtypes, and malformed identities are
    eval-only.  Registered IDs use their exact metadata assignment, so record text
    cannot relabel a held-out task into train.
    """

    data = _record_mapping(record)
    provenance = data.get("_provenance") or {}
    if not isinstance(provenance, Mapping):
        provenance = {}
    task_id = str(data.get("task_id") or provenance.get("task_id") or "").strip()
    if not task_id:
        # A record with no stable identity cannot safely enter train.
        task_id = "__unidentified_record__"
    registered = _discover().get(task_id)
    operation = str(
        (getattr(registered, "operation", None) if registered is not None else None)
        or data.get("operation")
        or provenance.get("op")
        or ""
    )
    product = (
        operator_family(registered)
        if registered is not None
        else operation_family_map().get(operation.lower())
    )
    architecture = (
        data.get("arch")
        or data.get("gpu")
        or provenance.get("arch")
        or (registered.gpu_target if registered is not None else None)
    )
    dtype = (
        data.get("dtype")
        or provenance.get("dtype")
        or (registered.dtype if registered is not None else None)
    )
    root = (
        data.get("provenance_root")
        or provenance.get("root")
        or (registered.provenance_root if registered is not None else task_id)
    )
    decision = taxonomy.split_decision_for_identity(
        task_id=task_id,
        operation=operation,
        product_family=product,
        architecture=architecture,
        dtype=dtype,
        provenance_root=root,
    )
    if task_id in set(extra_eval_ids) and not decision.heldout:
        decision = replace(decision, split="eval", reason="manifest_eval_id")
    return decision


def product_family_for_record(record: Any) -> str:
    return record_split_decision(record).product_family or "unclassified"


def is_heldout_record(record: Any, extra_eval_ids: Iterable[str] = ()) -> bool:
    return record_split_decision(record, extra_eval_ids).heldout


# --------------------------------------------------------------------------- #
# Train / held-out generalization split
# --------------------------------------------------------------------------- #
def heldout_tasks() -> list[Task]:
    """Tasks reserved for eval by family, task probe, arch, or dtype."""
    return [t for t in all_tasks() if is_heldout(t)]


def train_tasks() -> list[Task]:
    """Tasks available for training data-generation (complement of held-out)."""
    return [t for t in all_tasks() if not is_heldout(t)]


def heldout_families() -> set[str]:
    """Whole product leaves held out (not families containing task-level probes)."""
    return set(taxonomy.WHOLE_FAMILY_HOLDOUTS)


# --------------------------------------------------------------------------- #
# Zero-shot generalization scope (held-out MINUS contaminated)
# --------------------------------------------------------------------------- #
def contaminated_tasks() -> list[Task]:
    """Held-out tasks whose source already reached the pretraining corpus."""
    return [task for task in all_tasks() if is_contaminated(task)]


def generalization_tasks() -> list[Task]:
    """Held-out tasks that can still support a zero-shot generalization claim.

    This is the correct scope for any zero-shot number.  ``heldout_tasks()``
    deliberately keeps returning the full reservation, because a contaminated task
    must stay untrainable and must stay in the decontamination gate.
    """
    return [task for task in all_tasks() if split_decision(task).generalization_eligible]


def generalization_eval_ids() -> tuple[str, ...]:
    return tuple(task.task_id for task in generalization_tasks())


def generalization_exclusions(
    task_ids: Optional[Iterable[str]] = None,
) -> Mapping[str, str]:
    """``{task_id: contamination_reason}`` for IDs barred from a zero-shot claim.

    Defaults to the whole held-out reservation.  Unregistered IDs are reported as
    unknown rather than silently admitted, so a claim can never be built over an
    identity this registry cannot adjudicate.
    """

    if task_ids is None:
        candidates = [task.task_id for task in heldout_tasks()]
    else:
        candidates = [str(task_id or "").strip() for task_id in task_ids]
    out: dict[str, str] = {}
    for task_id in candidates:
        if not task_id:
            raise ContaminatedGeneralizationError(
                "generalization scope contains an empty task id"
            )
        task = find_task(task_id)
        if task is None:
            out[task_id] = "unregistered_task"
            continue
        record = contamination_record(task)
        if record is not None:
            out[task_id] = record.reason
    return MappingProxyType(dict(sorted(out.items())))


def assert_generalization_scope(task_ids: Iterable[str]) -> tuple[str, ...]:
    """Fail closed on a zero-shot claim built over contaminated/unknown tasks."""

    requested = tuple(str(task_id or "").strip() for task_id in task_ids)
    excluded = generalization_exclusions(requested)
    if excluded:
        detail = ", ".join(f"{task_id} ({reason})" for task_id, reason in excluded.items())
        raise ContaminatedGeneralizationError(
            f"zero-shot generalization scope includes {len(excluded)} task(s) that "
            f"cannot support the claim: {detail}"
        )
    return requested


def filter_generalization_scope(
    task_ids: Iterable[str],
) -> tuple[tuple[str, ...], Mapping[str, str]]:
    """Split ``task_ids`` into (scoreable, {dropped: reason})."""

    requested = [str(task_id or "").strip() for task_id in task_ids]
    excluded = generalization_exclusions(requested)
    kept = tuple(task_id for task_id in requested if task_id not in excluded)
    return kept, excluded


def generalization_claim_report(
    task_ids: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    """Auditable record of which held-out tasks a zero-shot claim may use."""

    if task_ids is None:
        requested = [task.task_id for task in heldout_tasks()]
    else:
        requested = [str(task_id or "").strip() for task_id in task_ids]
    kept, excluded = filter_generalization_scope(requested)
    return {
        "taxonomy_version": TAXONOMY_VERSION,
        "taxonomy_digest": taxonomy_digest(),
        "requested": len(requested),
        "scoreable": len(kept),
        "excluded": len(excluded),
        "scoreable_task_ids": list(kept),
        "excluded_task_ids": dict(excluded),
        "contamination_evidence": dict(taxonomy.MIDTRAIN_CURRICULUM_CONTAMINATION),
        "contamination_records": {
            task_id: record.as_dict()
            for task_id, record in sorted(taxonomy.CONTAMINATION_RECORDS.items())
        },
    }


# --------------------------------------------------------------------------- #
# Hardware-verification eligibility (TRAINING SELECTION ONLY)
# --------------------------------------------------------------------------- #
def hardware_verdict(task_id: str) -> verification.HardwareVerdict:
    """gfx950 verdict for ``task_id``; an unmeasured task is UNKNOWN, not PASS."""
    return verification.verdict_for(task_id)


def hardware_verdicts() -> Mapping[str, verification.HardwareVerdict]:
    """Verdicts for every registered task, including synthesized UNKNOWNs."""
    return MappingProxyType(
        {task_id: verification.verdict_for(task_id) for task_id in sorted(task_ids())}
    )


def hardware_verification_coverage() -> dict[str, Any]:
    """How much of the registry the gfx950 sweep actually measured."""
    return verification.coverage(task_ids())


def train_eligibility_exclusions(
    policy: Optional[verification.EligibilityPolicy | str] = None,
) -> Mapping[str, str]:
    """``{task_id: reason}`` for train tasks this policy would not select.

    The train split itself is unchanged; this is the explicit, opt-in view.
    """
    return verification.exclusions(
        (task.task_id for task in train_tasks()), policy
    )


def eligible_train_tasks(
    policy: Optional[verification.EligibilityPolicy | str] = None,
) -> list[Task]:
    """Train tasks admitted by a hardware-eligibility policy.

    Defaults to :data:`kore.tasks.verification.DEFAULT_ELIGIBILITY_POLICY`, which
    drops the tasks whose own seed is structurally broken or misses its declared
    SNR gate by more than :data:`~kore.tasks.verification.NEAR_GATE_MARGIN_DB`.
    Callers that must cover the entire registry train set keep using
    ``train_tasks()``; this function is never applied implicitly.
    """
    excluded = train_eligibility_exclusions(policy)
    return [task for task in train_tasks() if task.task_id not in excluded]


def hardware_eligibility_report(
    policy: Optional[verification.EligibilityPolicy | str] = None,
) -> dict[str, Any]:
    """Auditable eligibility report over the registry train split."""
    resolved = verification.resolve_policy(policy)
    report = verification.describe((task.task_id for task in train_tasks()), resolved)
    report["train_tasks"] = len(train_tasks())
    report["taxonomy_digest"] = taxonomy_digest()
    return report


def _canonical_tasks(items: Iterable[Task], label: str) -> tuple[Task, ...]:
    ids: list[str] = []
    for item in items:
        task_id = str(getattr(item, "task_id", "") or "").strip()
        if not task_id:
            raise SplitManifestError(f"{label} contains a task with no task_id")
        ids.append(task_id)
    if len(ids) != len(set(ids)):
        duplicates = sorted({task_id for task_id in ids if ids.count(task_id) > 1})
        raise SplitManifestError(f"{label} contains duplicate task IDs: {duplicates}")
    unknown = sorted(set(ids) - set(task_ids()))
    if unknown:
        raise SplitManifestError(f"{label} contains unknown task IDs: {unknown}")
    return tuple(get_task(task_id) for task_id in sorted(ids))


def build_split_manifest(
    train: Optional[Iterable[Task]] = None,
    eval: Optional[Iterable[Task]] = None,
) -> SplitManifest:
    """Create an immutable, lineage-checked split manifest.

    With no arguments this freezes the whole registry.  Supplying one side requires
    the other side explicitly; every ID must agree with its authoritative decision.
    """

    if train is None and eval is None:
        train_items = tuple(train_tasks())
        eval_items = tuple(heldout_tasks())
    elif train is None or eval is None:
        raise SplitManifestError("train and eval task collections must be supplied together")
    else:
        train_items = _canonical_tasks(train, "train")
        eval_items = _canonical_tasks(eval, "eval")

    train_items = _canonical_tasks(train_items, "train")
    eval_items = _canonical_tasks(eval_items, "eval")
    if not train_items:
        raise SplitManifestError("train split is empty; refusing all-heldout fallback")
    if not eval_items:
        raise SplitManifestError("eval split is empty")

    train_ids = tuple(task.task_id for task in train_items)
    eval_ids = tuple(task.task_id for task in eval_items)
    overlap = set(train_ids) & set(eval_ids)
    if overlap:
        raise SplitManifestError(f"train/eval task collision: {sorted(overlap)}")

    wrong_train = [
        task.task_id for task in train_items if split_decision(task).split != "train"
    ]
    wrong_eval = [
        task.task_id for task in eval_items if split_decision(task).split != "eval"
    ]
    contaminated_train = [
        task.task_id for task in train_items if is_contaminated(task)
    ]
    if contaminated_train:
        raise SplitManifestError(
            f"contaminated tasks placed in train: {contaminated_train}"
        )
    if wrong_train:
        raise SplitManifestError(f"eval-only tasks placed in train: {wrong_train}")
    if wrong_eval:
        raise SplitManifestError(f"train tasks placed in eval: {wrong_eval}")

    train_roots = tuple(
        (task.task_id, taxonomy.provenance_root_for_task(task))
        for task in train_items
    )
    eval_roots = tuple(
        (task.task_id, taxonomy.provenance_root_for_task(task))
        for task in eval_items
    )
    root_overlap = {root for _, root in train_roots} & {root for _, root in eval_roots}
    if root_overlap:
        raise SplitManifestError(
            f"train/eval provenance-lineage collision: {sorted(root_overlap)}"
        )

    return SplitManifest(
        taxonomy_version=TAXONOMY_VERSION,
        taxonomy_digest=taxonomy_digest(),
        train_ids=train_ids,
        eval_ids=eval_ids,
        train_provenance_roots=train_roots,
        eval_provenance_roots=eval_roots,
        whole_family_holdouts=tuple(sorted(taxonomy.WHOLE_FAMILY_HOLDOUTS)),
        near_generalization_ids=tuple(sorted(taxonomy.NEAR_GENERALIZATION_TASK_IDS)),
        contaminated_eval_ids=tuple(
            task.task_id for task in eval_items if is_contaminated(task)
        ),
    )


def split_manifest_for_selection(selected: Iterable[Task]) -> SplitManifest:
    """Freeze a campaign selection, using the global eval set when needed."""

    selected_items = _canonical_tasks(selected, "selection")
    train = [task for task in selected_items if not is_heldout(task)]
    selected_eval = [task for task in selected_items if is_heldout(task)]
    eval_items = selected_eval or heldout_tasks()
    return build_split_manifest(train, eval_items)


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SplitManifestError(f"{field} must be a list of strings")
    if value != sorted(value):
        raise SplitManifestError(f"{field} must be sorted canonically")
    if len(value) != len(set(value)):
        raise SplitManifestError(f"{field} contains duplicates")
    return value


def validate_split_manifest(
    payload: Mapping[str, Any],
    *,
    expected: Optional[SplitManifest] = None,
) -> SplitManifest:
    """Validate a serialized manifest against the live taxonomy and registry."""

    if not isinstance(payload, Mapping):
        raise SplitManifestError("split manifest must be a mapping")
    tax = payload.get("taxonomy")
    if not isinstance(tax, Mapping):
        raise StaleSplitManifestError("split manifest lacks taxonomy version/digest")
    version = tax.get("version")
    digest = tax.get("digest")
    if version != TAXONOMY_VERSION:
        raise StaleSplitManifestError(
            f"taxonomy version changed: manifest={version!r}, live={TAXONOMY_VERSION!r}"
        )
    live_digest = taxonomy_digest()
    if digest != live_digest:
        raise StaleSplitManifestError(
            f"taxonomy digest changed: manifest={digest!r}, live={live_digest!r}"
        )

    train_ids = _string_list(payload.get("train_ids"), "train_ids")
    eval_ids = _string_list(payload.get("eval_ids"), "eval_ids")
    roots = payload.get("provenance_roots")
    if not isinstance(roots, Mapping):
        raise SplitManifestError("provenance_roots must be a mapping")
    train_roots = roots.get("train")
    eval_roots = roots.get("eval")
    if not isinstance(train_roots, Mapping) or not isinstance(eval_roots, Mapping):
        raise SplitManifestError("provenance_roots.train/eval must be mappings")

    current = build_split_manifest(
        [get_task(task_id) for task_id in train_ids],
        [get_task(task_id) for task_id in eval_ids],
    )
    if dict(current.train_provenance_roots) != dict(train_roots):
        raise StaleSplitManifestError("train provenance roots changed")
    if dict(current.eval_provenance_roots) != dict(eval_roots):
        raise StaleSplitManifestError("eval provenance roots changed")

    policy = payload.get("policy")
    if not isinstance(policy, Mapping):
        raise SplitManifestError("split manifest lacks policy metadata")
    required_policy = current.as_dict()["policy"]
    if dict(policy) != required_policy:
        raise StaleSplitManifestError("split policy metadata changed")

    if expected is not None and (
        current.train_ids != expected.train_ids
        or current.eval_ids != expected.eval_ids
    ):
        raise StaleSplitManifestError(
            "manifest train/eval IDs differ from the requested campaign selection"
        )
    return current


def split_tasks(seed: int = 0) -> dict[str, object]:
    """Deterministic train/held-out split.

    The held-out set is FIXED (reserved operator families + arch-specific tasks)
    regardless of ``seed`` so training can never leak into it; ``seed`` only
    controls the reproducible ordering WITHIN each split (useful for sharding or
    cross-validation folds). Returns
    ``{"train": [...], "heldout": [...], "seed": seed}``.
    """
    manifest = build_split_manifest()
    train = [get_task(task_id) for task_id in manifest.train_ids]
    held = [get_task(task_id) for task_id in manifest.eval_ids]
    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(held)
    return {
        "train": train,
        "heldout": held,
        "seed": seed,
        "manifest": manifest,
    }
