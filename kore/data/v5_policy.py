"""The admission rules every v5 build stage shares.

**The backend suffix defeats the holdout, and that is the bug this module
exists to close.** :func:`taxonomy.split_decision_for_identity` tests membership
of the near-generalization probe set by exact task id, and
:func:`registry.record_split_decision` defaults ``provenance_root`` to the task
id when a record is not in the registry. A twin directory is named
``genb_attn2_cross_gqa_step_fp16__hip``, which is in neither set, so::

    genb_attn2_cross_gqa_step_fp16        -> eval,  near_probe
    genb_attn2_cross_gqa_step_fp16__hip   -> TRAIN, train

44 twin directories covering 39 of the 43 near-generalization probes, and all 11
tasks already marked contaminated, classify as trainable in suffixed form. That
compromises the held-out split this project measures zero-shot generalization
against -- a strictly worse problem than benchmark contamination, because it is
the yardstick itself. So identity is canonicalised to the base id before any
lookup, and the id guards run unconditionally rather than as a late
belt-and-braces check that a ``train`` verdict returns before ever reaching.

**The evaluation benchmark was never a decontamination target.** The repo's
decontaminator indexes KernelBench and KORE's own held-out sources, and nothing
compares the pool against AgentKernelArena, which is the benchmark every claim is
actually made on. It is contaminated: 12 arena tasks have a pool twin whose
PyTorch module is byte-identical after AST normalisation and 25 sit above 0.30
similarity, all in the ``gpumode`` sub-suites, because both corpora derive from
the same ``GPUMODE/KernelBook`` scrape. :func:`admits` therefore consults a frozen
arena index, and it does so for *every* record rather than only the fail-closed
ones -- the exact matches were already classified ``train``, so screening only
the rejects would recover thousands of tasks while leaving the real leak in place.

**Held-out admission.** Beyond those guards the registry classifier fails closed:
an operation absent from the taxonomy's family map is eval-only, so that record
text can never relabel a reserved task into train. That default is correct and
also indiscriminate -- it rejects 4,180 verified twins, 98.5% of them ``kbk_``
KernelBook modules held out for nothing worse than being named
``GateGRUSelectionLayer`` instead of ``gemm``. ``strict`` takes the classifier's
answer as final; ``audited`` overrides that one reason for cleared prefixes only,
with every other holdout reason still respected.

**Speedup credibility.** A kernel reported three orders of magnitude faster than
its reference is a statement about a broken baseline or a decoy that never ran,
not a kernel achievement. Training on it teaches the model to reproduce whatever
made the measurement absurd, which is exactly what a later RL stage is free to
exploit. The write-time flag caught only 137 of the 769 real wins above the
ceiling, so the ceiling is re-applied at build time to every source.
"""

from __future__ import annotations

import re
from typing import Any

#: Id prefixes cleared for the ``audited`` policy. ``genb_`` is deliberately
#: absent: those ids populate HELDOUT_TASKS and are the generalization probe, so
#: no override may reach them.
#:
#: ``flydsl_lib::`` is not a mined task at all -- it is Apache-2.0 library code
#: from the DSL's own test suite and tutorials, which carries no task identity for
#: the registry to classify and so fails closed for a reason that does not apply
#: to it. Its screening happens where it belongs, at extraction: the production
#: kernel directory is excluded wholesale because it is the corpus the benchmark
#: draws from, and what remains is checked against every arena FlyDSL task name.
AUDITED_PREFIXES: tuple[str, ...] = ("kbk_", "syn_synth_", "flydsl_lib::")

#: The only classifier verdict ``audited`` is allowed to overturn.
OVERRIDABLE_REASON = "unclassified_operation"

#: Above this a speedup is not credible. Applies to real wins, gold wins, and any
#: step whose gain is derived from a measured ratio.
CREDIBLE_SPEEDUP_MAX = 10.0

_SUFFIXES = ("__hipf", "__hip", "__flydsl", "__triton")


def strip_suffix(task_id: str) -> str:
    """The backend-independent identity of a twin."""
    for suffix in _SUFFIXES:
        if task_id.endswith(suffix):
            return task_id[: -len(suffix)]
    return task_id


def dialect(task_id: str) -> str:
    task_id = task_id or ""
    if task_id.endswith(("__hip", "__hipf")) or task_id.startswith("hip_"):
        return "HIP"
    if task_id.endswith("__flydsl"):
        return "FlyDSL"
    return "Triton"


def yaml_field(text: str, key: str, default: str = "") -> str:
    match = re.search(rf'^\s*"?{key}"?\s*:\s*"?([^",\n]+)"?', text, re.M)
    return match.group(1).strip() if match else default


def credible_speedup(value: Any) -> bool:
    """False only when a speedup is present and beyond the ceiling.

    An absent or unparseable speedup is not evidence of cheating -- most of the
    corpus predates the field -- so it passes and is judged on correctness alone.
    """
    try:
        if value is None:
            return True
        speedup = float(value)
    except (TypeError, ValueError):
        return True
    if speedup != speedup:  # NaN
        return True
    return speedup <= CREDIBLE_SPEEDUP_MAX


def admits(record: Any, policy: str = "strict", arena: Any = None) -> tuple[bool, str]:
    """Whether a record may enter training, and the reason for the verdict.

    ``record`` is any mapping carrying at least ``task_id``; ``operation``,
    ``arch`` and ``dtype`` sharpen the classification when present. ``arena`` is
    an optional :class:`kore.data.arena_index.ArenaIndex` -- when omitted, no
    benchmark screening happens and the caller is trusting an unscreened corpus.

    The guards below run before the classifier and unconditionally. Ordering is
    the whole point: a late check placed after "if not heldout: return train" is
    unreachable for exactly the records that need it most, since the suffix bug
    makes the classifier answer ``train`` for held-out probes.
    """
    from kore.tasks.registry import (CONTAMINATED_TASKS, HELDOUT_TASKS,
                                     heldout_families, record_split_decision)

    get = record.get if hasattr(record, "get") else (lambda _k, _d=None: None)
    task_id = str(get("task_id") or "").strip()
    base = strip_suffix(task_id)
    operation = str(get("operation") or "")

    if not task_id:
        return False, "no_identity"
    if base in HELDOUT_TASKS or task_id in HELDOUT_TASKS:
        return False, "heldout_task_id"
    if base in CONTAMINATED_TASKS or task_id in CONTAMINATED_TASKS:
        return False, "contaminated"
    blob = f"{task_id} {operation}".lower()
    if any(family.lower() in blob for family in heldout_families()):
        return False, "heldout_family_substring"
    if arena is not None:
        hit = arena.match(base)
        if hit is not None:
            return False, f"arena_contamination:{hit[0]}:{hit[1]:.2f}"

    # Canonicalise identity for the classifier too, so its own near-probe and
    # contamination lookups see the base id rather than a suffixed variant they
    # will not recognise.
    canonical = dict(record) if isinstance(record, dict) else {"task_id": task_id}
    canonical["task_id"] = base
    canonical.setdefault("provenance_root", base)
    try:
        decision = record_split_decision(canonical)
    except Exception:  # noqa: BLE001 - a classifier hiccup is not a licence to train
        return False, "classifier_error"
    if not decision.heldout:
        return True, "train"

    reason = str(getattr(decision, "reason", "") or "heldout")
    if policy != "audited" or reason != OVERRIDABLE_REASON:
        return False, reason
    if not base.startswith(AUDITED_PREFIXES):
        return False, "unaudited_prefix"
    return True, "train_audited"


__all__ = [
    "AUDITED_PREFIXES", "CREDIBLE_SPEEDUP_MAX", "OVERRIDABLE_REASON",
    "admits", "credible_speedup", "dialect", "strip_suffix", "yaml_field",
]
