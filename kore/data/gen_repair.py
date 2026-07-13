"""Generate repair data (KORE Stage 1: repair-weighted SFT).

Two sources of broken->fixed turns:
  1. INJECTED breakage: take the known-good seed, apply a ``mutate`` breakage,
     confirm via the verifier that it fails, then ask the teacher to repair it
     conditioned on the exact error, and confirm the fix.
  2. NATURAL failures: sample fresh candidates from the teacher; whenever one
     fails the verifier, mine it as a repair opportunity the same way.

Each accepted turn becomes a ``RepairRecord`` whose ``messages`` are the chat
that produced the fix (system + repair-user + assistant), so it can go straight
into SFT via ``build_datasets.build_sft``.
"""

from __future__ import annotations

import random
import threading
import time
from typing import Optional

from kore.data.prompts import (
    SYSTEM_PROMPT,
    build_turn_prompt,
    extract_kernel,
    format_assistant_turn,
)
from kore.data.mutate import apply_random_breakage, infer_family
from kore.data.schemas import RepairRecord
from kore.data.teacher import TeacherClient
from kore.env.replay import kernel_hash
from kore.obs import get_logger

log = get_logger("data.gen_repair")

# Per-attempt context (idx / mutator / broke_verified_fail) that the generation
# loops set right before calling ``make_repair_record`` so it can emit a fully
# populated ``repair_attempt`` event without a signature change. Thread-local so
# concurrent datagen never crosses wires. Purely observational.
_ctx = threading.local()


def _failure_class(obs) -> Optional[str]:
    """Map a verifier Observation to a failure bucket, or None if it passed."""
    if not obs.compiled:
        return "compile_fail"
    if not obs.validation_passed:
        return "snr_fail"
    return None


def _error_text(obs) -> str:
    if obs.error_text:
        return obs.error_text
    if not obs.compiled:
        return "kernel failed to compile"
    return f"correctness failed (snr_db={obs.snr_db})"


def _diagnostic_assistant(failure_class: str, error_text: str, fixed_src: str) -> str:
    """Fold the verifier's error into a self-diagnose-then-fix assistant turn,
    in the CANONICAL response contract (ANALYSIS / PROPOSED_CHANGE / FULL_KERNEL).

    The exact verifier diagnostic + fix-class reasoning becomes the ANALYSIS
    (self-diagnosis), the fix class becomes the PROPOSED_CHANGE, and the VERIFIED
    fixed kernel goes in FULL_KERNEL. SFT on this teaches the model to read the
    concrete failure and fix it in the SAME response shape it must emit at
    inference — not a separate ``<think>/<answer>`` format that drifts from the
    policy contract."""
    guidance = {
        "compile_fail": (
            "This is a COMPILE failure. I will fix the specific syntax/type/shape "
            "error the verifier reported without rewriting the kernel structure."
        ),
        "snr_fail": (
            "This is a CORRECTNESS (low-SNR) failure. The numerics are wrong — I "
            "will restore fp32 accumulation / correct masking / indexing / block "
            "alignment rather than change the algorithm."
        ),
    }.get(failure_class, "I will fix the specific failure the verifier reported.")
    analysis = (
        "The verifier rejected the current kernel. Diagnostic:\n"
        f"{error_text.strip()}\n"
        f"{guidance}"
    )
    proposed = {
        "compile_fail": "Fix the reported compile error without changing the kernel structure.",
        "snr_fail": ("Restore numerical correctness (fp32 accumulation / masking / "
                     "indexing / block alignment)."),
    }.get(failure_class, "Fix the specific failure the verifier reported.")
    return format_assistant_turn(analysis, proposed, fixed_src)


def make_repair_record(
    task,
    teacher: TeacherClient,
    env,
    broken_src: str,
    broken_obs,
    diagnostic: bool = True,
) -> Optional[RepairRecord]:
    """Given a known-broken kernel + its observation, get a teacher repair and
    emit a RepairRecord ONLY when the repair actually validates.

    The broken side must genuinely fail (``failure_class`` is not None) and the
    teacher's fix must pass full validation, otherwise we would mislabel a still-
    broken kernel as a correct SFT target. Returns None if the teacher produced
    no kernel, the fix crashed, or the fix did not pass validation.

    When ``diagnostic`` is True (default), the stored assistant turn is rendered in
    the canonical diagnose-then-fix contract (ANALYSIS / PROPOSED_CHANGE / FULL_KERNEL
    via :func:`format_assistant_turn`), folding the verifier's ``error_text`` into the
    ANALYSIS so SFT learns to self-diagnose. The emitted fix is always the VERIFIED
    kernel only — the "only emit verified fixes" rule is unchanged."""
    failure_class = _failure_class(broken_obs)
    if failure_class is None:
        return None

    error = _error_text(broken_obs)
    user_prompt = build_turn_prompt(
        parent_source=broken_src, feedback=error, mode="repair"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    ctx = getattr(_ctx, "attempt", None) or {}
    t_teacher = time.time()
    response = teacher.generate(messages)
    teacher_ms = round((time.time() - t_teacher) * 1000.0, 1)
    fixed_src = extract_kernel(response)

    def _emit(kept: bool, skip_reason: Optional[str], fixed_obs=None,
              child_snr=None) -> None:
        log.event(
            "repair_attempt",
            task=task.task_id, idx=ctx.get("idx"), mutator=ctx.get("mutator"),
            failure_class=failure_class,
            broke_verified_fail=ctx.get("broke_verified_fail"),
            teacher_ms=teacher_ms,
            fixed_compiled=(bool(getattr(fixed_obs, "compiled", False))
                            if fixed_obs is not None else False),
            fixed_correct=(bool(getattr(fixed_obs, "validation_passed", False))
                           if fixed_obs is not None else False),
            child_snr_db=child_snr, kept=kept, skip_reason=skip_reason,
        )

    if not fixed_src:
        _emit(False, "no_kernel")
        return None

    try:
        fixed_obs = env.step(fixed_src, full_validation=True, multi_shape=True)
    except Exception:
        fixed_obs = None

    # Only accept the repair as an SFT target if it genuinely passes validation;
    # otherwise it is a mislabeled still-broken kernel.
    if fixed_obs is None:
        _emit(False, "fix_crashed")
        return None
    if not getattr(fixed_obs, "validation_passed", False):
        _emit(False, "fix_unverified", fixed_obs=fixed_obs)
        return None
    child_snr = fixed_obs.snr_db
    _emit(True, None, fixed_obs=fixed_obs, child_snr=child_snr)

    if diagnostic:
        assistant = _diagnostic_assistant(failure_class, error, fixed_src)
    else:
        assistant = response
    messages = messages + [{"role": "assistant", "content": assistant}]
    return RepairRecord(
        task_id=task.task_id,
        failure_class=failure_class,
        parent_hash=kernel_hash(broken_src),
        error_text=error,
        messages=messages,
        child_snr_db=child_snr,
        gpu=task.gpu_target,
        operation=getattr(task, "operation", None),
        arch=getattr(task, "gpu_target", None),
    )


def generate_repairs(
    task,
    teacher: TeacherClient,
    env,
    n: int,
    seed: int = 0,
    natural_fraction: float = 0.3,
    diagnostic: bool = True,
) -> list[RepairRecord]:
    """Produce up to ``n`` RepairRecords for ``task``.

    A ``natural_fraction`` of attempts mine naturally-failed teacher generations;
    the rest inject a breakage into the seed and repair that. ``diagnostic``
    selects the diagnose-then-fix assistant format (see ``make_repair_record``).
    """
    with log.stage("generate_repairs", task=task.task_id, n=n,
                   natural_fraction=natural_fraction):
        rng = random.Random(seed)
        records: list[RepairRecord] = []
        seed_src = task.seed_source
        family = infer_family(getattr(task, "operation", "") or task.task_id)
        n_natural = int(round(n * natural_fraction))
        n_injected = n - n_natural

        # (1) Injected breakage repairs.
        t_start = time.time()
        attempts = 0
        while len([r for r in records]) < n_injected and attempts < n_injected * 5:
            attempts += 1
            broken_src, _hint, _name = apply_random_breakage(seed_src, rng, family=family)
            try:
                broken_obs = env.step(broken_src, full_validation=True, multi_shape=False)
            except Exception:
                log.debug("injected breakage crashed verifier",
                          task=task.task_id, idx=attempts, mutator=_name)
                continue
            if _failure_class(broken_obs) is None:
                log.debug("injected breakage did not fail verifier; skipping",
                          task=task.task_id, idx=attempts, mutator=_name)
                continue  # breakage didn't actually break — skip
            _ctx.attempt = {"idx": attempts, "mutator": _name,
                            "broke_verified_fail": True}
            rec = make_repair_record(task, teacher, env, broken_src, broken_obs,
                                     diagnostic=diagnostic)
            if rec is not None:
                records.append(rec)
            log.progress(attempts, max(1, n_injected * 5), "repair",
                         t_start=t_start, kept=len(records), target=n_injected)
        _ctx.attempt = {}
        injected_kept = len(records)

        # (2) Naturally-failed teacher turns.
        natural = mine_natural_failures(task, teacher, env, n_natural, seed=seed + 1,
                                        diagnostic=diagnostic)
        records += natural
        result = records[:n]
        log.metric(
            "repair_summary", task=task.task_id, attempts=attempts,
            kept=len(result), injected_kept=injected_kept,
            dropped_unverified=attempts - injected_kept,
            natural_mined=len(natural),
        )
        return result


def mine_natural_failures(
    task,
    teacher: TeacherClient,
    env,
    n: int,
    seed: int = 0,
    diagnostic: bool = True,
) -> list[RepairRecord]:
    """Sample teacher rewrites of the seed; whenever one fails, mine a repair."""
    with log.stage("mine_natural_failures", task=task.task_id, n=n):
        rng = random.Random(seed)
        records: list[RepairRecord] = []
        seed_src = task.seed_source
        t_start = time.time()
        attempts = 0
        budget = max(n * 5, 5)
        while len(records) < n and attempts < budget:
            attempts += 1
            mode = rng.choice(["exploit", "explore"])
            prompt = build_turn_prompt(parent_source=seed_src, mode=mode)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            response = teacher.generate(messages)
            cand_src = extract_kernel(response)
            if not cand_src:
                log.debug("natural sample had no kernel; skipping",
                          task=task.task_id, idx=attempts, mode=mode)
                continue
            try:
                obs = env.step(cand_src, full_validation=True, multi_shape=False)
            except Exception:
                log.debug("natural sample crashed verifier",
                          task=task.task_id, idx=attempts, mode=mode)
                continue
            if _failure_class(obs) is None:
                continue  # it worked — not a repair opportunity
            _ctx.attempt = {"idx": attempts, "mutator": f"natural:{mode}",
                            "broke_verified_fail": True}
            rec = make_repair_record(task, teacher, env, cand_src, obs,
                                     diagnostic=diagnostic)
            if rec is not None:
                records.append(rec)
            log.progress(attempts, budget, "natural_mine",
                         t_start=t_start, mined=len(records), target=n)
        _ctx.attempt = {}
        log.metric("natural_mine_summary", task=task.task_id, attempts=attempts,
                   mined=len(records), target=n)
        return records
