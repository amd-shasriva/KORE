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
from typing import Optional

from kore.data.prompts import SYSTEM_PROMPT, build_turn_prompt, extract_kernel
from kore.data.mutate import apply_random_breakage, infer_family
from kore.data.schemas import RepairRecord
from kore.data.teacher import TeacherClient
from kore.env.replay import kernel_hash


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


def make_repair_record(
    task,
    teacher: TeacherClient,
    env,
    broken_src: str,
    broken_obs,
) -> Optional[RepairRecord]:
    """Given a known-broken kernel + its observation, get a teacher repair and
    emit a RepairRecord ONLY when the repair actually validates.

    The broken side must genuinely fail (``failure_class`` is not None) and the
    teacher's fix must pass full validation, otherwise we would mislabel a still-
    broken kernel as a correct SFT target. Returns None if the teacher produced
    no kernel, the fix crashed, or the fix did not pass validation."""
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
    response = teacher.generate(messages)
    fixed_src = extract_kernel(response)
    if not fixed_src:
        return None

    try:
        fixed_obs = env.step(fixed_src, full_validation=True, multi_shape=True)
    except Exception:
        fixed_obs = None

    # Only accept the repair as an SFT target if it genuinely passes validation;
    # otherwise it is a mislabeled still-broken kernel.
    if fixed_obs is None or not getattr(fixed_obs, "validation_passed", False):
        return None
    child_snr = fixed_obs.snr_db

    messages = messages + [{"role": "assistant", "content": response}]
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
) -> list[RepairRecord]:
    """Produce up to ``n`` RepairRecords for ``task``.

    A ``natural_fraction`` of attempts mine naturally-failed teacher generations;
    the rest inject a breakage into the seed and repair that.
    """
    rng = random.Random(seed)
    records: list[RepairRecord] = []
    seed_src = task.seed_source
    family = infer_family(getattr(task, "operation", "") or task.task_id)
    n_natural = int(round(n * natural_fraction))
    n_injected = n - n_natural

    # (1) Injected breakage repairs.
    attempts = 0
    while len([r for r in records]) < n_injected and attempts < n_injected * 5:
        attempts += 1
        broken_src, _hint, _name = apply_random_breakage(seed_src, rng, family=family)
        try:
            broken_obs = env.step(broken_src, full_validation=True, multi_shape=False)
        except Exception:
            continue
        if _failure_class(broken_obs) is None:
            continue  # breakage didn't actually break — skip
        rec = make_repair_record(task, teacher, env, broken_src, broken_obs)
        if rec is not None:
            records.append(rec)

    # (2) Naturally-failed teacher turns.
    records += mine_natural_failures(task, teacher, env, n_natural, seed=seed + 1)
    return records[:n]


def mine_natural_failures(
    task,
    teacher: TeacherClient,
    env,
    n: int,
    seed: int = 0,
) -> list[RepairRecord]:
    """Sample teacher rewrites of the seed; whenever one fails, mine a repair."""
    rng = random.Random(seed)
    records: list[RepairRecord] = []
    seed_src = task.seed_source
    attempts = 0
    while len(records) < n and attempts < max(n * 5, 5):
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
            continue
        try:
            obs = env.step(cand_src, full_validation=True, multi_shape=False)
        except Exception:
            continue
        if _failure_class(obs) is None:
            continue  # it worked — not a repair opportunity
        rec = make_repair_record(task, teacher, env, cand_src, obs)
        if rec is not None:
            records.append(rec)
    return records
