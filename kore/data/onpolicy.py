"""On-policy relabeling for KORE (DAgger / iterative-DPO).

This is the iterative on-policy loop the plan promises: instead of only learning
from *teacher* samples, we relabel/repair on the states the CURRENT POLICY
actually visits, which is the DAgger no-regret argument (train on your own
distribution, not the expert's).

A "policy" here is any object with ``generate(messages: list[dict]) -> str`` —
the exact duck type of :class:`kore.data.teacher.TeacherClient`. So a live model
adapter (a served checkpoint, ``kore.policy.serve.load_generate``, or the GRPO
``_HFChatPolicy``) drops straight in wherever the teacher used to be.

Three entry points:

  * :func:`relabel_groups_on_policy` — sample candidate rewrites FROM THE POLICY
    via the same path as ``gen_groups``, verify + rank them, and emit
    ``RankedGroupRecord``s (DPO preferences on states the policy visits).
  * :func:`dagger_repairs` — roll the POLICY, collect the kernels it emits that
    FAIL the verifier, and get an EXPERT (teacher) correction for each; with a
    ``teacher_frac`` beta that decays 30%->0% across rounds (classic DAgger).
  * :func:`iterative_dpo` — orchestrate N rounds: relabel on-policy -> build DPO
    pairs -> (caller trains) -> refresh the reference; aggregating the union of
    all rounds' groups for the no-regret property.

Everything here is CPU-friendly and teacher/model-agnostic; the heavy training
is delegated to the caller (``kore.policy.dpo`` / ``kore.policy.rft``).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from kore.config import CONFIG
from kore.data.build_datasets import build_dpo
from kore.data.gen_groups import generate_groups
from kore.data.gen_repair import (
    _failure_class,
    make_repair_record,
    mine_natural_failures,
)
from kore.data.prompts import SYSTEM_PROMPT, build_turn_prompt, extract_kernel
from kore.data.schemas import RankedGroupRecord, RepairRecord
from kore.obs import get_logger

log = get_logger("data.onpolicy")


# A policy is anything with ``generate(messages) -> str`` (TeacherClient duck type).
Policy = Any


# --------------------------------------------------------------------------- #
# 1. Iterative on-policy DPO relabeling
# --------------------------------------------------------------------------- #
def relabel_groups_on_policy(
    task,
    policy: Policy,
    env,
    n_parents: int,
    k: int,
    seed: int = 0,
    cfg=CONFIG,
) -> list[RankedGroupRecord]:
    """Relabel ranked preference groups using candidates FROM THE POLICY.

    Identical generation/verification/ranking path as
    :func:`kore.data.gen_groups.generate_groups`, but the generator is the
    *current policy* rather than the teacher — so the preference pairs are over
    kernels the policy actually produces (the on-policy distribution). The
    verifier + ``rank_candidates`` / ``build_preferences`` (with the margin gate)
    are reused unchanged. Returns ``RankedGroupRecord``s ready for ``build_dpo``.
    """
    with log.stage("relabel_groups_on_policy", task=task.task_id,
                   n_parents=n_parents, k=k):
        return generate_groups(
            task, policy, env, n_parents=n_parents, k=k, seed=seed, cfg=cfg
        )


@dataclass
class IterativeDPORound:
    """One round of the iterative-DPO loop (data + reference-refresh bookkeeping)."""

    round: int
    ref_model_id: Optional[str]          # frozen reference for THIS round (prev ckpt)
    groups_new: list[RankedGroupRecord]  # groups relabeled this round
    groups_agg: list[RankedGroupRecord]  # union with all prior rounds (DAgger)
    dpo_pairs: list[dict]                 # build_dpo(groups_agg): trl preference rows
    n_pairs: int
    policy_ckpt: Optional[str] = None    # checkpoint the caller trained this round


def iterative_dpo(
    rounds: int,
    policy_factory: Callable[[int, Optional[str]], Policy],
    tasks,
    env_factory: Callable[[Any], Any],
    *,
    n_parents: int = 8,
    k: int = 4,
    seed: int = 0,
    cfg=CONFIG,
    train_fn: Optional[Callable[["IterativeDPORound"], Optional[str]]] = None,
    aggregate: bool = True,
    prompt_fn: Optional[Callable[[str], Any]] = None,
    extra_pairs: Optional[list] = None,
) -> list[IterativeDPORound]:
    """Run ``rounds`` of iterative on-policy DPO.

    Each round:
      1. ``policy = policy_factory(round_idx, prev_ckpt)`` — load the current
         policy (``prev_ckpt`` is the checkpoint trained last round, or ``None``
         on the first round). This is the reference-refresh handle.
      2. Relabel groups on-policy for every task (:func:`relabel_groups_on_policy`).
      3. AGGREGATE: union this round's groups with all prior rounds' groups
         (``aggregate=True``, the DAgger no-regret property — training on the
         union of every visited distribution). Set ``aggregate=False`` to train on
         the latest round only.
      4. Build DPO preference rows from the aggregated groups (``build_dpo``).
      5. If ``train_fn`` is given, call it with the round's data; its return value
         (a checkpoint id) becomes ``prev_ckpt`` for the next round — i.e. the
         REFERENCE REFRESH: the newly trained policy is both the next round's
         generator and its frozen DPO reference.

    ``train_fn`` is where the caller wires ``kore.policy.dpo.train`` (see the
    module docstring / campaign notes). When it is ``None`` the loop still runs
    relabel+build every round (useful for dry data generation / tests) but never
    refreshes the reference. Returns the per-round records.

    ``extra_pairs`` (the build stage's CURATED preference rows -- reward-hack hard
    negatives + fixed>broken repair pairs) are folded into EVERY round's training set
    alongside the on-policy relabeled group pairs, so iterative DPO keeps the
    correctness/anti-hack contrast instead of collapsing to among-correct speed pairs
    only (audit R2 dpo C2: the old loop trained on relabeled groups alone).
    """
    task_list = list(tasks) if isinstance(tasks, (list, tuple)) else [tasks]
    extra = list(extra_pairs or [])
    agg_groups: list[RankedGroupRecord] = []
    results: list[IterativeDPORound] = []
    prev_ckpt: Optional[str] = None

    with log.stage("iterative_dpo", rounds=rounds, tasks=len(task_list),
                   aggregate=aggregate):
        for r in range(rounds):
            policy = policy_factory(r, prev_ckpt)
            new_groups: list[RankedGroupRecord] = []
            for i, task in enumerate(task_list):
                env = env_factory(task)
                new_groups += relabel_groups_on_policy(
                    task, policy, env, n_parents=n_parents, k=k,
                    seed=seed + r * 100_003 + i, cfg=cfg,
                )
            agg_groups = (agg_groups + new_groups) if aggregate else list(new_groups)
            # Pillar 3: in-context DPO prompts (seed-kernel transcript) so iterative
            # on-policy preferences match the deployment context, exactly like the
            # first-round build stage. Falls back to the generic prompt if None.
            dpo_pairs = build_dpo(agg_groups, prompt_fn=prompt_fn)
            if extra:
                # union the curated hard-neg/repair contrast into every round
                dpo_pairs = list(dpo_pairs) + extra
            rd = IterativeDPORound(
                round=r,
                ref_model_id=prev_ckpt,
                groups_new=new_groups,
                groups_agg=list(agg_groups),
                dpo_pairs=dpo_pairs,
                n_pairs=len(dpo_pairs),
            )
            if train_fn is not None:
                rd.policy_ckpt = train_fn(rd)
                prev_ckpt = rd.policy_ckpt  # reference refresh for the next round
            results.append(rd)
            log.metric("iterative_dpo_round", round=r, groups_new=len(new_groups),
                       groups_agg=len(agg_groups), pairs=len(dpo_pairs),
                       ref_model_id=rd.ref_model_id, policy_ckpt=rd.policy_ckpt)
        return results


# --------------------------------------------------------------------------- #
# 2. DAgger-repair on the policy's own failures
# --------------------------------------------------------------------------- #
def dagger_teacher_frac(round_idx: int, rounds: int,
                        start: float = 0.30, end: float = 0.0) -> float:
    """Beta-decay schedule for the DAgger teacher-mixing fraction.

    Linearly anneals from ``start`` (round 0) to ``end`` (final round). Early
    rounds mix in teacher-sourced failures for exploration; later rounds rely
    purely on the policy's OWN visited failure states (beta -> 0)."""
    if rounds <= 1:
        return end
    frac = round_idx / (rounds - 1)
    return start + (end - start) * frac


def dagger_repairs(
    task,
    policy: Policy,
    teacher,
    env,
    n: int,
    seed: int = 0,
    teacher_frac: float = 0.0,
    diagnostic: bool = True,
) -> list[RepairRecord]:
    """Repair the POLICY's own verifier failures with EXPERT (teacher) fixes.

    Rolls the policy on ``task`` (seed-conditioned rewrites), collects the kernels
    it emits that FAIL the verifier (compile / SNR), and for each calls
    :func:`kore.data.gen_repair.make_repair_record` so the TEACHER produces a
    verified corrected kernel. This trains repair on the states KORE actually
    visits, not on injected/teacher-only breakages. A record is emitted only when
    the teacher's fix passes validation (``make_repair_record`` returns ``None``
    otherwise — an unfixed failure is simply dropped).

    ``teacher_frac`` is the DAgger beta: that fraction of the ``n`` target is
    sourced from the TEACHER's own natural failures instead (mixing expert
    exploration early; pass a schedule from :func:`dagger_teacher_frac`, decaying
    30%->0% across rounds). ``diagnostic`` selects the diagnose-then-fix SFT format.
    """
    teacher_frac = max(0.0, min(1.0, float(teacher_frac)))
    with log.stage("dagger_repairs", task=task.task_id, n=n,
                   teacher_frac=teacher_frac):
        rng = random.Random(seed)
        records: list[RepairRecord] = []
        n_teacher = int(round(n * teacher_frac))
        n_policy = n - n_teacher
        seed_src = task.seed_source
        t_start = time.time()
        attempts = 0
        budget = max(n_policy * 6, 6)

        # (1) Roll the POLICY; mine its OWN failures -> teacher-corrected repair.
        while len(records) < n_policy and attempts < budget:
            attempts += 1
            mode = rng.choice(["exploit", "explore"])
            prompt = build_turn_prompt(parent_source=seed_src, mode=mode)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            response = policy.generate(messages)
            cand_src = extract_kernel(response)
            if not cand_src:
                log.debug("dagger policy sample had no kernel; skipping",
                          task=task.task_id, idx=attempts, mode=mode)
                continue
            try:
                obs = env.step(cand_src, full_validation=True, multi_shape=False)
            except Exception:
                log.debug("dagger policy sample crashed verifier",
                          task=task.task_id, idx=attempts, mode=mode)
                continue
            if _failure_class(obs) is None:
                continue  # the policy succeeded here — not a repair state
            rec = make_repair_record(task, teacher, env, cand_src, obs,
                                     diagnostic=diagnostic)
            if rec is not None:
                records.append(rec)
            log.progress(attempts, budget, "dagger_policy",
                         t_start=t_start, kept=len(records), target=n_policy)
        n_policy_kept = len(records)

        # (2) DAgger beta-mixing: teacher-sourced failures (decays to 0 by the
        #     final round). Reuses the teacher natural-failure miner.
        if n_teacher > 0:
            records += mine_natural_failures(
                task, teacher, env, n_teacher, seed=seed + 1, diagnostic=diagnostic
            )

        result = records[:n]
        log.metric(
            "dagger_summary", task=task.task_id, attempts=attempts,
            policy_failures_repaired=n_policy_kept,
            teacher_mixed=len(result) - n_policy_kept, kept=len(result),
            teacher_frac=teacher_frac,
        )
        return result
