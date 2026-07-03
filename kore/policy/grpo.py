"""Multi-turn GRPO for KORE.

Pure, import-safe math (group advantages, discounted turn returns, asymmetric
"clip-higher" surrogate) lives at module top so it is unit-testable without any
heavy deps.

``train_grpo`` runs a single, self-contained multi-turn GRPO loop natively on
local AMD GPUs (transformers + PEFT), rolling out against the verified
:class:`KoreEnv`. There is NO external server, NO extra install, and NO config
to run it — it works out of the box on AMD. Memory is bounded by a per-sample
micro-batched backward, so it scales from LoRA bring-up to full-FT (FSDP via
``scripts/launch_distributed.sh``).
"""

from __future__ import annotations

import math
import time
from typing import Optional

from kore.obs import configure, get_logger, gpu_mem_snapshot

_EPS = 1e-6

log = get_logger("policy.grpo")


# --------------------------------------------------------------------------- #
# pure math (unit-tested)
# --------------------------------------------------------------------------- #
def group_advantages(rewards: list[float]) -> list[float]:
    """GRPO baseline: (r - mean) / (pop_std + eps). Degenerate group -> ~0."""
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = math.sqrt(var)
    return [(r - mean) / (std + _EPS) for r in rewards]


def discounted_returns(scores: list[float], gamma: float = 0.4) -> list[float]:
    """Turn-level returns R_t = s_t + gamma * R_{t+1} (credit later turns back)."""
    out = [0.0] * len(scores)
    running = 0.0
    for t in range(len(scores) - 1, -1, -1):
        running = scores[t] + gamma * running
        out[t] = running
    return out


def clip_higher_ratio(ratio, advantage, lo: float = 0.2, hi: float = 0.28):
    """DAPO-style asymmetric PPO surrogate (wider upper clip fights collapse).

    Returns ``min(ratio*A, clip(ratio, 1-lo, 1+hi)*A)`` — the surrogate whose
    NEGATION is the clip-higher policy-gradient loss. Works on plain floats
    (pure, unit-testable) and, in the training path, on a differentiable 0-dim
    torch tensor ``ratio``: in that case it stays a tensor (via ``clamp`` /
    ``minimum``) so the fully-clipped region contributes a well-defined ZERO
    gradient instead of dropping the autograd graph (which a Python ``min``
    would do by returning a bare float).
    """
    if hasattr(ratio, "clamp"):  # differentiable torch tensor
        import torch

        clipped = ratio.clamp(1.0 - lo, 1.0 + hi)
        return torch.minimum(ratio * advantage, clipped * advantage)
    clipped = min(max(ratio, 1.0 - lo), 1.0 + hi)
    return min(ratio * advantage, clipped * advantage)


# --------------------------------------------------------------------------- #
# Kevin multi-turn credit (best-kernel trajectory scoring)
# --------------------------------------------------------------------------- #
def kevin_trajectory_score(turn_rewards: list[float], correct_flags: list[bool]) -> float:
    """Trajectory value = the BEST *correct* kernel's reward; 0 if none correct.

    This is the Kevin-32B rule: performance is only credited when correctness is
    achieved, and the trajectory is judged by its best kernel (not its last).
    """
    correct = [r for r, c in zip(turn_rewards, correct_flags) if c]
    return max(correct) if correct else 0.0


def kevin_turn_returns(turn_rewards: list[float], correct_flags: list[bool],
                       gamma: float = 0.4) -> list[float]:
    """Per-turn discounted returns with performance gated on correctness.

    Each turn's immediate reward is zeroed unless that turn is correct, then the
    discounted-sum look-ahead (gamma) propagates later success back to the turns
    that set it up. Used as the per-turn-as-sample signal for GRPO.
    """
    gated = [r if c else 0.0 for r, c in zip(turn_rewards, correct_flags)]
    return discounted_returns(gated, gamma)


def build_kevin_samples(traj_rewards: list[list[float]], traj_correct: list[list[bool]],
                        gamma: float = 0.4,
                        traj_infra: Optional[list[list[bool]]] = None,
                        ) -> tuple[list[float], list[tuple[int, int]]]:
    """Flatten m trajectories x n turns into per-turn Kevin-credit samples.

    For each trajectory, per-turn returns are computed with
    :func:`kevin_turn_returns` (correctness-gated, gamma look-ahead). The
    returns are concatenated into a single flat list of ``m*n`` samples and an
    ``index`` list maps each flat position back to ``(traj_idx, turn_idx)``.

    The caller feeds ``returns`` to :func:`group_advantages` so the GRPO
    baseline is computed across ALL per-turn samples in the group (per-turn-as-
    sample), which is exactly the Kevin recipe.

    ``traj_infra`` (optional, same ragged shape as ``traj_rewards``) flags turns
    that hit an infrastructure error (timeout/OOM/segfault/import) rather than a
    kernel signal. Such turns are DROPPED from the emitted samples (they are not
    the policy's fault and must not be trained as reward-0 / included in
    :func:`group_advantages`). They still occupy a position in the gamma
    look-ahead chain — their gated reward is 0 anyway — so downstream credit for
    the real (kept) turns is unchanged.
    """
    returns: list[float] = []
    index: list[tuple[int, int]] = []
    for ti, (rewards, corrects) in enumerate(zip(traj_rewards, traj_correct)):
        infra = traj_infra[ti] if traj_infra is not None else None
        for tu, r in enumerate(kevin_turn_returns(rewards, corrects, gamma)):
            if infra is not None and tu < len(infra) and infra[tu]:
                continue  # infra sample: not a kernel signal — drop from the batch
            returns.append(r)
            index.append((ti, tu))
    return returns, index


def token_mean_logprob(seq_logprob: float, n_tokens: int) -> float:
    """DAPO length-debias: divide a sequence log-prob by its token count.

    Summed sequence log-probs scale with length, so longer completions get a
    disproportionate gradient. Dividing by the token count (token-mean) removes
    that length bias before the policy-gradient term is formed.
    """
    return seq_logprob / max(int(n_tokens), 1)


# --------------------------------------------------------------------------- #
# CoT masking: drop prior-turn thinking from the multi-turn context
# --------------------------------------------------------------------------- #
def mask_cot_turns(turns: list[dict]) -> list[dict]:
    """Drop prior-turn CoT/thinking from rollout turns before re-rendering context.

    Kevin keeps the durable turn artifact (the ``PROPOSED_CHANGE`` + the
    ``FULL_KERNEL`` source) but strips the verbose ANALYSIS and any
    ``<think>...</think>`` spans so context does not accumulate stale reasoning
    across turns. Returns NEW turn dicts (inputs are left untouched); each result
    carries an empty ``analysis`` plus the parsed ``proposed_change``/``kernel``
    so :func:`kore.policy.format.build_transcript` renders a thinking-free turn.
    """
    import re

    from kore.policy.format import parse_response

    out: list[dict] = []
    for turn in turns:
        nt = dict(turn)
        raw = turn.get("response")
        if raw:
            raw = re.sub(r"<think>[\s\S]*?</think>", "", raw)
            parsed = parse_response(raw)
            nt.pop("response", None)
            nt["analysis"] = ""
            nt["proposed_change"] = parsed.get("proposed_change", "")
            nt["kernel"] = parsed.get("kernel", "")
        else:
            nt["analysis"] = ""
        out.append(nt)
    return out


# --------------------------------------------------------------------------- #
# StarPO-S: Echo-Trap mitigation (variance filtering of rollout groups)
# --------------------------------------------------------------------------- #
def group_reward_std(rewards: list[float]) -> float:
    n = len(rewards)
    if n < 2:
        return 0.0
    mean = sum(rewards) / n
    return math.sqrt(sum((r - mean) ** 2 for r in rewards) / n)


def starpo_keep_group(rewards: list[float], min_std: float = 1e-3) -> bool:
    """Drop a collapsed group (all-equal reward -> zero learning signal)."""
    return group_reward_std(rewards) > min_std


def starpo_select_high_variance(groups: list[list[float]], keep_frac: float = 0.75,
                                min_std: float = 1e-3) -> list[int]:
    """Return indices of the highest-reward-variance groups to train on.

    First drops fully-collapsed groups (std<=min_std), then keeps the top
    ``keep_frac`` by std. This is the StarPO-S stability lever: train on the
    groups that carry signal, not the ones stuck in a reward-variance collapse.
    """
    live = [(i, group_reward_std(g)) for i, g in enumerate(groups) if starpo_keep_group(g, min_std)]
    if not live:
        return []
    live.sort(key=lambda x: x[1], reverse=True)
    k = max(1, int(round(keep_frac * len(live))))
    return sorted(i for i, _ in live[:k])


def dynamic_sampling_refill(roll_fn, target_groups: int, *, min_std: float = 1e-3,
                            max_attempts: int = 0, dynamic: bool = True,
                            std_key=None):
    """DAPO dynamic-sampling collector: OVERSAMPLE-AND-REFILL (item 2).

    Repeatedly calls ``roll_fn(attempt) -> group`` (a freshly rolled task-group)
    and keeps only NON-DEGENERATE groups (``std_key(group) > min_std``) until
    ``target_groups`` are collected or ``max_attempts`` rollouts are spent. This
    replaces StarPO-S drop-and-shrink: instead of rolling a fixed batch and
    discarding the collapsed groups (shrinking the update), it refills to keep the
    effective batch size stable. Returns ``(kept_groups, attempts)``.

    ``dynamic=False`` keeps every rolled group (legacy fixed batch of
    ``target_groups``). ``std_key(group)->float`` extracts the reward-std used for
    the degeneracy test (default: :func:`group_reward_std` over ``group`` when it
    is a list of rewards). Pure/CPU-testable via an injected ``roll_fn``.
    """
    if std_key is None:
        std_key = group_reward_std
    if max_attempts <= 0:
        max_attempts = (3 * target_groups) if dynamic else target_groups
    kept: list = []
    attempts = 0
    while len(kept) < target_groups and attempts < max_attempts:
        grp = roll_fn(attempts)
        attempts += 1
        if dynamic and std_key(grp) <= min_std:
            continue  # degenerate: refill (do not count toward target)
        kept.append(grp)
    return kept, attempts


# --------------------------------------------------------------------------- #
# KL-to-reference (retention anchor) — k3 estimator
# --------------------------------------------------------------------------- #
def kl_k3(logp: float, ref_logp: float) -> float:
    """Unbiased low-variance KL estimate (Schulman k3): E[exp(d) - d - 1], d=ref-logp.

    Anchoring RL to the post-SFT reference with a small coef preserves the chat/
    code/orchestration behavior learned in SFT while the policy specializes.
    """
    d = ref_logp - logp
    return math.exp(d) - d - 1.0


# --------------------------------------------------------------------------- #
# Measurement efficiency: value-model bench prefilter
# --------------------------------------------------------------------------- #
def value_prefilter(candidates: list, scorer, k: int) -> list[int]:
    """Return indices of the top-``k`` candidates by predicted utility.

    ``scorer(candidate)->float`` is the value model's expected-log-speedup (×
    pass prob) head. Only these k are actually compiled+benched on the GPU,
    which is the whole measurement-efficiency lever (≈4× fewer benches-to-best).
    """
    if k >= len(candidates):
        return list(range(len(candidates)))
    scored = sorted(range(len(candidates)), key=lambda i: scorer(candidates[i]), reverse=True)
    return sorted(scored[:k])


def _value_rank_order(candidate_codes: list[str], task) -> Optional[list[int]]:
    """Best-first candidate order from the value-model reranker (contract b).

    Calls ``kore.value.rerank.rank_candidates(items, task=None) -> list[int]``
    defensively: any import error, incompatible signature, or non-permutation
    result yields ``None`` so the caller can fall back to the natural order
    (a sibling agent wires the real reranker in parallel).
    """
    try:
        from kore.value.rerank import rank_candidates
    except Exception:  # noqa: BLE001 - value module unavailable
        return None
    try:
        order = list(rank_candidates(candidate_codes, task=task))
    except TypeError:
        return None  # signature not yet the contract-b form
    except Exception:  # noqa: BLE001 - reranker failed -> fallback
        return None
    if sorted(order) != list(range(len(candidate_codes))):
        return None
    return [int(i) for i in order]


def _prefilter_bench_indices(candidate_codes: list[str], task, k: int) -> list[int]:
    """Indices of the top-``k`` candidates to actually bench (value prefilter, item 6).

    Ranks all generated candidates best-first via the value model
    (:func:`_value_rank_order`, contract b) and keeps only the top-``k`` for real
    compilation+benching — realizing the ~``num_candidates/k``× fewer benches. The
    rank order is turned into a scorer so selection routes through the pure,
    unit-tested :func:`value_prefilter` primitive. Falls back to the natural
    generation order when the reranker is unavailable/incompatible.
    """
    n = len(candidate_codes)
    if n == 0:
        return []
    order = _value_rank_order(candidate_codes, task) or list(range(n))
    rank_pos = {idx: pos for pos, idx in enumerate(order)}
    # higher score = ranked earlier (better); value_prefilter returns sorted top-k
    return value_prefilter(list(range(n)), lambda i: -rank_pos.get(i, n), k)


def apply_reward_phase(rr, config):
    """Correctness->latency curriculum masking (item 8).

    ``reward_phase == "correctness"`` masks the SPEED term: a correct kernel is
    credited exactly the correctness base (no speedup contribution) so the phase
    trains correctness only. ``"latency"``/``"all"`` keep the full
    correctness+speed reward. Incorrect/compile-fail/hack tiers are untouched
    (their signal is correctness). The campaign runs GRPO twice (correctness
    phase, then latency phase) by flipping ``reward_phase``.
    """
    from dataclasses import replace

    phase = getattr(config, "reward_phase", "all")
    if phase == "correctness" and getattr(rr, "correct", False):
        base = getattr(config, "correctness_weight", 0.3)
        return replace(rr, reward=base, speedup=None, tier="correct_masked")
    return rr


# --------------------------------------------------------------------------- #
# ToolRL reward compositing (agentic orchestration)
# --------------------------------------------------------------------------- #
def composite_agentic_reward(kernel_reward: float, tool_reward: float = 0.0,
                             tool_weight: float = 0.2) -> float:
    """Fold ToolRL-style tool-use shaping into the verifiable kernel reward.

    The kernel (correctness×speedup) term dominates; the tool term rewards
    well-formed tool calls / valid params / correct keep-revert decisions so the
    orchestration behavior is trainable without overwhelming the ground truth.
    """
    return kernel_reward + tool_weight * tool_reward


# --------------------------------------------------------------------------- #
# training entrypoint
# --------------------------------------------------------------------------- #
def train_grpo(config, tasks: Optional[list[str]] = None, backend: str = "inprocess"):
    """Run multi-turn GRPO natively on AMD; return the output checkpoint dir (str).

    KORE uses ONE self-contained in-process transformers+PEFT GRPO loop that
    rolls out against the verified :class:`KoreEnv` on local AMD GPUs — no server,
    no extra install, no config. ``backend`` is accepted for back-compat and
    always routes to this native loop.
    """
    log.info("grpo backend: native in-process (transformers+PEFT on AMD)",
             backend=backend, model=getattr(config, "model_id", None))
    return _train_grpo_inprocess(config, tasks)


def _sample_field(sample, idx, default=None):
    """Defensive positional field access on a GRPO sample tuple.

    A GRPO sample is ``(ret, gen_inputs, ref_logp, old_logp, n_tokens, sc_w)``;
    older/short tuples (used by unit tests) simply fall back to ``default``.
    """
    return sample[idx] if len(sample) > idx else default


def _accumulate_grpo_grads(kept_groups, logp_fn, *, ref_anchor_coef: float,
                           clip_ratio_low: float = 0.2, clip_ratio_high: float = 0.28,
                           variance_floor: float = 0.0, avspo_virtual_k: int = 2,
                           adv_eps: float = _EPS, backward: bool = True):
    """Micro-batched GRPO loss: one ``backward()`` per sample, grads accumulated.

    ``kept_groups`` is a list of groups (the StarPO-S-kept task groups); each
    group is a list of samples
    ``(return, gen_inputs, ref_logp, old_logp, n_tokens, sc_weight)`` (trailing
    fields optional). ``logp_fn(gen_inputs) -> tensor`` recomputes that sample's
    differentiable *token-mean* log-prob against the LIVE policy (or ``None`` to
    skip).

    Objective (P0 upgrades):
      * **DAPO clip-higher importance ratio** (item 1): with the detached
        rollout-time ``old_logp`` (token-mean), the turn-level geometric-mean
        (Turn-PPO) ratio ``r = exp(logp - old_logp)`` drives the asymmetric
        clipped surrogate ``-min(r*A, clip(r, 1-lo, 1+hi)*A)`` via
        :func:`clip_higher_ratio`. When ``old_logp`` is absent the term degrades
        to the ratio-free ``-A*logp`` (back-compat / vanilla PG).
      * **Global token-mean normalization** (item 3): instead of a per-sequence
        token-mean averaged over samples, the loss is
        ``sum_samples(n_tokens * per_sample_term) / sum_samples(n_tokens)`` — one
        global token-mean across the whole kept batch (DAPO length-debias).
      * **AVSPO variance floor** (item 4a): group advantages come from
        :func:`anticollapse.avspo_advantages` (virtual-sample injection when the
        group std < ``variance_floor``).
      * **SC-GRPO** (item 4b): a per-sample multiplicative ``sc_weight`` (from
        per-token KL(teacher||student)) scales the PG term.
      * k3 KL anchor ``ref_anchor_coef * (exp(d)-d-1)``, ``d = ref-logp`` (item 5
        keeps this active on the agentic path too).

    Scaling each term by ``1/total_tokens`` before ``backward()`` makes the
    accumulated gradient IDENTICAL to a single backward on the full global
    token-mean loss, while only one sample's graph is ever materialized
    (activation memory O(1 sample)). Returns ``(loss_value, n_terms)``.

    Kept free of model/tokenizer coupling (log-prob recompute injected via
    ``logp_fn``) so the equivalence is unit-testable on CPU.
    """
    import torch

    from kore.policy import anticollapse as ac

    # Pass 1: per-group advantages (AVSPO floor) + global token normalizer.
    group_advs: list[list[float]] = []
    n_terms = 0
    total_tokens = 0
    for samples in kept_groups:
        returns = [s[0] for s in samples]
        advs = ac.avspo_advantages(returns, variance_floor, avspo_virtual_k, adv_eps)
        group_advs.append(advs)
        for s in samples:
            if s[1]:  # non-empty gen_inputs -> a learnable sample
                n_terms += 1
                total_tokens += max(int(_sample_field(s, 4, 1) or 1), 1)
    if n_terms == 0 or total_tokens == 0:
        return 0.0, 0

    # Pass 2: recompute each sample's log-prob, backward the 1/total_tokens term.
    loss_value = 0.0
    for samples, advs in zip(kept_groups, group_advs):
        for adv, sample in zip(advs, samples):
            gen_inputs = sample[1]
            if not gen_inputs:
                continue
            lp = logp_fn(gen_inputs)
            if lp is None:
                continue
            n_tok = max(int(_sample_field(sample, 4, 1) or 1), 1)
            old_lp = _sample_field(sample, 3)
            if old_lp is not None:
                ratio = torch.exp(lp - old_lp)  # turn-level geometric-mean (Turn-PPO)
                pg = -clip_higher_ratio(ratio, adv, clip_ratio_low, clip_ratio_high)
            else:
                pg = -adv * lp  # vanilla PG (no stored old_logp)
            sc_w = _sample_field(sample, 5)
            if sc_w is not None:
                pg = pg * sc_w  # SC-GRPO KL-weight on the PG term (item 4b)
            term = pg
            ref_lp = _sample_field(sample, 2)
            if ref_anchor_coef and ref_lp is not None:
                d = ref_lp - lp  # differentiable k3 KL anchor: exp(d) - d - 1
                term = term + ref_anchor_coef * (torch.exp(d) - d - 1.0)
            scaled = term * n_tok / total_tokens  # global token-mean (item 3)
            if backward:
                scaled.backward()  # frees this sample's graph before the next one
            loss_value += float(scaled.detach())
    return loss_value, n_terms


def _grpo_step_adv_stats(kept_groups, config):
    """Read-only recompute of mean|advantage| + KL-anchored sample count (logging).

    Mirrors pass-1 of :func:`_accumulate_grpo_grads` exactly (AVSPO variance-floor
    advantages) so the logged ``adv_absmean`` is faithful, without touching the
    training path or its return. Pure/CPU-cheap.
    """
    from kore.policy import anticollapse as ac

    tau = getattr(config, "variance_floor", 0.0)
    k = getattr(config, "avspo_virtual_k", 2)
    eps = getattr(config, "adv_eps", _EPS)
    total = 0.0
    n = 0
    n_kl = 0
    for samples in kept_groups:
        returns = [s[0] for s in samples]
        advs = ac.avspo_advantages(returns, tau, k, eps)
        for a, s in zip(advs, samples):
            total += abs(a)
            n += 1
            if len(s) > 2 and s[2] is not None:
                n_kl += 1
    return (total / n if n else 0.0), n_kl


def _grpo_step_kl_stat(kept_groups):
    """Mean k3 KL(policy||ref) diagnostic over kept samples (logging only).

    Wires the pure :func:`kl_k3` estimator: for every kept sample carrying both a
    detached rollout ``old_logp`` and a frozen-ref token-mean logp, accumulate
    ``exp(d)-d-1`` (``d = ref - old_logp``). Returns ``None`` when no sample has a
    ref (KL anchor inactive) so the logger records a null rather than 0.
    """
    total, n = 0.0, 0
    for samples in kept_groups:
        for s in samples:
            ref_lp = _sample_field(s, 2)
            old_lp = _sample_field(s, 3)
            if ref_lp is None or old_lp is None:
                continue
            total += kl_k3(float(old_lp), float(ref_lp))
            n += 1
    return (total / n) if n else None


def _rc_variance_floor_met(group) -> bool:
    """RC-GRPO variance-floor diagnostic — wires :func:`anticollapse.variance_floor`.

    Estimates the per-mode conditional means from the group's trajectory scores +
    reward-control tokens, then checks the realized group variance against the
    RC-GRPO floor ``(G-1)/G * p(1-p) * eps^2``. Used for logging only.
    """
    from kore.policy import anticollapse as ac

    rtoks = group.get("rtoks") or []
    scores = group.get("traj_scores") or []
    if len(rtoks) != len(scores) or not scores or any(t is None for t in rtoks):
        return False
    means: dict[str, float] = {}
    for name in set(rtoks):
        vals = [s for s, t in zip(scores, rtoks) if t == name]
        if vals:
            means[name] = sum(vals) / len(vals)
    return ac.variance_floor(scores, rtoks, means)


def _safe_seed_code(task) -> str:
    """Seed-kernel source for a task, or '' (used as the GTPO code-sim reference)."""
    try:
        return task.seed_source or ""
    except Exception:  # noqa: BLE001 - tasks without a seed file
        return ""


def _token_logp_dist(model, prompt_ids, gen_ids, temperature: float = 1.0):
    """Per-token log-softmax distribution over the vocab for ``gen_ids`` (GPU path)."""
    import torch

    full = torch.cat([prompt_ids[0], gen_ids]).unsqueeze(0)
    out = model(full)
    logits = out.logits[0, prompt_ids.shape[1] - 1:-1, :]
    if temperature and temperature > 0:
        logits = logits / temperature
    return torch.log_softmax(logits, dim=-1)


def _scgrpo_weight(model, tok, gen_inputs, demo_text, config):
    """Real SC-GRPO per-sample PG weight from per-token KL(teacher||student) (item 4b).

    The teacher is the SAME policy conditioned on a correct kernel used as an
    in-context demo (``demo_text`` prepended to the sample's prompt); the student
    is the policy on the original prompt. One extra (teacher) forward per weighted
    sample. Per-token ``KL(teacher||student) = sum_v p_teacher (logp_teacher -
    logp_student)`` is aggregated to a bounded multiplicative weight via
    :func:`anticollapse.scgrpo_weight_from_kl`. GPU-only; the caller guards it.
    """
    import torch

    from kore.policy import anticollapse as ac

    if not gen_inputs:
        return None
    prompt_ids, gen_ids = gen_inputs[0]
    student_lp = _token_logp_dist(model, prompt_ids, gen_ids, config.temperature)
    demo_ids = tok(demo_text, return_tensors="pt").input_ids.to(prompt_ids.device)
    teacher_prompt = torch.cat([prompt_ids[0], demo_ids[0]]).unsqueeze(0)
    teacher_lp = _token_logp_dist(model, teacher_prompt, gen_ids, config.temperature)
    pt = teacher_lp.exp()
    token_kls = (pt * (teacher_lp - student_lp)).sum(dim=-1)  # per-token KL(teacher||student)
    # The SC-GRPO weight is a DETACHED scalar multiplier on the PG term, so read
    # the KLs off the graph (also silences a requires_grad->scalar warning).
    return ac.scgrpo_weight_from_kl([float(x) for x in token_kls.detach()], scale=1.0,
                                    w_min=config.sc_grpo_w_min, w_max=config.sc_grpo_w_max)


def _activate_value_ranker(config):
    """Fix 1: install the TRAINED value model for the bench prefilter (or heuristic).

    ``config.value_model_path`` was a dead flag: :func:`_value_rank_order` calls
    ``rank_candidates(codes, task=task)`` with NO model, so the prefilter always
    fell back to the heuristic cold-start ranker. Here, at the start of the loop,
    when ``value_prefilter`` is on and a ``value_model_path`` is set, we
    ``load_default_model(path)`` (which ``set_default_model(...)`` installs) so
    ``rank_candidates`` routes through the TRAINED model. A missing/unset path
    degrades gracefully to the heuristic ranker — logged clearly, never silently.
    Returns the installed model (or ``None`` for the heuristic fallback).
    """
    if not getattr(config, "value_prefilter", False):
        return None
    try:
        from kore.value.rerank import load_default_model
    except Exception as e:  # noqa: BLE001 - value module unavailable
        print("[grpo] value-prefilter: heuristic cold-start fallback (value module unavailable)")
        log.warn("value-prefilter ranker: value module unavailable — heuristic fallback",
                 error=repr(e), ranker="heuristic")
        return None
    vpath = getattr(config, "value_model_path", None)
    model = load_default_model(vpath) if vpath else None
    if model is not None:
        print(f"[grpo] value-prefilter: TRAINED value model loaded from {vpath}")
        log.info("value-prefilter ranker: TRAINED value model active",
                 value_model_path=vpath, ranker="trained")
    else:
        reason = "value_model_path unset" if not vpath else f"load failed for {vpath!r}"
        print(f"[grpo] value-prefilter: heuristic cold-start fallback ({reason})")
        log.info("value-prefilter ranker: heuristic cold-start fallback",
                 value_model_path=vpath, ranker="heuristic", reason=reason)
    return model


def _model_dtype(config):
    """Fix 3 (bf16): honor ``config.bf16`` for the model dtype (was hardcoded)."""
    import torch

    return torch.bfloat16 if getattr(config, "bf16", True) else torch.float32


def _build_lr_scheduler(opt, config):
    """Fix 3 (scheduler): torch LR warmup + decay honoring the config flags.

    Wires the previously-dead ``lr_scheduler_type`` / ``warmup_ratio``: a linear
    warmup over ``warmup_ratio*total_steps`` steps, then ``constant`` (default),
    ``linear``, or ``cosine`` decay to 0 over the remaining steps. Pure torch —
    the native loop has no HF ``Trainer`` to own a scheduler. Stepped ONCE per
    optimizer step (i.e. per training step that actually updated).
    """
    import math

    import torch

    total = max(1, int(getattr(config, "total_steps", 1)))
    warmup = max(0, int(round(float(getattr(config, "warmup_ratio", 0.0)) * total)))
    sched = (getattr(config, "lr_scheduler_type", "constant") or "constant").lower()

    def lr_lambda(step):
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(warmup + 1)  # linear warmup
        progress = (step - warmup) / max(1, total - warmup)
        progress = min(max(progress, 0.0), 1.0)
        if sched == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if sched in ("linear", "linear_decay"):
            return 1.0 - progress
        return 1.0  # "constant" (and any unknown type) -> flat LR after warmup

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def _truncate_prompt_ids(ids, config):
    """Fix 3 (max_prompt_length): left-truncate a rendered prompt to the budget.

    Wires the previously-dead ``max_prompt_length``: keeps the most recent
    (rightmost) tokens — including the trailing generation prompt — so a long
    multi-turn context can't exceed the model's prompt budget. No-op when unset.
    """
    max_len = int(getattr(config, "max_prompt_length", 0) or 0)
    if max_len > 0 and ids.shape[1] > max_len:
        return ids[:, -max_len:]
    return ids


def _save_grpo_checkpoint(model, tok, config, step):
    """Fix 3 (save_steps): write a periodic checkpoint (never fatal on failure)."""
    import os

    ckpt = os.path.join(getattr(config, "output_dir", "runs/grpo"), f"checkpoint-{step}")
    try:
        model.save_pretrained(ckpt)
        tok.save_pretrained(ckpt)
        log.info("grpo periodic checkpoint saved", step=step, path=ckpt)
    except Exception as e:  # noqa: BLE001 - a checkpoint failure must not kill training
        log.warn("grpo periodic checkpoint failed", step=step, error=repr(e))
    return ckpt


def _train_grpo_fallback(config, tasks):
    """Compact in-process GRPO loop implementing the locked KORE recipe.

    Per step:
      1. DAPO dynamic sampling (oversample-and-refill) collects ``target_groups``
         non-degenerate groups — serial multi-turn refinement OR the agentic tool
         harness — both flattened to per-turn Kevin-credit samples (correctness-
         gated gamma returns, infra turns dropped);
      2. anti-collapse shaping: GTPO code-similarity partial reward for all-fail
         groups + real SC-GRPO KL-weighting for partial-solve groups;
      3. StarPO-S selects the highest-variance groups to train on;
      4. a GLOBAL TOKEN-MEAN DAPO clip-higher importance-ratio surrogate
         (``-min(r*A, clip(r,1-lo,1+hi)*A)``, ``r = exp(logp-old_logp)``) with an
         AVSPO variance floor and a differentiable k3 KL anchor to the frozen
         SFT/reference checkpoint, run for ``ppo_epochs`` minibatch passes that
         reuse the detached rollout ``old_logp``.

    Full-FT vs LoRA follows ``config.use_lora`` (the locked recipe is full-FT).
    When LoRA is used the adapter is merged before saving so soup/serve load a
    full model. Gradient checkpointing + PEFT needs ``enable_input_require_grads``
    or ``.backward()`` sees no grad path.

    When ``config.distributed`` full-FT is requested (``use_lora=False`` under an
    ``accelerate launch`` process group) this dispatches to
    :func:`_train_grpo_distributed`, which shards the policy + reference across all
    ranks (ZeRO-3-equivalent) so no full 14B replica ever lives on one GPU. Every
    single-process / LoRA / CPU path below is left byte-for-byte unchanged.
    """
    from kore.policy.configs import grpo_distributed_enabled

    if grpo_distributed_enabled(config):
        return _train_grpo_distributed(config, tasks)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kore.env.kore_env import KoreEnv
    from kore.policy import anticollapse as ac
    from kore.policy.configs import fsdp_enabled
    from kore.tasks.registry import get_task, task_ids

    tasks = tasks or task_ids()
    configure(run_dir=getattr(config, "output_dir", None))
    t_start = time.time()
    last_mean_r = None
    # Fix 4: full-FT GRPO launched distributed (distributed=True, use_lora=False)
    # must NOT use device_map="auto" — accelerate/FSDP owns placement (same as
    # sft.py). LoRA / single-GPU / CPU runs keep the legacy device_map path.
    use_fsdp = fsdp_enabled(config)
    log.info("grpo fallback: starting", model=config.model_id, total_steps=config.total_steps,
             agentic=bool(config.agentic), use_lora=bool(config.use_lora), n_tasks=len(tasks),
             tasks_per_step=max(1, getattr(config, "tasks_per_step", 1)),
             num_trajectories=config.num_trajectories, num_turns=config.num_turns,
             starpo_s=bool(config.starpo_s), ref_anchor_coef=config.ref_anchor_coef,
             distributed=bool(getattr(config, "distributed", False)), fsdp=bool(use_fsdp),
             bf16=bool(getattr(config, "bf16", True)), **gpu_mem_snapshot())
    # Fix 1: install the TRAINED value model (or the logged heuristic fallback)
    # BEFORE any rollout, so the value prefilter actually reranks with the model.
    _activate_value_ranker(config)
    tok = AutoTokenizer.from_pretrained(config.model_id)
    from kore.policy.configs import preferred_attn_impl
    model_kwargs = {"torch_dtype": _model_dtype(config),  # Fix 3: honor bf16
                    "attn_implementation": preferred_attn_impl()}
    if not use_fsdp:
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
    model.config.use_cache = False
    if getattr(config, "gradient_checkpointing", True):
        # REENTRANT layer-internal checkpointing: robust to the intermittent
        # flash_attention_2 -> SDPA per-worker downgrade (reentrant skips the
        # saved-tensor-count check that raises CheckpointError under kernel swaps).
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        model.enable_input_require_grads()  # critical for PEFT + grad-ckpt

    if config.use_lora:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(
            r=config.lora.r, lora_alpha=config.lora.lora_alpha, lora_dropout=0.0,
            target_modules=list(config.lora.target_modules), task_type="CAUSAL_LM"))
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=config.learning_rate)
    sched = _build_lr_scheduler(opt, config)  # Fix 3: real LR warmup + scheduler

    # ---- ref-model (KL anchor) gating: only pay the ~ref-model memory when a KL
    #      anchor is actually applied. The per-turn KL anchor is now ACTIVE on the
    #      agentic path too (item 5): ``_rollout_agentic`` records per-assistant-turn
    #      (prompt_ids, gen_ids), so a per-turn frozen-ref log-prob CAN be computed
    #      exactly as in the serial path. We therefore load the ref whenever
    #      ref_anchor_coef>0, regardless of agentic mode.
    want_ref = getattr(config, "ref_anchor_coef", 0.0) > 0
    ref_model = None
    if not want_ref:
        print("[grpo] ref_anchor_coef<=0: skipping frozen ref-model load (no KL anchor).")
        log.info("ref-model: skipped", reason="ref_anchor_coef<=0", kl_anchor="inactive")
    else:
        ref_model = _load_ref_model(config)  # frozen KL anchor (or None if unavailable)
        log.info("ref-model: loaded" if ref_model is not None else "ref-model: unavailable",
                 kl_anchor="active" if ref_model is not None else "inactive",
                 agentic=bool(config.agentic),
                 ref_checkpoint=config.ref_checkpoint or config.model_id)

    tasks_per_step = max(1, getattr(config, "tasks_per_step", 1))
    target_groups = getattr(config, "target_groups", None) or tasks_per_step
    dyn = bool(getattr(config, "dynamic_sampling", True)) and bool(config.starpo_s)
    max_attempts = getattr(config, "max_sampling_attempts", None) or (3 * target_groups if dyn else tasks_per_step)
    ppo_epochs = max(1, getattr(config, "ppo_epochs", 1))
    task_cursor = 0

    def _one_group(task, seed):
        """Roll out one task-group -> a dict with per-turn Kevin-credit samples.

        Serial and agentic rollouts are unified: both yield the same per-turn dict
        (rewards/correct/infra/gen_inputs/ref_logps/old_logps/n_tokens/codes), which
        is flattened into ``m*n`` per-turn samples via :func:`build_kevin_samples`
        (infra turns dropped). Samples are MUTABLE lists so GTPO code-sim shaping
        (item 4c) and SC-GRPO KL-weighting (item 4b) can be applied in-place before
        the loss.
        """
        env = KoreEnv(task)
        G = config.num_trajectories
        rtoks = ac.sample_reward_tokens(G, config.rc_p_high, seed=seed) if config.rc_grpo else [None] * G
        traj_scores: list[float] = []
        traj_rewards, traj_correct, traj_infra = [], [], []
        turn_inputs, turn_ref, turn_old, turn_ntok, turn_codes = [], [], [], [], []
        for g in range(G):
            if config.agentic:
                d = _rollout_agentic(model, tok, env, task, config, ref_model)
            else:
                d = _rollout(model, tok, env, task, config, rtoks[g], ref_model)
            traj_rewards.append(d["rewards"]); traj_correct.append(d["correct"])
            traj_infra.append(d["infra"]); turn_inputs.append(d["gen_inputs"])
            turn_ref.append(d["ref_logps"]); turn_old.append(d["old_logps"])
            turn_ntok.append(d["n_tokens"]); turn_codes.append(d["codes"])
            traj_scores.append(
                kevin_trajectory_score(d["rewards"], d["correct"])
                if config.kevin_best_kernel_scoring else (max(d["rewards"]) if d["rewards"] else 0.0))
            log.debug("rollout", task=task.task_id, traj=g, turns=len(d["rewards"]),
                      best_reward=traj_scores[-1], correct_turns=sum(1 for c in d["correct"] if c))
        returns, index = build_kevin_samples(traj_rewards, traj_correct, config.gamma, traj_infra=traj_infra)
        samples, codes = [], []
        for (ti, tu), ret in zip(index, returns):
            samples.append([ret, turn_inputs[ti][tu], turn_ref[ti][tu],
                            turn_old[ti][tu], turn_ntok[ti][tu], None])
            codes.append(turn_codes[ti][tu])
        correct_kernels = [turn_codes[ti][tu] for ti in range(G) for tu in range(len(traj_correct[ti]))
                           if traj_correct[ti][tu] and turn_codes[ti][tu]]
        return {"task": task, "traj_scores": traj_scores, "samples": samples, "codes": codes,
                "correct_kernels": correct_kernels, "any_correct": any(any(c) for c in traj_correct),
                "rtoks": rtoks, "infra": sum(1 for inf in traj_infra for x in inf if x)}

    for step in range(config.total_steps):
        # ---- 1. DAPO dynamic sampling: OVERSAMPLE-AND-REFILL (item 2) ---- #
        # Instead of drop-and-shrink (roll exactly tasks_per_step, then drop the
        # collapsed groups), keep rolling task-groups until ``target_groups``
        # NON-DEGENERATE (std>min_std) groups are collected, bounded by
        # ``max_attempts``. This keeps the effective batch size stable under
        # StarPO-S so a bad step no longer shrinks the update.
        def _roll(attempt):
            nonlocal task_cursor
            task = get_task(tasks[task_cursor % len(tasks)])
            task_cursor += 1
            return _one_group(task, step * 100003 + attempt)

        groups, attempts = dynamic_sampling_refill(
            _roll, target_groups, min_std=config.starpo_min_std,
            max_attempts=max_attempts, dynamic=dyn,
            std_key=lambda g: group_reward_std(g["traj_scores"]))
        if not groups:
            print(f"[grpo] step {step}: no non-degenerate groups in {attempts} attempts; skip")
            log.info("grpo step: dynamic-sampling exhausted — skipping", step=step,
                     attempts=attempts, target_groups=target_groups, reason="all groups collapsed")
            log.progress(step + 1, config.total_steps, "grpo", t_start=t_start)
            continue
        step_infra = sum(g["infra"] for g in groups)

        # ---- 1b. GTPO all-fail code-similarity shaping (item 4c) ---- #
        # For an all-fail group (no correct kernel -> Kevin returns all 0), replace
        # each sample's return with a graded partial reward = code shingle-cosine
        # similarity to the nearest correct kernel seen this step (or the seed).
        if config.gtpo_codesim:
            step_refs = [k for g in groups for k in g["correct_kernels"]]
            for g in groups:
                if g["any_correct"] or not g["samples"]:
                    continue
                refs = step_refs or ([_safe_seed_code(g["task"])] if _safe_seed_code(g["task"]) else [])
                partial = ac.gtpo_codesim_shaping(g["codes"], refs, config.gtpo_codesim_scale)
                for s, p in zip(g["samples"], partial):
                    s[0] = p

        # ---- 1c. Real SC-GRPO KL-weighting for partial-solve groups (item 4b) ---- #
        # GPU-only: one extra (demo-conditioned) forward per weighted sample; the
        # per-token KL(teacher||student) is aggregated to a bounded multiplicative
        # PG weight (:func:`anticollapse.scgrpo_weight_from_kl`). Guarded so a
        # failure degrades to weight 1.0 (plain PG).
        if config.sc_grpo:
            for g in groups:
                # partial-solve groups: at least one correct kernel to use as the
                # demo AND at least one non-correct sample to pull toward it.
                if not g["correct_kernels"] or all(s[0] > 0 for s in g["samples"]):
                    continue
                demo = g["correct_kernels"][0]
                for s in g["samples"]:
                    try:
                        s[5] = _scgrpo_weight(model, tok, s[1], demo, config)
                    except Exception:  # noqa: BLE001 - GPU path only; degrade to plain PG
                        s[5] = None

        group_rewards = [g["traj_scores"] for g in groups]
        group_samples = [g["samples"] for g in groups]
        group_tasks = [g["task"] for g in groups]

        # RC-GRPO variance-floor diagnostic (wire ac.variance_floor).
        if config.rc_grpo:
            met = sum(1 for g in groups if _rc_variance_floor_met(g))
            log.debug("rc_variance_floor", step=step, groups=len(groups), met=met)

        # ---- 2. StarPO-S selector: train on the highest-variance groups ---- #
        if config.starpo_s:
            keep = sorted(starpo_select_high_variance(
                group_rewards, config.starpo_keep_frac, config.starpo_min_std))
            if not keep:
                print(f"[grpo] step {step}: all {len(groups)} groups collapsed; skip")
                log.info("grpo step: all groups collapsed — skipping", step=step,
                         n_groups=len(group_rewards), reason="zero reward-variance (StarPO-S)",
                         keep_frac=config.starpo_keep_frac, min_std=config.starpo_min_std)
                log.progress(step + 1, config.total_steps, "grpo", t_start=t_start)
                continue
        else:
            keep = list(range(len(group_rewards)))
        log.debug("starpo_keep", step=step, kept=len(keep), of=len(group_rewards),
                  keep_frac=config.starpo_keep_frac, indices=keep, attempts=attempts)

        # ---- 3. MICRO-BATCHED clip-higher loss, multi-epoch (items 1, 3) ---- #
        # One backward() per sample (recomputing that sample's log-prob), grads
        # accumulated, one optimizer.step() per PPO epoch. Each term is scaled by
        # 1/total_tokens so the accumulated gradient equals a single backward on
        # the global token-mean loss, while only ONE sample's graph is ever alive.
        # ``ppo_epochs`` minibatch passes reuse the detached rollout ``old_logp``.
        kept_groups = [group_samples[gi] for gi in keep]

        def _logp_fn(gen_inputs):
            return _recompute_logp(model, tok, gen_inputs, config.temperature) if gen_inputs else None

        loss_value, n_terms, grad_norm = 0.0, 0, None
        for _epoch in range(ppo_epochs):
            opt.zero_grad()
            loss_value, n_terms = _accumulate_grpo_grads(
                kept_groups, _logp_fn,
                ref_anchor_coef=config.ref_anchor_coef,
                clip_ratio_low=config.clip_ratio_low, clip_ratio_high=config.clip_ratio_high,
                variance_floor=config.variance_floor, avspo_virtual_k=config.avspo_virtual_k,
                adv_eps=config.adv_eps)
            if n_terms == 0:
                break
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
            opt.step()
        if n_terms == 0:
            print(f"[grpo] step {step}: no learnable samples; skip")
            log.info("grpo step: no learnable samples — skipping", step=step,
                     n_kept_groups=len(keep), reason="all kept samples had empty gen_inputs")
            log.progress(step + 1, config.total_steps, "grpo", t_start=t_start)
            continue
        sched.step()  # Fix 3: advance the LR schedule once per real training step
        mean_r = sum(sum(g) / len(g) for g in group_rewards if g) / max(len(group_rewards), 1)
        last_mean_r = mean_r
        print(f"[grpo] step {step} kept={len(keep)}/{len(groups)} attempts={attempts} "
              f"epochs={ppo_epochs} meanR={mean_r:.3f} loss={loss_value:.4f}")

        # ---- observability: per-step metrics, emitted every logging_steps (Fix 3) ---- #
        log_every = max(1, int(getattr(config, "logging_steps", 1) or 1))
        if step % log_every == 0 or step == config.total_steps - 1:
            reward_flat = [s for g in group_rewards for s in g]
            reward_mean = sum(reward_flat) / len(reward_flat) if reward_flat else 0.0
            reward_std = group_reward_std(reward_flat)
            adv_absmean, n_kl_samples = _grpo_step_adv_stats(kept_groups, config)
            kl_val = _grpo_step_kl_stat(kept_groups)  # mean k3 KL(policy||ref) diagnostic
            try:
                grad_norm_val = float(grad_norm) if grad_norm is not None else None
            except Exception:  # noqa: BLE001 - some torch builds return a 0-dim tensor
                grad_norm_val = None
            lr_val = opt.param_groups[0]["lr"] if opt.param_groups else config.learning_rate
            log.event("grpo_step", step=step,
                      task=",".join(sorted({t.task_id for t in group_tasks})),
                      n_groups=len(group_rewards), n_kept_groups=len(keep), n_attempts=attempts,
                      ppo_epochs=ppo_epochs,
                      n_samples=sum(len(s) for s in group_samples), n_infra_dropped=step_infra,
                      reward_mean=reward_mean, reward_std=reward_std, adv_absmean=adv_absmean,
                      # Fix 3: the active anchor is ref_anchor_coef — the log used to
                      # mislabel it as ``kl_coef`` (a flag that never existed here).
                      kl=kl_val, ref_anchor_coef=config.ref_anchor_coef, n_kl_samples=n_kl_samples,
                      loss=loss_value, grad_norm=grad_norm_val, lr=lr_val, **gpu_mem_snapshot())
        # Fix 3: periodic checkpoint every save_steps (skips the final step — the
        # full model is saved below on loop exit).
        save_every = int(getattr(config, "save_steps", 0) or 0)
        if save_every > 0 and (step + 1) % save_every == 0 and step != config.total_steps - 1:
            _save_grpo_checkpoint(model, tok, config, step + 1)
        log.progress(step + 1, config.total_steps, "grpo", t_start=t_start)

    # Merge LoRA into the base before saving so soup/serve load a full model.
    out = config.output_dir
    if config.use_lora:
        log.info("saving: merging LoRA adapter into base", out=out)
        merged = model.merge_and_unload()
        merged.save_pretrained(out)
    else:
        log.info("saving: full-FT weights", out=out)
        model.save_pretrained(out)
    tok.save_pretrained(out)
    log.metric("grpo_done", steps=config.total_steps, mean_reward_last=last_mean_r, out=out,
               **gpu_mem_snapshot())
    return out


# Back-compat name kept; ``_train_grpo_inprocess`` is the preferred, first-class
# name for the in-process transformers+PEFT GRPO backend.
_train_grpo_inprocess = _train_grpo_fallback


# --------------------------------------------------------------------------- #
# FULL-PARAMETER SHARDED GRPO (distributed, ZeRO-3-equivalent)
#
# Sharding approach — chosen: torch FSDP FULL_SHARD (ZeRO-3-equivalent), with
# DeepSpeed ZeRO-3 fully wired and selectable (``sharding_backend="deepspeed"``).
# Why FSDP is the default for THIS loop (documented, deliberate):
#   * The KORE objective uses an O(1-sample) MICRO-BATCHED backward — many
#     per-sample ``backward()`` calls accumulate into ``.grad``, then ONE
#     ``optimizer.step()`` per PPO epoch (activation memory O(1 sample)). FSDP
#     reduce-scatters + accumulates grads on each backward and lets us own the
#     ``step()`` boundary; DeepSpeed's engine instead couples backward/step to a
#     FIXED ``gradient_accumulation_steps`` counter, which fights a dynamic
#     per-step sample count. FSDP maps onto the recipe with zero contortion.
#   * FSDP is torch-native -> guaranteed ROCm/gfx942 support (no compiled ops to
#     build, unlike DeepSpeed's fused/CPU-Adam kernels).
#   * It reuses the SAME FULL_SHARD recipe the KORE SFT/DPO stages already run.
# Generation robustness: both FSDP and ZeRO-3 GATHER each wrapped block's params
# per-forward and reshard after, so ``model.generate`` works out of the box
# (every decode step re-gathers). ``synced_gpus=True`` keeps ranks in lockstep on
# ragged completion lengths. The frozen REFERENCE is prepared as a sharded eval
# model (never a full replica per GPU). Cross-rank rewards are all-gathered so the
# group-relative GRPO baseline is over the FULL group (all trajectories/ranks).
# --------------------------------------------------------------------------- #
def merge_across_ranks(per_rank: list[list]) -> list:
    """Flatten a per-rank list-of-lists into one global list (rank-ordered)."""
    return [x for chunk in per_rank for x in chunk]


def distributed_group_advantages(per_rank_returns: list[list[float]],
                                 variance_floor: float = 0.0,
                                 avspo_virtual_k: int = 2,
                                 adv_eps: float = _EPS) -> list[list[float]]:
    """Global (cross-rank) group-relative advantages, split back per rank.

    ``per_rank_returns[r]`` is rank ``r``'s per-turn Kevin returns for ONE rollout
    group (the trajectories that rank rolled out). GRPO's baseline MUST be over the
    FULL group — every trajectory on every rank — so the returns are concatenated
    (rank order), normalized ONCE via the AVSPO variance-floor advantages
    (:func:`anticollapse.avspo_advantages`, ``tau=variance_floor``), and the
    resulting advantages are sliced back to each rank in the SAME order. Each rank
    therefore trains its own trajectories but with a globally-correct advantage.

    Pure (no torch/dist) so the cross-rank normalization math is unit-testable by
    simulating each rank's rewards.
    """
    from kore.policy import anticollapse as ac

    flat = merge_across_ranks(per_rank_returns)
    advs = ac.avspo_advantages(flat, variance_floor, avspo_virtual_k, adv_eps)
    out: list[list[float]] = []
    i = 0
    for chunk in per_rank_returns:
        out.append(advs[i:i + len(chunk)])
        i += len(chunk)
    return out


def _all_gather_object(obj, accelerator=None) -> list:
    """All-gather a picklable python object into a rank-ordered list.

    Resolution order (so the SAME call is correct on 1 rank, on the 2-proc gloo
    smoke, and on a real 8xMI300 ``accelerate launch``):
      1. the live default ``torch.distributed`` group (real GPU launch + the gloo
         smoke both init it) -> ``all_gather_object``;
      2. else, when an Accelerator reporting >1 processes is given, its managed
         group via ``accelerate.utils.gather_object`` (belt-and-suspenders for
         backends where the default group isn't the one in use);
      3. else identity ``[obj]`` (single process / uninitialized / CPU / tests).
    """
    try:
        import torch.distributed as dist
    except Exception:  # noqa: BLE001 - torch not available (pure CPU tests)
        dist = None
    if dist is not None and dist.is_available() and dist.is_initialized() \
            and dist.get_world_size() > 1:
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, obj)
        return list(gathered)
    if accelerator is not None and getattr(accelerator, "num_processes", 1) > 1:
        try:
            from accelerate.utils import gather_object

            gathered = list(gather_object([obj]))  # flat, rank-ordered
            if len(gathered) == accelerator.num_processes:
                return gathered
        except Exception:  # noqa: BLE001 - fall through to the identity return
            pass
    return [obj]


def _rank_slice(n: int, rank: int, world: int) -> list[int]:
    """Indices of the ``n`` items this ``rank`` owns under a strided partition.

    Strided (``rank, rank+world, ...``) so, with the Kevin group size ``G`` a
    multiple of ``world``, every rank rolls out exactly ``G/world`` trajectories
    of the SAME task-group in parallel — the group is split across ranks and its
    reward baseline is reconstructed by the cross-rank gather.
    """
    return list(range(rank, n, max(1, world)))


def build_fsdp_plugin(config):
    """Build an accelerate ``FullyShardedDataParallelPlugin`` (FULL_SHARD/ZeRO-3-eq).

    Shards params + grads + optimizer state across ranks, wraps one transformer
    decoder block per FSDP unit (auto-detected from ``model_id`` when
    ``fsdp_transformer_layer_cls`` is unset), reshards after forward (so
    ``model.generate`` re-gathers per decode step), and routes activation
    checkpointing through FSDP. ``cpu_offload`` moves params+optim to host RAM for
    32B/70B. Heavy import kept local so ``import kore.policy.grpo`` stays torch-free.
    """
    from accelerate import FullyShardedDataParallelPlugin

    from kore.policy.configs import detect_transformer_layer_cls

    layer_cls = getattr(config, "fsdp_transformer_layer_cls", None) or detect_transformer_layer_cls(
        getattr(config, "model_id", ""))
    version = int(getattr(config, "fsdp_version", 1) or 1)
    # FSDP1 expects the sharding strategy STRING; FSDP2 takes a bool. Both mean
    # "reshard params after forward" == FULL_SHARD == ZeRO-3-equivalent.
    reshard = True if version >= 2 else "FULL_SHARD"
    return FullyShardedDataParallelPlugin(
        fsdp_version=version,
        reshard_after_forward=reshard,              # FULL_SHARD (ZeRO-3 equivalent)
        auto_wrap_policy="transformer_based_wrap",
        transformer_cls_names_to_wrap=[layer_cls],
        cpu_offload=bool(getattr(config, "cpu_offload", False)),
        # Activation checkpointing is enabled on the MODEL (HF gradient_checkpointing,
        # use_reentrant=False) BEFORE accelerator.prepare — NOT here. The FSDP
        # plugin's external checkpoint_wrapper mismatches saved-tensor counts on an
        # FSDP1/use_orig_params unit (torch.utils.checkpoint CheckpointError).
        activation_checkpointing=False,
        use_orig_params=True,
        sync_module_states=True,
        cpu_ram_efficient_loading=True,
        limit_all_gathers=True,
        state_dict_type="FULL_STATE_DICT",          # gather a plain ckpt for soup/serve
    )


def build_deepspeed_plugin(config):
    """Build an accelerate ``DeepSpeedPlugin`` from the synthesized ZeRO config.

    ZeRO-3 shards params+grads+optim and gathers params per-forward (so rollout
    ``model.generate`` works on the sharded engine — the online-RL property). The
    ZeRO config comes from :func:`kore.policy.configs.build_deepspeed_config`
    (or the user's ``ds_config`` JSON). Heavy import kept local.
    """
    from accelerate import DeepSpeedPlugin

    from kore.policy.configs import build_deepspeed_config

    ds_cfg = build_deepspeed_config(config)
    return DeepSpeedPlugin(
        hf_ds_config=ds_cfg,
        zero_stage=int(getattr(config, "zero_stage", 3)),
        gradient_accumulation_steps=1,
        gradient_clipping=float(getattr(config, "max_grad_norm", 1.0)),
        offload_optimizer_device="cpu" if getattr(config, "cpu_offload", False) else None,
        offload_param_device="cpu" if getattr(config, "cpu_offload", False) else None,
        zero3_save_16bit_model=True,
    )


def build_grpo_accelerator(config):
    """Construct the ``accelerate.Accelerator`` for the sharded GRPO run.

    Picks the plugin from :func:`kore.policy.configs.grpo_sharding_backend`
    (``"fsdp"`` default, ``"deepspeed"`` opt-in) and requests bf16 mixed precision
    when ``config.bf16``. Heavy import kept local so the module import stays light.
    """
    from accelerate import Accelerator

    from kore.policy.configs import grpo_sharding_backend

    backend = grpo_sharding_backend(config)
    mp = "bf16" if getattr(config, "bf16", True) else "no"
    if backend == "deepspeed":
        return Accelerator(mixed_precision=mp, deepspeed_plugin=build_deepspeed_plugin(config))
    return Accelerator(mixed_precision=mp, fsdp_plugin=build_fsdp_plugin(config))


def _dummy_gen_inputs(tok, device):
    """A trivial ``[(prompt_ids, gen_ids)]`` for a padding (lockstep) forward.

    Used when a rank has FEWER learnable samples than its peers: it still must
    issue the SAME number of collective forwards (ZeRO-3/FSDP all-gather per
    forward), so it runs this 1-token forward whose loss is scaled by 0.
    """
    import torch

    ids = tok("x", return_tensors="pt").input_ids.to(device)
    return [(ids, ids[0][:1])]


def _accumulate_grpo_grads_distributed(local_terms, logp_fn, *, accelerator,
                                       global_total_tokens: int, grad_scale: float,
                                       max_micro_steps: int, ref_anchor_coef: float,
                                       clip_ratio_low: float, clip_ratio_high: float,
                                       tok, device):
    """Sharded O(1-sample) micro-batched backward, kept in LOCKSTEP across ranks.

    ``local_terms`` is this rank's list of ``(advantage, sample)`` pairs (sample =
    ``[ret, gen_inputs, ref_logp, old_logp, n_tokens, sc_weight]``). Each real
    sample recomputes its token-mean log-prob against the SHARDED live policy
    (``logp_fn`` -> a collective forward), forms the DAPO clip-higher surrogate
    ``-min(r*A, clip(r,1-lo,1+hi)*A)`` (+ k3 KL anchor), scales by
    ``n_tok / global_total_tokens`` for a GLOBAL token-mean over the whole batch
    across ALL ranks, and backprops via ``accelerator.backward`` (FSDP/DeepSpeed
    reduce-scatter accumulates into the sharded grad).

    Two distributed-correctness details:
      * **Grad scale** (``grad_scale = world_size``): FSDP/ZeRO AVERAGE grads over
        ranks, but the objective is a global SUM over all per-sample terms divided
        by the global token count. Multiplying each term by ``world_size`` converts
        the average back into the intended sum.
      * **Lockstep padding**: ranks can hold different sample counts (ragged
        infra-drops), yet every collective forward must fire on every rank the same
        number of times. Each rank runs exactly ``max_micro_steps`` forwards; the
        surplus steps are DUMMY forwards whose loss is multiplied by 0 (no grad,
        but the all-gather still happens), so no rank ever deadlocks.

    Returns ``(loss_value, n_real_terms)``.
    """
    import torch

    loss_value = 0.0
    n_real = 0
    for i in range(max_micro_steps):
        if i < len(local_terms):
            adv, sample = local_terms[i]
            gen_inputs = sample[1]
            lp = logp_fn(gen_inputs)
            if lp is None:
                # keep lockstep: run a zeroed dummy forward instead of skipping.
                dlp = logp_fn(_dummy_gen_inputs(tok, device))
                accelerator.backward(dlp * 0.0)
                continue
            n_tok = max(int(_sample_field(sample, 4, 1) or 1), 1)
            old_lp = _sample_field(sample, 3)
            if old_lp is not None:
                ratio = torch.exp(lp - old_lp)
                pg = -clip_higher_ratio(ratio, adv, clip_ratio_low, clip_ratio_high)
            else:
                pg = -adv * lp
            sc_w = _sample_field(sample, 5)
            if sc_w is not None:
                pg = pg * sc_w
            term = pg
            ref_lp = _sample_field(sample, 2)
            if ref_anchor_coef and ref_lp is not None:
                d = ref_lp - lp
                term = term + ref_anchor_coef * (torch.exp(d) - d - 1.0)
            scaled = term * (n_tok / global_total_tokens) * grad_scale
            accelerator.backward(scaled)
            loss_value += float(scaled.detach())
            n_real += 1
        else:
            # padding step: dummy collective forward, zero contribution.
            dlp = logp_fn(_dummy_gen_inputs(tok, device))
            accelerator.backward(dlp * 0.0)
    return loss_value, n_real


def _rollout_slice_distributed(model, tok, task, config, ref_model, rank, world, seed):
    """This rank's slice of a Kevin group: roll ``G/world`` trajectories.

    Mirrors the serial ``_one_group`` per-trajectory rollout but only for the
    trajectory indices this rank owns (:func:`_rank_slice`). Returns a dict with
    LOCAL parallel lists: ``traj_scores`` (per-trajectory best-kernel value, for
    the cross-rank StarPO-S/dynamic-sampling decision), ``returns`` + ``samples``
    (per-turn Kevin samples for THIS rank), ``correct_kernels``, and ``infra``.

    Only serial refinement is driven here: its per-trajectory forward COUNT is
    fixed (``num_turns``), so every rank issues the same number of collective
    forwards — the invariant ZeRO-3/FSDP requires. (Agentic tool rollouts have
    ragged per-trajectory turn counts and are not sharded here; see the module
    docstring / DISTRIBUTED notes.)
    """
    from kore.env.kore_env import KoreEnv
    from kore.policy import anticollapse as ac

    env = KoreEnv(task)
    G = config.num_trajectories
    my = _rank_slice(G, rank, world)
    rtoks = ac.sample_reward_tokens(G, config.rc_p_high, seed=seed) if config.rc_grpo else [None] * G

    traj_rewards, traj_correct, traj_infra = [], [], []
    turn_inputs, turn_ref, turn_old, turn_ntok, turn_codes = [], [], [], [], []
    traj_scores = []
    for g in my:
        d = _rollout(model, tok, env, task, config, rtoks[g], ref_model)
        traj_rewards.append(d["rewards"]); traj_correct.append(d["correct"])
        traj_infra.append(d["infra"]); turn_inputs.append(d["gen_inputs"])
        turn_ref.append(d["ref_logps"]); turn_old.append(d["old_logps"])
        turn_ntok.append(d["n_tokens"]); turn_codes.append(d["codes"])
        traj_scores.append(
            kevin_trajectory_score(d["rewards"], d["correct"])
            if config.kevin_best_kernel_scoring else (max(d["rewards"]) if d["rewards"] else 0.0))
    returns, index = build_kevin_samples(traj_rewards, traj_correct, config.gamma, traj_infra=traj_infra)
    samples, codes = [], []
    for (ti, tu), ret in zip(index, returns):
        samples.append([ret, turn_inputs[ti][tu], turn_ref[ti][tu],
                        turn_old[ti][tu], turn_ntok[ti][tu], None])
        codes.append(turn_codes[ti][tu])
    correct_kernels = [turn_codes[ti][tu]
                       for ti in range(len(my)) for tu in range(len(traj_correct[ti]))
                       if traj_correct[ti][tu] and turn_codes[ti][tu]]
    return {"traj_scores": traj_scores, "returns": [s[0] for s in samples], "samples": samples,
            "codes": codes, "correct_kernels": correct_kernels,
            "infra": sum(1 for inf in traj_infra for x in inf if x)}


def _train_grpo_distributed(config, tasks):
    """Sharded FULL-PARAMETER multi-turn GRPO across an ``accelerate`` process group.

    The policy (and frozen KL-anchor reference) are FULL-sharded across all ranks
    (FSDP FULL_SHARD by default, DeepSpeed ZeRO-3 opt-in) so no full 14B replica
    ever lives on one GPU. Per step:
      1. each rank rolls out its strided slice of every Kevin group's ``G``
         trajectories (serial refinement; ``synced_gpus`` generation);
      2. per-trajectory rewards + per-turn returns + correct-kernels are
         ALL-GATHERED so StarPO-S / dynamic-sampling and the group-relative
         advantage baseline are computed over the FULL group (all ranks);
      3. GTPO code-sim shaping runs against the gathered correct kernels;
      4. each rank backprops ITS local samples with the globally-normalized
         advantage via a lockstep-padded, world-scaled micro-batched
         ``accelerator.backward`` (activation memory O(1 sample)); one
         ``optimizer.step()`` per PPO epoch.

    The final model is gathered to a plain checkpoint on the main process. Every
    single-process / LoRA / CPU path is elsewhere; this function only runs under a
    real distributed launch (``grpo_distributed_enabled``).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kore.policy.configs import grpo_sharding_backend
    from kore.tasks.registry import get_task, task_ids

    tasks = tasks or task_ids()
    backend = grpo_sharding_backend(config)
    accelerator = build_grpo_accelerator(config)
    world = accelerator.num_processes
    rank = accelerator.process_index
    is_main = accelerator.is_main_process
    if is_main:
        configure(run_dir=getattr(config, "output_dir", None))

    # generation must run in lockstep across ranks under sharding.
    setattr(config, "_grpo_synced_gpus", bool(getattr(config, "synced_gpus", True)))
    t_start = time.time()
    log.info("grpo distributed: starting", backend=backend, world=world, rank=rank,
             model=config.model_id, total_steps=config.total_steps,
             num_trajectories=config.num_trajectories, num_turns=config.num_turns,
             use_lora=False, agentic=bool(config.agentic),
             cpu_offload=bool(getattr(config, "cpu_offload", False)),
             ref_anchor_coef=config.ref_anchor_coef, bf16=bool(getattr(config, "bf16", True)))
    if config.agentic:
        log.warn("grpo distributed: agentic rollouts have ragged per-trajectory turn "
                 "counts (non-symmetric collective forwards under sharding); running the "
                 "SERIAL refinement rollout on the sharded path instead", agentic=True)

    _activate_value_ranker(config)
    tok = AutoTokenizer.from_pretrained(config.model_id)

    from kore.policy.configs import preferred_attn_impl
    model = AutoModelForCausalLM.from_pretrained(config.model_id, torch_dtype=_model_dtype(config),
                                                 attn_implementation=preferred_attn_impl())
    model.config.use_cache = False
    if getattr(config, "gradient_checkpointing", True):
        # REENTRANT, layer-internal. Robust to the intermittent flash_attention_2 ->
        # SDPA per-worker downgrade (reentrant skips the saved-tensor-count check
        # that NON-REENTRANT does and that raises CheckpointError when SDPA swaps
        # fused kernels between forward/recompute). NOT the FSDP-plugin's external
        # checkpoint_wrapper (see build_grpo_fsdp_plugin / configs).
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        model.enable_input_require_grads()
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=config.learning_rate)
    # Shard params + optimizer state across ranks (ZeRO-3-equivalent).
    model, opt = accelerator.prepare(model, opt)
    sched = _build_lr_scheduler(opt, config)

    ref_model = None
    if getattr(config, "ref_anchor_coef", 0.0) > 0:
        ref_model = _load_ref_model(config)
        if ref_model is not None:
            # frozen reference is also sharded (eval mode) — never a full replica.
            ref_model = accelerator.prepare_model(ref_model, evaluation_mode=True)

    tasks_per_step = max(1, getattr(config, "tasks_per_step", 1))
    target_groups = getattr(config, "target_groups", None) or tasks_per_step
    dyn = bool(getattr(config, "dynamic_sampling", True)) and bool(config.starpo_s)
    max_attempts = getattr(config, "max_sampling_attempts", None) or (
        3 * target_groups if dyn else tasks_per_step)
    ppo_epochs = max(1, getattr(config, "ppo_epochs", 1))
    task_cursor = 0
    last_mean_r = None

    def _roll(attempt):
        nonlocal task_cursor
        task = get_task(tasks[task_cursor % len(tasks)])
        task_cursor += 1
        local = _rollout_slice_distributed(model, tok, task, config, ref_model,
                                            rank, world, seed=attempt * 100003 + rank)
        # gather across ranks -> the FULL group (every trajectory on every rank).
        all_scores = _all_gather_object(local["traj_scores"], accelerator)
        all_returns = _all_gather_object(local["returns"], accelerator)
        all_correct = _all_gather_object(local["correct_kernels"], accelerator)
        full_scores = merge_across_ranks(all_scores)
        return {"task": task, "local": local, "full_scores": full_scores,
                "per_rank_returns": all_returns,
                "correct_kernels": merge_across_ranks(all_correct),
                "any_correct": bool(merge_across_ranks(all_correct)),
                "infra": local["infra"]}

    for step in range(config.total_steps):
        groups, attempts = dynamic_sampling_refill(
            _roll, target_groups, min_std=config.starpo_min_std,
            max_attempts=max_attempts, dynamic=dyn,
            std_key=lambda g: group_reward_std(g["full_scores"]))
        if not groups:
            log.info("grpo(dist) step: dynamic-sampling exhausted — skipping", step=step,
                     attempts=attempts, rank=rank)
            log.progress(step + 1, config.total_steps, "grpo", t_start=t_start)
            continue

        # GTPO all-fail code-sim shaping against the gathered correct kernels.
        if config.gtpo_codesim:
            from kore.policy import anticollapse as ac
            step_refs = [k for g in groups for k in g["correct_kernels"]]
            for g in groups:
                loc = g["local"]
                if g["any_correct"] or not loc["samples"]:
                    continue
                refs = step_refs or ([_safe_seed_code(g["task"])] if _safe_seed_code(g["task"]) else [])
                partial = ac.gtpo_codesim_shaping(loc["codes"], refs, config.gtpo_codesim_scale)
                for s, p in zip(loc["samples"], partial):
                    s[0] = p
                loc["returns"] = [s[0] for s in loc["samples"]]

        # StarPO-S: identical decision on every rank (uses gathered full scores).
        group_full_scores = [g["full_scores"] for g in groups]
        if config.starpo_s:
            keep = sorted(starpo_select_high_variance(
                group_full_scores, config.starpo_keep_frac, config.starpo_min_std))
            if not keep:
                log.info("grpo(dist) step: all groups collapsed — skipping", step=step, rank=rank)
                log.progress(step + 1, config.total_steps, "grpo", t_start=t_start)
                continue
        else:
            keep = list(range(len(groups)))

        # Global (cross-rank) advantages per kept group -> this rank's slice.
        local_terms = []
        local_tokens = 0
        for gi in keep:
            g = groups[gi]
            # re-gather returns AFTER GTPO shaping so advantages see shaped returns.
            per_rank_returns = _all_gather_object(g["local"]["returns"], accelerator)
            per_rank_adv = distributed_group_advantages(
                per_rank_returns, config.variance_floor, config.avspo_virtual_k, config.adv_eps)
            my_adv = per_rank_adv[rank] if rank < len(per_rank_adv) else []
            for adv, sample in zip(my_adv, g["local"]["samples"]):
                if not sample[1]:
                    continue
                local_terms.append((adv, sample))
                local_tokens += max(int(_sample_field(sample, 4, 1) or 1), 1)

        # global token normalizer + lockstep bound, agreed across ranks.
        all_tokens = _all_gather_object(local_tokens, accelerator)
        global_total_tokens = sum(all_tokens)
        all_counts = _all_gather_object(len(local_terms), accelerator)
        max_micro = max(all_counts) if all_counts else 0
        if global_total_tokens == 0 or max_micro == 0:
            log.info("grpo(dist) step: no learnable samples — skipping", step=step, rank=rank)
            log.progress(step + 1, config.total_steps, "grpo", t_start=t_start)
            continue

        def _logp_fn(gen_inputs):
            return _recompute_logp(model, tok, gen_inputs, config.temperature) if gen_inputs else None

        loss_value, n_terms = 0.0, 0
        for _epoch in range(ppo_epochs):
            opt.zero_grad()
            loss_value, n_terms = _accumulate_grpo_grads_distributed(
                local_terms, _logp_fn, accelerator=accelerator,
                global_total_tokens=global_total_tokens, grad_scale=float(world),
                max_micro_steps=max_micro, ref_anchor_coef=config.ref_anchor_coef,
                clip_ratio_low=config.clip_ratio_low, clip_ratio_high=config.clip_ratio_high,
                tok=tok, device=accelerator.device)
            accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            opt.step()
        sched.step()

        mean_r = sum(sum(s) / len(s) for s in group_full_scores if s) / max(len(group_full_scores), 1)
        last_mean_r = mean_r
        if is_main:
            print(f"[grpo/dist] step {step} kept={len(keep)}/{len(groups)} world={world} "
                  f"epochs={ppo_epochs} meanR={mean_r:.3f} loss={loss_value:.4f}")
            log.event("grpo_step_dist", step=step, backend=backend, world=world,
                      n_groups=len(groups), n_kept_groups=len(keep), n_attempts=attempts,
                      reward_mean=mean_r, loss=loss_value, global_tokens=global_total_tokens,
                      **gpu_mem_snapshot())
        log.progress(step + 1, config.total_steps, "grpo", t_start=t_start)

    # Gather the sharded weights into a plain checkpoint on the main process.
    accelerator.wait_for_everyone()
    out = config.output_dir
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(
        out, is_main_process=is_main, save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model))
    if is_main:
        tok.save_pretrained(out)
        log.metric("grpo_done", steps=config.total_steps, mean_reward_last=last_mean_r,
                   out=out, backend=backend, world=world, **gpu_mem_snapshot())
    return out


def _load_ref_model(config):
    """Load the frozen KL-anchor reference (``ref_checkpoint`` or ``model_id``).

    Anchoring RL to the post-SFT reference preserves chat/code/orchestration
    behavior. Returns ``None`` (and logs) if the ref cannot be loaded so training
    degrades gracefully to no-KL rather than crashing.
    """
    from transformers import AutoModelForCausalLM

    from kore.policy.configs import fsdp_enabled, preferred_attn_impl

    ref_id = config.ref_checkpoint or config.model_id
    try:
        # Fix 3/4: honor bf16 and skip device_map under distributed FSDP.
        ref_kwargs = {"torch_dtype": _model_dtype(config),
                      "attn_implementation": preferred_attn_impl()}
        if not fsdp_enabled(config):
            ref_kwargs["device_map"] = "auto"
        ref = AutoModelForCausalLM.from_pretrained(ref_id, **ref_kwargs)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        return ref
    except Exception as e:  # noqa: BLE001
        print(f"[grpo] KL-anchor ref '{ref_id}' unavailable ({e}); training without KL anchor")
        log.warn("KL-anchor ref unavailable — training without KL anchor",
                 ref_id=ref_id, error=repr(e))
        return None


def _rollout(model, tok, env, task, config, reward_token, ref_model=None):
    """One multi-turn trajectory (serial refinement).

    Returns a per-turn dict with parallel lists (one entry per turn):
    ``rewards``, ``correct``, ``infra``, ``gen_inputs`` (each ``[(prompt_ids,
    gen_ids)]`` detached), ``ref_logps`` (frozen-ref token-mean log-prob or
    ``None``), ``old_logps`` (detached policy token-mean log-prob AT ROLLOUT — the
    DAPO importance-ratio anchor, item 1), ``n_tokens`` (for the global
    token-mean, item 3), and ``codes`` (parsed kernel source, for GTPO code-sim
    shaping, item 4c). Nothing differentiable is retained; the policy log-prob is
    RECOMPUTED at loss time so the rollout keeps no forward graph.

    When ``config.value_prefilter`` is set, each turn generates
    ``num_candidates_per_turn`` candidates, ranks them with the value model
    (:func:`_prefilter_bench_indices`, contract b) and only BENCHES the top-``k``
    — the measurement-efficiency lever (item 6). ``config.reward_phase`` applies
    the correctness->latency curriculum mask (item 8). ``config.cot_masking``
    drops prior-turn thinking from the re-rendered context.
    """
    import torch

    from kore.policy import anticollapse as ac
    from kore.policy.format import build_transcript, parse_response, build_turn_feedback
    from kore.reward.reward import compute_reward

    prompt = _task_prompt(task)
    if reward_token:
        prompt = ac.prepend_reward_token(prompt, reward_token)

    snr_threshold = getattr(task, "snr_threshold", None)
    prefilter = bool(getattr(config, "value_prefilter", False))
    n_cand = max(1, getattr(config, "num_candidates_per_turn", 1)) if prefilter else 1

    turns: list[dict] = []
    out = {"rewards": [], "correct": [], "infra": [], "gen_inputs": [],
           "ref_logps": [], "old_logps": [], "n_tokens": [], "codes": []}
    for _turn in range(config.num_turns):
        ctx_turns = mask_cot_turns(turns) if config.cot_masking else turns
        msgs = build_transcript(prompt, ctx_turns)
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
        ids = _truncate_prompt_ids(ids, config)  # Fix 3: honor max_prompt_length

        # Generate candidate(s) for this turn (no graph retained).
        cands: list[tuple] = []  # (seq_detached, text, code)
        for _c in range(n_cand):
            with torch.no_grad():
                # Fix 2: pass top_p (+temperature) so sampling matches the config.
                # synced_gpus keeps all ranks in lockstep under ZeRO-3/FSDP sharded
                # generation (default False -> single-process behavior unchanged).
                gen = model.generate(ids, max_new_tokens=config.max_response_length,
                                     do_sample=True, temperature=config.temperature,
                                     top_p=config.top_p,
                                     synced_gpus=getattr(config, "_grpo_synced_gpus", False),
                                     return_dict_in_generate=True, output_scores=True)
            seq = gen.sequences[0][ids.shape[1]:].detach()
            text = tok.decode(seq, skip_special_tokens=True)
            cands.append((seq, text, parse_response(text).get("kernel", "")))

        # Value-model prefilter: bench only the top-k candidates (item 6).
        if prefilter and n_cand > 1:
            bench_idx = _prefilter_bench_indices([c[2] for c in cands], task,
                                                 getattr(config, "value_prefilter_k", 1))
        else:
            bench_idx = list(range(len(cands)))

        best = None  # (seq, text, code, obs, rr)
        for ci in bench_idx:
            seq, text, code = cands[ci]
            obs = env.step(code, full_validation=True, multi_shape=True)
            rr = apply_reward_phase(
                compute_reward(obs, code, dtype=task.dtype, snr_threshold=snr_threshold), config)
            if best is None or (bool(rr.correct), rr.reward) > (bool(best[4].correct), best[4].reward):
                best = (seq, text, code, obs, rr)

        seq, text, code, obs, rr = best
        gen_inputs = [(ids.detach(), seq)]
        n_tok = max(int(seq.shape[0]), 1)
        # DAPO importance-ratio anchor: detached policy token-mean logp at rollout.
        with torch.no_grad():
            old_lp = _recompute_logp(model, tok, gen_inputs, config.temperature)
        old_lp = old_lp.detach() if old_lp is not None else None
        if ref_model is not None:
            with torch.no_grad():
                ref_lp = _recompute_logp(ref_model, tok, gen_inputs, config.temperature)
            ref_lp = ref_lp.detach() if ref_lp is not None else None
        else:
            ref_lp = None

        out["rewards"].append(rr.reward)
        out["correct"].append(bool(rr.correct))
        out["infra"].append(bool(getattr(obs, "infra_error", False)) or rr.tier == "infra")
        out["gen_inputs"].append(gen_inputs)
        out["ref_logps"].append(ref_lp)
        out["old_logps"].append(old_lp)
        out["n_tokens"].append(n_tok)
        out["codes"].append(code)
        turns.append({"response": text, "feedback": build_turn_feedback(obs)})
    return out


def _rollout_agentic(model, tok, env, task, config, ref_model=None):
    """One agentic trajectory via :class:`kore.agent.harness.AgentHarness`.

    Item 5: gives the agentic loop the SAME per-turn Kevin credit + per-turn KL
    anchor as the serial path. The episode's per-turn ``(reward, correct)`` trace
    (contract a: ``episode.turn_rewards`` / ``episode.turn_correct``) is aligned
    1:1 with the assistant-turn generations recorded by :class:`_HFChatPolicy`,
    and the caller feeds them through :func:`build_kevin_samples` exactly like the
    serial trajectories (correctness-gated gamma returns, per-turn-as-sample).
    The ToolRL-style tool-use shaping is folded (once) into the best correct
    turn's reward via :func:`composite_agentic_reward`.

    Returns the SAME per-turn dict shape as :func:`_rollout` (``rewards``,
    ``correct``, ``infra``, ``gen_inputs``, ``ref_logps``, ``old_logps``,
    ``n_tokens``, ``codes``). ``ref_logps`` are populated when ``ref_model`` is
    provided — the per-turn KL anchor is now ACTIVE in agentic mode (item 5). If
    the harness/episode exposes no per-turn trace, this degrades to a single
    terminal (best_reward, success) sample (Kevin trajectory value).
    """
    import torch

    from kore.agent.harness import AgentHarness
    from kore.agent.tools import tool_use_reward

    policy = _HFChatPolicy(model, tok, config)
    harness = AgentHarness(task, policy, env, max_turns=config.max_tool_turns)
    episode = harness.run()

    turn_rewards, turn_correct = _episode_turn_rewards(episode)
    turn_inputs = list(policy.turn_inputs)  # per assistant turn: (prompt_ids, gen_ids)

    # Fold ToolRL orchestration shaping ONCE into the best correct turn's reward.
    tool_total = tool_use_reward(episode).get("total", 0.0)
    if any(turn_correct):
        bi = max((i for i, c in enumerate(turn_correct) if c), key=lambda i: turn_rewards[i])
        turn_rewards = list(turn_rewards)
        turn_rewards[bi] = composite_agentic_reward(
            turn_rewards[bi], tool_total, config.tool_reward_weight)

    # Align the per-turn credit trace with the recorded assistant generations.
    n = min(len(turn_inputs), len(turn_rewards), len(turn_correct))
    out = {"rewards": [], "correct": [], "infra": [], "gen_inputs": [],
           "ref_logps": [], "old_logps": [], "n_tokens": [], "codes": []}
    for t in range(n):
        gen_inputs = [turn_inputs[t]]
        prompt_ids, gen_ids = turn_inputs[t]
        with torch.no_grad():
            old_lp = _recompute_logp(model, tok, gen_inputs, config.temperature)
        old_lp = old_lp.detach() if old_lp is not None else None
        if ref_model is not None:
            with torch.no_grad():
                ref_lp = _recompute_logp(ref_model, tok, gen_inputs, config.temperature)
            ref_lp = ref_lp.detach() if ref_lp is not None else None
        else:
            ref_lp = None
        out["rewards"].append(float(turn_rewards[t]))
        out["correct"].append(bool(turn_correct[t]))
        out["infra"].append(False)  # harness exposes no per-turn infra trace
        out["gen_inputs"].append(gen_inputs)
        out["ref_logps"].append(ref_lp)
        out["old_logps"].append(old_lp)
        out["n_tokens"].append(max(int(gen_ids.shape[0]), 1))
        out["codes"].append("")  # per-turn kernel source not exposed by the harness
    return out


def _episode_turn_rewards(episode):
    """Per-turn ``(reward, correct)`` trace from an AgentEpisode.

    Prefers the contract-(a) per-turn fields ``episode.turn_rewards`` /
    ``episode.turn_correct`` (a sibling agent records them on the harness) so the
    agentic loop gets the SAME per-turn Kevin credit as the serial path (item 5).
    Falls back to a single terminal ``(best_reward, success)`` sample — which
    :func:`kevin_trajectory_score` reduces to the best correct kernel — when the
    per-turn trace is absent (the harness records only the trajectory best today).
    """
    tr = getattr(episode, "turn_rewards", None)
    tc = getattr(episode, "turn_correct", None)
    if tr is not None and tc is not None and len(tr) == len(tc) and len(tr) > 0:
        return [float(x) for x in tr], [bool(x) for x in tc]
    best = getattr(episode, "best_reward", None)
    success = bool(getattr(episode, "success", False))
    if best is None:
        return [0.0], [False]
    return [float(best)], [success]


class _HFChatPolicy:
    """Adapter exposing ``generate(messages) -> str`` for the AgentHarness.

    Records the ``(prompt_ids, gen_ids)`` of each assistant turn so the GRPO loop
    can RECOMPUTE (at loss time) the token-mean log-prob to apply policy-gradient
    credit over the assistant turns — the rollout itself retains no graph (Fix 1).
    """

    def __init__(self, model, tok, config):
        self.model = model
        self.tok = tok
        self.config = config
        self.turn_inputs: list = []

    def generate(self, messages) -> str:
        import torch

        ids = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        ids = _truncate_prompt_ids(ids, self.config)  # Fix 3: honor max_prompt_length
        with torch.no_grad():
            # Fix 2: pass top_p (+temperature) so sampling matches the config.
            # synced_gpus keeps ranks in lockstep under sharded generation.
            gen = self.model.generate(
                ids, max_new_tokens=self.config.max_response_length, do_sample=True,
                temperature=self.config.temperature, top_p=self.config.top_p,
                synced_gpus=getattr(self.config, "_grpo_synced_gpus", False),
                return_dict_in_generate=True, output_scores=True)
        seq = gen.sequences[0][ids.shape[1]:]
        self.turn_inputs.append((ids.detach(), seq.detach()))
        return self.tok.decode(seq, skip_special_tokens=True)


def _sample_token_count(gen_inputs) -> int:
    """Total generated-token count across a sample's ``(prompt_ids, gen_ids)`` pairs."""
    return sum(max(int(gen_ids.shape[0]), 1) for _prompt_ids, gen_ids in gen_inputs)


def _recompute_logp(model, tok, gen_inputs, temperature: float = 1.0):
    """Recompute a sample's *token-mean* log-prob against the live policy.

    ``gen_inputs`` is a list of ``(prompt_ids, gen_ids)`` pairs (a single pair for
    a serial-refinement turn or an agentic assistant turn). The returned value is
    the global token-mean ``sum_pairs(sum_token_logp) / sum_pairs(n_tokens)`` — a
    single number per sample, matching the detached rollout-time ``old_logp`` so
    the importance ratio ``exp(logp - old_logp)`` is the turn-level geometric-mean
    (Turn-PPO) ratio. Recomputing here (not at rollout) keeps only ONE sample's
    forward graph alive during the micro-batched backward (activation O(1)).
    """
    total = None
    n_tok = 0
    for prompt_ids, gen_ids in gen_inputs:
        s = _seq_logprob(model, tok, prompt_ids, gen_ids, temperature)
        total = s if total is None else total + s
        n_tok += max(int(gen_ids.shape[0]), 1)
    if total is None:
        return None
    return token_mean_logprob(total, n_tok)  # DAPO length-debias (item 3)


def _seq_logprob(model, tok, prompt_ids, gen_ids, temperature: float = 1.0):
    """Summed log-prob of ``gen_ids`` under ``model`` (temperature-scaled logits).

    Divides the logits by ``temperature`` before ``log_softmax`` so the scored
    distribution matches the sampling distribution used to generate the tokens.
    """
    import torch

    full = torch.cat([prompt_ids[0], gen_ids]).unsqueeze(0)
    out = model(full)
    logits = out.logits[0, prompt_ids.shape[1] - 1:-1, :]
    if temperature and temperature > 0:
        logits = logits / temperature
    logp = torch.log_softmax(logits, dim=-1)
    idx = gen_ids.unsqueeze(-1)
    return logp.gather(-1, idx).squeeze(-1).sum()


def _task_prompt(task) -> str:
    return (f"Optimize a {task.dtype} {task.operation} kernel for AMD {task.gpu_target} "
            f"(backend: {task.backend}). Baseline to beat: {task.comparison_baseline}. "
            f"Return ANALYSIS, PROPOSED_CHANGE, and a complete FULL_KERNEL.\n\n"
            f"Seed kernel:\n```python\n{task.seed_source}\n```")


# --------------------------------------------------------------------------- #
# Distributed entry: `python -m kore.policy.grpo <config.json>`
#
# Used by scripts/launch_distributed.sh under `accelerate launch` so
# ``scripts/launch_distributed.sh grpo <config.json>`` drives FULL-PARAMETER GRPO
# across the sharded process group. Mirrors the sft.py / dpo.py entry pattern:
# pure-stdlib JSON parsing (NO torch at import time), heavy training only touched
# when train_grpo actually runs. The JSON is a flat map of GRPOConfig fields with
# an optional nested "lora" object.
# --------------------------------------------------------------------------- #
def grpo_config_from_dict(d: dict):
    """Build a :class:`kore.policy.configs.GRPOConfig` from a plain dict.

    A nested ``lora`` mapping is turned into a :class:`LoRAConfig`. Every other key
    is a GRPOConfig field (``model_id``, ``distributed``, ``use_lora``,
    ``sharding_backend``, ``zero_stage``, ``cpu_offload``, ``ds_config``, the Kevin
    rollout/objective knobs, the anti-collapse ladder, ...), so the same JSON the
    campaign renders can drive ``accelerate launch``-ed full-param GRPO.

    ``tasks`` is NOT a GRPOConfig field — the campaign threads the train-split task
    ids through the JSON so the sharded run trains on exactly the right tasks — so
    it is popped here (and surfaced by :func:`_main` to ``train_grpo(tasks=...)``).
    """
    from kore.policy.configs import GRPOConfig, LoRAConfig

    d = dict(d)
    d.pop("tasks", None)          # handled by _main -> train_grpo(tasks=...)
    lora = d.pop("lora", None)
    cfg = GRPOConfig(**d)
    if lora is not None:
        cfg.lora = LoRAConfig(**lora)
    return cfg


def _main(argv: Optional[list] = None) -> int:
    import json
    import sys
    from pathlib import Path

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m kore.policy.grpo <config.json>", file=sys.stderr)
        return 2
    raw = json.loads(Path(argv[0]).read_text())
    # Launched via accelerate -> default to the distributed full-FT path unless the
    # config explicitly opts out (mirrors sft.py / dpo.py).
    raw.setdefault("distributed", True)
    tasks = raw.get("tasks")      # optional train-split task ids threaded by the campaign
    cfg = grpo_config_from_dict(raw)
    out = train_grpo(cfg, tasks=tasks)
    print(f"[grpo] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
