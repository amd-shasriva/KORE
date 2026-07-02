"""Multi-turn GRPO for KORE.

Pure, import-safe math (group advantages, discounted turn returns, asymmetric
"clip-higher" surrogate) lives at module top so it is unit-testable without any
heavy deps. ``train_grpo`` prefers a verl backend and falls back to a compact
transformers+PEFT loop that rolls out against the verified :class:`KoreEnv`.
"""

from __future__ import annotations

import math
from typing import Optional

_EPS = 1e-6


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


def clip_higher_ratio(ratio: float, advantage: float, lo: float = 0.2, hi: float = 0.28) -> float:
    """DAPO-style asymmetric PPO surrogate (wider upper clip fights collapse)."""
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
    scored = [(i, group_reward_std(g)) for i, g in enumerate(groups)]
    live = [(i, s) for i, s in scored if s > min_std]
    if not live:
        return []
    live.sort(key=lambda x: x[1], reverse=True)
    k = max(1, int(round(keep_frac * len(live))))
    return sorted(i for i, _ in live[:k])


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
def train_grpo(config, tasks: Optional[list[str]] = None, backend: str = "auto"):
    """Dispatch to verl if available, else the built-in fallback loop."""
    if backend in ("auto", "verl"):
        try:
            import verl  # noqa: F401

            return _train_grpo_verl(config, tasks)
        except Exception as e:  # noqa: BLE001
            if backend == "verl":
                raise
            print(f"[grpo] verl unavailable ({e}); using fallback loop")
    return _train_grpo_fallback(config, tasks)


def _train_grpo_verl(config, tasks):
    raise NotImplementedError(
        "verl backend runs via the isolated server recipe in docs/rl_server.md; "
        "use backend='fallback' for the in-process loop.")


def _accumulate_grpo_grads(kept_groups, logp_fn, *, ref_anchor_coef: float,
                           sc_grpo_allfail: bool, sc_grpo_alpha: float,
                           backward: bool = True):
    """Micro-batched GRPO loss: one ``backward()`` per sample, grads accumulated.

    ``kept_groups`` is a list of groups (the StarPO-S-kept task groups); each
    group is a list of samples ``(return, gen_inputs, ref_logp_or_None)``.
    ``logp_fn(gen_inputs) -> tensor`` recomputes that sample's differentiable
    (token-mean) log-prob against the LIVE policy, or returns ``None`` to skip.

    The effective objective is the SAMPLE-MEAN over all kept samples of
    ``-adv*logp + ref_anchor_coef * k3_kl`` (k3 = ``exp(d)-d-1``, ``d=ref-logp``).
    Group-normalized advantages (+ optional SC-GRPO all-fail bonus) are computed
    per group exactly as before. Scaling each term by ``1/n_terms`` before
    ``backward()`` makes the accumulated gradient IDENTICAL to a single backward
    on the full mean loss, while only one sample's graph is ever materialized —
    bounding activation memory to O(1 sample). Returns ``(loss_value, n_terms)``.

    Kept intentionally free of model/tokenizer coupling (log-prob recompute is
    injected via ``logp_fn``) so the equivalence is unit-testable on CPU.
    """
    import torch

    from kore.policy import anticollapse as ac

    # Pass 1: per-group advantages + count learnable terms (sets the mean scale).
    group_advs: list[list[float]] = []
    n_terms = 0
    for samples in kept_groups:
        returns = [s[0] for s in samples]
        advs = group_advantages(returns)
        if sc_grpo_allfail:
            advs = [a + b for a, b in zip(advs, ac.sc_grpo_allfail_bonus(returns, sc_grpo_alpha))]
        group_advs.append(advs)
        for s in samples:
            if s[1]:  # non-empty gen_inputs -> a learnable sample
                n_terms += 1
    if n_terms == 0:
        return 0.0, 0

    # Pass 2: recompute each sample's log-prob, backward the 1/n_terms-scaled term.
    loss_value = 0.0
    for samples, advs in zip(kept_groups, group_advs):
        for adv, sample in zip(advs, samples):
            gen_inputs = sample[1]
            if not gen_inputs:
                continue
            lp = logp_fn(gen_inputs)
            if lp is None:
                continue
            term = -adv * lp
            ref_lp = sample[2] if len(sample) > 2 else None
            if ref_anchor_coef and ref_lp is not None:
                d = ref_lp - lp  # differentiable k3 KL anchor: exp(d) - d - 1
                term = term + ref_anchor_coef * (torch.exp(d) - d - 1.0)
            scaled = term / n_terms
            if backward:
                scaled.backward()  # frees this sample's graph before the next one
            loss_value += float(scaled.detach())
    return loss_value, n_terms


def _train_grpo_fallback(config, tasks):
    """Compact in-process GRPO loop implementing the locked KORE recipe.

    Per step:
      1. roll out ``num_trajectories`` groups for each of ``tasks_per_step`` tasks
         (serial multi-turn refinement, or the agentic tool harness);
      2. build per-turn Kevin-credit samples (correctness-gated gamma returns)
         and group-normalize advantages across the ``m*n`` samples of each group;
      3. apply StarPO-S variance selection across the step's task-groups;
      4. form a token-mean (DAPO length-debiased) policy-gradient loss with a
         differentiable k3 KL anchor to the frozen SFT/reference checkpoint.

    Full-FT vs LoRA follows ``config.use_lora`` (the locked recipe is full-FT).
    When LoRA is used the adapter is merged before saving so soup/serve load a
    full model. Gradient checkpointing + PEFT needs ``enable_input_require_grads``
    or ``.backward()`` sees no grad path.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kore.env.kore_env import KoreEnv
    from kore.policy import anticollapse as ac
    from kore.tasks.registry import get_task, task_ids

    tasks = tasks or task_ids()
    tok = AutoTokenizer.from_pretrained(config.model_id)
    model = AutoModelForCausalLM.from_pretrained(config.model_id, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    if getattr(config, "gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # critical for PEFT + grad-ckpt

    if config.use_lora:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(
            r=config.lora.r, lora_alpha=config.lora.lora_alpha, lora_dropout=0.0,
            target_modules=list(config.lora.target_modules), task_type="CAUSAL_LM"))
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=config.learning_rate)

    # ---- ref-model (KL anchor) gating: only pay the ~ref-model memory when a KL
    #      anchor is actually applied (Fix 2). The AGENTIC path pools assistant
    #      turns into ONE trajectory-level sample and exposes no per-turn ref
    #      log-prob, so a k3 KL term cannot be applied there — we therefore do NOT
    #      load the ref in agentic mode (it would sit idle wasting ~28GB) and log
    #      that the KL anchor is inactive for agentic GRPO. Retention in agentic
    #      mode is carried by the base-ward model soup (Stage-4), not a live KL.
    want_ref = getattr(config, "ref_anchor_coef", 0.0) > 0
    ref_model = None
    if not want_ref:
        print("[grpo] ref_anchor_coef<=0: skipping frozen ref-model load (no KL anchor).")
    elif config.agentic:
        print("[grpo] agentic mode: KL-anchor is INACTIVE (no per-turn ref log-prob "
              "through the tool harness); NOT loading the frozen ref to save memory. "
              "General-capability retention is handled by the Stage-4 base-ward soup.")
    else:
        ref_model = _load_ref_model(config)  # frozen KL anchor (or None if unavailable)

    tasks_per_step = max(1, getattr(config, "tasks_per_step", 1))
    task_cursor = 0
    for step in range(config.total_steps):
        # ---- 1. roll out a group per task, accumulate for StarPO-S ---- #
        # Rollout is done WITHOUT retaining a differentiable graph: each sample
        # stores only the (prompt_ids, gen_ids) needed to recompute its log-prob
        # at loss time (see ``_recompute_logp``). This is what bounds activation
        # memory to O(1 sample) during the micro-batched backward below (Fix 1).
        group_rewards: list[list[float]] = []   # trajectory-level reward per group (variance gate)
        group_samples: list[list[tuple]] = []   # per group: (return, gen_inputs, ref_logp) samples
        group_tasks: list = []
        for _ in range(tasks_per_step):
            task = get_task(tasks[task_cursor % len(tasks)])
            task_cursor += 1
            env = KoreEnv(task)
            G = config.num_trajectories
            rtoks = ac.sample_reward_tokens(G, config.rc_p_high, seed=step) \
                if config.rc_grpo else [None] * G

            traj_scores: list[float] = []
            samples: list[tuple] = []
            if config.agentic:
                for g in range(G):
                    r, gen_inputs = _rollout_agentic(model, tok, env, task, config)
                    traj_scores.append(r)
                    samples.append((r, gen_inputs, None))
            else:
                traj_rewards: list[list[float]] = []
                traj_correct: list[list[bool]] = []
                traj_infra: list[list[bool]] = []
                turn_inputs: list[list] = []
                turn_ref_logps: list[list] = []
                for g in range(G):
                    rewards, corrects, gen_inputs, ref_logps, infra = _rollout(
                        model, tok, env, task, config, rtoks[g], ref_model)
                    traj_rewards.append(rewards)
                    traj_correct.append(corrects)
                    traj_infra.append(infra)
                    turn_inputs.append(gen_inputs)
                    turn_ref_logps.append(ref_logps)
                    # trajectory score for the variance gate: best correct kernel.
                    traj_scores.append(
                        kevin_trajectory_score(rewards, corrects)
                        if config.kevin_best_kernel_scoring else (max(rewards) if rewards else 0.0))
                # per-turn Kevin-credit samples flattened across m*n; infra turns
                # are dropped from the batch (Fix 6) via ``traj_infra``.
                returns, index = build_kevin_samples(
                    traj_rewards, traj_correct, config.gamma, traj_infra=traj_infra)
                for (ti, tu), ret in zip(index, returns):
                    samples.append((ret, turn_inputs[ti][tu], turn_ref_logps[ti][tu]))

            group_rewards.append(traj_scores)
            group_samples.append(samples)
            group_tasks.append(task)

        # ---- 2. StarPO-S: keep only high-variance (signal-carrying) groups ---- #
        if config.starpo_s:
            keep = sorted(starpo_select_high_variance(
                group_rewards, config.starpo_keep_frac, config.starpo_min_std))
            if not keep:
                print(f"[grpo] step {step}: all {tasks_per_step} groups collapsed; skip")
                continue
        else:
            keep = list(range(len(group_rewards)))

        # ---- 3. MICRO-BATCHED policy-gradient loss over kept groups (Fix 1) ---- #
        # One backward() per sample (recomputing that sample's log-prob), grads
        # accumulated, a single optimizer.step() per step. Scaling each term by
        # 1/n_terms makes the accumulated gradient identical to a single backward
        # on the sample-mean loss, while only ONE sample's forward graph is ever
        # alive (activation memory O(1 sample), not O(tasks_per_step*G*turns)).
        kept_groups = [group_samples[gi] for gi in keep]

        def _logp_fn(gen_inputs):
            return _recompute_logp(model, tok, gen_inputs, config.temperature) if gen_inputs else None

        opt.zero_grad()
        loss_value, n_terms = _accumulate_grpo_grads(
            kept_groups, _logp_fn,
            ref_anchor_coef=config.ref_anchor_coef,
            sc_grpo_allfail=config.sc_grpo_allfail, sc_grpo_alpha=config.sc_grpo_alpha)
        if n_terms == 0:
            print(f"[grpo] step {step}: no learnable samples; skip")
            continue
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        mean_r = sum(sum(g) / len(g) for g in group_rewards if g) / max(len(group_rewards), 1)
        print(f"[grpo] step {step} kept={len(keep)}/{tasks_per_step} "
              f"meanR={mean_r:.3f} loss={loss_value:.4f}")

    # Merge LoRA into the base before saving so soup/serve load a full model.
    out = config.output_dir
    if config.use_lora:
        merged = model.merge_and_unload()
        merged.save_pretrained(out)
    else:
        model.save_pretrained(out)
    tok.save_pretrained(out)
    return out


def _load_ref_model(config):
    """Load the frozen KL-anchor reference (``ref_checkpoint`` or ``model_id``).

    Anchoring RL to the post-SFT reference preserves chat/code/orchestration
    behavior. Returns ``None`` (and logs) if the ref cannot be loaded so training
    degrades gracefully to no-KL rather than crashing.
    """
    import torch
    from transformers import AutoModelForCausalLM

    ref_id = config.ref_checkpoint or config.model_id
    try:
        ref = AutoModelForCausalLM.from_pretrained(ref_id, torch_dtype=torch.bfloat16,
                                                   device_map="auto")
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        return ref
    except Exception as e:  # noqa: BLE001
        print(f"[grpo] KL-anchor ref '{ref_id}' unavailable ({e}); training without KL anchor")
        return None


def _rollout(model, tok, env, task, config, reward_token, ref_model=None):
    """One multi-turn trajectory (serial refinement).

    Returns ``(rewards, correct_flags, gen_inputs, ref_logprobs, infra_flags)`` —
    one entry per turn. ``gen_inputs[t]`` is ``[(prompt_ids, gen_ids)]`` (detached
    token ids) from which the policy log-prob is RECOMPUTED at loss time; nothing
    differentiable is retained here, so the rollout does not accumulate a forward
    graph per turn (Fix 1). ``ref_logprobs`` are the frozen-reference token-mean
    log-probs (detached, ``None`` per turn if no ref model) used for the k3 KL
    anchor. ``infra_flags[t]`` is True when the turn hit an infrastructure error
    (timeout/OOM/segfault/import) — the caller drops those turns from the batch
    (Fix 6). When ``config.cot_masking`` is set, prior-turn thinking is dropped
    from the context that is re-rendered each turn.
    """
    import torch

    from kore.policy import anticollapse as ac
    from kore.policy.format import build_transcript, parse_response, build_turn_feedback
    from kore.reward.reward import compute_reward

    prompt = _task_prompt(task)
    if reward_token:
        prompt = ac.prepend_reward_token(prompt, reward_token)

    snr_threshold = getattr(task, "snr_threshold", None)
    turns: list[dict] = []
    rewards: list[float] = []
    corrects: list[bool] = []
    gen_inputs: list = []
    ref_logps: list = []
    infra_flags: list[bool] = []
    for _turn in range(config.num_turns):
        ctx_turns = mask_cot_turns(turns) if config.cot_masking else turns
        msgs = build_transcript(prompt, ctx_turns)
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=config.max_response_length, do_sample=True,
                                 temperature=config.temperature, return_dict_in_generate=True,
                                 output_scores=True)
        seq = gen.sequences[0][ids.shape[1]:]
        text = tok.decode(seq, skip_special_tokens=True)
        n_tok = max(int(seq.shape[0]), 1)
        # Defer the policy log-prob: store only the ids needed to recompute it
        # (with grad) one sample at a time during the micro-batched backward.
        gen_inputs.append([(ids.detach(), seq.detach())])
        if ref_model is not None:
            with torch.no_grad():
                ref_lp = token_mean_logprob(
                    _seq_logprob(ref_model, tok, ids, seq, config.temperature), n_tok)
            ref_logps.append(ref_lp.detach())
        else:
            ref_logps.append(None)

        parsed = parse_response(text)
        obs = env.step(parsed.get("kernel", ""), full_validation=True, multi_shape=True)
        rr = compute_reward(obs, parsed.get("kernel", ""), dtype=task.dtype,
                            snr_threshold=snr_threshold)
        rewards.append(rr.reward)
        corrects.append(bool(rr.correct))
        infra_flags.append(bool(getattr(obs, "infra_error", False)) or rr.tier == "infra")
        turns.append({"response": text, "feedback": build_turn_feedback(obs)})
    return rewards, corrects, gen_inputs, ref_logps, infra_flags


def _rollout_agentic(model, tok, env, task, config):
    """One agentic trajectory via :class:`kore.agent.harness.AgentHarness`.

    Drives the multi-turn build/test/bench/pmc tool loop, then folds the
    ToolRL-style tool-use shaping into the Kevin best-kernel trajectory score via
    :func:`composite_agentic_reward`. Policy-gradient credit is the token-mean
    sum of the assistant-turn log-probs (``PG on assistant turns``).

    Limitation: this applies a single trajectory-level composite reward as the PG
    signal on the pooled assistant turns rather than a full per-tool-turn
    advantage decomposition; the per-turn kernel-reward trace is not exposed by
    the harness/executor, so best-kernel scoring (Kevin) is used as the
    trajectory value. This is the documented minimal-agentic-PG path. Because the
    harness exposes no per-turn infra/tier trace either, an infra episode simply
    yields no positive Kevin signal (best_reward None -> reward 0) rather than
    being explicitly dropped; there is no per-turn signal to prune.

    Returns ``(reward, gen_inputs)`` where ``gen_inputs`` is the list of
    ``(prompt_ids, gen_ids)`` for the assistant turns; the summed token-mean
    log-prob is RECOMPUTED at loss time (Fix 1) so the rollout retains no graph.
    """
    from kore.agent.harness import AgentHarness
    from kore.agent.tools import tool_use_reward

    policy = _HFChatPolicy(model, tok, config)
    harness = AgentHarness(task, policy, env, max_turns=config.max_tool_turns)
    episode = harness.run()

    turn_rewards, turn_correct = _episode_turn_rewards(episode)
    kernel_score = kevin_trajectory_score(turn_rewards, turn_correct)
    tool_total = tool_use_reward(episode).get("total", 0.0)
    reward = composite_agentic_reward(kernel_score, tool_total, config.tool_reward_weight)

    return reward, list(policy.turn_inputs)


def _episode_turn_rewards(episode):
    """Best-effort per-turn (reward, correct) trace from an AgentEpisode.

    The harness exposes the trajectory's best kernel reward; a full per-turn
    kernel-reward trace is not recorded, so we surface a single terminal sample
    (best_reward, success) which :func:`kevin_trajectory_score` reduces to the
    best correct kernel — the Kevin trajectory value.
    """
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
        with torch.no_grad():
            gen = self.model.generate(
                ids, max_new_tokens=self.config.max_response_length, do_sample=True,
                temperature=self.config.temperature, return_dict_in_generate=True, output_scores=True)
        seq = gen.sequences[0][ids.shape[1]:]
        self.turn_inputs.append((ids.detach(), seq.detach()))
        return self.tok.decode(seq, skip_special_tokens=True)


def _recompute_logp(model, tok, gen_inputs, temperature: float = 1.0):
    """Recompute a sample's summed token-mean log-prob against the live policy.

    ``gen_inputs`` is a list of ``(prompt_ids, gen_ids)`` pairs (one per assistant
    turn that carries PG credit — a single pair for a serial-refinement turn, the
    whole assistant-turn list for an agentic trajectory). Recomputing here rather
    than at rollout time keeps only ONE sample's forward graph alive during the
    micro-batched backward (activation memory O(1 sample); Fix 1).
    """
    total = None
    for prompt_ids, gen_ids in gen_inputs:
        n_tok = max(int(gen_ids.shape[0]), 1)
        lp = token_mean_logprob(_seq_logprob(model, tok, prompt_ids, gen_ids, temperature), n_tok)
        total = lp if total is None else total + lp
    return total


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
