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


def _train_grpo_fallback(config, tasks):
    """Compact in-process GRPO loop: group rollouts -> reward -> policy-grad.

    Uses PEFT/LoRA on top of a frozen base. Gradient checkpointing + PEFT needs
    ``enable_input_require_grads`` or ``.backward()`` sees no grad path.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kore.env.kore_env import KoreEnv
    from kore.policy import anticollapse as ac
    from kore.policy.format import build_transcript, parse_response, build_turn_feedback
    from kore.reward.reward import compute_reward
    from kore.tasks.registry import get_task, task_ids

    tasks = tasks or task_ids()
    tok = AutoTokenizer.from_pretrained(config.model_id)
    model = AutoModelForCausalLM.from_pretrained(config.model_id, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    if getattr(config, "gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # critical for PEFT + grad-ckpt
    model = get_peft_model(model, LoraConfig(
        r=config.lora.r, lora_alpha=config.lora.lora_alpha, lora_dropout=0.0,
        target_modules=list(config.lora.target_modules), task_type="CAUSAL_LM"))
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.learning_rate)

    for step in range(config.total_steps):
        task = get_task(tasks[step % len(tasks)])
        env = KoreEnv(task)
        G = config.num_trajectories
        rtoks = ac.sample_reward_tokens(G, config.rc_p_high, seed=step) \
            if config.rc_grpo else [None] * G
        rewards, logps = [], []
        for g in range(G):
            r, lp = _rollout(model, tok, env, task, config, rtoks[g], build_transcript,
                             parse_response, build_turn_feedback, compute_reward)
            rewards.append(r)
            logps.append(lp)
        if config.starpo_s and not starpo_keep_group(rewards, config.starpo_min_std):
            print(f"[grpo] step {step} task={task.task_id} collapsed group (std<={config.starpo_min_std}); skip")
            continue
        advs = group_advantages(rewards)
        if config.sc_grpo_allfail:
            advs = [a + b for a, b in zip(advs, ac.sc_grpo_allfail_bonus(rewards))]
        loss = -sum(a * lp for a, lp in zip(advs, logps) if lp is not None) / max(G, 1)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        print(f"[grpo] step {step} task={task.task_id} meanR={sum(rewards)/G:.3f} loss={loss.item():.4f}")

    out = config.output_dir
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return out


def _rollout(model, tok, env, task, config, reward_token, build_transcript,
             parse_response, build_turn_feedback, compute_reward):
    """One multi-turn trajectory; returns (final_reward, summed_logprob_tensor)."""
    import torch
    from kore.policy import anticollapse as ac

    prompt = _task_prompt(task)
    if reward_token:
        prompt = ac.prepend_reward_token(prompt, reward_token)
    turns: list[dict] = []
    total_lp = None
    final_r = config.reward_compile_fail if hasattr(config, "reward_compile_fail") else -1.0
    for _turn in range(config.num_turns):
        msgs = build_transcript(prompt, turns)
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
        gen = model.generate(ids, max_new_tokens=config.max_response_length, do_sample=True,
                             temperature=config.temperature, return_dict_in_generate=True, output_scores=True)
        seq = gen.sequences[0][ids.shape[1]:]
        text = tok.decode(seq, skip_special_tokens=True)
        lp = _seq_logprob(model, tok, ids, seq)
        total_lp = lp if total_lp is None else total_lp + lp
        parsed = parse_response(text)
        obs = env.step(parsed.get("kernel", ""), full_validation=True, multi_shape=True)
        rr = compute_reward(obs, parsed.get("kernel", ""), dtype=task.dtype)
        final_r = rr.reward
        turns.append({"response": text, "feedback": build_turn_feedback(obs)})
        if obs.validation_passed and obs.wall_by_shape:
            break
    return final_r, total_lp


def _seq_logprob(model, tok, prompt_ids, gen_ids):
    import torch

    full = torch.cat([prompt_ids[0], gen_ids]).unsqueeze(0)
    out = model(full)
    logits = out.logits[0, prompt_ids.shape[1] - 1:-1, :]
    logp = torch.log_softmax(logits, dim=-1)
    idx = gen_ids.unsqueeze(-1)
    return logp.gather(-1, idx).squeeze(-1).sum()


def _task_prompt(task) -> str:
    return (f"Optimize a {task.dtype} {task.operation} kernel for AMD {task.gpu_target} "
            f"(backend: {task.backend}). Baseline to beat: {task.comparison_baseline}. "
            f"Return ANALYSIS, PROPOSED_CHANGE, and a complete FULL_KERNEL.\n\n"
            f"Seed kernel:\n```python\n{task.seed_source}\n```")
