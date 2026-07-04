"""Isolated multi-process reproduction of the DISTRIBUTED GRPO training step.

Mirrors the exact collective sequence of ``_train_grpo_distributed``'s per-step
update (the _all_gather_object calls + ``_accumulate_grpo_grads_distributed``'s
micro-batched forward/backward under FSDP SHARD_GRAD_OP) but with FAKE, RAGGED
per-rank samples so we can prove/fix the "7 busy, 1 idle" deadlock in ~3 min
instead of ~15. Run under the shipped FSDP accelerate launcher, e.g.:

    TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch \
        --config_file configs/accelerate_fsdp.yaml \
        -m scripts._repro_grpo_step <model_id>

TORCH_DISTRIBUTED_DEBUG=DETAIL converts any cross-rank collective mismatch into an
immediate explicit RuntimeError (tensor shapes / count), so a hang becomes a fast,
precise error.
"""

from __future__ import annotations

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kore.policy.configs import GRPOConfig
from kore.policy.grpo import (
    _accumulate_grpo_grads_distributed,
    _all_gather_object,
    _recompute_logp,
    build_grpo_accelerator,
)


def main() -> int:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-14B"
    cfg = GRPOConfig(
        model_id=model_id, distributed=True, use_lora=False,
        sharding_backend="fsdp", fsdp_version=1, bf16=True,
        gradient_checkpointing=True, ref_anchor_coef=1e-3,
        clip_ratio_low=0.2, clip_ratio_high=0.28,
    )
    acc = build_grpo_accelerator(cfg)
    world = acc.num_processes
    rank = acc.process_index
    dev = acc.device

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-6)
    model, opt = acc.prepare(model, opt)

    def _fake_sample(n_prompt: int, n_gen: int):
        # old_lp / ref_lp are PLAIN detached scalars (in the real code they are
        # computed during the rollout on the inner/unwrapped model, so building
        # samples issues NO FSDP collective). This isolates _accumulate's collectives.
        prompt_ids = torch.randint(0, 100, (1, n_prompt), device=dev)
        gen_ids = torch.randint(0, 100, (n_gen,), device=dev)
        gen_inputs = [(prompt_ids, gen_ids)]
        old_lp = torch.tensor(-1.5, device=dev)
        ref_lp = torch.tensor(-1.4, device=dev)
        return [0.0, gen_inputs, ref_lp, old_lp, n_gen, None]

    # RAGGED per-rank sample counts + varying seq lengths (the real-world trigger):
    # rank 0 -> 0 samples, rank 1 -> 1, rank 2 -> 2, ... (mod 3), different lengths.
    n_local = rank % 3
    local_terms = []
    local_tokens = 0
    for j in range(n_local):
        adv = float(j - 0.5)
        s = _fake_sample(n_prompt=8 + rank, n_gen=4 + j + rank)
        local_terms.append((adv, s))
        local_tokens += s[4]

    if rank == 0:
        print(f"[repro] world={world} building ragged local_terms (rank%3)", flush=True)

    # ------------------------------------------------------------------ #
    # Mirror the REAL step's pre-training sequence: inference-mode toggle +
    # summon-full-params + generate() + inner-model forward, THEN restore train
    # mode. This is the suspected deadlock trigger (FSDP state after summon/toggle).
    # ------------------------------------------------------------------ #
    from kore.policy.grpo import _summon_full_params_ctx
    _inner = getattr(model, "module", model)
    _inner.gradient_checkpointing_disable()
    _inner.config.use_cache = True
    _inner.eval()
    with _summon_full_params_ctx(model):
        prompt = torch.randint(0, 100, (1, 6 + rank), device=dev)  # ragged prompt len
        with torch.no_grad():
            gen = _inner.generate(prompt, max_new_tokens=4 + rank, do_sample=False,
                                  synced_gpus=True)
        # an inner-model logp forward too (mirrors old_lp recompute path)
        with torch.no_grad():
            _ = _inner(gen)
    _inner.config.use_cache = False
    _inner.train()
    _inner.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    _inner.enable_input_require_grads()
    acc.wait_for_everyone()
    if rank == 0:
        print("[repro] post-rollout toggle+summon done; entering training step", flush=True)

    all_tokens = _all_gather_object(local_tokens, acc)
    global_total_tokens = max(sum(all_tokens), 1)
    all_counts = _all_gather_object(len(local_terms), acc)
    max_micro = max(all_counts) if all_counts else 0
    print(f"[repro] rank={rank} n_local={len(local_terms)} max_micro={max_micro} "
          f"global_tokens={global_total_tokens}", flush=True)

    def _logp_fn(gen_inputs):
        return _recompute_logp(model, tok, gen_inputs, 1.0) if gen_inputs else None

    for step in range(2):  # two steps to exercise repeated collectives
        opt.zero_grad()
        loss_value, n_real = _accumulate_grpo_grads_distributed(
            local_terms, _logp_fn, accelerator=acc,
            global_total_tokens=global_total_tokens, grad_scale=float(world),
            max_micro_steps=max_micro, ref_anchor_coef=cfg.ref_anchor_coef,
            clip_ratio_low=cfg.clip_ratio_low, clip_ratio_high=cfg.clip_ratio_high,
            tok=tok, device=dev)
        acc.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        acc.wait_for_everyone()
        if rank == 0:
            print(f"[repro] STEP {step} OK loss={loss_value:.4f} n_real={n_real}", flush=True)

    acc.wait_for_everyone()
    if rank == 0:
        print("[repro] SUCCESS: distributed GRPO training step completed with ragged samples",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
