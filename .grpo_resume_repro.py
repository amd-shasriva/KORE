#!/usr/bin/env python3
"""Minimal reproduction of the GRPO resume crash, on a tiny real Qwen3.

Exercises the REAL helpers (_gather_full_optim_state / _load_full_optim_state)
under the REAL GRPO accelerate plugin, so the result is about kore's code rather
than about a hand-rolled FSDP setup.

Run with:  accelerate launch --config_file configs/accelerate_fsdp_grpo.yaml \
               --num_processes 2 .grpo_resume_repro.py
"""
from __future__ import annotations

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from kore.policy.grpo import (
    _gather_full_optim_state,
    _load_full_optim_state,
    build_grpo_accelerator,
)


class _Cfg:
    sharding_backend = "auto"
    fsdp_version = 1
    zero_stage = 2
    cpu_offload = False
    bf16 = True
    distributed = True
    use_lora = False
    output_dir = "/tmp/grpo_repro"
    mixed_precision = "bf16"


def _tiny_qwen3():
    config = AutoConfig.for_model(
        "qwen3", hidden_size=128, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        vocab_size=256, max_position_embeddings=128, tie_word_embeddings=True)
    return AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)


def _report(tag, optimizer, accelerator):
    inner = getattr(optimizer, "optimizer", optimizer)
    seen = {}
    for group in inner.param_groups:
        for param in group["params"]:
            st = inner.state.get(param)
            if not st:
                continue
            for key, value in st.items():
                if torch.is_tensor(value):
                    seen.setdefault(key, set()).add((str(value.dtype), str(value.device)))
            seen.setdefault("PARAM", set()).add((str(param.dtype), str(param.device)))
    if accelerator.is_main_process:
        print(f"[{tag}]")
        for key in sorted(seen):
            print(f"    {key:12s} -> {sorted(seen[key])}")


def _one_step(model, optimizer, accelerator):
    ids = torch.randint(0, 200, (1, 16), device=accelerator.device)
    loss = model(input_ids=ids, labels=ids).loss
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()


def main():
    accelerator = build_grpo_accelerator(_Cfg())
    torch.manual_seed(0)

    model = _tiny_qwen3()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model, opt = accelerator.prepare(model, opt)
    _one_step(model, opt, accelerator)          # populate exp_avg / exp_avg_sq
    _report("after a real step (the SAVE side)", opt, accelerator)

    full_osd = _gather_full_optim_state(model, opt, accelerator)
    if accelerator.is_main_process:
        any_state = next(iter(full_osd["state"].values()))
        print("[gathered full_osd] exp_avg ->",
              any_state["exp_avg"].dtype, any_state["exp_avg"].device)

    # ---- fresh process state, exactly like a requeued child ---- #
    torch.manual_seed(0)
    model2 = _tiny_qwen3()
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
    model2, opt2 = accelerator.prepare(model2, opt2)
    _load_full_optim_state(model2, opt2,
                           full_osd if accelerator.is_main_process else None,
                           accelerator)
    _report("after _load_full_optim_state (the RESUME side)", opt2, accelerator)

    try:
        _one_step(model2, opt2, accelerator)
    except RuntimeError as error:
        if accelerator.is_main_process:
            print(f"[REPRODUCED] opt.step() after resume raised: {error}")
        # ---- candidate fix: re-home every state tensor onto its parameter ---- #
        inner = getattr(opt2, "optimizer", opt2)
        moved = 0
        for group in inner.param_groups:
            for param in group["params"]:
                st = inner.state.get(param)
                if not st:
                    continue
                for key, value in list(st.items()):
                    if key == "step" or not torch.is_tensor(value):
                        continue
                    if value.is_floating_point() and value.dtype != torch.float32:
                        st[key] = value.to(dtype=torch.float32)
                        moved += 1
        if accelerator.is_main_process:
            print(f"[fix] restored {moved} state tensors to fp32")
        _report("after the fix", opt2, accelerator)
        _one_step(model2, opt2, accelerator)
        if accelerator.is_main_process:
            print("[fix] opt.step() after resume now SUCCEEDS")
    else:
        if accelerator.is_main_process:
            print("[NOT REPRODUCED] opt.step() after resume succeeded unchanged")


if __name__ == "__main__":
    main()
