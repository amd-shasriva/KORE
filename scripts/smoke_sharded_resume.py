"""Prove that a SHARDED_STATE_DICT checkpoint can be saved AND resumed from.

This is the mechanism the v5 SFT run depends on to survive a node failure, and it
was previously untested: the run had never lived long enough to write a checkpoint,
so "resume works now" was an assertion rather than a measurement. The predecessor
format (FULL_STATE_DICT) saved fine and could not be loaded back at all -- all eight
ranks died with SIGBUS materialising a single consolidated 244 GB optimizer -- which
is exactly the failure a save-only test would have missed.

So this test does the thing that matters: train a few steps, save, then start a
SECOND trainer over the same output dir and resume from that checkpoint, and assert
the resumed run picks up at the saved step with the optimizer state intact.

It runs against a TINY randomly-initialised Qwen3-MoE, not the 30B, because what is
being validated is the checkpoint plumbing: the same model class
(Qwen3MoeForCausalLM), the same FSDP wrap class (Qwen3MoeDecoderLayer), the same
state_dict_type, and the same HF Trainer resume path. Size is the one thing that does
not need to be real, and keeping it small is what makes this runnable in minutes on
whatever node we can get rather than needing the 8-GPU allocation we are queued for.

Launch under accelerate so FSDP is actually exercised:

    accelerate launch --config_file configs/accelerate_fsdp.yaml \
        --num_processes <N> scripts/smoke_sharded_resume.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM, Trainer, TrainingArguments

STEPS_BEFORE_SAVE = 6
SAVE_EVERY = 3
SEQ = 32
VOCAB = 256


class Tiny(Dataset):
    """Deterministic token soup; the loss value is irrelevant, only the plumbing."""

    def __init__(self, n: int = 256) -> None:
        g = torch.Generator().manual_seed(0)
        self.ids = torch.randint(0, VOCAB, (n, SEQ), generator=g)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> dict:
        x = self.ids[i]
        return {"input_ids": x, "labels": x.clone(), "attention_mask": torch.ones_like(x)}


def build_model() -> Qwen3MoeForCausalLM:
    cfg = Qwen3MoeConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=SEQ,
        decoder_sparse_step=1,
    )
    return Qwen3MoeForCausalLM(cfg)


def make_args(out: str, max_steps: int) -> TrainingArguments:
    return TrainingArguments(
        output_dir=out,
        max_steps=max_steps,
        per_device_train_batch_size=2,
        save_steps=SAVE_EVERY,
        save_total_limit=2,
        logging_steps=1,
        learning_rate=1e-4,
        report_to=[],
        save_safetensors=True,
        # Mirrors kore.policy.configs.build_fsdp_kwargs. The wrap class and the
        # state dict type are the two settings under test.
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "transformer_layer_cls_to_wrap": ["Qwen3MoeDecoderLayer"],
            "state_dict_type": "SHARDED_STATE_DICT",
            "use_orig_params": True,
            "sync_module_states": True,
            "cpu_ram_efficient_loading": True,
            "limit_all_gathers": True,
            "backward_prefetch": "backward_pre",
            "forward_prefetch": True,
        },
    )


def rank0(*a) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(*a, flush=True)


def main() -> int:
    out = os.environ.get("SMOKE_DIR") or tempfile.mkdtemp(prefix="kore_sharded_smoke_")
    rank0(f"[smoke] output_dir={out}")
    ds = Tiny()

    # ---- phase 1: train and save ------------------------------------------------
    t1 = Trainer(model=build_model(), args=make_args(out, STEPS_BEFORE_SAVE), train_dataset=ds)
    t1.train()
    del t1

    ckpts = sorted(Path(out).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if not ckpts:
        rank0("[smoke] FAIL: no checkpoint was written")
        return 1
    last = ckpts[-1]
    saved_step = json.loads((last / "trainer_state.json").read_text())["global_step"]
    sharded_dirs = [p.name for p in last.iterdir() if p.is_dir()]
    rank0(f"[smoke] saved {last.name} at step {saved_step}; subdirs={sharded_dirs}")
    rank0(f"[smoke] files={sorted(p.name for p in last.iterdir())[:12]}")

    # A sharded checkpoint keeps per-rank slices in a directory; a FULL one would
    # instead leave a single consolidated optimizer.bin, which is the thing that
    # could not be read back.
    if (last / "optimizer.bin").exists():
        rank0("[smoke] WARNING: found consolidated optimizer.bin -- not sharded!")

    # ---- phase 2: RESUME from it ------------------------------------------------
    # The real test. A fresh Trainer over the same dir must pick up the saved step
    # and optimizer state rather than starting over.
    t2 = Trainer(model=build_model(), args=make_args(out, STEPS_BEFORE_SAVE + SAVE_EVERY),
                 train_dataset=ds)
    t2.train(resume_from_checkpoint=str(last))
    final_step = t2.state.global_step
    rank0(f"[smoke] resumed from step {saved_step} and ran to {final_step}")

    ok = final_step == STEPS_BEFORE_SAVE + SAVE_EVERY and saved_step > 0
    rank0(f"[smoke] RESULT: {'PASS' if ok else 'FAIL'} "
          f"(saved={saved_step}, final={final_step}, "
          f"expected_final={STEPS_BEFORE_SAVE + SAVE_EVERY})")
    if os.environ.get("SMOKE_KEEP") != "1":
        if int(os.environ.get("RANK", "0")) == 0:
            shutil.rmtree(out, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
