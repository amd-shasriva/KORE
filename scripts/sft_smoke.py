"""Tiny real SFT run to prove the LoRA training path works end-to-end.

    PYTHONPATH=. python scripts/sft_smoke.py --data data/sft/train.jsonl
"""

from __future__ import annotations

import argparse

from kore.policy.configs import SFTConfig
from kore.policy.sft import train_sft


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft/train.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--out", default="runs/sft_smoke")
    args = ap.parse_args()

    cfg = SFTConfig(model_id=args.model, output_dir=args.out)
    cfg.num_train_epochs = 1
    cfg.per_device_train_batch_size = 1
    cfg.gradient_accumulation_steps = 1
    cfg.max_seq_length = 2048
    cfg.logging_steps = 1
    cfg.save_steps = 10_000
    out = train_sft(cfg, args.data)
    print(f"[sft_smoke] OK -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
