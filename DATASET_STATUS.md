# Dataset status

## Production direction

The product model is `Qwen/Qwen3-Coder-30B-A3B-Instruct`. It is trained by SFT
on the final mixture and then multi-turn RL; it does not consume a midtrain
corpus or a DPO-pair corpus. Those 14B artifacts remain cluster-side historical
data and test fixtures, not production inputs.

The retained 14B fixtures pin `Qwen/Qwen3-14B` at
`40c069824f4251a91eefaf281ebe4c544efd3e18`. This provenance preserves
reproducibility of the historical CPT/SFT/DPO tests; it is not a recommended
model target.

The next SFT output is `data/b05factory/sft/multicap_v3.jsonl`, built by
`scripts/build_sft_v3_mixture.py`. It admits the v2 base, recovered rows, and
step-centric AMD trajectories only after deduplication, held-out task/family
screening, and the 17,408-token length limit.

## Measured historical artifacts

These counts are pinned because the files are cluster-only and a changed artifact
must force an explicit review:

| Artifact | Rows | Status |
| --- | ---: | --- |
| `data/b05factory/midtrain/corpus.jsonl` | 86,010 | legacy 14B CPT corpus; not a production input |
| `data/b05factory/sft/multicap.jsonl` | 56,493 | earlier SFT base |
| `data/b05factory/dpo/pairs.jsonl` | 96,675 | legacy preference corpus; DPO is dropped for production |

The cluster paths are not expected in a fresh checkout. `data/release/reassemble.sh`
materializes packaged legacy artifacts where available.

## Data gates

- The task pool has 14,859 plannable tasks and 14,461 eligible tasks after
  screening. It contains 13,570 external tasks; 398 registry tasks are excluded
  because their seeds contaminate the held-out screen.
- The data driver has a six-node QoS ceiling. Measured agentic datagen is
  462–469 episodes per node-hour with 100% keep rate.
- `runs/DATA_NOT_FINAL` is an intentional training hold. The data driver never
  removes it; a person must review mixture counts before releasing SFT.
