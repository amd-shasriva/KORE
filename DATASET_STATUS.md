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

The production SFT mixture is v5 (`data/v5_sft.jsonl`, with its held-out half
at `data/v5_eval.jsonl`), which is what `configs/sft_coder30b_a3b.json` points
at. It is built by a seven-stage chain, in this order:

```text
scripts/v5_stage1_gather.py     -> runs/v5_build/stage1.pkl
scripts/v5_stage2_translate.py  -> the re-posed translate slice
scripts/v5_stage3_recover.py    -> runs/v5_build/
scripts/v5_stage4_mixture.py    -> the mixture (data/v5_sft.jsonl)
scripts/v5_split_eval.py        -> the eval half, and rewrites the mixture
scripts/v5_fix_truncated.py     -> both files
scripts/v5_verify.py            -> the twelve correctness gates
```

`data/b05factory/sft/multicap_v3.jsonl`, built by
`scripts/build_sft_v3_mixture.py`, is the superseded v3 mixture. It admits the
v2 base, recovered rows, and step-centric AMD trajectories only after
deduplication, held-out task/family screening, and the 17,408-token length
limit, and it is retained to reproduce that build, not as the next SFT input.
See [`docs/DATASET_SPEC.md`](docs/DATASET_SPEC.md) for the v5 contract and for
the v3/v4 naming trap around that filename.

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
