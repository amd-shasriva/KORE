# SFT launch readiness: 30B MoE direct instruct

**Verdict: data-gated GO.** The production stage starts from
`Qwen/Qwen3-Coder-30B-A3B-Instruct` and uses
`configs/sft_coder30b_a3b.json`. It must not inherit a 14B midtrain checkpoint:
the target has no Base sibling, and the historical instruct-CPT run destroyed
instruction-following. `runs/DATA_NOT_FINAL` is therefore an intentional stop
between datagen and training, not a failed automation.

This replaces the former 14B readiness narrative while preserving the
operational lessons that still constrain a 30B launch. Historical figures are
labelled as such; they are not evidence that the 30B run has completed.

## Launch contract

The input is `data/b05factory/sft/multicap_v3.jsonl`, produced by
`scripts/build_sft_v3_mixture.py` after task-pool decontamination,
step-centric extraction, deduplication, held-out-family exclusion, and the
17,408-token gate. It is cluster-only until materialized.

```json
{
  "model_id": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
  "dataset_path": "data/b05factory/sft/multicap_v3.jsonl",
  "output_dir": "runs/sft_coder30b_a3b",
  "max_seq_length": 17408,
  "num_train_epochs": 1,
  "save_total_limit": 1
}
```

The full pinned configuration and FSDP wrap class are in
[`configs/sft_coder30b_a3b.json`](../configs/sft_coder30b_a3b.json);
[`DISTRIBUTED.md`](DISTRIBUTED.md) is the executable launch reference.

## Durable failure modes

### Dataset validation must precede model loading

The old SFT launcher did not read the dataset until after every rank loaded the
model. A misspelled path therefore consumed minutes of multi-GPU allocation only
to raise `FileNotFoundError`. `train_sft` now checks the path before
`from_pretrained`. Do not bypass that preflight: a missing v3 mixture means the
data review has not released training.

### A checkpoint rotation needs headroom

Historical 14B full-state checkpoints were 221 GB, with a 55 GiB final model.
The relevant lesson scales up: rotation briefly holds the old and new complete
checkpoints simultaneously. For the 30.5B MoE model a checkpoint is about
488 GB, so `save_total_limit: 1` is mandatory: rotation peaks near 976 GB on
the ~1,090 GB shared volume. Raising the limit or assuming deletion happens
before writing turns a recoverable training interruption into an ENOSPC failure.

### Resume must reject half-written checkpoints

`latest_checkpoint` searches newest-first and accepts only a directory with
`trainer_state.json`. This is deliberate: a requeue during a save must fall
back to the preceding generation rather than restart at step zero or resume
partial state. Retain this check in any SFT refactor.

### MoE layers require explicit FSDP wrapping

The 30B configuration wraps `Qwen3MoeDecoderLayer`. Generic decoder-layer
matching can shard experts incorrectly, producing a run that starts but has the
wrong memory and communication topology. The wrap class is part of the launch
contract, not a cosmetic accelerator setting.

### The sequence cap is a data contract

`max_seq_length: 17408` matches the v3 mixture filter. Changing either side
silently changes the population that reaches training. The former 14B audit
found that a 16,384 cap removed 53.6% of its math slice; the general failure mode
is selective capability deletion, not merely a lower row count.

## Capacity estimate and observability

The v3 mixture yields roughly 478 SFT steps for one epoch at the configured
effective batch. The measured planning range is 90–120 seconds per step,
or roughly 12–16 hours, leaving margin inside SPUR's 23-hour allocation. These
are planning measurements, not a substitute for reading the first production
log; re-estimate if row count, sequence cap, batch, or FSDP topology changes.

Before submitting, verify:

1. `runs/DATA_NOT_FINAL` still exists until a person releases the mixture.
2. `data/b05factory/sft/multicap_v3.jsonl` exists and is the reviewed output.
3. the output filesystem has room for two 488 GB checkpoints during rotation.
4. the submitted command passes the production config and `-` as the starting
   model override, preserving the vendor instruct checkpoint.
5. logs report the Qwen3 MoE wrap class and a valid resume candidate, if one
   exists.

## Cluster launch

```bash
export SPUR_CONTROLLER_ADDR="http://crs-m2m-cpu-spur-005:6817"
cd /home/shasriva/Kore-RL/KORE
sbatch scripts/spur_sft_1node.sbatch \
  configs/sft_coder30b_a3b.json - runs/sft_coder30b_a3b
```

The login node is not a scheduler controller. Exporting the controller address
before `sbatch`, `squeue`, or a requeue check prevents a false “controller down”
diagnosis.

## Regression contract

`tests/test_sft_launch_readiness.py` currently collects **36 pass** in the
default suite plus one `release`-marked full-corpus check. The test count is
intentional: this document and `tests/test_docs_contract.py` keep it pinned, so
a readiness test cannot disappear without an explicit documentation change.

The old 14B measurements remain historical evidence only: full-state format can
inflate disk use, host-side dataset parsing is not free, and the first real
steps—not an extrapolation—are the authority for a new architecture.
