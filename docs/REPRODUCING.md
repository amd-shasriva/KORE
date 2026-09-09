# Reproducing KORE, end to end

What it takes to get from a fresh clone to each published number, what you need
that this repository cannot give you, and which parts genuinely cannot be
reproduced today.

Read this before `docs/DISTRIBUTED.md`, which is the operational detail for one
stage. This is the map.

## 0. Read this first: what is and is not reproducible

| Stage | Reproducible from a clone? |
| --- | --- |
| 1. Build the corpus | Yes for the mixture, no for generation. The v5 corpus ships committed; regenerating it needs an AMD-internal teacher endpoint. |
| 2. Supervised fine-tuning | Yes, given the hardware. Config, data and pinned model revision are all here. |
| 3. Multi-turn RL | Recipe yes, run no. The config is now committed; the SFT checkpoint it starts from is not in this repository. |
| 4. AgentKernelArena evaluation | Yes, given the benchmark checkout and a checkpoint to score. |

The single hard blocker is stage 3's input. The frontier run started from an SFT
checkpoint at `/mnt/vast/shasriva/models/sft_coder30b_a3b_v5`, on infrastructure
this cluster cannot see. You can rerun stage 2 to produce your own, but it will
not be bit-identical, so stage 3 and 4 numbers will differ. See
`docs/evidence/RL_RUN_PROVENANCE.md` for what the original run actually did.

Everything else below is honest about its prerequisites rather than assuming a
working SPUR account.

## 1. Environment

**Use `requirements-conductor.txt`, not `pyproject.toml`.** This matters and is
easy to get wrong. `pyproject.toml`'s dependencies float (`torch`,
`transformers>=4.44`) and omit `triton`, `matplotlib`, `vllm` and `aiter`
entirely, so `pip install -e .` alone produces an environment that imports but
does not work. `requirements-conductor.txt` is exact, and its ordering comments
are load-bearing: installing torch from PyPI first gets you a CUDA build that
silently fails on ROCm.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-conductor.txt   # follow the order in the file
pip install -e .
```

Python 3.10. ROCm 7.0 with `torch==2.10.0+rocm7.0` and
`pytorch-triton-rocm==3.5.1`. There is no container image; this cluster has no
container runtime.

Two known gaps: `tensorboard` is pinned in `requirements-conductor.txt` but
missing from at least one live venv, and `kore/policy/sft.py` hard-fails without
it. `flash_attn` has no ROCm wheel and its absence is expected.

## 2. Data

The corpus is committed as gzip parts, so no network and no hub account:

```bash
cd data/release && ./reassemble.sh
```

That writes `data/b05factory/sft/v5_sft.jsonl` (206,000 rows) and its held-out
half, and links both to the `data/v5_sft.jsonl` and `data/v5_eval.jsonl` paths
`configs/sft_coder30b_a3b.json` reads.

Regenerating the corpus rather than unpacking it is a different matter, and it
is the part an external reader cannot do. Generation drives a frontier teacher
through an AMD-internal gateway that requires `AMD_LLM_API_KEY` and `AMD_NTID`
in `.env.local`. There is no public substitute wired in. The entry points, for
completeness:

```bash
sbatch scripts/spur_build_task_pool.sbatch     # mine and screen the task pool
bash   scripts/spur_data_driver.sh             # agentic generation across nodes
python scripts/build_sft_v5_mixture.py         # assemble the mixture
python scripts/v5_split_eval.py                # carve the held-out slice
```

## 3. Supervised fine-tuning

```bash
nohup bash scripts/sft_supervise_v5.sh >/dev/null 2>&1 &
```

That chains through `scripts/spur_sft_1node.sbatch` with
`configs/sft_coder30b_a3b.json`. One epoch over 206,000 rows, 1,609 optimizer
steps, roughly 29 hours on 8 MI355X.

Do not pass a third positional argument overriding `output_dir`. The supervisor
derives its "did this finish" test from the config's own value, and a mismatch
means it cannot tell success from preemption.

The model revision is pinned (`b2cff646eb4bb1d68355c01b18ae02e7cf42d120`) and
the launcher runs offline, so the weights must already be in the HF cache.

## 4. Multi-turn RL

```bash
sbatch scripts/spur_grpo_1node.sbatch \
    configs/grpo_coder30b_a3b_trloo_frontier.json \
    <SFT_CHECKPOINT_DIR> <OUTPUT_DIR>
```

`configs/grpo_coder30b_a3b_trloo_frontier.json` is the recipe that produced
checkpoint-30. It was reconstructed by diffing the resolved config the trainer
received, and it differs from `configs/grpo_coder30b_a3b_trloo_burst.json` in
exactly three substantive keys: `learning_rate` 2e-06, `gradient_checkpointing`
false, `starpo_keep_frac` 0.875. `model_id` and `output_dir` are supplied by the
launcher's positional arguments.

Two things to know before you spend the allocation.

The learning rate contradicts its own lineage. The sibling configs argue for
1e-06 and the frontier run used 2e-06 anyway. That is recorded because it is
what ran, not because it is recommended.

`overlong_buffer_len` is 512 against a `max_response_length` of 1024, which
masks every response of 512 tokens or more rather than only truncated ones. It
cost the original run 908 training samples across 31 steps and is the leading
explanation for its flat speed result. Fix the buffer before rerunning unless
you are deliberately reproducing the original.

To run inside somebody else's allocation, the lender runs
`scripts/borrow_run_grpo.sh`; two nodes use `scripts/spur_grpo_2node.sbatch`.

## 5. Evaluation

Two passes into the **same** `--out`, baseline first:

```bash
sbatch scripts/spur_aka_1node.sbatch baseline <TYPES> <MODEL> <LIMIT> <ARM> <OUT>
sbatch scripts/spur_aka_1node.sbatch run      <TYPES> <MODEL> <LIMIT> <ARM> <OUT>
```

or queue the sweep with `scripts/queue_aka_final.sh`.

**The order is destructive if you get it wrong.** Running `run` first cost 246
of 302 correct kernels their speedup denominator permanently, because
workspaces are deleted after scoring. `run` now refuses below
`--min-baseline-coverage 0.5`. See `kore/eval/README.md`.

The benchmark is 416 tasks, measured against AgentKernelArena commit `b09f5eb`.
It is not vendored and not a submodule; clone it to `~/third_party/AgentKernelArena`
or set `KORE_AKA_ROOT`. Earlier figures of 402 and 413 in this repository are
stale.

## 6. External prerequisites

Nothing below is in this repository.

1. `Qwen/Qwen3-Coder-30B-A3B-Instruct` at revision `b2cff646eb4bb1d68355c01b18ae02e7cf42d120`, pre-fetched into the HF cache.
2. AgentKernelArena at `b09f5eb`.
3. `aiter`, for the vendor baselines. Without it the baseline silently falls back to torch, every measured speedup inflates, and RL optimises the wrong target.
4. HipKittens, for `scripts/build_hipkittens_sft.py`.
5. An AMD-internal LLM gateway key, for corpus generation only.
6. 8 MI355X (gfx950) per node, exclusive and idle. The launchers hard-exit otherwise.
7. A SLURM cluster. These scripts target SPUR, a modified controller, and encode workarounds that are wrong on stock Slurm.
8. Shared storage with 2-3 TB free for checkpoints.

## 7. Portability

These scripts were written for one cluster and say so. Known non-portable
surfaces, all of which need editing before this runs anywhere else:

`#SBATCH --output` paths are absolute and pre-resolved, and SLURM fails the job
if the directory does not exist. Roughly 283 occurrences of `/home/shasriva`
across `scripts/`, `kore/` and `configs/`. Account and QoS names
(`amd-general`, `amd-primus`, partition `amd-spur`) are site identities, and
their pairings are not interchangeable: `docs/CLUSTER_OPERATIONS.md` records
which combinations actually schedule. Hardware constants for gfx950 are baked
into `kore/analysis/roofline.py`, and the arch allowlists accept only gfx950 and
gfx942.

## 8. Determinism

No stage is deterministic, and the repository does not claim otherwise.

`seed` exists on all four training configs, defaults to 0, and is set by none of
the shipped recipes. It reaches `TrainingArguments` for SFT; GRPO uses it only
for task ordering, behind `KORE_TASK_ORDER=shuffled` which is not the default.
There is no `torch.use_deterministic_algorithms` and no MIOpen determinism flag
anywhere. SFT's `group_by_length` sampler draws from the global RNG rather than
`args.seed`, which on resume can retrain some rows and skip others; the code
adds a sampler fingerprint to make that detectable rather than to prevent it.

Determinism *is* enforced in one place, and it is the place that matters for
correctness rather than for repeatability: the oracle rejects any kernel whose
output drifts across a reseeded re-run, and task mining rejects any task whose
oracle is not deterministic under a fixed seed. That is determinism of the
verified artifact, not of the training run.
