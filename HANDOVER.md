# Handover

Where everything lives, what you need access to, and what it takes to pick this
up. Written on the last day of the internship that produced it, so it says what
is true rather than what was planned.

Start with `docs/REPRODUCING.md` for how to run the pipeline. This document is
about where the artifacts are and who to ask.

## 1. What exists, and where

| Artifact | Location | In this repo? |
| --- | --- | --- |
| Source, tests, scripts, configs | this repository | yes |
| v5 training corpus, 206,000 rows | `data/release/sft/v5_sft.jsonl.gz.part{aa..ad}` | yes, as sub-100 MB shards |
| v5 held-out eval slice, 899 rows | `data/release/sft/v5_eval.jsonl.gz` | yes |
| Earlier corpora (multicap, v4, midtrain, DPO) | `data/release/` | yes |
| Raw agentic episodes the corpus was distilled from | `data/release/episodes/` | yes, via `restore_upstream.sh` |
| Intermediate v5 build slices | `data/release/v5_stages/` | yes, via `restore_upstream.sh` |
| Contamination quarantines | `data/release/quarantine/` | yes, via `restore_upstream.sh` |
| 14B campaign | `data/release/campaign_14b/` | yes, via `restore_upstream.sh` |
| Arena evaluation runs behind the scores | `data/release/arena_runs/` | yes, via `restore_upstream.sh` |
| RL run evidence: resolved config, event log, three arena ledgers | `docs/evidence/rl_v5_frontier/` | yes |
| The RL recipe that produced checkpoint-30 | `configs/grpo_coder30b_a3b_trloo_frontier.json` | yes |
| RL checkpoint-30 (the model every result describes) | `/shared_nfs/shasriva/rl_checkpoint-30`, 342 GB | no, fetch with `scripts/fetch_weights.sh` |
| Arena ledgers for that checkpoint | `/shared_nfs/shasriva/aka_results_ckpt30`, 32 MB | no, on the cluster |
| SFT checkpoint the RL run started from | not located; see below | **no** |
| Base model, `Qwen3-Coder-30B-A3B-Instruct` @ `b2cff646` | HuggingFace Hub | no, and does not need to be |

Rebuild the corpus from a clean checkout with:

```bash
cd data/release && ./reassemble.sh
```

That writes `data/b05factory/sft/v5_sft.jsonl` and links it to the
`data/v5_sft.jsonl` path the SFT config reads. No network, no hub account.

### The committed corpus is the one that was trained on

Verified rather than assumed, because "the release matches the run" is the kind
of claim that is usually true and occasionally catastrophic. The file SFT
actually read was still on the origin machine at
`/home/shasriva/vultr_stage/KORE/data/v5_sft.jsonl`; reassembling the committed
shards reproduces it byte for byte.

| File | Rows | MD5 |
| --- | --- | --- |
| `v5_sft.jsonl` | 206,000 | `b91f1adb7d735c9a97f451ecc5e1ee01` |
| `v5_eval.jsonl` | 899 | `837186333e688fd16ff9ec538486d6ab` |

Nothing newer than this corpus exists. Every uncommitted data directory on the
origin machine predates it, so all of it is upstream of what is committed here.

### The upstream material is committed too

Everything the corpus was built *from* is in this repository as well, which was
not true for most of this project's life. 10.2 GB of raw material compresses to
941 MB across 15 sub-100 MB parts:

```bash
cd data/release && ./restore_upstream.sh                    # ~10 GB, check space
cd data/release && ./restore_upstream.sh /tmp/x episodes     # or one group
```

Nine bundles, each with a manifest carrying the SHA-256 of its reassembled
stream, which `restore_upstream.sh` verifies before extracting — a truncated
part is otherwise a silent corruption that surfaces much later as unparseable
JSON. The bundles hold the raw agentic episodes behind the corpus, each carrying
full multi-turn messages, phase traces, reflections and rewards verified on real
gfx950 (`episodes/`); the intermediate v5 build slices between mining and
mixture (`v5_stages/`); both contamination quarantines, kept as evidence the
gates ran and what they caught (`quarantine/`); the 14B campaign
(`campaign_14b/`); and the AgentKernelArena evaluation runs behind every
published score (`arena_runs/`).

This matters because `data/release/provenance/` holds only the *distilled*
groups, repair and wins — 12,286 entries. Rebuilding a corpus with a different
mixture, or auditing where one training row came from, needs the episodes, and
those cost GPU hours and gateway credits to produce.

Two exclusions inside the arena bundle, both deliberate: vendored `aiter`
subtrees and compiled objects were dropped as rebuildable, which is what takes
that material from 3.1 GB to 1 GB. The generated kernels and their
`eval_result.yaml` scores are kept, because those are the evidence.

Three things left out entirely as recreatable, so nobody hunts for them:
the replay pool, which is `allenai/tulu-3-sft-mixture` and redownloadable; the
13,570-task pool, which `scripts/build_task_pool.py` regenerates
deterministically from KernelBook at pinned revision
`b76504d85f7f14ef4b1fad81f136f638f2ce625b` plus template synthesis; and the
96,675 DPO pairs, which are already committed under `data/release/dpo/`.

## 2. The weights: where they are and how to get them

RL checkpoint-30 exists and is complete. It is at
`/shared_nfs/shasriva/rl_checkpoint-30` on the SPUR cluster, verified as 25
safetensors shards plus index, tokenizer, `optimizer.pt` and a
`trainer_state.json` reporting `global_step` 30, on a `qwen3_moe` config with
48 layers and 128 experts. One command fetches it:

```bash
./scripts/fetch_weights.sh /your/destination            # 114 GB, the model
./scripts/fetch_weights.sh /your/destination --resume   # 342 GB, + training state
```

**Take the 114 GB unless you intend to continue the interrupted RL run.** The
other 228 GB is optimizer, RNG and scheduler state, needed only to resume from
step 30 — not to evaluate, serve, or fine-tune. `KORE_SPUR_SSH` points the
script at a login node if you are not already on one.

`/shared_nfs/shasriva/KORE_HANDOFF/` collects the checkpoint, its arena
ledgers and a byte-size manifest behind symlinks, with its own README. It
duplicates nothing, because that volume is at 91%.

It cannot live in this repository: 342 GB against a 100 MB per-file limit, and
`LICENSE` separately forbids publishing a derived checkpoint to any public or
third-party registry without written AMD authorization.

Two things still open. **The SFT checkpoint was not found** — only the RL one
is on `/shared_nfs`, so it was likely rotated out, `save_total_limit` being 2.
Reproducing stage 3 from scratch therefore means rerunning stage 2 first,
roughly 29 hours on 8 MI355X. And **the path is a personal directory** on an
account being closed, so the single most urgent action in this document is
getting `/shared_nfs/shasriva/rl_checkpoint-30` copied somewhere team-owned, or
its ownership transferred.

**Every published number describes checkpoint-30 specifically.** If it is lost,
the figures in `docs/evidence/RL_RUN_PROVENANCE.md` remain attributable, since
the ledgers behind them are committed — but no stage is deterministic, so a
rerun produces a different checkpoint and the numbers would need re-measuring
rather than re-confirming.

## 3. Access you will need

| What | Why | Who to ask |
| --- | --- | --- |
| SPUR cluster account, `amd-general` | Every training and evaluation launcher | cluster ops |
| `/shared_nfs` read access | Holds RL checkpoint-30 and its arena ledgers. Files are world-readable; the risk is the directory, not the mode bits. | cluster ops |
| `AMD_LLM_API_KEY`, `AMD_NTID` | Corpus generation only. Not needed to train on the committed corpus. | AMD LLM gateway owners |
| AgentKernelArena checkout @ `b09f5eb` | Evaluation. Not vendored here. | the AKA repository |
| `aiter` source | Vendor baselines. Without it every speedup inflates. | the aiter repository |

`docs/CLUSTER_OPERATIONS.md` records which account and QoS pairings actually
schedule on SPUR, which is not obvious: a pairing can be accepted by `sbatch`
and then never become a scheduling candidate.

## 4. State of the work

Stages 1 and 2 are complete and reproducible. Stage 3 ran once, for 31 of a
configured 1,000 optimizer steps, and stopped; it was not converged. Stage 4
scored all four arms on 416 tasks.

The results, and the honest reading of them, are in
`docs/evidence/RL_RUN_PROVENANCE.md`. Correctness went 52.4% to 62.3% against
the fine-tuned model, paired at 63 tasks fixed and 22 broken. Compile rate went
82.7% to 93.8%. Speed did not move.

## 5. The first thing to fix

`overlong_buffer_len` is 512 against a `max_response_length` of 1024, so the
overlong mask dropped every response of 512 tokens or more rather than only the
truncated tail. It cost the run 908 training samples across 31 steps, counted
from the event log, and it falls hardest on long responses — which is where
optimised kernels live. It is the leading explanation for the flat speed
result.

Three changes, cheapest first:

1. Shrink the buffer to a small fraction of the limit rather than half of it.
2. Raise the generation limit. 1024 tokens is not room for a tiled kernel. This
   costs nodes, not code, or a trade of group width for context.
3. Log the mask rate as a live training metric, so the next run sees this at
   step one instead of in a post-mortem.

## 6. Things known to be true and not yet acted on

Recorded here because they were found and verified, not fixed.

`kore/policy/dpo.py` no longer consolidates its final save. Periodic
checkpoints moved to `SHARDED_STATE_DICT` so a run could resume, and the
consolidation that the cross-stage handoff needs was added to `train_sft` only.
A full fine-tune DPO run therefore writes a sharded final artifact that the
next stage cannot open on a different mesh. The parallel to copy is
`kore/policy/sft.py`'s final-save block.

`POOL_STREAMS=1` in `scripts/frontier_pipeline.sh` reintroduces a bug the tests
were written against: it seeds the pool roots but `GATE_ROOTS` still names only
the registry roots, so those seeds would never be gated. Under the default the
invariant holds. Making the switch safe to flip means growing `GATE_ROOTS`
alongside it.

`runs/shards_frontier_twins` is still refreshed by the pipeline and nothing
mines it. The merged `frontierhip` shard set is built by
`scripts/build_balanced_stream.py`, which is invoked by hand, so twins gated
after the stream merge do not automatically join a mined set.

## 7. Conventions worth keeping

Two mechanisms in this repository do real work and are cheap to preserve.

`tests/test_docs_contract.py` asserts that every filesystem path named in a
markdown file exists or is explicitly explained. It is the reason the
documentation here can be trusted; it was red for a month and is now green.
When it fails, the doc is usually right and the allowlist is stale — read the
failure before editing prose.

`scripts/operations_registry.json` names every operational script and says
whether it is active, diagnostic, deprecated or destructive, and a test derives
its expectation from the filesystem. Add a script, add an entry, in the same
change. Two scripts in it rewrite the committed corpus in place with no dry
run; that is exactly the kind of thing the registry exists to make visible.
