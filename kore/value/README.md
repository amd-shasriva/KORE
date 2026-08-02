# `kore/value` — the bench-prefilter surrogate

Benchmarking a kernel on real silicon is the expensive step. This package is a **cheap 3-head cost model** that ranks candidate kernels *before* they hit the GPU, so GRPO can bench only the top-k of the N candidates generated per turn instead of measuring all of them. It is a learned cost model in the standard autotuning lineage (Ansor / Tenset), trained from the run's *own* verified ranked groups; when there is not yet enough data it falls back to a hand-coded schedule heuristic, and both paths are fail-safe.

> **No trained model is shipped, and the flagship GRPO recipe has both consumers off.** See [Neither consumer is live in the flagship config](#neither-consumer-is-live-in-the-flagship-config) before assuming this package affects a run.

---

## Files

| File | Purpose |
| --- | --- |
| `features.py` | Fixed-length featurization: problem context + schedule features from kernel source |
| `model.py` | `ValueModel` (3 heads + optional pairwise ranker) |
| `rerank.py` | The GRPO-facing `rank_candidates` / `score_candidates` contract + replay-validation metrics |
| `train_value.py` | Offline training from a JSONL value table + online refit |
| `replay_train.py` | Train a schedule-conditioned `ValueModel` from the run's own verified ranked-group shards |
| `tests/` | `test_rerank_source_prior.py` (schedule-conditioned scoring, degenerate-prior fallback, throughput weighting) |

---

## The model

```python
class ValueModel:
    def predict(X) -> {"p_compile", "p_snr_pass", "e_log_speedup"}
utility = p_compile * p_snr_pass * exp(e_log_speedup)
```

Three heads — P(compile), P(SNR pass), E[log speedup] — trained with sklearn `HistGradientBoosting` (a pure-numpy logistic/ridge fallback when sklearn is absent), throughput-weighted (`sample_weight = max(speedup, 0.1)`) so fast kernels dominate the fit. An optional `PairwiseRanker` learns within-group ordering directly (RankNet-style pairwise logistic loss over same-group candidate pairs), supplying an ordering signal the pointwise regressor lacks.

**Features** (`features.py`): operator / dtype / shape / parent stats / PMC bottleneck (problem context) plus schedule features parsed from the kernel source (BLOCK sizes, `num_warps`, `num_stages`, `tl.dot` presence, tiling multiples, …). The schedule block makes the model **action-conditioned** — it sees the actual schedule a candidate encodes, not only the problem it targets. When a candidate carries no source, that block is all-zero and the vector layout stays backward-compatible.

---

## Use in GRPO

```python
def rank_candidates(items, task=None, model=None) -> list[int]   # best-first indices
def score_candidates(items, task=None, model=None) -> list[float]
def load_default_model(path=None)                                 # install a global default
```

GRPO calls `rank_candidates` to pick which of N generated kernels to bench when `value_prefilter` is on (`value_prefilter_k` sets how many survive; the `GRPOConfig` dataclass default is 4). With no trained model it falls back to `_heuristic_scores` (prefer `tl.dot`, 64-multiple tiles, sane warp/stage counts, an fp32 accumulator, bounds masking, a K reduction loop) so cold start never benches blindly. A bounded structural tie-breaker keeps genuinely distinct sources from collapsing to an identical score, and a usable-but-degenerate model (one that returns a near-constant utility over distinct candidates) defers to the heuristic so the PUCT prior stays informative.

**Offline training + validation** (`train_value.py`): `train_from_table` fits from a JSONL value table and reports held-out Spearman correlation, *benches-to-best* (how many benches the reranker saves vs. random order), and top-k recall. A `refit_online` path was removed: the replay JSONL stores only `(task_id -> Observation)` with no candidate source, so it cannot learn to separate sibling candidates and reproduces a degenerate model by construction. Refit from ranked groups via `replay_train.train_value_from_groups` instead.

---

## Training on the run's own ranked groups

`replay_train.py` trains the model directly from the campaign's **verified ranked-group shards** (`groups/*.jsonl`, the `RankedGroupRecord`s datagen already writes):

```python
def train_value_from_groups(groups_dir, out_path, *, cap=None, use_sklearn=None) -> dict   # -> heldout_group_rank_corr, ...
```

- **Why the ranked groups, not the replay cache.** The replay JSONL stores only `(task_id → Observation)` with **no source**, so it cannot learn to differentiate sibling candidates. A ranked group carries each candidate's **source** (hence the schedule features: block sizes, warps, stages, `tl.dot`, fp32-accum), its measured wall/speedup (a rank-based fallback when timing is absent), and the group structure — exactly the within-group ranking signal the top-k bench selector and the search prior consume. Every candidate in a ranked group is already verified-correct, so `compiled`/`snr_pass` are True and the differentiating outcome is the measured speedup.
- **Auto-trained pre-GRPO.** `run_campaign.py` trains the model from this run's `groups/` when `value_prefilter` is on and no `value_model_path` was supplied, installs it globally via `grpo._activate_value_ranker`, and reports the held-out group Spearman. (This is the path that would populate the missing artifact — but it is gated on `value_prefilter`, which the flagship config now sets to `false`, so it does not currently run.) It is pure/CPU (no GPU, no torch) and safe to run while datagen is still appending shards (append-only; it snapshots what exists). It is fail-safe: any shortfall (too little data, a malformed shard) leaves `value_model_path` unset and the ranker degrades to the source heuristic.

## Neither consumer is live in the flagship config

There are two consumers of a trained model — the bench-prefilter reranker and the [AlphaKernel](../search/README.md) PUCT search prior, both via `kore.value.rerank.score_candidates` — and `configs/grpo_14b_full.json` currently disables **both** (`value_prefilter: false`, `search_value_prior: false`). Two independent reasons, either alone decisive:

- **No artifact.** `value_model_path` is null, and the previous `runs/value/value_model.pkl` was deleted because it had been fit under a 28-feature layout while `kore.value.features.N_FEATURES` is now 47. `rerank._model_is_serviceable` rejects exactly that mismatch, and `load_default_model(None)` returns `None`, so `grpo._activate_value_ranker` installs nothing.
- **No prefilter consumer under `agentic: true`.** Only the serial `_rollout` calls `_prefilter_bench_indices`; `_rollout_agentic` drives the tool harness and never generates `num_candidates_per_turn` candidates to filter. `kore.policy.capabilities.validate_grpo_config` encodes this as the hard error *"agentic `value_prefilter` has no on-policy consumer"*. So turning it on did not merely degrade the ranker to the heuristic — the whole generate-N-bench-top-k step was absent, and the recipe's implied measurement economics never existed.

The search prior is the easier of the two to restore, because its consumer (the off-policy search-then-distill hook) *is* live: train an artifact with `kore.value.train_value` or `replay_train.train_value_from_groups` against the current featurizer, set `value_model_path`, and flip `search_value_prior` back with it. Restoring `value_prefilter` additionally needs either a serial run (`agentic: false`) or a prefilter consumer added to the agentic harness.

Both paths remain fail-safe in the meantime: with no serviceable model, `score_candidates` falls back to the source heuristic, which is still a sane best-first order.

See also: [`kore/policy`](../policy/README.md) (the `value_prefilter` and search-then-distill consumers), [`kore/search`](../search/README.md) (the AlphaKernel search prior + its production `TransformProposePolicy`), [`kore/env`](../env/README.md) (produces the labels).
