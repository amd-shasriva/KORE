# `kore/value` — the bench-prefilter surrogate

Benchmarking a kernel on real silicon is the expensive step. This package is a **cheap 3-head cost model** that ranks candidate kernels *before* they hit the GPU, so GRPO benches only the top-k of the N candidates generated per turn instead of measuring all of them. It is a learned cost model in the standard autotuning lineage (Ansor / Tenset), trained from the run's *own* verified ranked groups; when there is not yet enough data it falls back to a hand-coded schedule heuristic, and both paths are fail-safe.

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

GRPO calls `rank_candidates` to pick which of N generated kernels to bench (`value_prefilter=true`, `value_prefilter_k=4`). With no trained model it falls back to `_heuristic_scores` (prefer `tl.dot`, 64-multiple tiles, sane warp/stage counts, an fp32 accumulator, bounds masking, a K reduction loop) so cold start never benches blindly. A bounded structural tie-breaker keeps genuinely distinct sources from collapsing to an identical score, and a usable-but-degenerate model (one that returns a near-constant utility over distinct candidates) defers to the heuristic so the PUCT prior stays informative.

**Offline training + validation** (`train_value.py`): `train_from_table` fits from a JSONL value table and reports held-out Spearman correlation and *benches-to-best* (how many benches the reranker saves vs. random order); `refit_online` grows the buffer from live env replay.

---

## Training on the run's own ranked groups

`replay_train.py` trains the model directly from the campaign's **verified ranked-group shards** (`groups/*.jsonl`, the `RankedGroupRecord`s datagen already writes):

```python
def train_value_from_groups(groups_dir, out_path, *, cap=None, use_sklearn=None) -> dict   # -> heldout_group_rank_corr, ...
```

- **Why the ranked groups, not the replay cache.** The replay JSONL stores only `(task_id → Observation)` with **no source**, so it cannot learn to differentiate sibling candidates. A ranked group carries each candidate's **source** (hence the schedule features: block sizes, warps, stages, `tl.dot`, fp32-accum), its measured wall/speedup (a rank-based fallback when timing is absent), and the group structure — exactly the within-group ranking signal the top-k bench selector and the search prior consume. Every candidate in a ranked group is already verified-correct, so `compiled`/`snr_pass` are True and the differentiating outcome is the measured speedup.
- **Auto-trained pre-GRPO.** `run_campaign.py` trains the model from this run's `groups/` when `value_prefilter` is on and no `value_model_path` was supplied, installs it globally via `grpo._activate_value_ranker`, and reports the held-out group Spearman. It is pure/CPU (no GPU, no torch) and safe to run while datagen is still appending shards (append-only; it snapshots what exists). It is fail-safe: any shortfall (too little data, a malformed shard) leaves `value_model_path` unset and the ranker degrades to the source heuristic.

In the flagship 14B configuration (`configs/grpo_14b_full.json`) both `value_prefilter` and `use_search` (`search_budget=16`, `search_every=50`) are enabled, so a model trained from `groups/` backs **both** consumers — the bench-prefilter reranker and the [AlphaKernel](../search/README.md) PUCT search prior — through `kore.value.rerank.score_candidates`. If the run's `groups/` are too sparse to train a model, both consumers fall back to the source heuristic (still a sane best-first order).

See also: [`kore/policy`](../policy/README.md) (the `value_prefilter` and search-then-distill consumers), [`kore/search`](../search/README.md) (the AlphaKernel search prior + its production `TransformProposePolicy`), [`kore/env`](../env/README.md) (produces the labels).
