# `kore/value` - the bench-prefilter surrogate

Benchmarking a kernel on real silicon is the expensive step. This package is a **cheap 3-head cost model** (Ansor/NLTSP-style) that ranks candidate kernels *before* they hit the GPU, so GRPO benches only the top-k of N generated candidates per turn - roughly a 4× measurement-efficiency win.

> **Honest framing (no novelty claim).** This is an *incremental learned cost model* in the standard autotuning lineage (Ansor, Tenset, Compiler-World-Models) - a surrogate that ranks candidates so fewer are measured. KORE's only twist is training it from the run's *own* verified ranked groups (below); the model, features, and utility are deliberately conventional. It is **trained-from-groups when enough data exists, else a hand-coded schedule heuristic** - both paths are fail-safe.

---

## Files

| File | Purpose |
| --- | --- |
| `features.py` | Fixed-length featurization: problem context + schedule features from kernel source |
| `model.py` | `ValueModel` (3 heads + optional pairwise ranker) |
| `rerank.py` | The GRPO-facing `rank_candidates` / `score_candidates` contract + replay-validation metrics |
| `train_value.py` | Offline training from a JSONL value table + online refit |
| `replay_train.py` | Train a schedule-conditioned `ValueModel` from the run's own verified ranked-group shards |

---

## The model

```python
class ValueModel:
    def predict(X) -> {"p_compile", "p_snr_pass", "e_log_speedup"}
utility = p_compile * p_snr_pass * exp(e_log_speedup)
```

Three heads - P(compile), P(SNR pass), E[log speedup] - trained with sklearn `HistGradientBoosting` (numpy fallback if sklearn is absent), throughput-weighted (`sample_weight = max(speedup, 0.1)`) so fast kernels dominate the fit. An optional `PairwiseRanker` learns within-group ordering.

**Features** (`features.py`): operator / dtype / shape / parent stats / PMC bottleneck (problem context) plus schedule features parsed from the kernel source (BLOCK sizes, `num_warps`, `num_stages`, `tl.dot` presence, tiling multiples, …).

---

## Use in GRPO

```python
def rank_candidates(items, task=None, model=None) -> list[int]   # best-first indices
def score_candidates(items, task=None, model=None) -> list[float]
def load_default_model(path=None)                                 # install a global default
```

GRPO calls `rank_candidates` to pick which of N generated kernels to bench. With no trained model it falls back to `_heuristic_scores` (gfx942 discipline: prefer `tl.dot`, 64-multiple tiles, sane warp/stage counts) so cold start never benches blindly.

**Offline training + validation** (`train_value.py`): `train_from_table` fits from a JSONL value table and reports held-out Spearman correlation and *benches-to-best* (how many benches the reranker saves vs. random order); `refit_online` grows the buffer from live env replay.

---

## Paradigm-v2: training on the run's own ranked groups

`train_value.py` fits from a JSONL *value table*; paradigm-v2 adds `replay_train.py`, which trains the model directly from the campaign's own **verified ranked-group shards** (`groups/*.jsonl`, the `RankedGroupRecord`s datagen already writes):

```python
def train_value_from_groups(groups_dir, out_path, *, cap=None, use_sklearn=None) -> dict   # -> heldout_group_rank_corr, ...
```

- **Why the ranked groups, not the replay cache.** The replay JSONL stores only `(task_id → Observation)` with **no source**, so it cannot learn to differentiate sibling candidates. A ranked group carries each candidate's **source** (hence the schedule features: block sizes, warps, stages, `tl.dot`, fp32-accum), its measured wall/speedup (a rank-based fallback when timing is absent), and the group structure - exactly the within-group ranking signal the top-k bench selector and the search prior consume. Every candidate in a ranked group is already verified-correct, so `compiled`/`snr_pass` are True and the differentiating outcome is the measured speedup.
- **Auto-trained pre-GRPO.** `run_campaign.py` trains it from this run's `groups/` when `value_prefilter` is on and no `value_model_path` was supplied, then installs it globally via `grpo._activate_value_ranker` and reports the held-out group Spearman. Pure/CPU (no GPU, no torch), and safe to run while datagen is still appending shards (append-only; it snapshots what exists). Fully fail-safe: any shortfall (too little data, malformed shard) leaves `value_model_path` unset and the ranker degrades to the source heuristic.

> **Previously dormant/untrained → now grounded.** Before paradigm-v2 nothing trained the model on real run data - `value_model_path` stayed unset, so `rank_candidates`/`score_candidates` fell back to the hand-coded `_heuristic_scores` cold start. `replay_train.py` fixes the *data* half so the model is fit on this run's own verified measurements. *Active in the live run* (`configs/grpo_14b_full.json`): `value_prefilter = true` **and** `use_search = true` (`search_budget = 16`, `search_every = 50`), so the trained value model backs **both** consumers - the bench-prefilter reranker **and** the live **AlphaKernel search prior** - *when the pre-GRPO training-from-groups produced a model*; if the run's `groups/` were too sparse, both consumers fall back to the source heuristic (still a sane best-first order, just ungrounded). The search prior consumes the same model through `kore.value.rerank.score_candidates` (the AlphaKernel value/PUCT prior) for the production `TransformProposePolicy` (`kore/search/propose.py` + `search_from_kernel`), run as a throttled, fail-safe, off-policy search-then-distill hook in `grpo.py`.

See also: [`kore/policy/grpo.py`](../policy/README.md) (search-then-distill consumer), [`kore/env`](../env/README.md) (produces the labels), `kore/search/alphakernel.py` + `kore/search/propose.py` (the now-live AlphaKernel search prior + its production `TransformProposePolicy`).
