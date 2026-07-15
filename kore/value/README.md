# `kore/value` - the bench-prefilter surrogate

Benchmarking a kernel on real silicon is the expensive step. This package is a **cheap 3-head cost model** (Ansor/NLTSP-style) that ranks candidate kernels *before* they hit the GPU, so GRPO benches only the top-k of N generated candidates per turn - roughly a 4× measurement-efficiency win.

---

## Files

| File | Purpose |
| --- | --- |
| `features.py` | Fixed-length featurization: problem context + schedule features from kernel source |
| `model.py` | `ValueModel` (3 heads + optional pairwise ranker) |
| `rerank.py` | The GRPO-facing `rank_candidates` / `score_candidates` contract + replay-validation metrics |
| `train_value.py` | Offline training from a JSONL value table + online refit |

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

See also: [`kore/policy/grpo.py`](../policy/README.md) (consumer), [`kore/env`](../env/README.md) (produces the labels).
