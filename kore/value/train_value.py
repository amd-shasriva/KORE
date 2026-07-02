"""KORE value-model training + replay evaluation.

Loads a JSONL "value table" of logged moves and their real GPU outcomes, trains
the 3-head `ValueModel` with *throughput weighting* (weight = max(speedup, 0.1)
so the fit is accurate about the fast kernels), and reports how much cheaper the
learned ranker makes the search.

Analogue (KORE.pdf Sec 4.5):
  - This is the offline cost-model training loop from Ansor / TVM: accumulate
    (schedule -> measured throughput) pairs, fit a predictor, and validate it by
    replaying the log -- measuring benches-to-best and rank correlation against a
    random baseline. A good surrogate reaches the best kernel in far fewer real
    measurements, the core "measurement efficiency" claim of Ansor and the
    Compiler-World-Models line of work.

`synthesize_table(n)` fabricates a plausible table so this whole path (and the
tests) runs with zero real GPU data.
"""

from __future__ import annotations

import json
import os
import random
from typing import Optional

import numpy as np

from kore.value.features import featurize_many
from kore.value.model import ValueModel
from kore.value.rerank import (
    benches_to_best,
    rank_candidates,
    rank_correlation,
    score_candidates,
)

# Keys in a table row that describe the OUTCOME (everything else is meta).
_OUTCOME_KEYS = {"compiled", "snr_pass", "log_speedup", "speedup"}

_OPERATIONS = ["gemm", "matmul", "conv", "attention", "reduction", "elementwise", "norm"]
_DTYPES = ["fp32", "fp16", "bf16", "fp8"]
_BOTTLENECKS = ["compute", "balanced", "memory", "unknown"]


def _split_row(row: dict) -> tuple[dict, dict]:
    meta = {k: v for k, v in row.items() if k not in _OUTCOME_KEYS}
    outcome = {k: row[k] for k in _OUTCOME_KEYS if k in row}
    return meta, outcome


# --------------------------------------------------------------------------- #
# Synthetic table (smoke-test data with real signal)
# --------------------------------------------------------------------------- #
def synthesize_table(n: int, seed: int = 0) -> list[dict]:
    """Fabricate `n` plausible (meta + outcome) rows for smoke-testing.

    Outcomes are driven by a latent "quality" of the move so a fitted model can
    actually learn a positive rank correlation:
      - bigger diffs are riskier (lower P(compile), lower P(SNR-pass));
      - memory-bound kernels with big shapes have more speedup headroom.
    """
    rng = random.Random(seed)
    nprng = np.random.RandomState(seed)
    rows: list[dict] = []
    for _ in range(n):
        op = rng.choice(_OPERATIONS)
        dtype = rng.choice(_DTYPES)
        bottleneck = rng.choice(_BOTTLENECKS)
        M = int(2 ** rng.randint(6, 13))
        N = int(2 ** rng.randint(6, 13))
        K = int(2 ** rng.randint(6, 13))
        diff_size = int(abs(nprng.normal(80, 60))) + 1
        parent_snr = float(nprng.uniform(20, 60))
        parent_wall_ms = float(abs(nprng.normal(2.0, 1.5)) + 0.05)
        parent_vgpr = int(nprng.uniform(32, 256))

        # --- latent generative process ---
        risk = diff_size / 200.0
        p_compile = 1.0 / (1.0 + np.exp(-(2.2 - 3.0 * risk)))
        compiled = nprng.rand() < p_compile

        snr_logit = 1.5 - 2.5 * risk + 0.03 * (parent_snr - 30.0)
        p_snr = 1.0 / (1.0 + np.exp(-snr_logit))
        snr_pass = bool(compiled and (nprng.rand() < p_snr))

        # speedup headroom: memory-bound + large shapes -> more room to optimize
        mem_bonus = {"memory": 0.7, "balanced": 0.35, "compute": 0.08, "unknown": 0.2}[bottleneck]
        size_bonus = 0.08 * np.log10(max(M * N * K, 1) / 1e6)
        mu_ls = mem_bonus + size_bonus - 0.5 * risk
        log_speedup = float(nprng.normal(mu_ls, 0.10))
        if not snr_pass:
            # invalid kernels: no usable speedup
            log_speedup = float(nprng.normal(-0.3, 0.1))
        speedup = float(np.exp(log_speedup))

        rows.append(
            {
                "operation": op,
                "M": M,
                "N": N,
                "K": K,
                "dtype": dtype,
                "diff_size": diff_size,
                "parent_snr": parent_snr,
                "parent_wall_ms": parent_wall_ms,
                "parent_vgpr": parent_vgpr,
                "pmc_bottleneck": bottleneck,
                "compiled": bool(compiled),
                "snr_pass": bool(snr_pass),
                "log_speedup": log_speedup,
                "speedup": speedup,
            }
        )
    return rows


def _load_table(table_path: str) -> list[dict]:
    rows: list[dict] = []
    with open(table_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _realized_utility(outcome: dict) -> float:
    """What we actually get from benching this move: speedup if valid, else 0."""
    if outcome.get("compiled") and outcome.get("snr_pass"):
        return float(outcome.get("speedup", 0.0))
    return 0.0


def train_from_table(
    table_path: str,
    out_path: str,
    test_frac: float = 0.25,
    seed: int = 0,
    use_sklearn: Optional[bool] = None,
) -> dict:
    """Train a ValueModel from a JSONL table; save it; return + print metrics."""
    rows = _load_table(table_path)
    if not rows:
        raise ValueError(f"empty value table: {table_path}")

    rng = np.random.RandomState(seed)
    idx = np.arange(len(rows))
    rng.shuffle(idx)
    n_test = max(1, int(len(rows) * test_frac)) if len(rows) > 1 else 0
    test_idx = set(int(i) for i in idx[:n_test])

    train_metas, train_out = [], []
    test_metas, test_out = [], []
    for i, row in enumerate(rows):
        meta, outcome = _split_row(row)
        if i in test_idx:
            test_metas.append(meta)
            test_out.append(outcome)
        else:
            train_metas.append(meta)
            train_out.append(outcome)

    X = featurize_many(train_metas)
    y_compile = np.array([1 if o.get("compiled") else 0 for o in train_out])
    y_snr = np.array([1 if o.get("snr_pass") else 0 for o in train_out])
    y_ls = np.array([float(o.get("log_speedup", 0.0)) for o in train_out])
    # throughput weighting: emphasize accuracy on the fast kernels.
    sample_weight = np.array([max(float(o.get("speedup", 0.1)), 0.1) for o in train_out])

    model = ValueModel(use_sklearn=use_sklearn)
    model.fit(X, y_compile, y_snr, y_ls, sample_weight=sample_weight)
    model.save(out_path)

    metrics: dict = {"backend": model.backend, "n_train": len(train_metas), "n_test": len(test_metas)}

    if test_metas:
        pred_scores = score_candidates(model, test_metas)
        true_util = np.array([_realized_utility(o) for o in test_out])
        rc = rank_correlation(pred_scores, true_util)
        b2b = benches_to_best(pred_scores, true_util)
        # random baseline: expected position of the true best under random order
        rand_b2b = (len(test_metas) + 1) / 2.0
        metrics.update(
            {
                "rank_corr": rc,
                "benches_to_best": b2b,
                "benches_to_best_random": rand_b2b,
                "n_candidates": len(test_metas),
            }
        )
        print(
            f"[value] backend={model.backend} "
            f"train={len(train_metas)} test={len(test_metas)}"
        )
        print(f"[value] held-out Spearman rank-corr : {rc:+.3f}")
        print(
            f"[value] benches-to-best (ranked)    : {b2b} / {len(test_metas)}  "
            f"(random baseline ~{rand_b2b:.1f})"
        )
    else:
        print(f"[value] backend={model.backend} trained on {len(train_metas)} rows (no held-out split)")

    print(f"[value] saved model -> {out_path}")
    return metrics


if __name__ == "__main__":
    from kore.config import CONFIG

    out_dir = CONFIG.runs_dir / "value"
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = str(out_dir / "synth_table.jsonl")
    model_path = str(out_dir / "value_model.pkl")

    rows = synthesize_table(600, seed=1)
    with open(table_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[value] synthesized {len(rows)} rows -> {table_path}")

    train_from_table(table_path, model_path, seed=1)
