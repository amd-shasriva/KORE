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
import math
import random
from typing import Optional

import numpy as np

from kore.value.features import featurize_many
from kore.value.model import ValueModel
from kore.value.rerank import (
    benches_to_best,
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
        pred_scores = score_candidates(test_metas, model=model)
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


# --------------------------------------------------------------------------- #
# Within-group RANKING objective (pairwise/listwise, throughput-weighted)
#
# The pointwise regressor fits an absolute E[log speedup]; the ranking head fits
# the *order within a group of siblings* (the candidates generated for one
# parent) - which is what the top-k bench selector actually consumes. Both are
# kept: the pointwise heads gate validity, the ranking head orders the survivors.
# --------------------------------------------------------------------------- #
def _synth_source(bm: int, bn: int, bk: int, warps: int, stages: int,
                  use_dot: bool, fp32_acc: bool) -> str:
    """A tiny but structurally-real Triton kernel string whose SCHEDULE knobs
    (block sizes, warps, stages, tl.dot/MFMA, fp32 accumulate) are recoverable by
    features.extract_schedule_features - so the ranker is schedule-conditioned."""
    acc = "float32" if fp32_acc else "bfloat16"
    inner = "acc += tl.dot(x, x)" if use_dot else "acc += x * x"
    return (
        "import triton\n"
        "import triton.language as tl\n\n"
        "@triton.jit\n"
        "def _k(a_ptr, b_ptr, c_ptr, M, N, K,\n"
        "       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):\n"
        "    offs = tl.arange(0, BLOCK_K)\n"
        f"    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.{acc})\n"
        "    for kk in range(0, K, BLOCK_K):\n"
        "        x = tl.load(a_ptr + offs, mask=offs < K, other=0.0)\n"
        f"        {inner}\n"
        "    tl.store(c_ptr + offs, acc, mask=offs < N)\n\n"
        "def entry(a, b):\n"
        f"    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = {bm}, {bn}, {bk}, 8\n"
        f"    return _k[(1,)](a, b, num_warps={warps}, num_stages={stages})\n"
    )


def synthesize_groups(n_groups: int, group_size: int = 6, seed: int = 0) -> list[list[dict]]:
    """Fabricate ranked candidate GROUPS for the ranking objective.

    Each group shares a parent context (op/dtype/shape); its ``group_size``
    candidates vary their SCHEDULE (tile sizes, warps, stages, tl.dot, fp32
    accumulate), and the latent quality of that schedule drives the outcome. A
    fitted ranker should recover the within-group order from schedule features.
    Rows are table-like (meta + ``source`` + outcome), so they also train the
    pointwise heads."""
    rng = random.Random(seed)
    nprng = np.random.RandomState(seed)
    groups: list[list[dict]] = []
    for g in range(n_groups):
        op = rng.choice(_OPERATIONS)
        dtype = rng.choice(_DTYPES)
        bottleneck = rng.choice(_BOTTLENECKS)
        M = int(2 ** rng.randint(7, 13))
        N = int(2 ** rng.randint(7, 13))
        K = int(2 ** rng.randint(7, 13))
        rows: list[dict] = []
        for _ in range(group_size):
            bm = rng.choice([64, 128, 256, 96])   # 96 is not a 64-multiple (bad)
            bn = rng.choice([64, 128, 256, 96])
            bk = rng.choice([32, 64, 128, 48])    # 48 is not a 64-multiple (bad)
            warps = rng.choice([4, 8, 2, 16])
            stages = rng.choice([1, 2, 3])
            use_dot = rng.random() < 0.7
            fp32 = rng.random() < 0.7

            # latent schedule quality (what the ranker must recover)
            q = 0.0
            q += 0.5 if use_dot else -0.3
            q += 0.4 if fp32 else -0.5
            q += 0.4 if (bm % 64 == 0 and bn % 64 == 0 and bk % 32 == 0) else -0.7
            q += 0.2 if warps in (4, 8) else -0.3
            q += 0.1 * (stages - 1)

            p_compile = 1.0 / (1.0 + math.exp(-(2.0 + 2.0 * q)))
            compiled = nprng.rand() < p_compile
            p_snr = 1.0 / (1.0 + math.exp(-(1.0 + 2.5 * q)))
            snr_pass = bool(compiled and (nprng.rand() < p_snr))
            mem_bonus = {"memory": 0.6, "balanced": 0.3, "compute": 0.1, "unknown": 0.2}[bottleneck]
            mu_ls = 0.5 * q + mem_bonus + 0.05 * np.log10(max(M * N * K, 1) / 1e6)
            log_speedup = float(nprng.normal(mu_ls, 0.08))
            if not snr_pass:
                log_speedup = float(nprng.normal(-0.3, 0.1))
            speedup = float(math.exp(log_speedup))
            rows.append({
                "operation": op, "M": M, "N": N, "K": K, "dtype": dtype,
                "pmc_bottleneck": bottleneck,
                "source": _synth_source(bm, bn, bk, warps, stages, use_dot, fp32),
                "compiled": bool(compiled), "snr_pass": snr_pass,
                "log_speedup": log_speedup, "speedup": speedup,
            })
        groups.append(rows)
    return groups


def _flatten_groups(groups: list[list[dict]]):
    """(metas, group_ids, utils, sample_weight, outcomes) for a list of groups."""
    metas: list[dict] = []
    group_ids: list[int] = []
    utils: list[float] = []
    sw: list[float] = []
    outcomes: list[dict] = []
    for gid, rows in enumerate(groups):
        for row in rows:
            meta, outcome = _split_row(row)
            metas.append(meta)
            group_ids.append(gid)
            u = _realized_utility(outcome)
            utils.append(u)
            sw.append(max(float(outcome.get("speedup", 0.1)), 0.1))
            outcomes.append(outcome)
    return metas, np.array(group_ids), np.array(utils), np.array(sw), outcomes


def groupwise_rank_corr(score_fn, groups: list[list[dict]]) -> float:
    """Mean within-group Spearman correlation between predicted scores and the
    realized utility. ``score_fn(metas) -> scores`` (list/array)."""
    corrs: list[float] = []
    for rows in groups:
        metas = [_split_row(r)[0] for r in rows]
        util = np.array([_realized_utility(_split_row(r)[1]) for r in rows])
        scores = np.asarray(score_fn(metas), dtype=np.float64)
        if scores.shape[0] >= 2 and np.ptp(util) > 0:
            corrs.append(rank_correlation(scores, util))
    return float(np.mean(corrs)) if corrs else 0.0


def train_ranking(groups: list[list[dict]], use_sklearn: Optional[bool] = None) -> ValueModel:
    """Fit a full ValueModel (pointwise heads) PLUS the pairwise ranking head
    from within-group order. Returns the model (with ``.ranker`` attached)."""
    metas, gids, utils, sw, outcomes = _flatten_groups(groups)
    X = featurize_many(metas)
    y_compile = np.array([1 if o.get("compiled") else 0 for o in outcomes])
    y_snr = np.array([1 if o.get("snr_pass") else 0 for o in outcomes])
    y_ls = np.array([float(o.get("log_speedup", 0.0)) for o in outcomes])

    model = ValueModel(use_sklearn=use_sklearn)
    model.fit(X, y_compile, y_snr, y_ls, sample_weight=sw)
    # throughput-weighted within-group ranking supervision
    model.fit_ranker(X, gids, utils, sample_weight=sw)
    return model


# --------------------------------------------------------------------------- #
# Online refit from freshly-benched candidates (env replay cache)
# --------------------------------------------------------------------------- #
def _obs_speedup(obs) -> Optional[float]:
    """Worst-shape speedup from an Observation (mirrors reward._worst_speedup)."""
    try:
        from kore.reward.reward import _worst_speedup
        return _worst_speedup(obs)
    except Exception:
        return None


def row_from_observation(meta: dict, obs) -> dict:
    """Turn a (meta, Observation) benched candidate into a value-table row."""
    row = dict(meta or {})
    compiled = bool(getattr(obs, "compiled", False))
    correct = bool(getattr(obs, "validation_passed", False))
    su = _obs_speedup(obs)
    su = float(su) if (su and su > 0) else (0.0 if not correct else 1.0)
    row["compiled"] = compiled
    row["snr_pass"] = bool(correct)
    row["speedup"] = su
    row["log_speedup"] = math.log(su) if su > 0 else 0.0
    return row


def _coerce_row(item) -> dict:
    """Accept a table row dict, a (meta, Observation) pair, or {meta..., 'obs':Observation}."""
    if isinstance(item, tuple) and len(item) == 2:
        return row_from_observation(item[0], item[1])
    if isinstance(item, dict):
        if "obs" in item:
            meta = {k: v for k, v in item.items() if k != "obs"}
            return row_from_observation(meta, item["obs"])
        return dict(item)
    raise TypeError(f"cannot coerce {type(item)!r} into a value-table row")


def refit_online(
    new_rows,
    model: Optional[ValueModel] = None,
    history: Optional[list] = None,
    use_sklearn: Optional[bool] = None,
    fit_ranker: bool = True,
) -> tuple[ValueModel, list[dict]]:
    """Refit the value model from freshly benched candidates logged during a run.

    ``new_rows`` may be table-row dicts, ``(meta, Observation)`` pairs, or dicts
    with an embedded ``obs`` Observation (as produced by the env replay cache).
    ``history`` is the accumulated buffer of prior rows; the model is retrained on
    ``history + new_rows`` (GBT can't warm-start, so a full refit on the growing
    buffer is the robust online update - same 3-head + sklearn/numpy fallback).

    If rows carry a ``group_id`` and ``fit_ranker`` is set, the pairwise ranking
    head is refit too. Returns ``(model, buffer)`` so the caller can thread the
    buffer through the next refit."""
    buffer: list[dict] = list(history or [])
    for it in new_rows:
        buffer.append(_coerce_row(it))
    if not buffer:
        raise ValueError("refit_online: no rows to fit")

    metas, outs = zip(*[_split_row(r) for r in buffer])
    X = featurize_many(list(metas))
    y_compile = np.array([1 if o.get("compiled") else 0 for o in outs])
    y_snr = np.array([1 if o.get("snr_pass") else 0 for o in outs])
    y_ls = np.array([float(o.get("log_speedup", 0.0)) for o in outs])
    sw = np.array([max(float(o.get("speedup", 0.1)), 0.1) for o in outs])

    if model is None:
        model = ValueModel(use_sklearn=use_sklearn)
    model.fit(X, y_compile, y_snr, y_ls, sample_weight=sw)

    if fit_ranker:
        gids = [r.get("group_id") for r in buffer]
        if all(g is not None for g in gids) and len(set(gids)) >= 1:
            utils = np.array([max(float(o.get("speedup", 0.0)), 0.0)
                              if (o.get("compiled") and o.get("snr_pass")) else 0.0
                              for o in outs])
            model.fit_ranker(X, np.array(gids), utils, sample_weight=sw)
    return model, buffer


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
