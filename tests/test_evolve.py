"""CPU-only tests for the evolutionary datagen module (kore.data.evolve).

Covers the three review pieces:
  - D-MAB bandit (UCB1 + Page-Hinkley) selecting better operators over time and
    re-adapting after a reward-regime change;
  - MAP-Elites / island archive insert / elite / migrate semantics;
  - the operator registry exposed by mutate for the bandit to select from;
  - evolve_task end-to-end with a StubTeacher + a deterministic fake env,
    producing verified WinRecords and RankedGroupRecords.

No GPU, no teacher model: the environment is a pure-python fake and the generator
is a StubTeacher.
"""

from __future__ import annotations

import hashlib
import random
from types import SimpleNamespace

from kore.data import mutate
from kore.data.evolve import (
    DMABBandit,
    EliteRecord,
    EvolveConfig,
    MapElitesArchive,
    behavior_descriptor,
    evolve_task,
    migrate,
)
from kore.data.schemas import RankedGroupRecord, WinRecord
from kore.data.teacher import StubTeacher
from kore.reward.reward import Observation


# --------------------------------------------------------------------------- #
# operator registry (mutate) — the bandit's action set
# --------------------------------------------------------------------------- #
_KERNEL = """
import triton
import triton.language as tl

@triton.jit
def _mm(a_ptr, b_ptr, c_ptr, M, N, K,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    offs = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in range(0, K, BLOCK_K):
        x = tl.load(a_ptr + offs, mask=offs < K, other=0.0)
        acc += tl.dot(x, x)
    tl.store(c_ptr + offs, acc, mask=offs < N)

def entry(a, b):
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 128, 128, 64, 8
    return _mm[(1,)](a, b, num_warps=4, num_stages=2)
"""


def test_operator_registry_exposes_optimize_and_break():
    opt = set(mutate.list_operators("optimize"))
    brk = set(mutate.list_operators("break"))
    allops = set(mutate.list_operators("all"))
    assert {"tile_change", "vectorize", "pipeline_depth", "num_warps_sweep"} <= opt
    assert "break_block_size" in brk
    assert allops == opt | brk
    assert all(name in mutate.OPERATOR_REGISTRY for name in allops)


def test_apply_operator_changes_source():
    rng = random.Random(0)
    for name in mutate.list_operators("optimize"):
        new_src, hint = mutate.apply_operator(name, _KERNEL, rng)
        assert new_src != _KERNEL
        assert hint == "optimize"


def test_apply_operator_unknown_raises():
    import pytest

    with pytest.raises(KeyError):
        mutate.apply_operator("does_not_exist", _KERNEL)


# --------------------------------------------------------------------------- #
# D-MAB bandit
# --------------------------------------------------------------------------- #
def test_ucb1_selects_better_operator_over_time():
    rng = random.Random(0)
    bandit = DMABBandit(["a", "b", "c"], seed=0)
    true_p = {"a": 0.2, "b": 0.8, "c": 0.3}
    for _ in range(600):
        op = bandit.select()
        reward = 1.0 if rng.random() < true_p[op] else 0.0
        bandit.update(op, reward)
    # the genuinely-best operator must dominate cumulative pulls
    assert bandit.best_operator() == "b"
    assert bandit.pulls["b"] == max(bandit.pulls.values())
    assert bandit.pulls["b"] > 0.5 * sum(bandit.pulls.values())


def test_page_hinkley_detects_reward_regime_change():
    rng = random.Random(1)
    bandit = DMABBandit(["x", "y"], seed=1)
    # phase 1: x is best; phase 2: y is best
    post_change_x = 0
    post_change_y = 0
    for i in range(400):
        p = {"x": 0.8, "y": 0.2} if i < 200 else {"x": 0.2, "y": 0.8}
        op = bandit.select()
        reward = 1.0 if rng.random() < p[op] else 0.0
        bandit.update(op, reward)
        if i >= 260:  # give the detector time to react after the switch
            post_change_x += 1 if op == "x" else 0
            post_change_y += 1 if op == "y" else 0
    # the Page-Hinkley detector must have fired at least once (a restart)
    assert bandit.n_resets >= 1
    # after the change the bandit re-explores and favors the new best (y)
    assert post_change_y > post_change_x


def test_bandit_plays_every_arm_before_exploiting():
    bandit = DMABBandit(["a", "b", "c"], seed=3)
    seen = set()
    for _ in range(3):
        op = bandit.select()
        bandit.update(op, 0.5)
        seen.add(op)
    assert seen == {"a", "b", "c"}


# --------------------------------------------------------------------------- #
# MAP-Elites / island archive
# --------------------------------------------------------------------------- #
def test_archive_insert_and_elite_keeps_best_per_cell():
    a = MapElitesArchive(speedup_bins=(1.0, 1.2, 1.5, 2.0, 3.0), seed=0)
    # first insert into a cell
    assert a.insert("s1", True, 1.6, 40.0, "gemm") is True
    # same cell (speedup-bin 2), worse -> rejected
    assert a.insert("s2", True, 1.55, 39.0, "gemm") is False
    # same cell, better -> replaces
    assert a.insert("s3", True, 1.9, 41.0, "gemm") is True
    # a different speedup bin -> a new cell
    assert a.insert("s4", True, 2.5, 42.0, "gemm") is True
    # an incorrect candidate lands in its own (correct=False) cell
    assert a.insert("s5", False, None, 5.0, "gemm") is True

    assert len(a) == 3  # {bin2, bin3, incorrect}
    best = a.best(1)[0]
    assert best.source == "s4"  # highest speedup among the correct elites
    # the bin-2 cell holds the better of s1/s3
    srcs = {e.source for e in a.elites()}
    assert "s3" in srcs and "s2" not in srcs and "s1" not in srcs


def test_behavior_descriptor_op_family_speedup_correctness():
    bins = (1.0, 1.2, 1.5, 2.0, 3.0)
    d_fast = behavior_descriptor("gemm", 2.5, True, bins)
    d_slow = behavior_descriptor("gemm", 1.3, True, bins)
    d_bad = behavior_descriptor("gemm", None, False, bins)
    assert d_fast != d_slow  # different speedup bins -> different cells
    assert d_bad[2] is False and d_bad[1] == -1
    assert d_fast[0] == "gemm"


def test_migrate_moves_top_elites_between_islands():
    src = MapElitesArchive(seed=0)
    dst = MapElitesArchive(seed=1)
    src.insert("fast", True, 2.5, 40.0, "gemm")
    src.insert("mid", True, 1.3, 38.0, "gemm")
    moved = migrate(src, dst, n=2)
    assert moved == 2
    assert len(dst) == 2
    # migrating again is idempotent (dst already has equal-or-better elites)
    assert migrate(src, dst, n=2) == 0


# --------------------------------------------------------------------------- #
# evolve_task end-to-end (StubTeacher + fake env)
# --------------------------------------------------------------------------- #
_SEED = """import triton
import triton.language as tl

@triton.jit
def _mm(a_ptr, b_ptr, c_ptr, M, N, K,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    offs = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in range(0, K, BLOCK_K):
        x = tl.load(a_ptr + offs, mask=offs < K, other=0.0)
        acc += tl.dot(x, x)
    tl.store(c_ptr + offs, acc, mask=offs < N)

def entry(a, b):
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 128, 128, 64, 8
    return _mm[(1,)](a, b, num_warps=4, num_stages=2)
"""


def _fake_task():
    return SimpleNamespace(
        task_id="gemm_bf16",
        dtype="bf16",
        operation="gemm",
        gpu_target="gfx942",
        seed_source=_SEED,
        shapes=[SimpleNamespace(name="s", dims={"M": 1024, "N": 1024, "K": 1024})],
    )


class _FakeEnv:
    """Deterministic verifier: everything compiles + is correct; the wall time is
    a stable function of the source so different candidates get different speeds
    (and 'TUNE' directives / bigger tiles run faster), producing diversity."""

    def __init__(self, task):
        self.task = task

    def step(self, source, full_validation=True, multi_shape=True):
        h = int(hashlib.sha256(source.encode()).hexdigest(), 16)
        wall = 2.0 - 0.05 * source.count("TUNE") - 0.0015 * (len(source) % 90) - (h % 100) / 400.0
        wall = max(0.15, wall)
        return Observation(
            compiled=True, dtype=self.task.dtype, validation_passed=True,
            snr_db=40.0, snr_by_shape={"s": 40.0},
            wall_ms=wall, baseline_ms=3.0,
            wall_by_shape={"s": wall}, baseline_by_shape={"s": 3.0},
        )


def _teacher():
    def fn(messages):
        # emit a valid, improving kernel (adds a TUNE directive so it benches fast)
        return (
            "ANALYSIS: memory bound.\nCHANGE: tune\n"
            "FULL_KERNEL:\n```python\n" + _SEED + "\n# TUNE: num_stages += 1\n```"
        )

    return StubTeacher(fn=fn)


def test_evolve_task_produces_verified_records():
    task = _fake_task()
    res = evolve_task(task, _teacher(), _FakeEnv(task), generations=6,
                      cfg=EvolveConfig(seed=0, islands=2))
    # verified WinRecord(s) and RankedGroupRecord(s) were produced
    assert len(res.wins) >= 1
    assert all(isinstance(w, WinRecord) for w in res.wins)
    w = res.wins[0]
    assert w.task_id == "gemm_bf16"
    assert w.speedup is not None and w.speedup > 1.0
    assert w.final_source and w.operation == "gemm"

    assert len(res.groups) >= 1
    assert all(isinstance(g, RankedGroupRecord) for g in res.groups)
    g = res.groups[0]
    assert g.candidates and g.preferences

    # the bandit was actually exercised and the archive populated
    assert res.stats["n_benched"] > 0
    assert res.stats["n_correct"] > 0
    assert sum(res.bandit.pulls.values()) > 0
    assert len(res.archive) >= 1


def test_evolve_task_prefilter_limits_benches():
    task = _fake_task()
    cfg = EvolveConfig(seed=1, islands=1, candidates_per_gen=5, prefilter_k=2)
    res = evolve_task(task, _teacher(), _FakeEnv(task), generations=4, cfg=cfg)
    # with prefilter_k=2 we bench at most 2 candidates per generation
    assert res.stats["n_benched"] <= 2 * 4


def test_evolve_task_deterministic_with_seed():
    task = _fake_task()
    r1 = evolve_task(task, _teacher(), _FakeEnv(task), generations=4,
                     cfg=EvolveConfig(seed=7, islands=2))
    r2 = evolve_task(task, _teacher(), _FakeEnv(task), generations=4,
                     cfg=EvolveConfig(seed=7, islands=2))
    assert r1.stats["n_benched"] == r2.stats["n_benched"]
    assert r1.stats["operator_pulls"] == r2.stats["operator_pulls"]
    assert (r1.stats["best_speedup"] or 0) == (r2.stats["best_speedup"] or 0)
