"""Re-verify / re-baseline of existing kernels (reuse, no teacher)."""

from __future__ import annotations

import json

from kore.config import CONFIG
from kore.data import reverify as RV
from kore.reward.reward import Observation


class _Task:
    task_id = "gen_relu_fp16"
    dtype = "fp16"
    operation = "relu"
    gpu_target = "gfx942"
    backend = "triton"
    comparison_baseline = "torch_relu"
    seed_source = "def s(): pass"


class _StubEnv:
    """Correctness + speed keyed off markers in the source (deterministic)."""

    def step(self, source, full_validation=True, multi_shape=True):
        if "__WRONG__" in source:  # fails adversarial correctness now
            return Observation(compiled=True, validation_passed=False, snr_db=5.0,
                               snr_by_shape={"primary": 5.0}, dtype="fp16")
        if "__SLOW__" in source:  # correct but not faster than the strong baseline
            return Observation(compiled=True, validation_passed=True, snr_db=60.0,
                               snr_by_shape={"primary": 60.0}, wall_ms=2.0,
                               baseline_ms=2.0, dtype="fp16")
        return Observation(compiled=True, validation_passed=True, snr_db=60.0,
                           snr_by_shape={"primary": 60.0}, wall_ms=1.0,
                           baseline_ms=2.0, dtype="fp16")


def test_reverify_group_reranks_and_prefs_honest():
    env, task = _StubEnv(), _Task()
    g = {"task_id": "t", "parent_id": "p", "candidates": [
        {"source": "cand FAST", "wall_us": 1.0, "snr_db": 99, "rank": 1},
        {"source": "cand __WRONG__", "wall_us": 0.5, "snr_db": 99, "rank": 0}],  # v1 "best"
        "preferences": [[1, 0]], "type": "ranked_group", "gpu": "gfx942",
        "operation": "relu", "arch": "gfx942"}
    ng = RV.reverify_group(g, task, env, CONFIG)
    ranks = {c["source"]: c["rank"] for c in ng["candidates"]}
    assert ranks["cand FAST"] == 0 and ranks["cand __WRONG__"] == 1  # WRONG sinks
    assert [1, 0] not in ng["preferences"]  # old dishonest pref gone
    assert [0, 1] in ng["preferences"]      # FAST > WRONG


def test_reverify_win_rebaselines_and_culls():
    env, task = _StubEnv(), _Task()
    fast = RV.reverify_win({"final_source": "cand FAST", "speedup": 9.9}, task, env, CONFIG)
    slow = RV.reverify_win({"final_source": "cand __SLOW__", "speedup": 9.9}, task, env, CONFIG)
    wrong = RV.reverify_win({"final_source": "cand __WRONG__", "speedup": 9.9}, task, env, CONFIG)
    assert fast is not None and fast["speedup"] == 2.0  # re-baselined vs strong baseline
    assert slow is None    # no longer beats the strong baseline -> culled
    assert wrong is None   # fails adversarial correctness -> culled


def test_reverify_repair_drops_lucky_pass():
    env, task = _StubEnv(), _Task()
    ok = RV.reverify_repair(
        {"messages": [{"role": "assistant", "content": "FULL_KERNEL:\n```python\ncand FAST\n```"}]},
        task, env, CONFIG)
    bad = RV.reverify_repair(
        {"messages": [{"role": "assistant", "content": "FULL_KERNEL:\n```python\ncand __WRONG__\n```"}]},
        task, env, CONFIG)
    assert ok is not None and bad is None


def test_reverify_shard_drops_and_backs_up(tmp_path):
    env, task = _StubEnv(), _Task()
    wp = tmp_path / "wins" / "t.jsonl"
    wp.parent.mkdir(parents=True)
    wp.write_text("\n".join(json.dumps(x) for x in [
        {"final_source": "cand FAST", "speedup": 9.9},
        {"final_source": "cand __SLOW__", "speedup": 9.9}]) + "\n")
    n_in, n_keep = RV._reverify_shard(
        wp, lambda r: RV.reverify_win(r, task, env, CONFIG), drop_none=True, backup=True)
    assert n_in == 2 and n_keep == 1  # slow win culled
    assert (tmp_path / "wins" / "t.jsonl.pre_reverify.bak").exists()
    out = [json.loads(x) for x in wp.read_text().splitlines() if x.strip()]
    assert len(out) == 1 and out[0]["final_source"] == "cand FAST"


def test_shard_tasks_pins_round_robin():
    from kore.data.parallel_datagen import shard_tasks
    shards = shard_tasks(["a", "b", "c", "d", "e"], 3)
    assert [len(s) for s in shards] == [2, 2, 1]
    assert set(sum(shards, [])) == {"a", "b", "c", "d", "e"}
