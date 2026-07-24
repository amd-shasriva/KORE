"""Focused CPU regressions for the Phase-0 verifier/driver fixes."""

from __future__ import annotations

import contextlib
import dataclasses
import math
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kore.agent.tools import ToolExecutor, tool_use_reward
from kore.config import CONFIG
from kore.env.kore_env import (
    KoreEnv,
    _parse_driver_capabilities,
    _parse_timing_pairs,
    _supports_batch_bench,
)
from kore.policy import grpo
from kore.reward.reward import Observation, compute_reward
from kore.reward.stats import paired_timing_stats
from kore.tasks import _genops
from kore.tasks.base import Shape, Task


_PROMOTED_BY_COMMON = {
    "_attn_common": (
        "flash_attn_varlen_noncausal_bf16",
        "flash_attn_noncausal_prefill_bf16",
        "flash_attn_sliding_decode_bf16",
        "flash_attn_sink_prefill_bf16",
        "flash_attn_mqa_decode_bf16",
        "flash_attn_decode_fp8",
        "flash_attn_headdim_prefill_bf16",
        "flash_attn_noncausal_fp8",
        "flash_attn_mqa_prefill_bf16",
        "flash_attn_mha_prefill_bf16",
        "flash_attn_chunked_prefill_bf16",
    ),
    "_quant_common": (
        "gemm_w4a8_fp8",
        "gemm_int8_a8w8",
        "gemm_w4a16_g128",
        "gemm_fp8_a8w8_pertensor",
        "gemm_fp8_requant_epilogue",
        "gemm_fp8_a8w8_blockscale",
        "gemm_mxfp4_a4w4",
        "gemm_fp8_a8w8_pertoken",
    ),
    "_moe_common": (
        "moe_gelu_bf16",
        "moe_grouped_gemm_fp8",
        "moe_topk_softmax_norenorm_bf16",
        "moe_permute_bf16",
        "moe_biased_grouped_topk_bf16",
        "moe_grouped_gemm_bf16",
        "moe_sum_combine_bf16",
        "moe_batched_gemm_bf16",
    ),
    "_training_common": (
        "softmax_backward_bf16",
        "layernorm_backward_bf16",
        "gemm_backward_bf16",
        "flash_attn_backward_bf16",
    ),
}


def test_generated_driver_advertises_full_versioned_batch_protocol(capsys):
    ref = SimpleNamespace(mutates_input=False)
    assert _genops.driver_main(
        ref, "/unused", ["--kore-driver-capabilities"]) == 0
    caps = _parse_driver_capabilities(capsys.readouterr().out)
    assert caps["protocol"] == _genops.DRIVER_CAPABILITY_PROTOCOL == 2
    assert caps["protocol_id"] == _genops.DRIVER_PROTOCOL_ID
    assert _supports_batch_bench(caps)


def test_all_31_promoted_drivers_are_full_protocol_or_explicitly_ineligible(
        capsys, monkeypatch):
    """Every promoted thin driver makes an explicit vendor-grade admission claim."""
    tasks_root = Path(_genops.__file__).resolve().parent
    promoted = [task for tasks in _PROMOTED_BY_COMMON.values() for task in tasks]
    assert len(promoted) == 31 and len(set(promoted)) == 31
    tested = 0

    for _common, task_ids in _PROMOTED_BY_COMMON.items():
        for task_id in task_ids:
            driver = tasks_root / task_id / "driver.py"
            assert driver.is_file()
            old_path = list(sys.path)
            sys.modules.pop("reference", None)
            monkeypatch.setattr(
                sys, "argv", [str(driver), "--kore-driver-capabilities"])
            try:
                with pytest.raises(SystemExit) as exc:
                    runpy.run_path(str(driver), run_name="__main__")
                assert exc.value.code == 0
            finally:
                sys.path[:] = old_path
                sys.modules.pop("reference", None)
            caps = _parse_driver_capabilities(capsys.readouterr().out)
            full = _supports_batch_bench(caps)
            ineligible = (
                caps.get("performance_eligible") is False
                and bool(caps.get("ineligible_reason")))
            assert full or ineligible, task_id
            tested += 1
    assert tested == 31


def test_legacy_capability_is_explicit_screening_not_vendor_grade(
        tmp_path, monkeypatch):
    task_dir = tmp_path / "task"
    workdir = tmp_path / "work"
    task_dir.mkdir()
    workdir.mkdir()
    (task_dir / "driver.py").write_text("# driver_main appears here but proves nothing\n")
    shapes = [Shape("small", {"N": 8}), Shape("large", {"N": 16})]
    task = Task(
        task_id="promoted", operation="op", dtype="bf16", backend="triton",
        gpu_target="gfx950", dir=task_dir, seed_kernel_name="seed.py",
        snr_threshold=25.0, comparison_baseline="vendor", shapes=shapes,
    )
    cfg = dataclasses.replace(CONFIG, verifier_determinism_check=False)
    env = KoreEnv(task, config=cfg, use_replay=False)
    monkeypatch.setenv("KORE_NO_BENCH_BOTH", "1")
    legacy = (
        'KORE_DRIVER_CAPABILITIES: {"bench_both":false,"fresh_inputs":false,'
        '"interleaved":false,"multi_shape":false,"postcheck_all_shapes":false,'
        '"protocol":1}\n'
    )

    def fake_exec(cmd, *_args):
        if "--kore-driver-capabilities" in cmd:
            return 0, legacy, False
        return 0, "SNR: 80.0 dB\nallclose: True\n", False

    calls = []

    def fake_bench(_driver, sh, impl, _workdir, _env):
        calls.append((sh.name, impl))
        return (1.0 if impl == "candidate" else 2.0), 0.0, False

    monkeypatch.setattr(env, "_exec", fake_exec)
    monkeypatch.setattr(env, "_env", lambda: {})
    monkeypatch.setattr(env, "_bench_multi", fake_bench)
    obs = env._run(task, "kernel source", shapes, workdir, do_bench=True)

    assert not obs.infra_error
    assert set(obs.wall_by_shape) == set(obs.baseline_by_shape) == {"small", "large"}
    assert calls == [
        ("small", "candidate"), ("small", "reference"),
        ("large", "candidate"), ("large", "reference"),
    ]
    assert obs.timing_grade == "screening"
    assert obs.performance_eligible is False
    rr = compute_reward(obs, dtype="bf16")
    assert rr.tier == "correct_screening" and rr.speedup is None


def _publication_obs(candidate, baseline):
    stats = paired_timing_stats(candidate, baseline)
    cand_med = sorted(candidate)[len(candidate) // 2]
    base_med = sorted(baseline)[len(baseline) // 2]
    return Observation(
        compiled=True, validation_passed=True, dtype="bf16",
        snr_by_shape={"s": 80.0}, requested_shapes=["s"],
        timing_requested=True,
        timing_protocol=_genops.DRIVER_PROTOCOL_ID,
        timing_protocol_version=_genops.DRIVER_CAPABILITY_PROTOCOL,
        timing_guarantees=dict(_genops.PUBLICATION_GUARANTEES),
        timing_grade="publication", performance_eligible=True,
        timing_pair_count=len(candidate),
        wall_by_shape={"s": cand_med}, baseline_by_shape={"s": base_med},
        wall_ms=cand_med, baseline_ms=base_med,
        candidate_samples_by_shape={"s": list(candidate)},
        baseline_samples_by_shape={"s": list(baseline)},
        paired_ratio_samples_by_shape={"s": stats["paired_ratios"]},
        paired_log_speedup_samples_by_shape={"s": stats["paired_log_speedups"]},
        candidate_cv_by_shape={"s": stats["candidate_cv_pct"]},
        baseline_cv_by_shape={"s": stats["baseline_cv_pct"]},
        paired_ratio_cv_by_shape={"s": stats["paired_ratio_cv_pct"]},
        paired_log_ci_by_shape={
            "s": [stats["log_ci_lo"], stats["log_ci_hi"]]},
        timing_classification_by_shape={"s": stats["classification"]},
        cv_pct=stats["candidate_cv_pct"],
        baseline_cv_pct=stats["baseline_cv_pct"],
        paired_ratio_cv_pct=stats["paired_ratio_cv_pct"],
        paired_ci_half_width_pct=stats["ci_half_width_pct"],
    )


def test_paired_count_mismatch_rejects_publication_reward():
    obs = _publication_obs([1.0] * 5, [2.0] * 5)
    obs.baseline_samples_by_shape["s"] = [2.0] * 4
    rr = compute_reward(obs, dtype="bf16")
    assert rr.tier == "infra" and not rr.correct
    assert "sample count" in rr.detail


def test_baseline_high_variance_rejects_even_with_stable_candidate():
    obs = _publication_obs(
        [1.0] * 5,
        [1.5, 2.5, 1.5, 2.5, 1.5],
    )
    rr = compute_reward(obs, dtype="bf16")
    assert rr.tier == "infra" and not rr.correct
    assert "baseline CV" in rr.detail


def test_paired_ci_classifies_statistical_tie_and_clamps_micro_win():
    stats = paired_timing_stats(
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
    )
    assert stats["classification"] == "tie"
    assert stats["log_ci_lo"] <= 0.0 <= stats["log_ci_hi"]
    obs = _publication_obs([1.0] * 5, [1.0] * 5)
    rr = compute_reward(obs, dtype="bf16")
    assert rr.tier == "correct_timed" and rr.speedup == 1.0
    assert not any(f.startswith("fast_p") for f in rr.flags)


def test_parse_timing_pairs_rejects_count_and_unbalanced_order():
    one = (
        'KORE_TIMING_PAIR: {"baseline_ms":2.0,"candidate_ms":1.0,'
        '"log_speedup":0.6931471805599453,"order":"AB","pair":0,"ratio":2.0}\n'
    )
    assert _parse_timing_pairs(one, 2)[1] is not None
    repeated = one + one.replace('"pair":0', '"pair":1')
    assert "alternating" in _parse_timing_pairs(repeated, 2)[1]


def test_raw_samples_and_protocol_identity_survive_environment(tmp_path, monkeypatch):
    task_dir = tmp_path / "task_pub"
    workdir = tmp_path / "work_pub"
    task_dir.mkdir()
    workdir.mkdir()
    (task_dir / "driver.py").write_text("# paired driver\n")
    shape = Shape("primary", {"N": 8})
    task = Task(
        task_id="paired", operation="op", dtype="bf16", backend="triton",
        gpu_target="gfx950", dir=task_dir, seed_kernel_name="seed.py",
        snr_threshold=25.0, comparison_baseline="vendor", shapes=[shape],
    )
    cfg = dataclasses.replace(CONFIG, verifier_determinism_check=False)
    env = KoreEnv(task, config=cfg, use_replay=False)
    caps = _genops.publication_driver_capabilities()
    pairs = [
        {"pair": i, "order": "AB" if i % 2 == 0 else "BA",
         "candidate_ms": 1.0, "baseline_ms": 2.0,
         "ratio": 2.0, "log_speedup": math.log(2.0)}
        for i in range(cfg.max_variance_runs)
    ]
    monkeypatch.delenv("KORE_NO_BENCH_BOTH", raising=False)
    monkeypatch.setattr(env, "_env", lambda: {})
    monkeypatch.setattr(
        env, "_exec",
        lambda *_args: (0, "SNR: 80.0 dB\nallclose: True\n", False))
    monkeypatch.setattr(
        env, "_driver_capabilities", lambda *_args: caps)
    monkeypatch.setattr(
        env, "_bench_all", lambda *_args: ({"primary": pairs}, False))
    obs = env._run(task, "kernel source", [shape], workdir, do_bench=True)

    assert obs.timing_grade == "publication"
    assert obs.timing_protocol == _genops.DRIVER_PROTOCOL_ID
    assert obs.candidate_samples_by_shape["primary"] == [1.0] * 5
    assert obs.baseline_samples_by_shape["primary"] == [2.0] * 5
    assert obs.paired_ratio_samples_by_shape["primary"] == [2.0] * 5
    assert compute_reward(obs, dtype="bf16").tier == "correct_timed"


def test_partial_or_unknown_capabilities_are_performance_ineligible(
        tmp_path, monkeypatch):
    task_dir = tmp_path / "task_no_perf"
    workdir = tmp_path / "work_no_perf"
    task_dir.mkdir()
    workdir.mkdir()
    (task_dir / "driver.py").write_text("# no full protocol\n")
    shape = Shape("primary", {"N": 8})
    task = Task(
        task_id="no_perf", operation="op", dtype="bf16", backend="triton",
        gpu_target="gfx950", dir=task_dir, seed_kernel_name="seed.py",
        snr_threshold=25.0, comparison_baseline="vendor", shapes=[shape],
    )
    cfg = dataclasses.replace(CONFIG, verifier_determinism_check=False)
    env = KoreEnv(task, config=cfg, use_replay=False)
    monkeypatch.delenv("KORE_NO_BENCH_BOTH", raising=False)
    monkeypatch.setattr(env, "_env", lambda: {})
    monkeypatch.setattr(
        env, "_exec",
        lambda *_args: (0, "SNR: 80.0 dB\nallclose: True\n", False))
    monkeypatch.setattr(
        env, "_driver_capabilities",
        lambda *_args: {"protocol": 2, "protocol_id": "partial"})
    obs = env._run(task, "kernel source", [shape], workdir, do_bench=True)

    assert obs.timing_grade == "ineligible"
    assert obs.performance_eligible is False
    rr = compute_reward(obs, dtype="bf16")
    assert rr.tier == "correct_perf_ineligible" and rr.speedup is None


def test_unknown_probe_is_normalized_to_explicit_ineligibility(monkeypatch):
    env = object.__new__(KoreEnv)
    env.task = SimpleNamespace(task_id="unknown")
    env.correctness_timeout = 1
    monkeypatch.setattr(env, "_exec", lambda *_args: (2, "unknown option", False))
    caps = env._driver_capabilities(Path("/driver.py"), Path("/tmp"), {})
    assert caps["performance_eligible"] is False
    assert caps["protocol_id"] == "unknown"
    assert caps["ineligible_reason"]


def test_old_observation_schema_loads_with_timing_defaults():
    from kore.env.replay import _obs_from_dict

    obs = _obs_from_dict({
        "compiled": True,
        "validation_passed": True,
        "snr_db": 80.0,
        "wall_ms": 1.0,
        "baseline_ms": 2.0,
        "unknown_future_field": "ignored",
    })
    assert obs.timing_grade == "screening"
    assert obs.timing_protocol == "legacy-unpaired-v0"
    assert obs.candidate_samples_by_shape == {}
    assert compute_reward(obs, dtype="bf16").tier == "correct_screening"


def test_output_contract_rejects_arity_shape_dtype_and_nonfinite_adversaries():
    torch = pytest.importorskip("torch")
    ref = torch.tensor([float("nan"), float("inf"), float("-inf"), 1.0])
    assert _genops._compare_outputs(ref.clone(), ref)[2]

    adversaries = [
        (ref.clone(), ref.reshape(2, 2)),                    # shape
        (ref.double(), ref),                                # dtype
        ((ref.clone(), ref.clone()), (ref,)),               # tuple arity
        ((ref.clone(),), ref),                              # tuple vs scalar ABI
        (torch.tensor([float("inf"), float("inf"),
                       float("-inf"), 1.0]), ref),           # NaN -> +Inf
        (torch.tensor([float("nan"), float("-inf"),
                       float("-inf"), 1.0]), ref),           # +Inf sign
    ]
    for out, expected in adversaries:
        assert not _genops._compare_outputs(out, expected)[2]


@pytest.mark.parametrize(
    "walls,bases",
    [
        ({"a": 1.0}, {"a": 2.0, "b": 2.0}),
        ({"a": 1.0, "b": 1.0}, {"a": 2.0}),
        ({"a": 1.0}, {"a": 2.0}),
        ({"a": 1.0, "b": 1.0, "extra": 1.0},
         {"a": 2.0, "b": 2.0, "extra": 2.0}),
    ],
)
def test_missing_or_extra_timing_keys_are_retryable_infra(walls, bases):
    obs = Observation(
        compiled=True, validation_passed=True,
        snr_by_shape={"a": 80.0, "b": 80.0},
        requested_shapes=["a", "b"], timing_requested=True,
        wall_by_shape=walls, baseline_by_shape=bases,
    )
    rr = compute_reward(obs, dtype="bf16")
    assert rr.tier == "infra" and not rr.correct and rr.speedup is None
    assert "incomplete_timing" in rr.flags


def test_correctness_keys_must_cover_every_requested_shape():
    obs = Observation(
        compiled=True, validation_passed=True,
        snr_by_shape={"a": 80.0},
        requested_shapes=["a", "b"],
    )
    assert compute_reward(obs, dtype="bf16").tier == "incorrect"


def test_batched_inputs_are_storage_isolated(monkeypatch):
    torch = pytest.importorskip("torch")
    seen = []
    candidate_ptrs = []

    class Ref:
        entry_name = "op"
        mutates_input = False

        @staticmethod
        def get_inputs(_shape, device="cuda", seed=0):
            return (torch.tensor([1.0]),)

        @staticmethod
        def baseline_fn(x):
            seen.append(float(x.item()))
            return x

    def candidate(x):
        candidate_ptrs.append(x.data_ptr())
        x.add_(10.0)
        return x

    monkeypatch.setattr(_genops, "_load_candidate",
                        lambda _task_dir, _entry: candidate)
    for pair_index in (0, 1):
        cand, refr = _genops._build_bench_pair(
            Ref, "/unused", {}, pair_index)
        cand()
        refr()
    assert seen == [1.0, 1.0]
    assert candidate_ptrs[0] != candidate_ptrs[1]  # fresh storage each pair


def test_batched_pair_order_alternates_from_randomized_side(monkeypatch):
    import random

    events = []
    monkeypatch.setattr(
        _genops, "_build_bench_pair",
        lambda _ref, _task_dir, _shape, _pair:
            (lambda: events.append("C"), lambda: events.append("R")),
    )

    def fake_time(fn, _warmup, _iters):
        fn()
        return 1.0

    monkeypatch.setattr(_genops, "_time_median", fake_time)
    monkeypatch.setattr(random, "getrandbits", lambda _n: 1)
    monkeypatch.setattr(random, "randint", lambda lo, _hi: lo)
    _genops._run_bench_both(object(), "/unused", {}, 4, 8, repeat=4)
    assert events == ["C", "R", "R", "C", "C", "R", "R", "C"]


def test_all_shape_batch_postchecks_every_shape(monkeypatch):
    checked = []

    class Ref:
        @staticmethod
        def parse_shape(spec):
            return {"spec": spec}

    monkeypatch.setattr(
        _genops, "_build_bench_pair",
        lambda *_args: ((lambda: None), (lambda: None)),
    )
    monkeypatch.setattr(_genops, "_time_median",
                        lambda _fn, _warmup, _iters: 1.0)
    monkeypatch.setattr(
        _genops, "_run_correctness",
        lambda _ref, _task_dir, shape: checked.append(shape) or 0,
    )
    _genops._run_paired_bench_all_shapes(
        Ref, "/unused", ["N=8", "N=16", "N=31"], 4, 8, repeat=1,
        build_pair=_genops._build_bench_pair,
        postcheck=_genops._run_correctness)
    assert checked == [
        {"spec": "N=8"}, {"spec": "N=16"}, {"spec": "N=31"},
    ]


def test_env_rejects_one_failed_postcheck_hidden_before_later_pass(monkeypatch):
    env = object.__new__(KoreEnv)
    env.task = SimpleNamespace(
        task_id="t", dtype="bf16", snr_threshold=25.0)
    env.cfg = SimpleNamespace(
        max_variance_runs=1, warmup_iters=4, bench_iters=8)
    env.bench_timeout = 10
    env._gpu = None
    monkeypatch.setattr(env, "_timing_lock", lambda: contextlib.nullcontext())
    out = """\
SHAPE_BEGIN N=8
CAND_median_ms: 1.0
REF_median_ms: 2.0
SNR: 0.0 dB
allclose: False
SHAPE_BEGIN N=16
CAND_median_ms: 1.0
REF_median_ms: 2.0
SNR: 80.0 dB
allclose: True
"""
    monkeypatch.setattr(env, "_exec", lambda *_args: (0, out, False))
    result, poisoned = env._bench_all(
        Path("/driver.py"),
        [Shape("a", {"N": 8}), Shape("b", {"N": 16})],
        Path("/tmp"), {},)
    assert result == {} and poisoned


class _InfraTask:
    task_id = "infra_task"
    operation = "op"
    dtype = "bf16"
    gpu_target = "gfx950"


class _InfraEnv:
    def step(self, *_args, **_kwargs):
        return Observation(
            compiled=True, validation_passed=False, infra_error=True,
            error_text="infra: timed out", dtype="bf16")


def test_agent_tools_surface_infra_as_failed_retryable_calls(monkeypatch):
    monkeypatch.delenv("KORE_REWARD_MODE", raising=False)
    ex = ToolExecutor(_InfraEnv(), _InfraTask())
    trace = []
    for name in ("build", "test", "bench", "pmc"):
        result = ex.dispatch(
            {"name": name, "arguments": {"kernel_src": "src"}})
        assert result["ok"] is False
        assert result["infra_error"] is True
        trace.append({"name": name, "result": result})
    # Infra is not counted as a failed kernel/tool decision penalty.
    comp = tool_use_reward({"tool_trace": trace})
    assert comp["n_failed"] == 0


def test_agentic_infra_trace_is_dropped_before_advantages():
    episode = SimpleNamespace(tool_trace=[
        {"turn": 0, "result": {"ok": True, "infra_error": False}},
        {"turn": 1, "result": {"ok": False, "infra_error": True}},
        {"turn": 2, "result": {"ok": True, "tier": "correct_timed"}},
    ])
    infra = grpo._agentic_turn_infra(episode, 3)
    assert infra == [False, True, False]
    returns, index = grpo.build_kevin_samples(
        [[1.0, 2.0, 3.0]], [[True, True, True]],
        gamma=0.4, traj_infra=[infra])
    assert index == [(0, 0), (0, 2)]
    assert len(grpo.group_advantages(returns)) == 2
