"""Focused CPU regressions for the Phase-0 verifier/driver fixes."""

from __future__ import annotations

import contextlib
import dataclasses
import importlib.util
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
    _supports_batch_bench,
)
from kore.policy import grpo
from kore.reward.reward import Observation, compute_reward
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
    assert _genops.driver_main(
        object(), "/unused", ["--kore-driver-capabilities"]) == 0
    caps = _parse_driver_capabilities(capsys.readouterr().out)
    assert caps["protocol"] == _genops.DRIVER_CAPABILITY_PROTOCOL == 1
    assert _supports_batch_bench(caps)


def test_all_31_promoted_drivers_handshake_to_supported_legacy_path(
        capsys, monkeypatch):
    """Every promoted thin driver explicitly declines unsupported batch flags."""
    tasks_root = Path(_genops.__file__).resolve().parent
    promoted = [task for tasks in _PROMOTED_BY_COMMON.values() for task in tasks]
    assert len(promoted) == 31 and len(set(promoted)) == 31
    have_torch = importlib.util.find_spec("torch") is not None
    tested = 0

    for common, task_ids in _PROMOTED_BY_COMMON.items():
        for task_id in task_ids:
            driver = tasks_root / task_id / "driver.py"
            assert driver.is_file()
            # The quant common has an intentional top-level torch dependency;
            # exercise those eight too in the normal GPU/test environment.
            if common == "_quant_common" and not have_torch:
                continue
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
            assert caps["protocol"] == 1, task_id
            assert not _supports_batch_bench(caps), task_id
            tested += 1
    assert tested == (31 if have_torch else 23)


def test_legacy_capability_uses_per_impl_bench_even_if_source_mentions_driver_main(
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
    assert compute_reward(obs, dtype="bf16").tier == "correct_timed"


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
        x.add_(10.0)
        return x

    monkeypatch.setattr(_genops, "_load_candidate",
                        lambda _task_dir, _entry: candidate)
    cand = _genops._build_bench_fn(Ref, "/unused", {}, "candidate")
    refr = _genops._build_bench_fn(Ref, "/unused", {}, "reference")
    cand()
    refr()
    assert seen == [1.0]


def test_batched_pair_order_alternates_from_randomized_side(monkeypatch):
    import random

    events = []
    monkeypatch.setattr(
        _genops, "_build_bench_fn",
        lambda _ref, _task_dir, _shape, impl:
            (lambda: events.append("C" if impl == "candidate" else "R")),
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
        _genops, "_build_bench_fn",
        lambda *_args: (lambda: None),
    )
    monkeypatch.setattr(_genops, "_time_median",
                        lambda _fn, _warmup, _iters: 1.0)
    monkeypatch.setattr(
        _genops, "_run_correctness",
        lambda _ref, _task_dir, shape: checked.append(shape) or 0,
    )
    _genops._run_bench_all_shapes(
        Ref, "/unused", ["N=8", "N=16", "N=31"], 4, 8, repeat=1)
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
