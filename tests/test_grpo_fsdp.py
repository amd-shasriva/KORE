"""Tests for KORE FULL-PARAMETER sharded (distributed) GRPO.

Coverage (all CPU / no-GPU unless noted):
  * backend selection + config->plugin wiring (FSDP FULL_SHARD default, DeepSpeed
    ZeRO-3 opt-in) - the sharded full-param path is gated to distributed full-FT;
  * ``grpo_config_from_dict`` JSON round-trip (nested ``lora``, ``tasks`` pop) and
    the ``python -m kore.policy.grpo <config.json>`` entrypoint;
  * the CROSS-RANK group-relative advantage gather math (simulate N ranks' rewards
    -> a single global GRPO normalization split back per rank);
  * a real 2-process (gloo/CPU) smoke: all-gather rollout rewards across ranks,
    compute the global advantages, and take one training step - validating the
    multiprocess wiring + gather primitive without needing GPUs or a real 14B.

The non-distributed / LoRA / CPU GRPO path is unchanged and covered by
``tests/test_rl_core.py``; a couple of guards here re-assert it stays untouched.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path

import pytest

from kore.policy import grpo
from kore.policy.configs import (
    GRPOConfig,
    build_deepspeed_config,
    grpo_distributed_enabled,
    grpo_sharding_backend,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _restore_dist_env():
    """Building an accelerate FSDP/DeepSpeed plugin exports FSDP_*/ACCELERATE_*
    env vars; snapshot + restore them around every test so this module can never
    leak distributed config into the rest of the suite (e.g. test_distributed.py)."""
    prefixes = ("FSDP_", "ACCELERATE_", "DEEPSPEED_")
    keys = [k for k in os.environ if k.startswith(prefixes)]
    snapshot = {k: os.environ[k] for k in keys}
    try:
        yield
    finally:
        for k in list(os.environ):
            if k.startswith(prefixes):
                os.environ.pop(k, None)
        os.environ.update(snapshot)


# --------------------------------------------------------------------------- #
# gating: sharded full-param path only for distributed full-FT
# --------------------------------------------------------------------------- #
def test_grpo_distributed_enabled_only_for_distributed_full_ft():
    assert grpo_distributed_enabled(GRPOConfig(use_lora=False, distributed=True)) is True
    assert grpo_distributed_enabled(GRPOConfig(use_lora=True, distributed=True)) is False
    assert grpo_distributed_enabled(GRPOConfig(use_lora=False, distributed=False)) is False
    # default config (single-process LoRA) never takes the sharded path.
    assert grpo_distributed_enabled(GRPOConfig()) is False


def test_grpo_sharding_backend_selection():
    base = dict(use_lora=False, distributed=True)
    # "auto" -> FSDP (default) for the O(1-sample) micro-batched RL loop.
    assert grpo_sharding_backend(GRPOConfig(**base)) == "fsdp"
    assert grpo_sharding_backend(GRPOConfig(sharding_backend="auto", **base)) == "fsdp"
    assert grpo_sharding_backend(GRPOConfig(sharding_backend="fsdp", **base)) == "fsdp"
    assert grpo_sharding_backend(GRPOConfig(sharding_backend="fsdp2", **base)) == "fsdp"
    assert grpo_sharding_backend(GRPOConfig(sharding_backend="deepspeed", **base)) == "deepspeed"
    # not a distributed full-FT run -> "none" (keep the legacy in-process path).
    assert grpo_sharding_backend(GRPOConfig(use_lora=True, distributed=True)) == "none"
    assert grpo_sharding_backend(GRPOConfig()) == "none"


# --------------------------------------------------------------------------- #
# DeepSpeed ZeRO-3 config dict builder (pure, no deepspeed import)
# --------------------------------------------------------------------------- #
def test_build_deepspeed_config_zero3_defaults():
    cfg = build_deepspeed_config(GRPOConfig(use_lora=False, distributed=True, max_grad_norm=0.5))
    assert cfg["zero_optimization"]["stage"] == 3
    assert cfg["bf16"]["enabled"] is True
    assert cfg["fp16"]["enabled"] is False
    assert cfg["gradient_clipping"] == 0.5
    assert cfg["train_micro_batch_size_per_gpu"] == 1
    # ZeRO-3 gathers full weights for a plain checkpoint at save time.
    assert cfg["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] is True
    # no offload by default (14B fits on 8xMI300 without it).
    assert "offload_param" not in cfg["zero_optimization"]


def test_build_deepspeed_config_cpu_offload():
    cfg = build_deepspeed_config(GRPOConfig(use_lora=False, distributed=True, cpu_offload=True))
    z = cfg["zero_optimization"]
    assert z["offload_param"] == {"device": "cpu", "pin_memory": True}
    assert z["offload_optimizer"] == {"device": "cpu", "pin_memory": True}


def test_build_deepspeed_config_explicit_ds_config_path(tmp_path):
    p = tmp_path / "ds.json"
    p.write_text(json.dumps({"zero_optimization": {"stage": 2}, "custom": 1}))
    cfg = build_deepspeed_config(GRPOConfig(use_lora=False, distributed=True, ds_config=str(p)))
    assert cfg == {"zero_optimization": {"stage": 2}, "custom": 1}  # verbatim override


def test_build_deepspeed_config_zero2_omits_stage3_keys():
    cfg = build_deepspeed_config(GRPOConfig(use_lora=False, distributed=True, zero_stage=2))
    assert cfg["zero_optimization"]["stage"] == 2
    assert "stage3_gather_16bit_weights_on_model_save" not in cfg["zero_optimization"]


# --------------------------------------------------------------------------- #
# accelerate plugin wiring (needs accelerate; CPU only, no GPU / no launch)
# --------------------------------------------------------------------------- #
def test_build_fsdp_plugin_full_shard():
    pytest.importorskip("accelerate")
    plug = grpo.build_fsdp_plugin(GRPOConfig(model_id="Qwen/Qwen3-14B", use_lora=False,
                                             distributed=True))
    assert plug.transformer_cls_names_to_wrap == ["Qwen3DecoderLayer"]


def test_build_fsdp_plugin_autodetects_llama_layer():
    pytest.importorskip("accelerate")
    plug = grpo.build_fsdp_plugin(GRPOConfig(
        model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-70B", use_lora=False,
        distributed=True, cpu_offload=True))
    assert plug.transformer_cls_names_to_wrap == ["LlamaDecoderLayer"]


def test_build_deepspeed_plugin_zero3():
    pytest.importorskip("accelerate")
    pytest.importorskip("deepspeed")
    plug = grpo.build_deepspeed_plugin(GRPOConfig(
        model_id="Qwen/Qwen3-14B", use_lora=False, distributed=True,
        sharding_backend="deepspeed"))
    assert int(plug.zero_stage) == 3


def test_build_grpo_accelerator_routes_to_selected_backend(monkeypatch):
    # Route selection without constructing a real (distributed) Accelerator.
    import accelerate

    seen = {}

    class FakeAcc:
        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setattr(accelerate, "Accelerator", FakeAcc)
    monkeypatch.setattr(grpo, "build_fsdp_plugin", lambda c: "FSDP_PLUGIN")
    monkeypatch.setattr(grpo, "build_deepspeed_plugin", lambda c: "DS_PLUGIN")

    grpo.build_grpo_accelerator(GRPOConfig(use_lora=False, distributed=True))
    assert seen.get("fsdp_plugin") == "FSDP_PLUGIN" and seen.get("mixed_precision") == "bf16"

    seen.clear()
    grpo.build_grpo_accelerator(GRPOConfig(use_lora=False, distributed=True,
                                           sharding_backend="deepspeed", bf16=False))
    assert seen.get("deepspeed_plugin") == "DS_PLUGIN" and seen.get("mixed_precision") == "no"


# --------------------------------------------------------------------------- #
# grpo_config_from_dict JSON round-trip + `-m` entry
# --------------------------------------------------------------------------- #
def test_grpo_config_from_dict_ignores_stale_lora_and_tasks():
    cfg = grpo.grpo_config_from_dict({
        "model_id": "Qwen/Qwen3-14B",
        "use_lora": False,
        "distributed": True,
        "sharding_backend": "deepspeed",
        "zero_stage": 3,
        "cpu_offload": True,
        "ref_anchor_coef": 1e-3,
        "tasks": ["rmsnorm_aiter", "gemm_bf16"],   # threaded by the campaign (NOT a field)
        "lora": {"r": 8, "lora_alpha": 16},        # GRPO is full-FT only: ignored
    })
    assert cfg.model_id == "Qwen/Qwen3-14B"
    assert cfg.use_lora is False and cfg.distributed is True
    assert cfg.sharding_backend == "deepspeed" and cfg.zero_stage == 3 and cfg.cpu_offload is True
    # GRPO has no `lora` field (full-FT only) and `tasks` is consumed by _main.
    assert not hasattr(cfg, "lora")
    assert not hasattr(cfg, "tasks")


def test_grpo_config_from_dict_defaults_are_sharded_ready():
    cfg = grpo.grpo_config_from_dict({"model_id": "m", "use_lora": False, "distributed": True})
    assert cfg.sharding_backend == "auto" and cfg.zero_stage == 3
    assert cfg.synced_gpus is True and cfg.cpu_offload is False


def test_grpo_main_no_args_returns_usage():
    assert grpo._main([]) == 2


def test_grpo_main_reads_json_and_runs(monkeypatch, tmp_path):
    # _main reads the JSON, defaults distributed=True, threads `tasks` into
    # train_grpo, and returns 0 - WITHOUT importing torch (train_grpo stubbed).
    seen = {}

    def fake_train(cfg, tasks=None):
        seen["cfg"] = cfg
        seen["tasks"] = tasks
        return "runs/grpo_out"

    monkeypatch.setattr(grpo, "train_grpo", fake_train)
    p = tmp_path / "grpo.json"
    p.write_text(json.dumps({"model_id": "Qwen/Qwen3-14B", "use_lora": False,
                             "tasks": ["rmsnorm_aiter"]}))
    assert grpo._main([str(p)]) == 0
    assert seen["cfg"].distributed is True           # defaulted by the entry
    assert seen["cfg"].model_id == "Qwen/Qwen3-14B"
    assert seen["tasks"] == ["rmsnorm_aiter"]         # threaded through


def test_grpo_entry_module_imports_without_torch():
    code = (
        "import sys; import kore.policy.grpo, kore.policy.configs; "
        "assert 'torch' not in sys.modules; "
        "assert 'accelerate' not in sys.modules; "
        "assert 'deepspeed' not in sys.modules; print('ok')"
    )
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


# --------------------------------------------------------------------------- #
# cross-rank group-relative advantage gather (the RL-math correctness bit)
# --------------------------------------------------------------------------- #
def test_distributed_group_advantages_matches_centralized():
    # Two ranks each rolled part of ONE group. The GRPO baseline MUST be over the
    # FULL group (all trajectories), so gathering then normalizing == doing it
    # centrally, and each rank keeps its own slice in order.
    per_rank = [[1.0, 2.0], [3.0, 4.0]]           # rank0 rewards, rank1 rewards
    split = grpo.distributed_group_advantages(per_rank)
    central = grpo.group_advantages([1.0, 2.0, 3.0, 4.0])
    assert grpo.merge_across_ranks(split) == central
    assert split[0] == central[:2] and split[1] == central[2:]
    # normalized: zero mean, unit-ish std over the FULL group.
    assert abs(sum(central)) < 1e-9


def test_distributed_group_advantages_uneven_and_three_ranks():
    per_rank = [[5.0], [1.0, 1.0], [3.0]]         # ragged per-rank counts
    split = grpo.distributed_group_advantages(per_rank)
    central = grpo.group_advantages([5.0, 1.0, 1.0, 3.0])
    assert [len(s) for s in split] == [1, 2, 1]
    assert grpo.merge_across_ranks(split) == central


def test_distributed_group_advantages_uses_avspo_variance_floor():
    # A near-degenerate global group: with the AVSPO floor (tau>0) the advantages
    # do NOT explode / collapse - they route through anticollapse.avspo_advantages.
    per_rank = [[1.0, 1.0], [1.0, 1.0]]
    floored = grpo.distributed_group_advantages(per_rank, variance_floor=0.5, avspo_virtual_k=2)
    from kore.policy import anticollapse as ac
    central = ac.avspo_advantages([1.0, 1.0, 1.0, 1.0], 0.5, 2, grpo._EPS)
    assert grpo.merge_across_ranks(floored) == central


def test_rank_slice_strided_partition():
    # G=8 across world=4 -> each rank owns 2 trajectories, disjoint + covering.
    slices = [grpo._rank_slice(8, r, 4) for r in range(4)]
    assert slices == [[0, 4], [1, 5], [2, 6], [3, 7]]
    assert sorted(i for s in slices for i in s) == list(range(8))


def test_all_gather_object_single_process_fallback():
    # No live process group -> identity list (correct on 1 rank / CPU / tests).
    assert grpo._all_gather_object([1.0, 2.0]) == [[1.0, 2.0]]
    assert grpo._all_gather_object(7) == [7]


# --------------------------------------------------------------------------- #
# RAGGED AGENTIC rollouts on the sharded path (the agentic-at-scale enablement)
# --------------------------------------------------------------------------- #
def test_build_kevin_samples_handles_ragged_agentic_trajectories():
    """Agentic rollouts have VARIABLE turn counts. Kevin credit must flatten ragged
    trajectories into per-turn samples so the agentic path can share the serial
    per-turn pipeline on the distributed sharded loop (the enablement's core)."""
    traj_rewards = [[0.0, 1.0], [0.0, 0.0, 0.0, 2.0, 0.0]]        # 2-turn + 5-turn
    traj_correct = [[False, True], [False, False, False, True, False]]
    returns, index = grpo.build_kevin_samples(traj_rewards, traj_correct, gamma=0.5)
    # one per-turn sample per turn across BOTH ragged trajectories (2 + 5)
    assert len(returns) == len(index) == 7
    # index maps back to the correct (trajectory, turn) for the ragged shapes
    assert index == [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]
    # correctness-gated gamma returns: the pre-correct turns still carry look-ahead
    # credit (>0) toward the correct turn, and every trajectory keeps its own length.
    assert returns[0] > 0.0 and returns[1] > 0.0


def test_accumulator_lockstep_pads_ragged_ranks():
    """THE invariant that makes ragged agentic rollouts shard-safe: every rank runs
    EXACTLY ``max_micro_steps`` backward calls regardless of its local sample count
    (surplus = zeroed dummy forwards), so the FSDP/ZeRO collectives stay symmetric
    across ranks even when agentic trajectories yield different per-rank counts."""
    torch = pytest.importorskip("torch")

    class _FakeAcc:
        def __init__(self):
            self.n_backward = 0

        def backward(self, loss):
            self.n_backward += 1

    class _FakeTok:
        def __call__(self, s, return_tensors=None):
            class _R:
                input_ids = torch.tensor([[1]])
            return _R()

    def _logp_fn(gen_inputs):  # ignores input; fixed scalar logp
        return torch.tensor(0.5, requires_grad=True)

    def _run(n_local, max_micro):
        acc = _FakeAcc()
        # sample = [ret, gen_inputs, ref_logp, old_logp, n_tokens, sc_weight]
        terms = [(0.3, [0.3, [("p", "g")], None, None, 4, None]) for _ in range(n_local)]
        grpo._accumulate_grpo_grads_distributed(
            terms, _logp_fn, accelerator=acc, global_total_tokens=100, grad_scale=2.0,
            max_micro_steps=max_micro, ref_anchor_coef=0.0, clip_ratio_low=0.2,
            clip_ratio_high=0.28, tok=_FakeTok(), device="cpu")
        return acc.n_backward

    # A "slow" rank (2 samples), a "full" rank (5), and a fully-idle rank (0) all run
    # the SAME max_micro=5 collective forwards -> no rank desyncs under ragged agentic.
    assert _run(2, 5) == 5
    assert _run(5, 5) == 5
    assert _run(0, 5) == 5


def test_agentic_per_turn_signal_recovers_aligned_speedup_and_code():
    """The agentic rollout now surfaces per-turn MEASURED speedup + kernel source
    (previously dropped as None/""), so co-evolution distillation gets a real
    best_kernel_src and the controller sees achieved speedups. Speedups are
    correctness-gated exactly like the serial _rollout."""
    from kore.agent.harness import AgentEpisode

    ep = AgentEpisode(
        task_id="t",
        turn_rewards=[0.0, 0.5, 1.0],
        turn_correct=[False, True, True],
        turn_speedups=[None, 1.5, 2.0],
        turn_codes=["srcA", "srcB", "srcC"],
    )
    codes, sus, phis = grpo._agentic_per_turn_signal(ep, ep.turn_correct, 3)
    assert codes == ["srcA", "srcB", "srcC"]
    assert sus == [None, 1.5, 2.0]


def test_agentic_per_turn_signal_gates_speedup_on_correctness():
    # A measured speedup on a NON-correct turn is dropped (a kernel that is not
    # correct can never contribute a rewarded/distilled speedup).
    from kore.agent.harness import AgentEpisode

    ep = AgentEpisode(
        task_id="t",
        turn_rewards=[1.0, 0.0],
        turn_correct=[True, False],
        turn_speedups=[2.0, 3.0],     # 3.0 sits on an incorrect turn -> must drop
        turn_codes=["good", "bad"],
    )
    codes, sus, phis = grpo._agentic_per_turn_signal(ep, ep.turn_correct, 2)
    assert codes == ["good", "bad"]
    assert sus == [2.0, None]


def test_agentic_per_turn_signal_degrades_when_trace_misaligned():
    # Terminal-only fallback (turn_rewards length 1) vs a longer harness trace ->
    # NOT index-aligned, so we must degrade to ("", None) rather than misattribute
    # a speedup/source to the wrong turn.
    from kore.agent.harness import AgentEpisode

    ep = AgentEpisode(
        task_id="t",
        turn_rewards=[1.0],                 # collapsed (best_reward, success)
        turn_correct=[True],
        turn_speedups=[None, 2.0, None],    # full harness trace (len 3) -> misaligned
        turn_codes=["a", "b", "c"],
    )
    codes, sus, phis = grpo._agentic_per_turn_signal(ep, [True], 1)
    assert codes == [""] and sus == [None]


def test_agentic_per_turn_signal_handles_missing_fields():
    # An episode object without the per-turn fields (legacy) degrades cleanly.
    class _LegacyEp:
        pass

    codes, sus, phis = grpo._agentic_per_turn_signal(_LegacyEp(), [True, True], 2)
    assert codes == ["", ""] and sus == [None, None] and phis == [None, None]
    assert grpo._agentic_per_turn_signal(_LegacyEp(), [], 0) == ([], [], [])


def test_agentic_per_turn_signal_matches_live_harness_trace():
    """End-to-end contract: a real harness episode's recorded trace maps through
    _agentic_per_turn_signal to the per-turn (code, speedup) the GRPO agentic
    rollout emits - the fast bench turn carries its 2x speedup + source. Uses
    local fakes (no cross-test import) so it is robust to the test layout."""
    import json as _json

    from kore.agent.harness import AgentHarness
    from kore.data.teacher import StubTeacher
    from kore.reward.reward import Observation

    class _Task:
        task_id, operation, dtype, gpu_target = "fake_gemm_bf16", "gemm", "bf16", "gfx942"

    class _Env:
        def step(self, source, full_validation=True, multi_shape=True):
            if "__WRONG__" in source:
                return Observation(compiled=True, dtype="bf16", validation_passed=False,
                                   snr_by_shape={"primary": 5.0}, snr_db=5.0,
                                   error_text="worst SNR 5.0 < 25.0")
            if "__FAST__" in source:
                return Observation(compiled=True, dtype="bf16", validation_passed=True,
                                   snr_by_shape={"primary": 41.0}, snr_db=41.0,
                                   wall_by_shape={"primary": 1.0},
                                   baseline_by_shape={"primary": 2.0},
                                   wall_ms=1.0, baseline_ms=2.0)
            return Observation(compiled=True, dtype="bf16", validation_passed=True,
                               snr_by_shape={"primary": 40.0}, snr_db=40.0,
                               wall_by_shape={"primary": 2.0},
                               baseline_by_shape={"primary": 2.0},
                               wall_ms=2.0, baseline_ms=2.0)

    def _call(name, args):
        return f'<tool_call>\n{_json.dumps({"name": name, "arguments": args})}\n</tool_call>'

    script = [
        _call("build", {"kernel_src": "cand __WRONG__"}),
        _call("test", {"kernel_src": "cand __WRONG__"}),
        _call("test", {"kernel_src": "cand fixed"}),
        _call("bench", {"kernel_src": "cand __FAST__"}) + "\n" + _call("keep", {}),
        "All done - no further changes.",
    ]
    state = {"i": 0}

    def _fn(_messages):
        i = state["i"]; state["i"] = i + 1
        return script[i] if i < len(script) else "done."

    ep = AgentHarness(_Task(), StubTeacher(fn=_fn), _Env(),
                      max_turns=8, use_kb=False).run()
    n = ep.turns_used
    codes, sus, phis = grpo._agentic_per_turn_signal(ep, ep.turn_correct, n)
    assert len(codes) == len(sus) == n
    assert sus[3] == 2.0 and codes[3] == "cand __FAST__"
    assert sus[0] is None and sus[1] is None and sus[2] is None


def test_shipped_14b_config_enables_agentic_rollouts():
    # The 14B GRPO template requests agentic tool-use RL; the distributed path now
    # HONORS it (it previously silently fell back to serial refinement).
    raw = json.loads((REPO_ROOT / "configs" / "grpo_14b_full.json").read_text())
    cfg = grpo.grpo_config_from_dict(raw)
    assert cfg.agentic is True
    assert cfg.max_tool_turns >= 1


# --------------------------------------------------------------------------- #
# shipped 14B full-FT sharded template
# --------------------------------------------------------------------------- #
def test_shipped_grpo_14b_full_config_is_sharded_full_ft():
    raw = json.loads((REPO_ROOT / "configs" / "grpo_14b_full.json").read_text())
    cfg = grpo.grpo_config_from_dict(raw)
    assert cfg.use_lora is False and cfg.distributed is True
    assert grpo_distributed_enabled(cfg) is True
    assert grpo_sharding_backend(cfg) in ("fsdp", "deepspeed")
    assert cfg.zero_stage == 3 and cfg.synced_gpus is True
    # Kevin rollout defaults for the 8-GPU 14B run.
    assert cfg.num_trajectories == 16 and cfg.num_turns == 4


# --------------------------------------------------------------------------- #
# 2-process (gloo/CPU) smoke - real multiprocess gather + one training step
# --------------------------------------------------------------------------- #
def _smoke_worker(rank: int, world: int, port: int, q):
    """One rank of the 2-proc smoke (module-level so it is fork-safe).

    Inits a gloo process group, all-gathers this rank's rollout rewards across
    ranks, computes the GLOBAL group-relative advantages, and takes ONE real
    gradient step on a tiny local model using its own advantage slice. Puts
    ``(rank, "ok", ...)`` on the queue, or ``(rank, "ERR:...")`` on failure.

    NB: this validates the DISTRIBUTED WIRING (process group + cross-rank reward
    gather + a training step under 2 ranks) on CPU/gloo. Real FSDP/ZeRO-3 param
    sharding needs GPUs and is exercised by the 8xMI300 run, not here.
    """
    try:
        import torch
        import torch.distributed as dist

        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group("gloo", rank=rank, world_size=world)

        from kore.policy import grpo as g

        # each rank "rolled out" 2 trajectories of the SAME group.
        local_rewards = [[1.0, 2.0], [3.0, 4.0]][rank]
        gathered = g._all_gather_object(local_rewards)
        # the gathered full group must be identical + rank-ordered on every rank.
        assert gathered == [[1.0, 2.0], [3.0, 4.0]], gathered

        split = g.distributed_group_advantages(gathered)
        central = g.group_advantages([1.0, 2.0, 3.0, 4.0])
        assert g.merge_across_ranks(split) == central
        my_adv = split[rank]

        # one real training step driven by THIS rank's global-advantage slice.
        torch.manual_seed(rank)
        w = torch.nn.Parameter(torch.zeros(2))
        opt = torch.optim.SGD([w], lr=0.1)
        opt.zero_grad()
        # surrogate ~ -A * logp with a trivial logp = sum(w) per sample.
        loss = -sum(a * w.sum() for a in my_adv)
        loss.backward()
        opt.step()
        changed = bool((w.detach() != 0).any().item())

        dist.destroy_process_group()
        q.put((rank, "ok", [round(x, 6) for x in my_adv], changed))
    except Exception as e:  # noqa: BLE001 - surface to the parent assertion
        q.put((rank, "ERR:" + repr(e)))


def test_two_process_gloo_smoke_gather_and_step():
    pytest.importorskip("torch")
    if sys.platform == "win32":
        pytest.skip("multiprocess gloo smoke is POSIX-only")
    # Prefer 'spawn': fork is fundamentally incompatible with a process that has
    # already touched autograd (PyTorch raises "Unable to handle autograd's
    # threading in combination with fork-based multiprocessing"), which happens
    # whenever an earlier test in the session ran a backward pass. The real GRPO
    # launcher uses spawn/torchrun too, so spawn is the faithful path; fall back
    # to fork only if spawn is unavailable.
    ctx = None
    for method in ("spawn", "fork"):
        try:
            ctx = mp.get_context(method)
            break
        except ValueError:
            continue
    if ctx is None:
        pytest.skip("no multiprocess start method available")

    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]          # a free ephemeral port (avoid TIME_WAIT reuse)
    world = 2
    q = ctx.Queue()
    procs = [ctx.Process(target=_smoke_worker, args=(r, world, port, q)) for r in range(world)]
    for p in procs:
        p.start()
    try:
        results = [q.get(timeout=60) for _ in range(world)]
    except Exception:  # noqa: BLE001 - a hung child must not hang the suite
        for p in procs:
            if p.is_alive():
                p.terminate()
        raise
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()

    results.sort()
    for r in results:
        assert r[1] == "ok", f"rank {r[0]} failed: {r[1]}"
    # each rank got its correct global-advantage slice and took a real step.
    central = grpo.group_advantages([1.0, 2.0, 3.0, 4.0])
    assert results[0][2] == [round(x, 6) for x in central[:2]]
    assert results[1][2] == [round(x, 6) for x in central[2:]]
    assert results[0][3] is True and results[1][3] is True


# --------------------------------------------------------------------------- #
# non-distributed path is untouched
# --------------------------------------------------------------------------- #
def test_inprocess_alias_preserved():
    # The distributed dispatch lives INSIDE _train_grpo_fallback; the alias that
    # the existing suite asserts on must still hold.
    assert grpo._train_grpo_inprocess is grpo._train_grpo_fallback


def test_default_config_takes_single_process_path():
    # Default (LoRA / non-distributed) config never resolves to a sharded backend.
    assert grpo_sharding_backend(GRPOConfig()) == "none"
    assert grpo_distributed_enabled(GRPOConfig()) is False
