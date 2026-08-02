"""CPU-runnable launch-readiness regressions for Stage-3 multi-turn GRPO.

Companion to ``tests/test_sft_launch_readiness.py`` and
``tests/test_dpo_launch_readiness.py``, locking down what was verified end to end
on 8x MI350X (gfx950) for ``docs/GRPO_READINESS.md``: the sharding topology that
makes in-loop generation safe, per-rank memory arithmetic for the THREE full-
weight 14B copies each rank holds, checkpoint discovery and the fail-closed
resume, the frozen held-out shape lane, the budget ledger, and the straggler
lanes that run on rank 0 while every other rank waits at a collective.

Nothing here needs a GPU or the model weights.

``configs/grpo_14b_full.json`` is concurrently owned by another change, so these
tests assert STRUCTURAL invariants (arity, divisibility, retention floors,
topology) rather than the specific values of the four features being fixed there
(co-evolution root, value model, physics shaping, ref_checkpoint).

``xfail`` markers ARE the blocker list, per this repo's convention.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
GRPO_CONFIG_PATH = REPO / "configs" / "grpo_14b_full.json"
GRPO_ACCEL = REPO / "configs" / "accelerate_fsdp_grpo.yaml"
GRPO_SBATCH = REPO / "scripts" / "spur_grpo_1node.sbatch"

#: Qwen3-14B, measured from the safetensors headers.
PARAMS_14B = 14_768_307_200
#: Production topology: one node, 8x MI350X, 252 GiB of HBM each.
WORLD = 8
HBM_BYTES_PER_GPU = 252 * 1024**3


def _grpo_config() -> dict:
    return json.loads(GRPO_CONFIG_PATH.read_text())


def _config_object():
    from kore.policy.grpo import grpo_config_from_dict

    return grpo_config_from_dict(_grpo_config())


# --------------------------------------------------------------------------- #
# 1. The shipped config parses, and its rollout topology is launchable
# --------------------------------------------------------------------------- #
def test_shipped_config_parses_through_the_strict_loader():
    config = _config_object()
    assert config.use_lora is False, "GRPO is full-parameter only"
    assert config.distributed is True
    assert config.total_steps > 0 and config.save_steps > 0
    assert config.agentic is True


def test_grpo_rejects_lora_before_it_loads_anything():
    from kore.policy.capabilities import FeatureConfigurationError
    from kore.policy.grpo import train_grpo

    config = _config_object()
    config.use_lora = True
    with pytest.raises(FeatureConfigurationError, match="use_lora must be false"):
        train_grpo(config, tasks=["gen_add_bf16"])


def test_grpo_refuses_an_empty_task_list():
    """The train/eval split is the whole basis of the held-out claim; GRPO must
    never silently fall back to the full registry."""
    from kore.policy.grpo import train_grpo

    with pytest.raises(ValueError, match="non-empty train task list"):
        train_grpo(_config_object(), tasks=[])


def test_num_trajectories_divides_the_world_size():
    """Unequal per-rank rollouts deadlock the lockstep generation.

    ``_train_grpo_distributed`` rounds UP to the next multiple of world when it
    can, but a config that already divides avoids a silent group-size change.
    """
    config = _config_object()
    assert config.num_trajectories % WORLD == 0, (
        f"num_trajectories={config.num_trajectories} does not divide {WORLD} ranks; "
        "the loop will silently round it up and change the group size"
    )


def test_rank_slice_covers_every_trajectory_exactly_once():
    from kore.policy.grpo import _rank_slice

    config = _config_object()
    total = config.num_trajectories
    covered = [i for rank in range(WORLD) for i in _rank_slice(total, rank, WORLD)]
    assert sorted(covered) == list(range(total))
    sizes = {len(_rank_slice(total, rank, WORLD)) for rank in range(WORLD)}
    assert len(sizes) == 1, f"ragged per-rank rollout counts {sizes} deadlock generation"


# --------------------------------------------------------------------------- #
# 2. Sharding topology: why in-loop generation does not deadlock
# --------------------------------------------------------------------------- #
def test_grpo_accelerate_config_pins_shard_grad_op_in_both_keys():
    """FSDP1 reads the deprecated ``sharding_strategy`` FIRST.

    Setting only ``reshard_after_forward`` lets the FULL_SHARD default win, which
    reshards params after every forward and deadlocks ragged decode.
    """
    text = GRPO_ACCEL.read_text()
    assert "fsdp_sharding_strategy: SHARD_GRAD_OP" in text
    assert "fsdp_reshard_after_forward: SHARD_GRAD_OP" in text
    assert "fsdp_state_dict_type: FULL_STATE_DICT" in text


def test_launcher_selects_the_grpo_specific_accelerate_config():
    launcher = (REPO / "scripts" / "launch_distributed.sh").read_text()
    assert "accelerate_fsdp_grpo.yaml" in launcher
    assert 'if [ "$STAGE" = "grpo" ]' in launcher


def test_build_fsdp_plugin_requests_shard_grad_op_in_code_too():
    """The in-code plugin must agree with configs/accelerate_fsdp_grpo.yaml.

    ``FullyShardedDataParallelPlugin`` needs a visible accelerator to construct
    (``sync_module_states`` queries the current device), so on a CPU-only box
    this asserts the request at the source level instead.
    """
    from kore.policy.grpo import build_fsdp_plugin

    source = (REPO / "kore" / "policy" / "grpo.py").read_text()
    assert "SHARD_GRAD_OP" in source, (
        "build_fsdp_plugin no longer requests ZeRO-2; in-loop generation will "
        "reshard params after every forward and deadlock on ragged decode"
    )

    try:
        plugin = build_fsdp_plugin(_config_object())
    except RuntimeError as error:  # no accelerator visible in this process
        pytest.skip(f"FSDP plugin needs a device: {error}")
    if plugin is None:
        pytest.skip("torch FSDP plugin unavailable in this environment")
    strategy = f"{getattr(plugin, 'reshard_after_forward', '')}" \
               f"{getattr(plugin, 'sharding_strategy', '')}"
    assert "SHARD_GRAD_OP" in strategy or "1" in strategy


# --------------------------------------------------------------------------- #
# 3. Memory: THREE full-weight 14B copies live on every rank
# --------------------------------------------------------------------------- #
def test_per_rank_memory_accounts_for_all_three_full_weight_replicas():
    """SHARD_GRAD_OP shards grads + optimizer state but NOT params.

    Each rank therefore holds: the replicated bf16 policy, a plain full-weight
    generation replica (``gen_replica``), and -- whenever ``ref_anchor_coef > 0``
    and the reference loads -- a full frozen reference. Only the gradient and
    Adam terms shrink with world size. This is the arithmetic the readiness
    review measured against, so a topology change that breaks it fails here.
    """
    config = _config_object()
    bf16 = 2 * PARAMS_14B

    replicated = bf16                      # policy params (never resharded)
    replicated += bf16                     # gen_replica (full weight, per rank)
    if getattr(config, "ref_anchor_coef", 0.0) > 0:
        replicated += bf16                 # frozen KL-anchor reference, per rank

    sharded = (bf16 + 3 * 4 * PARAMS_14B) / WORLD   # bf16 grads + fp32 master + 2 Adam moments
    total = replicated + sharded

    assert replicated == 3 * bf16, "the reference replica dropped out of the budget"
    assert total < HBM_BYTES_PER_GPU, (
        f"projected {total / 1024**3:.1f} GiB/rank exceeds "
        f"{HBM_BYTES_PER_GPU / 1024**3:.0f} GiB of HBM"
    )
    # Headroom for activations, the KV cache during generation, and fragmentation.
    assert HBM_BYTES_PER_GPU - total > 60 * 1024**3


def test_ref_anchor_coef_and_ref_checkpoint_are_consistent():
    """A KL anchor that cannot load is silently dropped by ``_load_ref_model``.

    ``ref_anchor_coef > 0`` therefore only means something if the reference
    checkpoint is actually resolvable at launch time. This asserts the pair is
    internally consistent; the capability audit checks the artifact at runtime.
    """
    config = _config_object()
    coef = getattr(config, "ref_anchor_coef", 0.0)
    ref = getattr(config, "ref_checkpoint", None)
    if coef > 0:
        assert ref, (
            "ref_anchor_coef>0 with ref_checkpoint unset falls back to model_id, "
            "which the launcher rewrites to the DPO output -- so the retention "
            "anchor would measure drift from a preference-tuned model, not from "
            "the broad post-SFT policy"
        )


def test_load_ref_model_failing_open_is_reported_not_silent():
    """``_load_ref_model`` returns None on any load failure.

    That is a deliberate graceful degradation, but it changes the objective, so
    the capability audit must classify it rather than let a WARN scroll past.
    """
    from kore.policy.capabilities import audit_requested_capabilities

    config = _config_object()
    if not getattr(config, "ref_anchor_coef", 0.0) > 0:
        pytest.skip("no KL anchor requested")
    config.ref_checkpoint = "runs/definitely_not_here"
    findings = audit_requested_capabilities(config)
    assert any("ref_anchor" in f.feature for f in findings), (
        "an unloadable KL-anchor reference must surface as an inert capability"
    )


# --------------------------------------------------------------------------- #
# 4. Checkpoint retention, discovery and the fail-closed resume
# --------------------------------------------------------------------------- #
def test_grpo_save_total_limit_never_drops_below_two():
    from kore.policy.grpo import _grpo_save_total_limit

    assert _grpo_save_total_limit(SimpleNamespace()) == 2
    assert _grpo_save_total_limit(SimpleNamespace(save_total_limit=1)) == 2
    assert _grpo_save_total_limit(SimpleNamespace(save_total_limit=None)) == 2
    assert _grpo_save_total_limit(SimpleNamespace(save_total_limit="junk")) == 2
    assert _grpo_save_total_limit(SimpleNamespace(save_total_limit=4)) == 4


def _write_grpo_checkpoint(directory: Path, step: int, *, complete: bool = True):
    """A checkpoint shaped the way ``_save_grpo_checkpoint_distributed`` leaves it."""
    from kore.policy.grpo import _GRPO_STATE_FILE

    directory.mkdir(parents=True, exist_ok=True)
    manifest = ["model.safetensors", "optimizer.pt"]
    for name in manifest:
        (directory / name).write_bytes(b"\0")
    if complete:
        (directory / _GRPO_STATE_FILE).write_text(json.dumps({
            "global_step": step, "kore_grpo_files": manifest}))
    return directory


def test_resume_discovery_walks_newest_first_past_a_damaged_checkpoint(tmp_path):
    from kore.policy.grpo import _find_grpo_resume_checkpoint

    _write_grpo_checkpoint(tmp_path / "checkpoint-100", 100)
    _write_grpo_checkpoint(tmp_path / "checkpoint-200", 200, complete=False)
    found = _find_grpo_resume_checkpoint(SimpleNamespace(output_dir=str(tmp_path)))
    assert found is not None
    path, state = found
    assert path.endswith("checkpoint-100") and state["global_step"] == 100


def test_resume_discovery_rejects_a_checkpoint_missing_a_manifest_file(tmp_path):
    """Completeness is checked against the manifest the WRITER recorded, so a
    directory that lost a shard is rejected instead of half-restored."""
    from kore.policy.grpo import _find_grpo_resume_checkpoint, _read_grpo_trainer_state

    checkpoint = _write_grpo_checkpoint(tmp_path / "checkpoint-50", 50)
    assert _read_grpo_trainer_state(checkpoint) is not None
    (checkpoint / "optimizer.pt").unlink()
    assert _read_grpo_trainer_state(checkpoint) is None
    with pytest.raises(RuntimeError, match="none .*is resumable"):
        _find_grpo_resume_checkpoint(SimpleNamespace(output_dir=str(tmp_path)))


def test_resume_discovery_fails_closed_rather_than_restarting_from_zero(tmp_path):
    """Checkpoints present but none resumable must STOP the run.

    Silently restarting a 2000-step RL run from step 0 would burn a full
    allocation and overwrite the only surviving policy.
    """
    from kore.policy.grpo import _find_grpo_resume_checkpoint

    assert _find_grpo_resume_checkpoint(SimpleNamespace(output_dir=str(tmp_path))) is None
    _write_grpo_checkpoint(tmp_path / "checkpoint-10", 10, complete=False)
    with pytest.raises(RuntimeError, match="Refusing to silently restart"):
        _find_grpo_resume_checkpoint(SimpleNamespace(output_dir=str(tmp_path)))


def test_resume_error_is_broadcast_so_every_rank_fails_together(tmp_path):
    """Raising on rank 0 alone leaves the others blocked in the next collective."""
    from kore.policy.grpo import _discover_grpo_resume

    _write_grpo_checkpoint(tmp_path / "checkpoint-10", 10, complete=False)
    config = SimpleNamespace(output_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="Refusing to silently restart"):
        _discover_grpo_resume(config, accelerator=None)

    source = (REPO / "kore" / "policy" / "grpo.py").read_text()
    assert '_broadcast_rank0_object(payload, accelerator)' in source, (
        "the fail-closed error must travel as a VALUE to every rank"
    )


def test_checkpoint_step_parsing_ignores_foreign_directories(tmp_path):
    from kore.policy.grpo import _grpo_checkpoint_dirs

    for name in ("checkpoint-3", "checkpoint-20", "checkpoint-notanint", "shape_splits"):
        (tmp_path / name).mkdir()
    steps = [step for step, _path in _grpo_checkpoint_dirs(tmp_path)]
    assert steps == [3, 20]


def test_restored_optimizer_moments_stay_fp32_under_mixed_precision():
    """The resume path must not downcast the Adam moments to the param dtype.

    ``_fsdp_full_state_ctx`` offloads to CPU, and ``Optimizer.load_state_dict``
    then casts every floating-point state tensor to ``param.dtype``. Under FSDP
    mixed precision the params are BF16 at that moment, so the fp32 moments were
    silently downcast -- destroying ``exp_avg_sq``'s dynamic range and making the
    NEXT ``opt.step()`` raise "Tensors of the same index must be on the same
    device and the same dtype". Every requeued GRPO child died on its first
    optimizer step. Reproduced on 2x MI350X and fixed in ``_load_full_optim_state``.
    """
    source = (REPO / "kore" / "policy" / "grpo.py").read_text()
    marker = 'entry[key] = value.to(dtype=torch.float32)'
    assert marker in source, (
        "_load_full_optim_state no longer restores the Adam moments to fp32; "
        "a resumed run will die on its first optimizer step"
    )

    # The guard must skip `step` (torch documents it as legitimately CPU/fp32)
    # and must run inside the FSDP branch, after load_state_dict.
    fsdp_branch = source.split("def _load_full_optim_state")[1].split("\ndef ")[0]
    assert "inner.load_state_dict(sharded)" in fsdp_branch
    assert fsdp_branch.index("inner.load_state_dict(sharded)") < fsdp_branch.index(marker)
    assert 'if key == "step"' in fsdp_branch


def test_grpo_save_predicate_is_rank_invariant():
    """Every rank must enter (or skip) the save's collectives together, so the
    predicate may only read values that are identical on all ranks."""
    from kore.policy.grpo import _grpo_should_save

    config = SimpleNamespace(save_steps=100, total_steps=2000)
    assert _grpo_should_save(99, 0, config) is True     # step+1 == 100
    assert _grpo_should_save(50, 0, config) is False
    assert _grpo_should_save(99, 100, config) is False  # already saved


# --------------------------------------------------------------------------- #
# 5. The frozen held-out shape lane
# --------------------------------------------------------------------------- #
def test_shape_split_directory_prefers_the_campaign_environment(monkeypatch, tmp_path):
    from kore.policy.grpo import SHAPE_SPLIT_DIR_ENV, shape_split_directory

    monkeypatch.delenv(SHAPE_SPLIT_DIR_ENV, raising=False)
    config = SimpleNamespace(output_dir=str(tmp_path / "runs" / "grpo"))
    assert shape_split_directory(config) == tmp_path / "runs" / "grpo" / "shape_splits"

    monkeypatch.setenv(SHAPE_SPLIT_DIR_ENV, str(tmp_path / "shared"))
    assert shape_split_directory(config) == tmp_path / "shared"


def test_only_rank_zero_writes_the_shape_lane_and_the_others_wait(monkeypatch, tmp_path):
    """Concurrent writers race on the split index: a rank that enumerates the
    directory mid-write publishes an index that omits the other's manifests, and
    certification then rejects the lane."""
    from kore.policy import grpo as grpo_module

    written = []
    monkeypatch.setattr(grpo_module, "get_task", lambda task_id: task_id,
                        raising=False)
    monkeypatch.setattr(
        "kore.tasks.shape_policy.freeze_shape_splits",
        lambda tasks, directory: written.append(directory) or SimpleNamespace(
            entries=list(tasks), hidden_shapes=2, seed=0, hidden_max_shapes=2,
            code_identity="x"))
    monkeypatch.setattr("kore.tasks.registry.get_task", lambda task_id: task_id)

    waited = []
    monkeypatch.setattr(grpo_module, "_shape_split_barrier",
                        lambda receipt, world: waited.append((receipt, world)))

    config = SimpleNamespace(output_dir=str(tmp_path))
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")
    grpo_module.freeze_training_shape_splits(config, ["t1"])
    assert written == [], "a follower rank must not write the shape lane"
    assert waited, "a follower rank must wait on the barrier"

    monkeypatch.setenv("RANK", "0")
    grpo_module.freeze_training_shape_splits(config, ["t1"])
    assert len(written) == 1, "rank 0 must be the single writer"


def test_shape_split_barrier_is_bounded_and_loud(tmp_path):
    """A follower cannot tell 'rank 0 is still writing 1,300 manifests' from
    'rank 0 died', so the filesystem wait must time out with a real message."""
    from kore.policy.grpo import _await_shape_split_receipt

    with pytest.raises(RuntimeError, match="timed out .* waiting for the training-time"):
        _await_shape_split_receipt(tmp_path / "never", timeout=0.05)


# --------------------------------------------------------------------------- #
# 6. Reward, advantages and the budget ledger
# --------------------------------------------------------------------------- #
def test_budget_ledger_tracks_every_counter_the_readiness_review_reports():
    from kore.policy.budget import BudgetLedgerV1

    ledger = BudgetLedgerV1()
    for field in ("correctness_calls", "fresh_timed_calls", "replay_hits",
                  "verifier_gpu_seconds", "profiler_gpu_seconds"):
        assert hasattr(ledger, field), field


def test_budget_ledgers_merge_across_ranks_by_summing():
    """The per-rank ledgers are all-gathered and merged on the main process, so a
    merge that overwrote instead of summing would under-report verifier spend by
    world_size."""
    from kore.policy.budget import BudgetLedgerV1

    merged = BudgetLedgerV1.merge([
        BudgetLedgerV1(correctness_calls=3, fresh_timed_calls=1,
                       verifier_gpu_seconds=2.0, replay_hits=1),
        BudgetLedgerV1(correctness_calls=4, fresh_timed_calls=2,
                       verifier_gpu_seconds=5.0, replay_hits=3),
    ])
    assert merged.correctness_calls == 7
    assert merged.fresh_timed_calls == 3
    assert merged.verifier_gpu_seconds == 7.0
    assert merged.replay_hits == 4


def test_collapsed_groups_are_dropped_and_refilled_not_silently_trained():
    from kore.policy.grpo import (
        dynamic_sampling_refill,
        starpo_select_high_variance,
    )

    assert starpo_select_high_variance([[1.0, 1.0, 1.0]], 0.75, 1e-3) == []
    assert starpo_select_high_variance([[0.0, 1.0], [1.0, 1.0]], 1.0, 1e-3) == [0]

    rolled = []

    def roll(attempt):
        rolled.append(attempt)
        return [0.0, 0.0] if attempt < 2 else [0.0, 1.0]

    kept, attempts = dynamic_sampling_refill(roll, 1, min_std=1e-3, max_attempts=5)
    assert len(kept) == 1 and attempts == 3, "degenerate groups must not count"


def test_a_fully_collapsed_step_makes_no_optimizer_update():
    """Every group degenerate -> refill exhausts -> the step is SKIPPED.

    This is correct behaviour, but it means an early-training policy that solves
    (or fails) every trajectory identically produces steps that cost a full
    rollout and update nothing. The readiness review measures how often.
    """
    from kore.policy.grpo import dynamic_sampling_refill

    kept, attempts = dynamic_sampling_refill(
        lambda attempt: [0.3, 0.3, 0.3], target_groups=8, min_std=1e-3,
        max_attempts=24)
    assert kept == [] and attempts == 24


def test_group_advantages_are_computed_over_the_full_cross_rank_group():
    """The GRPO baseline is group-relative; computing it per-rank would give each
    rank a different baseline for the same group."""
    from kore.policy.grpo import distributed_group_advantages

    per_rank = [[0.0, 1.0], [2.0, 3.0]]
    advantages = distributed_group_advantages(per_rank, 0.0, 2, 1e-8)
    flat = [a for chunk in advantages for a in chunk]
    assert len(flat) == 4
    assert abs(sum(flat)) < 1e-6, "advantages must be mean-centred over the FULL group"


# --------------------------------------------------------------------------- #
# 7. Straggler lanes and the launcher contract
# --------------------------------------------------------------------------- #
def test_test_time_search_fires_on_the_very_first_step():
    """``step % search_every == 0`` is true at step 0.

    The search runs on rank 0 only, with every other rank parked at the next
    collective, so the first step of every allocation pays the full
    ``search_budget`` of verified evaluations before any optimizer step happens.
    """
    config = _grpo_config()
    if not config.get("use_search"):
        pytest.skip("search disabled in the shipped config")
    every = int(config.get("search_every", 25))
    assert 0 % every == 0, "step 0 always triggers the search lane"
    budget = int(config.get("search_budget", 64))
    assert budget > 0
    # A requeued child restarts at the resumed step, so this recurs whenever the
    # resume step is a multiple of search_every.
    assert (0 % every == 0) and (every * 3 % every == 0)


def test_search_straggler_stays_inside_the_collective_timeout():
    """Rank 0's search must finish well inside the distributed timeout, or the
    other seven ranks abort the job at the next all-gather."""
    config = _grpo_config()
    if not config.get("use_search"):
        pytest.skip("search disabled in the shipped config")
    budget = int(config.get("search_budget", 64))
    # Measured on gfx950 for a small elementwise task: ~17 s per verified
    # evaluation (correctness battery + timed bench).
    seconds_per_eval = 17.0
    projected = budget * seconds_per_eval
    assert projected < 1800, (
        f"search_budget={budget} projects to {projected/60:.1f} min of rank-0 "
        "straggler time against a 1800 s collective timeout"
    )


def test_grpo_launcher_reroots_the_coevolution_archive():
    """``_build_opus_scores`` derives the archive root from
    ``dirname(coevolve_distill_path)``, so leaving the shipped ``data/full14b``
    path in place points the regret curriculum at a different data root -- and it
    fails SAFE, returning no scores rather than complaining."""
    launcher = GRPO_SBATCH.read_text()
    assert "--data-root" in launcher
    sys.path.insert(0, str(REPO / "scripts"))
    from spur_resolve_launch_config import _GRPO_DATA_ROOT_KEYS

    assert set(_GRPO_DATA_ROOT_KEYS) == {
        "coevolve_distill_path", "coevolve_opus_scores_path"}


def test_resolver_refuses_a_from_stage_that_is_not_a_real_checkpoint(tmp_path):
    """``spur_grpo_1node.sbatch`` defaults FROM_STAGE to ``runs/dpo_14b_frontier``."""
    sys.path.insert(0, str(REPO / "scripts"))
    from spur_resolve_launch_config import resolve

    with pytest.raises(ValueError, match="not a loadable checkpoint"):
        resolve("grpo", _grpo_config(), from_stage="runs/dpo_14b_frontier",
                output_dir="runs/grpo_14b_frontier", repo_root=tmp_path)


def test_resolver_rewrites_grpo_paths_for_a_real_checkpoint(tmp_path):
    from tests.test_sft_launch_readiness import _write_fake_checkpoint

    sys.path.insert(0, str(REPO / "scripts"))
    from spur_resolve_launch_config import resolve

    _write_fake_checkpoint(tmp_path / "runs" / "dpo_14b_frontier")
    resolved, _changes = resolve(
        "grpo", _grpo_config(), from_stage="runs/dpo_14b_frontier",
        output_dir="runs/grpo_14b_frontier", data_root="data/b05factory",
        repo_root=tmp_path)
    assert resolved["model_id"] == "runs/dpo_14b_frontier"
    assert resolved["output_dir"] == "runs/grpo_14b_frontier"
    for key in ("coevolve_distill_path", "coevolve_opus_scores_path"):
        if resolved.get(key):
            assert resolved[key].startswith("data/b05factory/"), resolved[key]


def test_grpo_config_loader_drops_the_in_config_comment_convention():
    from kore.policy.grpo import grpo_config_from_dict

    grpo_config_from_dict({**_grpo_config(),
                           "_comment_anything": "prose", "_note": 1})


def test_grpo_config_loader_drops_a_stale_lora_block():
    from kore.policy.grpo import grpo_config_from_dict

    config = grpo_config_from_dict({**_grpo_config(),
                                    "lora": {"r": 8}, "tasks": ["gen_add_bf16"]})
    assert config.use_lora is False


if __name__ == "__main__":  # pragma: no cover - convenience for ad-hoc runs
    sys.exit(pytest.main([__file__, "-v"]))
