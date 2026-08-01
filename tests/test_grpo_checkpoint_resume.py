"""Preemption safety for GRPO: periodic checkpoints + full resume.

GRPO runs on a preemptible Slurm QoS with ``--requeue``, so the run MUST survive
being killed at step 1,900 of 2,000. These tests pin the contract that makes that
true, all on CPU with tiny fakes (no GPU, no 14B, no real FSDP):

  * periodic saves fire on the configured cadence in BOTH loops, and a step that
    was skipped on the save boundary does not push the checkpoint a full period out;
  * a resume restores weights source, optimizer moments, LR-schedule position and
    the step counter, and the loop continues from that step instead of step 0;
  * a corrupt / half-written checkpoint FAILS CLOSED (raises) instead of silently
    restarting from scratch - and a damaged NEWEST checkpoint falls back to the
    previous good one rather than losing the run;
  * retention keeps >= 2 checkpoints and only rotates the old one out AFTER the
    replacement is published and re-validated;
  * the distributed save runs an IDENTICAL sequence of collectives on every rank
    (the invariant that keeps rank 0 from deadlocking inside a gather its peers
    never enter), with only the file writes rank-0 gated.

Local fakes throughout (no cross-test imports), mirroring tests/test_grpo_fsdp.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from kore.policy import grpo
from kore.policy.configs import GRPOConfig, latest_checkpoint


# --------------------------------------------------------------------------- #
# tiny CPU fakes
# --------------------------------------------------------------------------- #
def _tiny_model():
    """A minimal trainable torch module with an HF-ish ``save_pretrained``."""
    import torch

    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(4, 4)
            self.config = SimpleNamespace(use_cache=True)

        def forward(self, x):
            return self.head(x)

        def save_pretrained(self, path, is_main_process=True, save_function=None,
                            state_dict=None):
            if not is_main_process:
                return                      # followers hold an empty rank0-only dict
            os.makedirs(path, exist_ok=True)
            Path(path, "model.safetensors").write_text("weights")

    return TinyLM()


class _FakeTok:
    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        Path(path, "tokenizer.json").write_text("{}")


def _cfg(tmp_path, **kw):
    base = dict(model_id="tiny", output_dir=str(tmp_path), use_lora=False,
                total_steps=10, save_steps=1)
    base.update(kw)
    return GRPOConfig(**base)


def _optim_and_sched(model, config, n_steps=0):
    """A real AdamW + the loop's own LambdaLR, advanced ``n_steps`` times."""
    import torch

    opt = torch.optim.AdamW(model.parameters(), lr=0.1)
    sched = grpo._build_lr_scheduler(opt, config)
    for _ in range(n_steps):
        loss = model(torch.ones(1, 4)).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
        sched.step()
    return opt, sched


def _write_checkpoint(tmp_path, step, *, valid=True, drop_file=None, bad_json=False,
                      optimizer_state_saved=True):
    """Hand-build a checkpoint dir, optionally damaged in a specific way."""
    ckpt = Path(tmp_path, f"checkpoint-{step}")
    ckpt.mkdir(parents=True, exist_ok=True)
    Path(ckpt, "model.safetensors").write_text("weights")
    Path(ckpt, grpo._GRPO_OPTIM_FILE).write_text("opt")
    if not valid:
        return str(ckpt)                                   # no trainer_state.json at all
    if bad_json:
        Path(ckpt, grpo._GRPO_STATE_FILE).write_text("{truncated")
        return str(ckpt)
    files = ["model.safetensors", grpo._GRPO_OPTIM_FILE]
    if drop_file:
        files.append(drop_file)                            # manifest names a missing file
    Path(ckpt, grpo._GRPO_STATE_FILE).write_text(json.dumps({
        "global_step": step,
        "optimizer_state_saved": optimizer_state_saved,
        "kore_grpo_files": files,
    }))
    return str(ckpt)


# --------------------------------------------------------------------------- #
# save cadence
# --------------------------------------------------------------------------- #
def test_should_save_fires_on_the_configured_cadence():
    cfg = _cfg("out", save_steps=100, total_steps=2000)
    assert grpo._grpo_should_save(98, 0, cfg) is False
    assert grpo._grpo_should_save(99, 0, cfg) is True       # step 99 completes step 100
    assert grpo._grpo_should_save(100, 100, cfg) is False   # just saved
    assert grpo._grpo_should_save(199, 100, cfg) is True


def test_should_save_recovers_a_boundary_step_that_was_skipped():
    # Step 99 (the exact boundary) was skipped - all groups collapsed - so no save
    # happened. The next completed step must still checkpoint instead of waiting
    # another full period, which would double the work a preemption destroys.
    cfg = _cfg("out", save_steps=100, total_steps=2000)
    assert grpo._grpo_should_save(100, 0, cfg) is True
    assert grpo._grpo_should_save(150, 0, cfg) is True


def test_should_save_skips_the_final_step_and_a_disabled_cadence():
    cfg = _cfg("out", save_steps=100, total_steps=200)
    assert grpo._grpo_should_save(199, 100, cfg) is False    # loop-exit save covers it
    assert grpo._grpo_should_save(99, 0, _cfg("out", save_steps=0, total_steps=200)) is False


def test_periodic_save_writes_a_resumable_checkpoint(tmp_path):
    pytest.importorskip("torch")
    cfg = _cfg(tmp_path, total_steps=10)
    model = _tiny_model()
    opt, sched = _optim_and_sched(model, cfg, n_steps=3)

    ckpt = grpo._save_grpo_checkpoint(model, _FakeTok(), cfg, 3, optimizer=opt,
                                      scheduler=sched)

    assert ckpt == str(tmp_path / "checkpoint-3")
    state = grpo._read_grpo_trainer_state(ckpt)
    assert state is not None and state["global_step"] == 3
    assert state["optimizer_state_saved"] is True
    for name in (grpo._GRPO_OPTIM_FILE, grpo._GRPO_SCHED_FILE, grpo._GRPO_RNG_FILE,
                 "model.safetensors", "tokenizer.json"):
        assert Path(ckpt, name).exists(), name
    # the manifest is what makes a truncated dir detectable, so it must be complete
    assert set(state["kore_grpo_files"]) >= {grpo._GRPO_OPTIM_FILE, grpo._GRPO_SCHED_FILE,
                                             grpo._GRPO_RNG_FILE, "model.safetensors"}
    # nothing half-written is left behind for discovery to trip over
    assert not [p for p in os.listdir(tmp_path) if p.startswith(grpo._GRPO_STAGING_PREFIX)]


# --------------------------------------------------------------------------- #
# resume: step, optimizer, scheduler
# --------------------------------------------------------------------------- #
def test_resume_restores_step_optimizer_moments_and_lr_schedule(tmp_path):
    torch = pytest.importorskip("torch")
    cfg = _cfg(tmp_path, total_steps=10, lr_scheduler_type="linear")
    model = _tiny_model()
    opt, sched = _optim_and_sched(model, cfg, n_steps=4)
    ckpt = grpo._save_grpo_checkpoint(model, _FakeTok(), cfg, 4, optimizer=opt,
                                      scheduler=sched)
    saved_moment = opt.state[model.head.weight]["exp_avg"].clone()
    sched.step()
    lr_after_save = opt.param_groups[0]["lr"]

    # exactly what a requeued process builds: fresh model/optimizer/scheduler.
    model2 = _tiny_model()
    opt2, sched2 = _optim_and_sched(model2, cfg, n_steps=0)
    fresh_lr = opt2.param_groups[0]["lr"]

    start = grpo._restore_grpo_training_state(
        (ckpt, grpo._read_grpo_trainer_state(ckpt)), cfg, optimizer=opt2, scheduler=sched2)

    assert start == 4
    assert torch.allclose(opt2.state[model2.head.weight]["exp_avg"], saved_moment)
    sched2.step()
    assert opt2.param_groups[0]["lr"] == pytest.approx(lr_after_save)
    # and the restored LR is genuinely different from a schedule restarted at 0,
    # so this assertion would catch a silent "resume" that lost the scheduler.
    assert lr_after_save != pytest.approx(fresh_lr)


def test_resume_restores_rng_so_rollout_sampling_continues(tmp_path):
    torch = pytest.importorskip("torch")
    cfg = _cfg(tmp_path, total_steps=10)
    model = _tiny_model()
    opt, sched = _optim_and_sched(model, cfg)
    torch.manual_seed(1234)
    ckpt = grpo._save_grpo_checkpoint(model, _FakeTok(), cfg, 1, optimizer=opt,
                                      scheduler=sched)
    expected = torch.randn(3)

    torch.manual_seed(9999)                                  # a different process
    grpo._restore_grpo_training_state(
        (ckpt, grpo._read_grpo_trainer_state(ckpt)), cfg, optimizer=opt, scheduler=sched)

    assert torch.allclose(torch.randn(3), expected)


def test_find_resume_returns_none_for_a_fresh_output_dir(tmp_path):
    assert grpo._find_grpo_resume_checkpoint(_cfg(tmp_path)) is None
    assert grpo._discover_grpo_resume(_cfg(tmp_path)) is None


def test_find_resume_picks_the_newest_valid_checkpoint(tmp_path):
    _write_checkpoint(tmp_path, 100)
    _write_checkpoint(tmp_path, 300)
    _write_checkpoint(tmp_path, 200)
    path, state = grpo._find_grpo_resume_checkpoint(_cfg(tmp_path))
    assert path.endswith("checkpoint-300") and state["global_step"] == 300


# --------------------------------------------------------------------------- #
# fail closed on a damaged checkpoint
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("damage", [
    {"valid": False},                                   # save died before trainer_state
    {"bad_json": True},                                 # truncated trainer_state
    {"drop_file": "scheduler.pt"},                      # manifest names a lost file
])
def test_damaged_checkpoint_fails_closed_instead_of_restarting(tmp_path, damage):
    _write_checkpoint(tmp_path, 1900, **damage)
    cfg = _cfg(tmp_path)
    assert grpo._read_grpo_trainer_state(tmp_path / "checkpoint-1900") is None
    with pytest.raises(RuntimeError, match="none is resumable"):
        grpo._find_grpo_resume_checkpoint(cfg)
    with pytest.raises(RuntimeError, match="none is resumable"):
        grpo._discover_grpo_resume(cfg)


def test_damaged_newest_checkpoint_falls_back_to_the_previous_good_one(tmp_path):
    # A half-written newest checkpoint must not cost the run its history.
    # configs.latest_checkpoint used to inspect only the highest-numbered dir and
    # report "nothing to resume" here, which is why GRPO grew its own discovery;
    # that helper now walks newest-first too, so both agree on checkpoint-1800.
    good = _write_checkpoint(tmp_path, 1800)
    _write_checkpoint(tmp_path, 1900, valid=False)
    assert latest_checkpoint(tmp_path) == str(good)

    # GRPO's discovery remains the stricter of the two: it additionally verifies
    # the trainer state parses and the manifest's files are all present, so it
    # rejects checkpoints that merely *have* a trainer_state.json.
    path, state = grpo._find_grpo_resume_checkpoint(_cfg(tmp_path))
    assert path == good and state["global_step"] == 1800


def test_resume_refuses_a_checkpoint_without_consolidated_optimizer_state(tmp_path):
    # Written by a backend whose optimizer state could not be gathered. Resuming
    # would silently reinitialize AdamW; refuse instead.
    ckpt = _write_checkpoint(tmp_path, 500, optimizer_state_saved=False)
    with pytest.raises(RuntimeError, match="no consolidated optimizer state"):
        grpo._restore_grpo_training_state(
            (ckpt, grpo._read_grpo_trainer_state(ckpt)), _cfg(tmp_path),
            optimizer=object(), scheduler=None)


def test_publish_rejects_a_staging_dir_that_is_not_complete(tmp_path):
    staging = Path(grpo._stage_grpo_checkpoint_dir(tmp_path, 7))
    Path(staging, "model.safetensors").write_text("weights")   # no trainer_state.json
    with pytest.raises(RuntimeError, match="completeness check"):
        grpo._publish_grpo_checkpoint(_cfg(tmp_path), str(staging), 7)
    assert not Path(tmp_path, "checkpoint-7").exists()
    assert not staging.exists()


# --------------------------------------------------------------------------- #
# retention
# --------------------------------------------------------------------------- #
def test_retention_default_is_at_least_two_and_configurable():
    assert grpo._grpo_save_total_limit(_cfg("out")) == 2            # no field set
    assert grpo._grpo_save_total_limit(SimpleNamespace(save_total_limit=5)) == 5
    # 1 is the dangerous setting the old rotation hardcoded: clamped up to 2 so a
    # crash during the next save always leaves a predecessor to resume from.
    assert grpo._grpo_save_total_limit(SimpleNamespace(save_total_limit=1)) == 2
    assert grpo._grpo_save_total_limit(SimpleNamespace(save_total_limit="bad")) == 2


def test_rotation_keeps_the_newest_n_checkpoints(tmp_path):
    pytest.importorskip("torch")
    cfg = _cfg(tmp_path, total_steps=10)
    model = _tiny_model()
    opt, sched = _optim_and_sched(model, cfg)
    for step in (1, 2, 3, 4):
        grpo._save_grpo_checkpoint(model, _FakeTok(), cfg, step, optimizer=opt,
                                   scheduler=sched)
    kept = sorted(s for s, _p in grpo._grpo_checkpoint_dirs(tmp_path))
    assert kept == [3, 4]                              # default retention of 2


def test_rotation_runs_only_after_the_replacement_is_published_and_valid(tmp_path,
                                                                         monkeypatch):
    # The ordering that makes rotation safe: at the moment old checkpoints become
    # deletable, the new one must already be on disk under its final name AND pass
    # the same completeness check discovery applies.
    pytest.importorskip("torch")
    cfg = _cfg(tmp_path, total_steps=10)
    model = _tiny_model()
    opt, sched = _optim_and_sched(model, cfg)
    grpo._save_grpo_checkpoint(model, _FakeTok(), cfg, 1, optimizer=opt, scheduler=sched)

    seen = {}
    real_rotate = grpo._rotate_grpo_checkpoints

    def spy(config, keep_path):
        seen["new_is_valid"] = grpo._read_grpo_trainer_state(keep_path) is not None
        seen["old_still_present"] = Path(tmp_path, "checkpoint-1").exists()
        return real_rotate(config, keep_path)

    monkeypatch.setattr(grpo, "_rotate_grpo_checkpoints", spy)
    grpo._save_grpo_checkpoint(model, _FakeTok(), cfg, 2, optimizer=opt, scheduler=sched)

    assert seen == {"new_is_valid": True, "old_still_present": True}


def test_a_failed_save_leaves_every_existing_checkpoint_intact(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    cfg = _cfg(tmp_path, total_steps=10)
    model = _tiny_model()
    opt, sched = _optim_and_sched(model, cfg)
    grpo._save_grpo_checkpoint(model, _FakeTok(), cfg, 1, optimizer=opt, scheduler=sched)
    grpo._save_grpo_checkpoint(model, _FakeTok(), cfg, 2, optimizer=opt, scheduler=sched)

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(grpo, "_write_grpo_resume_state", boom)
    assert grpo._save_grpo_checkpoint(model, _FakeTok(), cfg, 3, optimizer=opt,
                                      scheduler=sched) is None

    assert sorted(s for s, _p in grpo._grpo_checkpoint_dirs(tmp_path)) == [1, 2]
    assert grpo._find_grpo_resume_checkpoint(cfg)[0].endswith("checkpoint-2")


def test_a_failed_save_raises_under_a_strict_profile(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    cfg = _cfg(tmp_path, total_steps=10, resume_state_required=True)
    model = _tiny_model()
    opt, sched = _optim_and_sched(model, cfg)
    monkeypatch.setattr(grpo, "_write_grpo_resume_state",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        grpo._save_grpo_checkpoint(model, _FakeTok(), cfg, 1, optimizer=opt, scheduler=sched)


# --------------------------------------------------------------------------- #
# distributed save: identical collectives on every rank
# --------------------------------------------------------------------------- #
class _FakeAccelerator:
    """Records every collective so two ranks' sequences can be compared."""

    def __init__(self, rank, world, trace):
        self.process_index = rank
        self.num_processes = world
        self.is_main_process = (rank == 0)
        self.device = "cpu"
        self._trace = trace

    def wait_for_everyone(self):
        self._trace.append("barrier")

    def get_state_dict(self, model):
        self._trace.append("get_state_dict")
        return {"w": 1} if self.is_main_process else {}     # FSDP rank0_only gather

    def unwrap_model(self, model):
        return model

    def save(self, obj, f):
        pass


def _run_distributed_save(monkeypatch, tmp_path, rank, world=4, step=100):
    trace = []
    monkeypatch.setattr(grpo, "_all_gather_object",
                        lambda obj, acc=None: (trace.append("all_gather"), [obj] * world)[1])
    monkeypatch.setattr(grpo, "_broadcast_rank0_object",
                        lambda obj, acc=None, src=0: (trace.append("broadcast"), obj)[1])
    monkeypatch.setattr(grpo, "_gather_full_optim_state",
                        lambda m, o, acc=None: (trace.append("optim_gather"),
                                                o.state_dict() if acc.is_main_process
                                                else {})[1])
    cfg = _cfg(tmp_path, total_steps=2000, save_steps=100)
    model = _tiny_model()
    opt, sched = _optim_and_sched(model, cfg, n_steps=2)
    acc = _FakeAccelerator(rank, world, trace)
    grpo._save_grpo_checkpoint_distributed(model, _FakeTok(), cfg, step, accelerator=acc,
                                           optimizer=opt, scheduler=sched)
    return trace


def test_distributed_save_runs_identical_collectives_on_every_rank(tmp_path, monkeypatch):
    # THE deadlock invariant: the weight gather, optimizer gather, RNG all-gather,
    # verdict broadcast and barriers all live OUTSIDE the rank guard, so a follower
    # never sits at a collective rank 0 is not in (or vice versa).
    traces = [_run_distributed_save(monkeypatch, tmp_path / f"r{r}", r, world=4)
              for r in range(4)]

    assert traces[0] == traces[1] == traces[2] == traces[3]
    assert traces[0].count("get_state_dict") == 1
    assert traces[0].count("optim_gather") == 1
    assert traces[0].count("all_gather") == 1              # per-rank RNG states
    assert traces[0].count("broadcast") == 1               # rank-0 save verdict
    assert traces[0].count("barrier") >= 2


def test_distributed_save_writes_only_on_rank_zero(tmp_path, monkeypatch):
    main, follower = tmp_path / "main", tmp_path / "follower"
    _run_distributed_save(monkeypatch, main, rank=0, world=4)
    _run_distributed_save(monkeypatch, follower, rank=3, world=4)

    published = main / "checkpoint-100"
    assert grpo._read_grpo_trainer_state(published) is not None
    assert grpo._read_grpo_trainer_state(published)["world_size"] == 4
    # the follower published nothing and left no staging dir behind
    assert not (follower / "checkpoint-100").exists()
    follower_entries = os.listdir(follower) if follower.exists() else []
    assert not [p for p in follower_entries if p.startswith(grpo._GRPO_STAGING_PREFIX)]


def test_distributed_save_failure_is_broadcast_so_all_ranks_raise(tmp_path, monkeypatch):
    # Rank 0 owns the writes, so only rank 0 sees the error. It must be broadcast:
    # a rank 0 that raised alone would leave its peers blocked on the next
    # collective forever instead of failing the job.
    trace = []
    monkeypatch.setattr(grpo, "_all_gather_object", lambda obj, acc=None: [obj])
    monkeypatch.setattr(grpo, "_gather_full_optim_state", lambda m, o, acc=None: {})
    monkeypatch.setattr(grpo, "_broadcast_rank0_object",
                        lambda obj, acc=None, src=0: "OSError('disk full')")
    monkeypatch.setattr(grpo, "_write_grpo_resume_state",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    cfg = _cfg(tmp_path, total_steps=2000)
    model = _tiny_model()
    opt, sched = _optim_and_sched(model, cfg)

    for rank in (0, 3):                                    # writer AND follower raise
        acc = _FakeAccelerator(rank, 4, trace)
        with pytest.raises(RuntimeError, match="disk full"):
            grpo._save_grpo_checkpoint_distributed(model, _FakeTok(), cfg, 100,
                                                   accelerator=acc, optimizer=opt,
                                                   scheduler=sched)


def test_gather_full_optim_state_single_process_and_unsupported_backend(tmp_path):
    pytest.importorskip("torch")
    cfg = _cfg(tmp_path)
    model = _tiny_model()
    opt, _sched = _optim_and_sched(model, cfg, n_steps=1)

    assert grpo._gather_full_optim_state(model, opt) is not None
    assert grpo._gather_full_optim_state(model, None) is None
    # multi-rank but not FSDP (e.g. a DeepSpeed engine): the state cannot be
    # consolidated, so None is recorded and a later resume fails closed.
    multi = SimpleNamespace(num_processes=8, process_index=0, is_main_process=True)
    assert grpo._gather_full_optim_state(model, opt, multi) is None


# --------------------------------------------------------------------------- #
# loop parity: overlong masking + adaptive horizon
# --------------------------------------------------------------------------- #
def test_overlong_mask_drops_truncated_samples_but_keeps_their_returns():
    cfg = _cfg("out", max_response_length=16384, overlong_buffer_len=512,
               overlong_mask=True)
    # sample = [ret, gen_inputs, ref_logp, old_logp, n_tokens, sc_weight]
    short = [0.5, [("p", "g")], None, None, 100, None]
    truncated = [0.9, [("p", "g")], None, None, 16384, None]
    groups = [[short, truncated]]

    assert grpo._apply_overlong_mask(groups, cfg) == 1
    assert short[1] and truncated[1] is None      # only the truncated one is dropped
    # the return survives, so the group's advantage baseline is unchanged - exactly
    # what the distributed loop does (it masks AFTER the cross-rank normalization).
    assert [s[0] for s in groups[0]] == [0.5, 0.9]


def test_overlong_mask_is_inert_when_disabled_or_misconfigured():
    sample = [0.5, [("p", "g")], None, None, 3, None]
    off = _cfg("out", overlong_mask=False, max_response_length=16384)
    assert grpo._apply_overlong_mask([[list(sample)]], off) == 0
    # buffer >= cap collapses the is_overlong threshold to one token, which would
    # mask the ENTIRE batch. That is a misconfiguration, not a batch of truncations.
    assert grpo._overlong_masked(3, _cfg("out", max_response_length=4,
                                         overlong_buffer_len=512)) is False
    assert grpo._overlong_masked(16384, _cfg("out", max_response_length=16384,
                                             overlong_buffer_len=512)) is True


def test_step_controller_is_built_from_the_same_config_in_both_loops():
    assert grpo._build_step_controller(_cfg("out", adaptive_steps=False)) is None
    ctrl = grpo._build_step_controller(_cfg("out", adaptive_steps=True, total_steps=500,
                                            min_steps=2, plateau_patience=3,
                                            plateau_min_delta=0.01))
    assert (ctrl.min_steps, ctrl.max_steps, ctrl.patience) == (2, 500, 3)
    assert ctrl.update(0, 1.0) is False                       # below min_steps
    for step in range(1, 6):
        stop = ctrl.update(step, 1.0)                         # flat -> plateau
    assert stop is True and "plateau" in ctrl.stopped_reason


# --------------------------------------------------------------------------- #
# end-to-end: the REAL single-process loop checkpoints, is killed, and resumes
# --------------------------------------------------------------------------- #
def _install_tiny_loop(monkeypatch, decode_text=None):
    """Point the real GRPO loop at a tiny CPU model + fake env; return call spies."""
    import torch
    import transformers

    import kore.env.kore_env as ke
    import kore.tasks.registry as reg
    from kore.reward.reward import Observation

    decode_text = decode_text or (
        "ANALYSIS:\nx\n\nPROPOSED_CHANGE:\ny\n\n"
        "FULL_KERNEL:\n```python\ndef k():\n    return 0\n```")
    spy = {"loaded": [], "rollouts": 0}

    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vocab = 32
            self.emb = torch.nn.Embedding(self.vocab, 8)
            self.head = torch.nn.Linear(8, self.vocab)
            self.device = torch.device("cpu")
            self.config = SimpleNamespace(use_cache=True)

        def forward(self, input_ids):
            return SimpleNamespace(logits=self.head(self.emb(input_ids)))

        def generate(self, input_ids, max_new_tokens=4, **kw):
            gen = torch.randint(0, self.vocab, (1, max(1, min(int(max_new_tokens), 3))))
            return SimpleNamespace(sequences=torch.cat([input_ids, gen], dim=1))

        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
            pass

        def enable_input_require_grads(self):
            pass

        def save_pretrained(self, path, **kw):
            os.makedirs(path, exist_ok=True)
            Path(path, "model.safetensors").write_text("weights")

    class TinyTok:
        def _encode(self, text):
            return [(ord(c) % 29) + 1 for c in (text or "x")[:12]] or [1]

        def apply_chat_template(self, messages, add_generation_prompt=True,
                                return_tensors="pt", **kw):
            return torch.tensor([self._encode(
                " ".join(str(m.get("content", "")) for m in messages))])

        def __call__(self, text, return_tensors="pt", **kw):
            return SimpleNamespace(input_ids=torch.tensor([self._encode(text)]))

        def decode(self, seq, skip_special_tokens=True):
            return decode_text

        def save_pretrained(self, path):
            os.makedirs(path, exist_ok=True)
            Path(path, "tokenizer.json").write_text("{}")

    class FakeEnv:
        """First rollout is correct+fast, the rest fail -> real reward variance."""

        def __init__(self, task, **kw):
            self.task = task
            self.n = 0

        def step(self, source, full_validation=True, multi_shape=True):
            self.n += 1
            if self.n == 1:
                return Observation(compiled=True, dtype="bf16", validation_passed=True,
                                   snr_by_shape={"primary": 100.0}, snr_db=100.0,
                                   wall_by_shape={"primary": 1.0},
                                   baseline_by_shape={"primary": 2.0},
                                   wall_ms=1.0, baseline_ms=2.0)
            return Observation(compiled=True, dtype="bf16", validation_passed=False,
                               snr_by_shape={"primary": 3.0}, snr_db=3.0,
                               error_text="worst SNR 3.0 < gate")

    class FakeTask:
        task_id = "fake_gemm_bf16"
        operation, dtype, gpu_target, backend = "gemm", "bf16", "gfx942", "triton"
        comparison_baseline = "aiter"
        seed_source = "def seed():\n    return 0"
        shapes: list = []
        snr_threshold = None

    def load_model(model_id, **kw):
        spy["loaded"].append(str(model_id))
        return TinyLM()

    monkeypatch.setattr(transformers, "AutoModelForCausalLM",
                        SimpleNamespace(from_pretrained=load_model))
    monkeypatch.setattr(transformers, "AutoTokenizer",
                        SimpleNamespace(from_pretrained=lambda mid, **kw: TinyTok()))
    monkeypatch.setattr(ke, "KoreEnv", FakeEnv)
    monkeypatch.setattr(reg, "get_task", lambda tid: FakeTask())
    monkeypatch.setattr(reg, "task_ids", lambda: ["fake_gemm_bf16"])

    real_rollout = grpo._rollout
    monkeypatch.setattr(grpo, "_rollout",
                        lambda *a, **k: (spy.__setitem__("rollouts", spy["rollouts"] + 1)
                                         or real_rollout(*a, **k)))
    return spy


def _e2e_cfg(tmp_path, **kw):
    base = dict(
        model_id="tiny", output_dir=str(tmp_path), use_lora=False,
        num_trajectories=2, num_turns=1, tasks_per_step=1,
        gradient_checkpointing=False, bf16=False, learning_rate=0.1,
        warmup_ratio=0.0, lr_scheduler_type="constant", max_response_length=4,
        max_prompt_length=16, temperature=0.9, top_p=1.0, agentic=False,
        ref_anchor_coef=0.0, starpo_s=True, dynamic_sampling=True, logging_steps=1,
    )
    base.update(kw)
    return GRPOConfig(**base)


def test_e2e_loop_checkpoints_then_resumes_at_the_right_step(monkeypatch, tmp_path):
    """Run the REAL single-process loop, stop it, and restart it: the second run
    must load the checkpoint, start at the saved step, and roll out only the
    REMAINING steps instead of redoing the whole run."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    spy = _install_tiny_loop(monkeypatch)
    grpo._train_grpo_inprocess(_e2e_cfg(tmp_path, total_steps=2, save_steps=1),
                               tasks=["fake_gemm_bf16"])

    assert grpo._read_grpo_trainer_state(tmp_path / "checkpoint-1") is not None
    first_run_rollouts = spy["rollouts"]
    assert first_run_rollouts == 4               # 2 steps x 2 trajectories

    spy["loaded"].clear()
    spy["rollouts"] = 0
    grpo._train_grpo_inprocess(_e2e_cfg(tmp_path, total_steps=4, save_steps=1),
                               tasks=["fake_gemm_bf16"])

    # the policy was loaded FROM the checkpoint, not from the base model id
    assert spy["loaded"][0] == str(tmp_path / "checkpoint-1")
    # and only steps 1..3 ran (6 rollouts); a silent restart from 0 would be 8
    assert spy["rollouts"] == 6
    assert sorted(s for s, _p in grpo._grpo_checkpoint_dirs(tmp_path)) == [2, 3]


def test_e2e_loop_fails_closed_on_a_corrupt_checkpoint(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    spy = _install_tiny_loop(monkeypatch)
    _write_checkpoint(tmp_path, 1900, valid=False)

    with pytest.raises(RuntimeError, match="none is resumable"):
        grpo._train_grpo_inprocess(_e2e_cfg(tmp_path, total_steps=2, save_steps=1),
                                   tasks=["fake_gemm_bf16"])
    assert spy["rollouts"] == 0                  # it never silently trained from zero


def test_e2e_loop_stops_early_on_an_adaptive_plateau(monkeypatch, tmp_path):
    """adaptive_steps existed only in the distributed loop; the single-process loop
    must honor it too, or the two GRPO recipes are not the same contract."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    spy = _install_tiny_loop(monkeypatch)
    grpo._train_grpo_inprocess(
        _e2e_cfg(tmp_path, total_steps=20, save_steps=0, adaptive_steps=True,
                 min_steps=1, plateau_patience=1, plateau_min_delta=10.0),
        tasks=["fake_gemm_bf16"])

    assert 0 < spy["rollouts"] < 40              # stopped well before 20 full steps
