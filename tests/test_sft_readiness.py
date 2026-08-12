

def test_every_key_in_the_30b_sft_config_is_recognised():
    """A config key the dataclass does not declare is silently ignored.

    This is not hypothetical. `save_only_model` was added to
    configs/sft_coder30b_a3b.json to stop 488GB checkpoint writes from being
    killed by other users filling the shared volume, and it did nothing at all
    until SFTConfig declared it -- the JSON looked correct, the run looked
    configured, and the setting was dropped on the floor. Anything that ships a
    number in a config file and reads it through getattr can fail this way, so
    the whole file is checked rather than the one key.
    """
    import dataclasses
    import json
    import pathlib

    from kore.policy.configs import SFTConfig
    from kore.policy.model_spec import IDENTITY_CONFIG_KEYS
    from kore.policy.resources import PREFLIGHT_CONFIG_KEYS

    path = pathlib.Path(__file__).resolve().parent.parent / "configs" / "sft_coder30b_a3b.json"
    raw = json.loads(path.read_text())
    # Three legitimate homes for a key: an SFTConfig field, or an identity /
    # preflight key that sft_config_from_dict splits off and attaches as an
    # attribute, or an underscore-prefixed comment. Anything else is dropped.
    allowed = ({f.name for f in dataclasses.fields(SFTConfig)}
               | set(IDENTITY_CONFIG_KEYS) | set(PREFLIGHT_CONFIG_KEYS))
    unknown = sorted(k for k in raw
                     if not k.startswith("_") and k not in allowed)
    assert not unknown, (
        f"{len(unknown)} config key(s) are neither SFTConfig fields nor "
        f"identity/preflight keys, so they are silently ignored at load: {unknown}")


def test_the_30b_sft_config_keeps_resumable_checkpoints():
    """A killed run must resume, not restart.

    Run 33992 died at step 400 with 'Disk quota exceeded' after 200M tokens of
    healthy training. The tempting fix was save_only_model, which shrinks the
    write 8x by dropping Adam state -- but it also makes every future failure
    unrecoverable, and the premise was wrong: there is no per-user quota. Both
    volumes are NFSv3 with no quota tooling, and NFSv3 reports a full volume as
    EDQUOT, so the write simply landed while other users had the volume at zero.
    With 42T free against a ~1.46TB rotation peak, the space is there and the
    optimizer state should stay.
    """
    import json
    import pathlib

    path = pathlib.Path(__file__).resolve().parent.parent / "configs" / "sft_coder30b_a3b.json"
    raw = json.loads(path.read_text())
    assert raw.get("save_only_model") is False, (
        "optimizer state must be kept so a killed run resumes; if disk pressure "
        "returns, fix the disk rather than the resumability")
    # >= 2, and this reverses what this test used to assert. limit=1 looked
    # self-cleaning -- the Trainer writes the new checkpoint and then deletes the
    # previous one, so steady state is one -- but that ordering is exactly the
    # problem: there is a window in which the only complete checkpoint has been
    # deleted, and on a partition that kills jobs at the ~2h mark, a run with 32
    # save points will eventually be killed inside it. The premise that forced
    # limit=1 (~1090GB free, where two checkpoints did not fit) was stale by three
    # generations; /shared_nfs has 42T, so ~1.46TB is 3.5% of it.
    assert raw.get("save_total_limit", 0) >= 2, (
        "save_total_limit must be >= 2: at limit=1 the rotation window leaves "
        "nothing resumable, and preemption is the expected way this run ends")
    # A 662-step run with save_steps=400 has exactly ONE save point, and run
    # 33992 died on it. Three save points bound the worst-case loss to ~200 steps.
    assert raw.get("save_steps") <= 200, (
        f"save_steps={raw.get('save_steps')} leaves too few checkpoints on a "
        "662-step run; the previous failure landed on the only one")


def test_moe_router_stability_settings():
    """Warmup and gradient clipping must be the MoE values, not the dense defaults.

    Qwen3-Coder-30B-A3B routes each token to a subset of 128 experts through a
    small linear router. Two dense-model defaults are actively harmful there:

    * a 3% warmup lets a large early step collapse the router onto a few experts
      before the loss has said anything useful, and the experts it stops routing
      to then get no gradient to recover with;
    * a 1.0 gradient clip permits spikes that are noise in a dense model but
      permanently re-route tokens in an MoE.

    Published MoE fine-tuning guidance is 0.10-0.15 warmup and 0.5 clip. v3 ran
    at 0.03/1.0 and converged, which is exactly why this is a test rather than a
    comment -- it is the kind of setting that looks fine until the run that
    matters, and v4 is 3x the tokens.
    """
    import json
    import pathlib

    path = (pathlib.Path(__file__).resolve().parent.parent
            / "configs" / "sft_coder30b_a3b.json")
    raw = json.loads(path.read_text())
    assert 0.10 <= raw.get("warmup_ratio", 0) <= 0.20, (
        f"warmup_ratio={raw.get('warmup_ratio')} is outside the MoE band; the "
        "dense 0.03 default risks router collapse")
    assert raw.get("max_grad_norm") == 0.5, (
        f"max_grad_norm={raw.get('max_grad_norm')} -- MoE wants 0.5, and an "
        "absent key silently inherits HF's dense 1.0")


def test_packing_stays_off_without_flash_attention():
    """Packing on SDPA cross-contaminates documents, silently.

    Packing concatenates examples into one sequence and depends on the attention
    mask to isolate them. kore/policy/sft.py refuses to enable it outside
    flash_attention_2 for that reason. This asserts the config does not ASK for
    it either, so the intent and the guard agree rather than relying on the guard
    to quietly override a config that says otherwise.
    """
    import json
    import pathlib

    from kore.policy.configs import preferred_attn_impl

    path = (pathlib.Path(__file__).resolve().parent.parent
            / "configs" / "sft_coder30b_a3b.json")
    raw = json.loads(path.read_text())
    if preferred_attn_impl() != "flash_attention_2":
        assert raw.get("packing") is False, (
            "packing is requested but the backend is SDPA; the trainer would "
            "disable it anyway, and a config that asks for something it cannot "
            "have is a trap for the next reader")


def test_checkpoint_rotation_bounds_disk_to_one_checkpoint():
    """32 save events must not mean 32 retained checkpoints.

    save_steps=50 over a 1,613-step run fires ~32 times, and a 30B checkpoint with
    optimizer state is ~488GB. Retaining them all would be ~15TB. What makes the
    frequency safe is that rotation is BOUNDED, not that it is bounded to one: at
    limit=2 steady state is two checkpoints and the rotation window briefly holds
    three, ~1.46TB against 42T free.

    The bound is two-sided on purpose. Too high and the footprint grows on a shared
    volume that has already had another user consume the margin under a write; too
    low -- specifically one -- and the rotation window has no complete checkpoint in
    it, which on a preempting partition means a kill can destroy the only resume
    point. Both failures have precedent in this project, which is why neither edge
    is left unasserted.
    """
    import json
    import pathlib

    path = (pathlib.Path(__file__).resolve().parent.parent
            / "configs" / "sft_coder30b_a3b.json")
    raw = json.loads(path.read_text())
    steps = raw.get("save_steps")
    limit = raw.get("save_total_limit")
    assert limit is not None and 2 <= limit <= 3, (
        f"save_total_limit={limit} with save_steps={steps}: at ~488GB per 30B "
        "checkpoint it must stay small, but 1 is unsafe because the rotation "
        "window would leave nothing resumable when a preemption lands in it")
    # Frequent saves are only defensible while rotation is bounded. If someone
    # raises the limit, they must also justify the frequency.
    assert steps is not None and steps <= 50, (
        f"save_steps={steps} -- preemption on this cluster kills runs at the "
        "~2h mark, so the interval bounds how much work a kill costs")
