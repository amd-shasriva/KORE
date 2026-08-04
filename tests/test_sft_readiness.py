

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
    With 17T free against a 976GB rotation peak, the space is there and the
    optimizer state should stay.
    """
    import json
    import pathlib

    path = pathlib.Path(__file__).resolve().parent.parent / "configs" / "sft_coder30b_a3b.json"
    raw = json.loads(path.read_text())
    assert raw.get("save_only_model") is False, (
        "optimizer state must be kept so a killed run resumes; if disk pressure "
        "returns, fix the disk rather than the resumability")
    assert raw.get("save_total_limit") == 1
