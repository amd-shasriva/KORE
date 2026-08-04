

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


def test_the_30b_sft_config_writes_weights_only_checkpoints():
    """Weights-only is what makes a checkpoint survivable on this filesystem.

    Run 33992 died at step 400 with 'Disk quota exceeded' after 200M tokens of
    healthy training, writing 4 of 25 shards. /shared_nfs is 146T of 150T used
    and shared, so a ~488GB write is long enough for someone else to take the
    margin; ~61GB is not.
    """
    import json
    import pathlib

    path = pathlib.Path(__file__).resolve().parent.parent / "configs" / "sft_coder30b_a3b.json"
    raw = json.loads(path.read_text())
    assert raw.get("save_only_model") is True
    assert raw.get("save_total_limit") == 1
