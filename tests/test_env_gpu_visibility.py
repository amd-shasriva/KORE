from __future__ import annotations

from types import SimpleNamespace

from kore.env.kore_env import KoreEnv


def test_gpu_pin_drops_inherited_rocr_visibility_mask(monkeypatch):
    # SPUR restricts a job with ROCR_VISIBLE_DEVICES, while KORE pins each verifier
    # child with HIP_VISIBLE_DEVICES. Leaving both masks set can make their
    # intersection empty (hipErrorNoDevice), so the child must inherit only KORE's
    # single-GPU mask.
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "7")
    task = SimpleNamespace(task_id="test", gpu_target="gfx950")
    env = KoreEnv(task, use_replay=False, gpu="0")

    child_env = env._env()

    assert "ROCR_VISIBLE_DEVICES" not in child_env
    assert child_env["HIP_VISIBLE_DEVICES"] == "0"
    assert child_env["CUDA_VISIBLE_DEVICES"] == "0"


def test_kore_env_uses_trusted_label_and_drops_ambient_secrets(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "do-not-inherit")
    monkeypatch.setenv("HTTPS_PROXY", "http://do-not-inherit")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/do-not-inherit")
    monkeypatch.setenv("LD_PRELOAD", "/tmp/do-not-inherit.so")
    monkeypatch.setenv("PYTHONUSERBASE", "/tmp/do-not-inherit-site")
    task = SimpleNamespace(task_id="test", gpu_target="gfx950")
    env = KoreEnv(task, use_replay=False, gpu="0")

    child_env = env._env()

    assert env.isolation_controller.backend_label == "trusted-code-only"
    for key in (
        "ANTHROPIC_API_KEY",
        "HTTPS_PROXY",
        "SLURM_JOB_ID",
        "SSH_AUTH_SOCK",
        "LD_PRELOAD",
        "PYTHONUSERBASE",
    ):
        assert key not in child_env
