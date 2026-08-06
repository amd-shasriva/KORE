"""The environment a task's own commands run in.

Arena tasks invoke `python3`, not an interpreter we pick. With the ambient
environment that is /usr/bin/python3, which has no torch, so whole families failed
without any kernel being judged: a repository task died with "No module named
'torch'" inside its own runner, and 19 torch2flydsl tasks failed every case with
"No module named 'aiter'".
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kore.eval.agent_kernel_arena import _task_env  # noqa: E402


def test_the_interpreter_with_torch_comes_first_on_path():
    env = _task_env()
    first = env["PATH"].split(os.pathsep)[0]
    assert first == str(Path(sys.executable).parent)


def test_python3_on_that_path_is_the_one_running_us():
    """A task runs `python3`; it must resolve to the interpreter that has torch,
    not to /usr/bin/python3."""
    env = _task_env()
    first = Path(env["PATH"].split(os.pathsep)[0])
    assert (first / "python3").exists() or (first / "python").exists()


def test_aiter_is_importable_by_tasks_that_use_it_as_a_library():
    """Several tasks import aiter rather than editing it, and the checkout is not
    pip-installed."""
    env = _task_env()
    aiter = Path.home() / "third_party" / "aiter"
    if not aiter.is_dir():
        return
    assert str(aiter) in env.get("PYTHONPATH", "").split(os.pathsep)


def test_the_ambient_environment_is_preserved():
    """Tasks need the rest of the environment -- ROCm paths, HIP_VISIBLE_DEVICES the
    worker set, the scheduler's variables -- so this augments rather than replaces."""
    env = _task_env()
    for key in ("HOME", "PATH"):
        assert key in env
    if "HIP_VISIBLE_DEVICES" in os.environ:
        assert env["HIP_VISIBLE_DEVICES"] == os.environ["HIP_VISIBLE_DEVICES"]


def test_repeated_calls_do_not_stack_path_entries():
    a, b = _task_env(), _task_env()
    assert a["PATH"] == b["PATH"]
    venv = str(Path(sys.executable).parent)
    assert a["PATH"].split(os.pathsep).count(venv) == 1
