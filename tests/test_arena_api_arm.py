"""An API model must run through the same harness as a checkpoint.

The point of scoring a frontier model on this arena is comparison, and a
comparison is only worth something if everything either side of the model is
identical: the same task list, the same prompt, the same three attempts with
verifier feedback between them, and the same scorer. model_policy already
exposes a ``generate`` seam for exactly this, so the API arm injects one and
changes nothing else.

Getting the seam wrong is silent. model_policy calls
``gen(messages, max_tokens=..., temperature=...)``; a generate that rejects
those keywords raises inside the first attempt of the first task and the whole
sweep records generation failures that look like the model being bad.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SRC = (REPO / "scripts" / "run_agent_kernel_arena.py").read_text()

#: mirrors _API_MODEL_PREFIXES; kept here so the test fails if they diverge
PREFIXES = ("claude-", "anthropic/", "gpt-", "opus", "sonnet")


def _is_api(model: str) -> bool:
    m = (model or "").lower()
    return any(m.startswith(p) for p in PREFIXES)


@pytest.mark.parametrize("model,expected", [
    ("claude-opus-4.8", True),
    ("claude-opus-5", True),
    ("claude-sonnet-4.5", True),
    ("Qwen/Qwen3-Coder-30B-A3B-Instruct", False),
    ("/shared_nfs/shasriva/kore/runs/sft_v4", False),
    ("/home/shasriva/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-30B", False),
])
def test_api_models_are_told_apart_from_checkpoints(model, expected):
    """A local checkpoint must never be routed to the gateway, and an API model
    must never be handed to the checkpoint loader -- either way the run dies
    before it scores a task."""
    assert _is_api(model) is expected


def test_prefix_list_matches_the_script():
    for p in PREFIXES:
        assert f'"{p}"' in SRC, f"prefix {p} missing from the script"


def test_api_arm_injects_generate_rather_than_serving():
    assert "_api_generate(args) if _is_api_model(args.model)" in SRC, \
        "the API arm does not reach model_policy's generate seam"


def test_generate_accepts_the_keywords_model_policy_passes():
    """model_policy calls gen(messages, max_tokens=..., temperature=...)."""
    block = SRC.split("def _api_generate")[1].split("\ndef ")[0]
    assert "def gen(messages, max_tokens=None, temperature=None, **_)" in block, \
        "injected generate would raise TypeError on model_policy's call"


def test_teacher_is_built_once_not_per_call():
    """Per-call construction re-reads .env.local and rebuilds the HTTP client
    for every attempt of every task."""
    block = SRC.split("def _api_generate")[1].split("\ndef ")[0]
    before, after = block.split("def gen(")
    assert "ClaudeTeacher(" in before, "teacher is constructed inside gen()"
    assert "ClaudeTeacher(" not in after


def test_the_rest_of_the_run_is_unchanged():
    """Same attempts, same feedback, same scorer -- only the model differs."""
    assert "_attempt_task(task, ws, dst_rel, prompt, policy, args," in SRC
    assert "reply = policy(prompt) if feedback is None else policy(prompt, feedback)" in SRC


def test_the_arm_runs_the_model_it_is_labelled_with():
    """ClaudeTeacher resolves os.environ.get("KORE_TEACHER_MODEL", model), so
    .env.local silently wins over the constructor argument. Right for datagen,
    wrong for a benchmark arm: a ledger row naming claude-opus-4.8 while the
    gateway served claude-opus-5 is worse than no row at all."""
    block = SRC.split("def _api_generate")[1].split("\ndef ")[0]
    assert 'os.environ["KORE_TEACHER_MODEL"] = args.model' in block, \
        "the env default can override the requested model"
    assert "refusing to run" in block, "no check that the model actually resolved"
