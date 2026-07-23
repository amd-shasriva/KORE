from __future__ import annotations

import sys
import types

import pytest

from kore.data.teacher import ClaudeTeacher, _retry_call


def test_claude_teacher_uses_gateway_supported_api_version(monkeypatch):
    captured = {}

    class FakeAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules, "anthropic", types.SimpleNamespace(Anthropic=FakeAnthropic)
    )
    monkeypatch.setenv("AMD_LLM_API_KEY", "test-key")
    monkeypatch.setenv("AMD_NTID", "test-user")
    monkeypatch.delenv("AMD_LLM_API_VERSION", raising=False)

    ClaudeTeacher()

    assert captured["default_headers"]["anthropic-version"] == "2023-06-01"


def test_retry_call_fails_fast_on_non_retryable_client_error():
    calls = 0
    stats = {}

    class BadRequestError(Exception):
        status_code = 400

    def fail():
        nonlocal calls
        calls += 1
        raise BadRequestError("invalid request")

    with pytest.raises(RuntimeError, match="non-retryable HTTP 400"):
        _retry_call(fail, what="test", stats=stats)

    assert calls == 1
    assert stats == {"attempt": 1, "retries": 0}
