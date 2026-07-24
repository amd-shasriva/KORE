"""CPU-only serving ownership/interface tests."""

from __future__ import annotations

import sys
import types

import pytest

from kore.policy.serve import (
    GenerationClient,
    GenerationProtocol,
    as_generation_client,
    load_generate,
)


def test_generation_client_is_callable_generate_and_releasable():
    calls = []
    closes = []

    def generate(messages, **kwargs):
        calls.append((messages, kwargs))
        return "ok"

    client = GenerationClient(generate, lambda: closes.append("closed"))
    assert isinstance(client, GenerationProtocol)
    assert client("prompt", temperature=0.0) == "ok"
    assert client.generate([{"role": "user", "content": "p"}]) == "ok"
    assert len(calls) == 2
    client.release()
    client.close()  # idempotent
    assert closes == ["closed"]
    assert client.closed is True
    with pytest.raises(RuntimeError, match="closed"):
        client("again")


def test_generation_adapter_accepts_one_generate_protocol():
    class LegacyPolicy:
        def __init__(self):
            self.closed = False

        def generate(self, messages, **kwargs):
            return messages[-1]["content"]

        def close(self):
            self.closed = True

    legacy = LegacyPolicy()
    client = as_generation_client(legacy)
    assert client.generate([{"role": "user", "content": "answer"}]) == "answer"
    client.close()
    assert legacy.closed is True
    with pytest.raises(TypeError):
        as_generation_client(object())


def test_load_generate_returns_owned_client_for_vllm(monkeypatch):
    instances = []

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Completion:
        def __init__(self, text):
            self.text = text

    class Output:
        def __init__(self, text):
            self.outputs = [Completion(text)]

    class LLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.shutdown_called = False
            instances.append(self)

        def generate(self, prompts, _params):
            return [Output(f"completion:{prompt}") for prompt in prompts]

        def chat(self, batches, _params, **_kwargs):
            return [Output(f"chat:{batch[-1]['content']}") for batch in batches]

        def shutdown(self):
            self.shutdown_called = True

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = LLM
    fake_vllm.SamplingParams = SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    client = load_generate("fixture/model", backend="vllm")
    assert isinstance(client, GenerationClient)
    assert callable(client)
    assert client("hello") == "completion:hello"
    assert (
        client.generate([{"role": "user", "content": "hello"}])
        == "chat:hello"
    )
    client.close()
    assert instances[0].shutdown_called is True
