"""CPU-only serving ownership/interface tests."""

from __future__ import annotations

import json
import os
import sys
import types

import pytest

from kore.policy.model_spec import FloatingRevisionError
from kore.policy.serve import (
    DeviceVisibilityError,
    GenerationClient,
    GenerationProtocol,
    as_generation_client,
    configure_rocm_env,
    load_generate,
)

REVISION = "a" * 40
BASE_REVISION = "b" * 40


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
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)

    client = load_generate(
        "fixture/model", backend="vllm", revision=REVISION
    )
    assert isinstance(client, GenerationClient)
    assert callable(client)
    assert client("hello") == "completion:hello"
    assert (
        client.generate([{"role": "user", "content": "hello"}])
        == "chat:hello"
    )
    client.close()
    assert instances[0].shutdown_called is True
    assert instances[0].kwargs["revision"] == REVISION


def test_rocm_visibility_rejects_double_or_composed_masks(monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0")
    with pytest.raises(DeviceVisibilityError, match="double masks"):
        configure_rocm_env([0])

    monkeypatch.delenv("ROCR_VISIBLE_DEVICES")
    with pytest.raises(DeviceVisibilityError, match="refusing to compose"):
        configure_rocm_env([1])
    assert os.environ["HIP_VISIBLE_DEVICES"] == "0"


def test_rocm_visibility_normalizes_one_rocr_mask(monkeypatch):
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "2,4")
    result = configure_rocm_env([2, 4])
    assert result["HIP_VISIBLE_DEVICES"] == "2,4"
    assert "ROCR_VISIBLE_DEVICES" not in os.environ


def test_load_generate_requires_and_forwards_adapter_revisions(
    monkeypatch, tmp_path
):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "fixture/base",
                "revision": BASE_REVISION,
            }
        )
    )
    calls = []

    class Tokenizer:
        pad_token = None
        eos_token = "<eos>"

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("tokenizer", model_id, kwargs))
            return Tokenizer()

    class BaseModel:
        def eval(self):
            calls.append(("base-eval",))

    class AutoModel:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("base", model_id, kwargs))
            return BaseModel()

    class MergedModel:
        def eval(self):
            calls.append(("eval",))

    class AdapterLoad:
        def merge_and_unload(self):
            return MergedModel()

    class PeftModel:
        @classmethod
        def from_pretrained(cls, base, model_id, **kwargs):
            calls.append(("adapter", model_id, kwargs))
            return AdapterLoad()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = AutoTokenizer
    fake_transformers.AutoModelForCausalLM = AutoModel
    fake_peft = types.ModuleType("peft")
    fake_peft.PeftModel = PeftModel
    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = "bf16"
    fake_torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    client = load_generate(
        str(adapter),
        backend="hf",
        revision=REVISION,
        base_revision=BASE_REVISION,
    )
    assert calls[0][2]["revision"] == BASE_REVISION
    assert calls[1][2]["revision"] == BASE_REVISION
    assert calls[2][2]["revision"] == REVISION
    client.close()

    full = load_generate(
        "fixture/full", backend="hf", revision=REVISION
    )
    full_loads = [call for call in calls if call[0] in ("tokenizer", "base")]
    assert full_loads[-2][2]["revision"] == REVISION
    assert full_loads[-1][2]["revision"] == REVISION
    full.close()

    with pytest.raises(ValueError, match="conflicts"):
        load_generate(
            str(adapter),
            backend="hf",
            revision=REVISION,
            base_revision="d" * 40,
        )
    with pytest.raises(FloatingRevisionError, match="full 40"):
        load_generate("fixture/model", backend="vllm")
