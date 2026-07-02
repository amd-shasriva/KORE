"""Teacher clients for cold-start data generation.

A teacher maps a chat ``messages`` list to a completion string. Four backends:
    StubTeacher   - deterministic, dependency-free (tests / dry-runs)
    VLLMTeacher   - an OpenAI-compatible vLLM/SGLang endpoint (self-hosted)
    HFTeacher     - a local transformers model
    ClaudeTeacher - Anthropic frontier model via AMD's internal LLM gateway

All conform to the :class:`TeacherClient` protocol so the datagen pipeline is
backend-agnostic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Protocol, runtime_checkable


@runtime_checkable
class TeacherClient(Protocol):
    def generate(self, messages: list[dict]) -> str: ...


def load_env_local(path: Optional[Path] = None) -> None:
    """Load KEY=VALUE lines from .env.local into os.environ (no overwrite)."""
    path = path or (Path(__file__).resolve().parents[2] / ".env.local")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_CANNED = (
    "ANALYSIS:\nSeed is memory-bound; increase vectorized width and occupancy.\n\n"
    "PROPOSED_CHANGE:\nTune BLOCK sizes and num_warps.\n\n"
    "FULL_KERNEL:\n```python\nimport triton\nimport triton.language as tl\n\n"
    "@triton.jit\ndef _k():\n    pass\n```\n"
)


def _ensure_system(messages: list[dict], system: Optional[str]) -> list[dict]:
    if system and not (messages and messages[0].get("role") == "system"):
        return [{"role": "system", "content": system}] + messages
    return messages


class StubTeacher:
    """Deterministic teacher. Either returns canned text or a user fn(messages)."""

    def __init__(self, fn: Optional[Callable[[list[dict]], str]] = None):
        self._fn = fn
        self.calls: list[list[dict]] = []

    def generate(self, messages: list[dict]) -> str:
        self.calls.append(list(messages))
        return self._fn(messages) if self._fn else _CANNED


class VLLMTeacher:
    def __init__(self, model: str, base_url: str = "http://localhost:8000/v1",
                 api_key: str = "EMPTY", temperature: float = 0.7,
                 max_tokens: int = 8192, system: Optional[str] = None):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system = system

    def generate(self, messages: list[dict]) -> str:
        r = self.client.chat.completions.create(
            model=self.model, messages=_ensure_system(messages, self.system),
            temperature=self.temperature, max_tokens=self.max_tokens,
        )
        return r.choices[0].message.content or ""


class HFTeacher:
    def __init__(self, model_id: str, temperature: float = 0.7, max_new_tokens: int = 4096,
                 system: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto")
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.system = system

    def generate(self, messages: list[dict]) -> str:
        msgs = _ensure_system(messages, self.system)
        inputs = self.tok.apply_chat_template(msgs, add_generation_prompt=True,
                                              return_tensors="pt").to(self.model.device)
        out = self.model.generate(inputs, max_new_tokens=self.max_new_tokens,
                                  do_sample=self.temperature > 0, temperature=self.temperature)
        return self.tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)


class ClaudeTeacher:
    """Anthropic frontier model via AMD's internal LLM gateway.

    Requires (from env / .env.local):
        AMD_LLM_GATEWAY_URL, AMD_LLM_API_KEY, and optionally AMD_NTID.
    """

    def __init__(self, model: str = "claude-opus-4-8", temperature: float = 0.7,
                 max_tokens: int = 8192, system: Optional[str] = None):
        import anthropic

        load_env_local()
        base_url = os.environ.get("AMD_LLM_GATEWAY_URL")
        api_key = os.environ.get("AMD_LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("AMD_LLM_API_KEY not set (put it in .env.local)")
        headers = {}
        sub = os.environ.get("AMD_LLM_SUBSCRIPTION_KEY", api_key)
        if sub:
            headers["Ocp-Apim-Subscription-Key"] = sub
        if os.environ.get("AMD_NTID"):
            headers["X-User-Id"] = os.environ["AMD_NTID"]
        kw = {"api_key": api_key, "default_headers": headers}
        if base_url:
            kw["base_url"] = base_url
        self.client = anthropic.Anthropic(**kw)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system = system

    def generate(self, messages: list[dict]) -> str:
        system = self.system
        conv = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                conv.append({"role": m["role"], "content": m["content"]})
        kw = dict(model=self.model, max_tokens=self.max_tokens,
                  temperature=self.temperature, messages=conv)
        if system:
            kw["system"] = system
        r = self.client.messages.create(**kw)
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


def make_teacher(kind: str = "stub", **kwargs) -> TeacherClient:
    kind = (kind or "stub").lower()
    if kind == "stub":
        return StubTeacher(**kwargs)
    if kind == "vllm":
        return VLLMTeacher(**kwargs)
    if kind == "hf":
        return HFTeacher(**kwargs)
    if kind in ("claude", "anthropic", "opus"):
        return ClaudeTeacher(**kwargs)
    raise ValueError(f"unknown teacher kind: {kind}")
