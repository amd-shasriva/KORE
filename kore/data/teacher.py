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
import time
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from kore.obs import get_logger

log = get_logger("data.teacher")


# Bounded exponential-backoff retry defaults for the network-backed teachers.
# Sized to ride out multi-minute gateway hiccups / rate-limit windows: a multi-day
# datagen makes tens of thousands of teacher calls and WILL hit transient 5xx /
# timeouts / rate limits - the run must not die on one. 8 attempts with the backoff
# below span ~4-5 minutes of resilience per call before giving up.
_MAX_RETRIES = 8          # total attempts = _MAX_RETRIES
_BACKOFF_BASE = 1.0       # seconds; delay = _BACKOFF_BASE * 2**attempt
_BACKOFF_CAP = 60.0       # seconds; never wait longer than this
_REQUEST_TIMEOUT = 180.0  # per-request wall timeout (seconds)


def _retry_call(fn: Callable[[], Any], *, what: str, stats: Optional[dict] = None):
    """Call ``fn`` with bounded exponential backoff on transient failures.

    Retries up to ``_MAX_RETRIES`` times, sleeping ``_BACKOFF_BASE * 2**attempt``
    (capped at ``_BACKOFF_CAP``) between attempts. Re-raises the last exception
    if every attempt fails so the caller can decide to skip the sample.

    Deterministic client errors (most HTTP 4xx responses) fail immediately.
    Retrying an invalid model, header, or request can never succeed and otherwise
    wastes roughly two minutes per sample. 408, 409, and 429 remain retryable.

    ``stats`` (optional) is populated with ``attempt`` (1-based count of the call
    that succeeded / the total made) and ``retries`` (number of failed attempts)
    for observability; it never affects control flow.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(_MAX_RETRIES):
        try:
            result = fn()
            if stats is not None:
                stats["attempt"] = attempt + 1
                stats["retries"] = attempt
            return result
        except Exception as e:  # noqa: BLE001 - transient network/server errors
            last_exc = e
            status = getattr(e, "status_code", None)
            non_retryable = (
                isinstance(status, int)
                and 400 <= status < 500
                and status not in (408, 409, 429)
            )
            if non_retryable:
                if stats is not None:
                    stats["attempt"] = attempt + 1
                    stats["retries"] = attempt
                log.error(
                    f"{what} non-retryable client failure",
                    what=what, attempt=attempt + 1, status_code=status,
                    exc_type=type(e).__name__, exc=str(e)[:200],
                )
                raise RuntimeError(
                    f"{what} failed with non-retryable HTTP {status}"
                ) from e
            if attempt == _MAX_RETRIES - 1:
                break
            delay = min(_BACKOFF_BASE * (2 ** attempt), _BACKOFF_CAP)
            log.warn(
                f"{what} transient failure; backing off",
                what=what, attempt=attempt + 1, max_retries=_MAX_RETRIES,
                next_delay_s=round(delay, 3),
                exc_type=type(e).__name__, exc=str(e)[:200],
            )
            time.sleep(delay)
    if stats is not None:
        stats["attempt"] = _MAX_RETRIES
        stats["retries"] = _MAX_RETRIES - 1
    log.error(
        f"{what} failed after {_MAX_RETRIES} attempts",
        what=what, attempts=_MAX_RETRIES,
        exc_type=type(last_exc).__name__ if last_exc is not None else None,
        exc=str(last_exc)[:200] if last_exc is not None else None,
    )
    raise RuntimeError(f"{what} failed after {_MAX_RETRIES} attempts") from last_exc


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
        t0 = time.time()
        out = self._fn(messages) if self._fn else _CANNED
        log.debug(
            "stub teacher_call",
            backend="stub", model="stub", n_messages=len(messages),
            prompt_chars=sum(len(str(m.get("content", ""))) for m in messages),
            latency_ms=round((time.time() - t0) * 1000.0, 3),
            resp_chars=len(out) if isinstance(out, str) else 0,
            call_idx=len(self.calls),
        )
        return out


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
        msgs = _ensure_system(messages, self.system)
        prompt_chars = sum(len(str(m.get("content", ""))) for m in msgs)

        def _call():
            return self.client.chat.completions.create(
                model=self.model, messages=msgs,
                temperature=self.temperature, max_tokens=self.max_tokens,
                timeout=_REQUEST_TIMEOUT,
            )

        stats: dict = {}
        t0 = time.time()
        r = _retry_call(_call, what="VLLMTeacher.generate", stats=stats)
        latency_ms = round((time.time() - t0) * 1000.0, 1)

        finish_reason = None
        truncated = False
        if not r.choices:
            result = ""
        else:
            choice = r.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            # Truncation guard: a cut-off completion is an incomplete kernel; drop
            # it rather than let a half-written kernel enter the corpus.
            if finish_reason == "length":
                truncated = True
                result = ""
            else:
                result = choice.message.content or ""

        if truncated:
            log.warn(
                "teacher completion truncated (finish_reason=length); dropping",
                backend="vllm", model=self.model, finish_reason=finish_reason,
                max_tokens=self.max_tokens,
            )
        log.event(
            "teacher_call", backend="vllm", model=self.model,
            n_messages=len(msgs), prompt_chars=prompt_chars,
            latency_ms=latency_ms, resp_chars=len(result),
            finish_reason=finish_reason, truncated=truncated,
            attempt=stats.get("attempt"), retries=stats.get("retries"),
        )
        return result


class HFTeacher:
    def __init__(self, model_id: str, temperature: float = 0.7, max_new_tokens: int = 4096,
                 system: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto")
        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.system = system

    def generate(self, messages: list[dict]) -> str:
        msgs = _ensure_system(messages, self.system)
        prompt_chars = sum(len(str(m.get("content", ""))) for m in msgs)
        inputs = self.tok.apply_chat_template(msgs, add_generation_prompt=True,
                                              return_tensors="pt").to(self.model.device)
        gen_tokens = int(inputs.shape[1])
        t0 = time.time()
        out = self.model.generate(inputs, max_new_tokens=self.max_new_tokens,
                                  do_sample=self.temperature > 0, temperature=self.temperature)
        latency_ms = round((time.time() - t0) * 1000.0, 1)
        result = self.tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        new_tokens = int(out.shape[-1]) - gen_tokens
        truncated = new_tokens >= self.max_new_tokens
        if truncated:
            log.warn(
                "teacher generation hit max_new_tokens (possible truncation)",
                backend="hf", model=self.model_id, max_new_tokens=self.max_new_tokens,
            )
        log.event(
            "teacher_call", backend="hf", model=self.model_id,
            n_messages=len(msgs), prompt_chars=prompt_chars,
            latency_ms=latency_ms, resp_chars=len(result),
            finish_reason="length" if truncated else "stop", truncated=truncated,
            attempt=1, retries=0, new_tokens=new_tokens,
        )
        return result


class ClaudeTeacher:
    """Anthropic frontier model via AMD's internal LLM gateway.

    Requires (from env / .env.local):
        AMD_LLM_GATEWAY_URL, AMD_LLM_API_KEY, and optionally AMD_NTID.
    """

    GATEWAY_URL = "https://llm-api.amd.com/Anthropic"

    def __init__(self, model: str = "claude-opus-4.8", temperature: float = 0.7,
                 max_tokens: int = 8192, system: Optional[str] = None):
        import anthropic

        load_env_local()
        api_key = (os.environ.get("AMD_LLM_API_KEY")
                   or os.environ.get("LLM_GATEWAY_KEY"))
        if not api_key:
            raise RuntimeError("AMD_LLM_API_KEY not set (put it in .env.local)")
        base_url = os.environ.get("AMD_LLM_GATEWAY_URL", self.GATEWAY_URL)
        user = self._resolve_user()
        # 2023-06-01 is the Anthropic native API version accepted by AMD's
        # AzureOpenAI-backed gateway. The former 2023-10-16 default is invalid.
        version = os.environ.get("AMD_LLM_API_VERSION", "2023-06-01")
        # AMD gateway auth is via the Ocp-Apim header; the SDK api_key is a dummy.
        self.client = anthropic.Anthropic(
            api_key="dummy", base_url=base_url,
            default_headers={
                "Ocp-Apim-Subscription-Key": api_key,
                "user": user,
                "anthropic-version": version,
            },
        )
        self.model = os.environ.get("KORE_TEACHER_MODEL", model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system = system

    @staticmethod
    def _resolve_user() -> str:
        for var in ("AMD_NTID", "GEAK_USER", "USER"):
            v = (os.environ.get(var) or "").strip()
            if v and v != "root":
                return v if "@" in v else f"{v}@amd.com"
        return "unknown"

    def generate(self, messages: list[dict]) -> str:
        system = self.system
        conv: list[dict] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                system = content
                continue
            if role == "tool":
                # Anthropic accepts only user/assistant; render Hermes tool
                # results as a user turn (our protocol is text-based, not native
                # tool-use blocks).
                name = m.get("name") or m.get("tool_call_id") or "tool"
                role = "user"
                content = f"[tool:{name}] {content}"
            elif role not in ("user", "assistant"):
                role = "user"
            conv.append({"role": role, "content": content})
        # Merge consecutive same-role turns (Anthropic requires alternation).
        merged: list[dict] = []
        for m in conv:
            if merged and merged[-1]["role"] == m["role"]:
                merged[-1]["content"] = f"{merged[-1]['content']}\n\n{m['content']}"
            else:
                merged.append(dict(m))
        # The conversation must start with a user turn.
        if merged and merged[0]["role"] == "assistant":
            merged.insert(0, {"role": "user", "content": "Continue."})
        kw = dict(model=self.model, max_tokens=self.max_tokens,
                  temperature=self.temperature, messages=merged,
                  timeout=_REQUEST_TIMEOUT)
        if system:
            kw["system"] = system

        prompt_chars = sum(len(str(m.get("content", ""))) for m in merged)
        if system:
            prompt_chars += len(str(system))
        stats: dict = {}
        t0 = time.time()
        r = _retry_call(lambda: self.client.messages.create(**kw),
                        what="ClaudeTeacher.generate", stats=stats)
        latency_ms = round((time.time() - t0) * 1000.0, 1)

        stop_reason = getattr(r, "stop_reason", None)
        # Truncation guard: if Anthropic stopped because it hit max_tokens, the
        # kernel is cut off mid-source. Treat it as truncated and skip it so a
        # half-written kernel never enters the corpus.
        truncated = stop_reason == "max_tokens"
        if truncated:
            result = ""
        else:
            content = getattr(r, "content", None)
            if not content:
                result = ""
            else:
                result = "".join(b.text for b in content if getattr(b, "type", "") == "text")

        if truncated:
            log.warn(
                "teacher completion truncated (stop_reason=max_tokens); dropping",
                backend="claude", model=self.model, stop_reason=stop_reason,
                max_tokens=self.max_tokens,
            )
        log.event(
            "teacher_call", backend="claude", model=self.model,
            n_messages=len(merged), prompt_chars=prompt_chars,
            latency_ms=latency_ms, resp_chars=len(result),
            finish_reason=stop_reason, truncated=truncated,
            attempt=stats.get("attempt"), retries=stats.get("retries"),
        )
        return result


class ResilientTeacher:
    """Wrap a teacher so a TRANSIENT total-failure (all retries exhausted on one
    call) SKIPS that sample (returns "") instead of crashing the whole run - but a
    SUSTAINED outage (``max_consecutive_failures`` in a row) still raises, so we
    never silently produce empty datagen (no silent degradation).

    A multi-day datagen makes tens of thousands of teacher calls; occasional 5xx /
    rate-limit / timeout errors are inevitable and must not kill the campaign. One
    failed generation among thousands is noise (skip it); an hour of consecutive
    failures is a real outage (stop, resumably). Delegates everything else to the
    wrapped teacher.
    """

    def __init__(self, inner, max_consecutive_failures: int = 15):
        self._inner = inner
        self._max_consec = int(max_consecutive_failures)
        self._consec = 0
        self._total_skipped = 0

    def __getattr__(self, name):  # delegate non-overridden attrs to the inner teacher
        return getattr(self._inner, name)

    def generate(self, messages, **kwargs) -> str:
        try:
            out = self._inner.generate(messages, **kwargs)
            self._consec = 0  # reset the outage counter on any success
            return out
        except Exception as e:  # noqa: BLE001 - retries already exhausted inside inner
            self._consec += 1
            self._total_skipped += 1
            log.error("teacher.generate exhausted retries - SKIPPING this sample "
                      "(resilient); will hard-stop if the outage is sustained",
                      exc_type=type(e).__name__, exc=str(e)[:200],
                      consecutive_failures=self._consec,
                      max_consecutive=self._max_consec, total_skipped=self._total_skipped)
            if self._consec >= self._max_consec:
                raise RuntimeError(
                    f"teacher unavailable: {self._consec} consecutive generation "
                    f"failures (sustained outage) - stopping so the run can resume "
                    f"later rather than produce empty data") from e
            return ""


def make_teacher(kind: str = "stub", *, resilient: bool = False, **kwargs) -> TeacherClient:
    kind = (kind or "stub").lower()
    if kind == "stub":
        base = StubTeacher(**kwargs)
    elif kind == "vllm":
        base = VLLMTeacher(**kwargs)
    elif kind == "hf":
        base = HFTeacher(**kwargs)
    elif kind in ("claude", "anthropic", "opus"):
        base = ClaudeTeacher(**kwargs)
    else:
        raise ValueError(f"unknown teacher kind: {kind}")
    return ResilientTeacher(base) if resilient else base
