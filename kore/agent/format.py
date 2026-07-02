"""Tool-call parsing + chat rendering for the agentic loop — PURE, dep-free.

The policy speaks Qwen3-native Hermes tool-calling:

    <tool_call>
    {"name": "test", "arguments": {"kernel_src": "..."}}
    </tool_call>

and tool results are fed back as ``role: "tool"`` messages. This module owns:

  - ``parse_tool_calls``  : robust Hermes + fenced-json + OpenAI ``tool_calls``
                            extraction into ``[{"name","arguments","malformed",...}]``.
  - ``render_tool_result``: wrap an executor result dict as a ``role: "tool"`` msg.
  - ``episode_to_chat``   : normalize an episode's messages into a training-ready
                            multi-turn transcript (thinking / no-think variants).
  - ``build_agent_system_prompt``: Hermes system prompt advertising the tools.

Nothing here touches the GPU, torch, or a model.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from kore.agent.tools import TOOL_SCHEMAS, TOOL_NAMES

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FENCED_JSON_RE = re.compile(
    r"```(?:json|tool_call|tool)?\s*\n?(\{.*?\})\s*```", re.DOTALL
)
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _coerce_arguments(raw: Any) -> Optional[dict]:
    """Arguments may be a dict already or a JSON-encoded string (OpenAI style)."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            val = json.loads(raw)
            return val if isinstance(val, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def _norm_call(obj: Any, raw: str) -> dict:
    """Normalize one decoded object into ``{name, arguments, malformed, raw}``.

    Supports both the Hermes ``{"name","arguments"}`` shape and the OpenAI
    ``{"function": {"name","arguments"}}`` shape.
    """
    if isinstance(obj, dict) and "function" in obj and isinstance(obj["function"], dict):
        fn = obj["function"]
        name = fn.get("name")
        args = _coerce_arguments(fn.get("arguments"))
    elif isinstance(obj, dict):
        name = obj.get("name") or obj.get("tool") or obj.get("tool_name")
        args = _coerce_arguments(obj.get("arguments", obj.get("parameters")))
    else:
        name, args = None, None

    malformed = name is None or args is None
    return {
        "name": name,
        "arguments": args if isinstance(args, dict) else {},
        "malformed": bool(malformed),
        "raw": raw,
    }


def parse_tool_calls(text: str) -> list[dict]:
    """Extract tool calls from assistant ``text``. Robust and never raises.

    Recognizes, in priority order:
      1. Hermes ``<tool_call>{json}</tool_call>`` blocks (one or many).
      2. Fenced ```json { ... }``` blocks that look like a tool call.
      3. A single top-level JSON object with a ``name`` / ``tool_calls`` field.

    Each returned dict is ``{"name", "arguments", "malformed", "raw"}``. A block
    whose JSON is broken is returned with ``malformed=True`` and ``name=None`` so
    reward shaping can see the format error.
    """
    if not text:
        return []
    calls: list[dict] = []

    # 1. Hermes <tool_call> blocks
    for m in _TOOL_CALL_RE.finditer(text):
        body = m.group(1).strip()
        obj = _try_json(body)
        if obj is None:
            calls.append({"name": None, "arguments": {}, "malformed": True, "raw": body})
        else:
            calls.append(_norm_call(obj, body))
    if calls:
        return calls

    # 2. Fenced json blocks that carry a tool-call-ish object
    for m in _FENCED_JSON_RE.finditer(text):
        body = m.group(1).strip()
        obj = _try_json(body)
        if obj is None:
            continue
        if _looks_like_call(obj):
            calls.append(_norm_call(obj, body))
    if calls:
        return calls

    # 3. A bare top-level object (possibly OpenAI {"tool_calls": [...]})
    obj = _try_json(text.strip())
    if isinstance(obj, dict):
        if isinstance(obj.get("tool_calls"), list):
            return [_norm_call(c, json.dumps(c)) for c in obj["tool_calls"]]
        if _looks_like_call(obj):
            return [_norm_call(obj, text.strip())]
    return calls


def _looks_like_call(obj: Any) -> bool:
    return isinstance(obj, dict) and (
        "name" in obj or "tool" in obj or "tool_name" in obj or "function" in obj
    )


def _try_json(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_tool_result(name: str, result: Any) -> dict:
    """Wrap an executor result as a Hermes/OpenAI ``role: "tool"`` message."""
    content = result if isinstance(result, str) else json.dumps(result)
    return {"role": "tool", "name": name, "content": content}


def render_tool_call_message(calls: list[dict], thinking: str = "") -> dict:
    """Render an assistant turn that emits one or more Hermes ``<tool_call>``s.

    ``thinking`` (optional) is wrapped in ``<think>...</think>`` so the no-think
    variant can strip it cleanly in ``episode_to_chat``.
    """
    parts: list[str] = []
    if thinking:
        parts.append(f"<think>\n{thinking.strip()}\n</think>")
    for c in calls:
        payload = {"name": c.get("name"), "arguments": c.get("arguments", {})}
        parts.append(f"<tool_call>\n{json.dumps(payload)}\n</tool_call>")
    return {"role": "assistant", "content": "\n".join(parts)}


def strip_thinking(content: str) -> str:
    """Remove ``<think>...</think>`` spans from an assistant message."""
    return _THINK_RE.sub("", content or "").strip()


def episode_to_chat(episode: Any, thinking: bool = True) -> list[dict]:
    """Return a training-ready multi-turn transcript for ``episode``.

    The transcript is system + user + alternating assistant(tool_call) + tool
    messages, exactly as the harness recorded them. When ``thinking`` is False,
    ``<think>...</think>`` spans are stripped from assistant messages (the
    no-think SFT variant); when True they are preserved.
    """
    messages = _episode_messages(episode)
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "assistant" and not thinking:
            content = strip_thinking(content)
        new = {"role": role, "content": content}
        if role == "tool" and m.get("name"):
            new["name"] = m["name"]
        out.append(new)
    return out


def _episode_messages(episode: Any) -> list[dict]:
    if isinstance(episode, dict):
        return list(episode.get("messages", []) or [])
    return list(getattr(episode, "messages", []) or [])


# --------------------------------------------------------------------------- #
# System prompt (Hermes tool advertisement)
# --------------------------------------------------------------------------- #
_HERMES_HEADER = """\
You are KORE, an expert AMD MI325X (gfx942 / CDNA3) kernel engineer operating as \
an autonomous agent. You optimize a Triton kernel by ORCHESTRATING your own loop: \
plan a change, then CALL a tool to build / test / bench / pmc it, read the result, \
and decide to keep or revert. Correctness first, then speed vs the production \
baseline. The trajectory is scored by the BEST correct kernel you reach.

You may call these functions. To call one, emit a Hermes tool call:
<tool_call>
{"name": <tool-name>, "arguments": <args-json>}
</tool_call>
Tool results are returned to you as messages with role "tool". Make ONE focused \
change per turn, verify it, then keep (commit) or revert (discard) it."""


def build_agent_system_prompt(tools: Optional[list[dict]] = None) -> str:
    """Hermes-style system prompt advertising the tool schemas as JSON."""
    tools = tools or TOOL_SCHEMAS
    lines = [_HERMES_HEADER, "", "<tools>"]
    for t in tools:
        lines.append(json.dumps(t))
    lines.append("</tools>")
    lines.append("")
    lines.append(f"Available tools: {', '.join(t['function']['name'] for t in tools)}.")
    return "\n".join(lines)
