"""Normalise public tool-use corpora into this model's own tool-call surface.

Every public tool-use dataset encodes calls in its own schema, and this model emits
a bespoke nested-XML block. Training on a foreign schema teaches the model to
produce calls the inference stack cannot parse -- a regression that presents
exactly like forgetting while being entirely self-inflicted. So the calls are
re-rendered here rather than passed through.

Three properties of ``Agent-Ark/Toucan-1.5M`` each independently break a naive
reader, which is why an earlier pass over it kept zero rows out of 12,000:

* ``messages`` is a JSON **string**, not a list, so it must be decoded first.
* The role vocabulary differs by config. ``SFT`` uses ``tool_call`` and
  ``tool_response`` as *roles*; ``Kimi-K2``, ``OSS`` and ``Qwen3`` instead use
  ``function`` plus a ``function_call`` key on the assistant message.
* ``tool_call`` bodies are Python ``repr`` output, not JSON -- single-quoted outer
  dict with a JSON-encoded ``arguments`` string inside -- so ``json.loads`` raises
  and ``ast.literal_eval`` followed by a second parse is required.

``NousResearch/hermes-function-calling-v1`` is simpler: ShareGPT ``conversations``
with ``from``/``value`` keys, and calls already tagged but carrying a JSON body,
which is a different model's surface and has to be rewritten.

The target surface, taken from the model's own chat template::

    <tool_call>
    <function=NAME>
    <parameter=ARG>
    value
    </parameter>
    </function>
    </tool_call>

Scalars go in verbatim; dicts and lists are JSON-encoded. Tool *results* render as
``<tool_response>`` inside a **user** turn, because that is where the template puts
them, and consecutive results merge into one turn.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Optional

TOOL_PREAMBLE = (
    "You have access to the following functions. To call one, emit a "
    "<tool_call> block.\n\n<tools>\n%s\n</tools>"
)

_HERMES_ROLE = {"system": "system", "human": "user", "user": "user",
                "gpt": "assistant", "assistant": "assistant",
                "tool": "tool", "observation": "tool", "function_response": "tool"}

_TAG_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_TAG_RESP = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.S)


def _coerce(obj: Any) -> Any:
    """Parse a cell that may be JSON, a Python literal repr, or already decoded."""
    if obj is None or isinstance(obj, (list, dict)):
        return obj
    if not isinstance(obj, str):
        return obj
    s = obj.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001 - fall through to the literal parser
        pass
    try:
        return ast.literal_eval(s)
    except Exception:  # noqa: BLE001 - not structured; caller decides
        return None


def _args_to_dict(args: Any) -> dict:
    d = _coerce(args)
    if isinstance(d, dict):
        return d
    if isinstance(d, str):
        d2 = _coerce(d)
        return d2 if isinstance(d2, dict) else {}
    return {}


def _render_value(v: Any) -> str:
    """A parameter value as the template wants it, for any Python object.

    Scalars go in verbatim and containers are JSON-encoded, but the encoder must
    not be trusted blindly: ``ast.literal_eval`` is what parses these payloads and
    it happily produces objects ``json`` refuses, so a literal ``...`` in an
    argument becomes ``Ellipsis`` and ``json.dumps`` raises. That exception used to
    escape normalisation and abort the whole source, which is why one corpus
    contributed zero rows rather than merely losing the offending ones.
    """
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (dict, list, tuple)):
        try:
            return json.dumps(v, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(v)
    if isinstance(v, (int, float, str)):
        return str(v)
    return str(v)


def render_tool_call(name: str, arguments: Any) -> str:
    parts = ["<tool_call>", f"<function={name}>"]
    for k, v in _args_to_dict(arguments).items():
        parts.append(f"<parameter={k}>\n{_render_value(v)}\n</parameter>")
    parts += ["</function>", "</tool_call>"]
    return "\n".join(parts)


def render_tool_response(content: Any) -> str:
    if isinstance(content, (dict, list)):
        content = _render_value(content)
    return f"<tool_response>\n{str(content).strip()}\n</tool_response>"


def _tools_block(tools: Any) -> str:
    t = _coerce(tools)
    if not t:
        return ""
    try:
        return json.dumps(t, ensure_ascii=False, indent=None)[:6000]
    except Exception:  # noqa: BLE001
        return ""


def _flush(out: list, role: str, buf: list) -> list:
    if buf:
        out.append({"role": role, "content": "\n".join(buf).strip()})
    return []


def _finalize(msgs: list, min_turns: int = 2) -> Optional[list]:
    """Merge same-role runs, drop empties, require a real assistant answer."""
    merged: list = []
    for m in msgs:
        if not m["content"]:
            continue
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n" + m["content"]
        else:
            merged.append(dict(m))
    while merged and merged[-1]["role"] != "assistant":
        merged.pop()
    if len(merged) < min_turns or not any(m["role"] == "assistant" for m in merged):
        return None
    return merged


def normalize_toucan(row: dict) -> Optional[list]:
    """Toucan row -> messages, covering the SFT, Kimi-K2, OSS and Qwen3 dialects."""
    msgs = _coerce(row.get("messages"))
    if not isinstance(msgs, list) or not msgs:
        return None

    out: list = []
    sys_parts: list = []
    pending_assistant: list = []
    pending_tool: list = []
    block = _tools_block(row.get("tools") or row.get("available_tools"))
    if block:
        sys_parts.append(TOOL_PREAMBLE % block)

    for m in msgs:
        if not isinstance(m, dict):
            continue
        role, content = m.get("role"), m.get("content")
        if isinstance(content, list):
            content = "\n".join(p.get("text", "") for p in content
                               if isinstance(p, dict))
        content = "" if content is None else str(content)

        if role == "system":
            pending_tool = _flush(out, "user", pending_tool)
            pending_assistant = _flush(out, "assistant", pending_assistant)
            if content.strip():
                sys_parts.append(content.strip())
        elif role == "user":
            pending_tool = _flush(out, "user", pending_tool)
            pending_assistant = _flush(out, "assistant", pending_assistant)
            out.append({"role": "user", "content": content.strip()})
        elif role == "assistant":
            pending_tool = _flush(out, "user", pending_tool)
            if content.strip():          # OSS `reasoning_content` is intentionally dropped
                pending_assistant.append(content.strip())
            fc = m.get("function_call") or m.get("tool_calls")
            if fc:
                for call in (fc if isinstance(fc, list) else [fc]):
                    call = _coerce(call) if isinstance(call, str) else call
                    if not isinstance(call, dict):
                        continue
                    call = call.get("function", call)
                    if call.get("name"):
                        pending_assistant.append(
                            render_tool_call(call["name"], call.get("arguments")))
        elif role == "tool_call":
            pending_tool = _flush(out, "user", pending_tool)
            call = _coerce(content)
            if isinstance(call, dict) and call.get("name"):
                pending_assistant.append(
                    render_tool_call(call["name"], call.get("arguments")))
        elif role in ("tool_response", "tool", "function"):
            pending_assistant = _flush(out, "assistant", pending_assistant)
            pending_tool.append(render_tool_response(content))

    _flush(out, "user", pending_tool)
    _flush(out, "assistant", pending_assistant)
    if sys_parts:
        out.insert(0, {"role": "system", "content": "\n\n".join(sys_parts)})
    return _finalize(out)


def _hermes_assistant(text: str) -> str:
    """Rewrite Hermes JSON-body tool calls into this model's XML surface."""
    calls = _TAG_CALL.findall(text)
    if not calls:
        return text.strip()
    prose = _TAG_CALL.sub("", text).strip()
    parts = [prose] if prose else []
    for raw in calls:
        obj = _coerce(raw)
        parts.append(render_tool_call(obj["name"], obj.get("arguments"))
                     if isinstance(obj, dict) and obj.get("name")
                     else f"<tool_call>\n{raw.strip()}\n</tool_call>")
    return "\n".join(parts).strip()


def normalize_hermes(row: dict) -> Optional[list]:
    """Hermes ShareGPT row -> messages with calls re-rendered."""
    conv = _coerce(row.get("conversations") or row.get("messages"))
    if not isinstance(conv, list) or not conv:
        return None
    out: list = []
    pending_tool: list = []
    for m in conv:
        if not isinstance(m, dict):
            continue
        role = _HERMES_ROLE.get(m.get("from") or m.get("role"))
        text = m.get("value") if m.get("value") is not None else m.get("content")
        if not role or text is None:
            continue
        text = str(text)
        if role == "tool":
            bodies = _TAG_RESP.findall(text)
            pending_tool.append("\n".join(render_tool_response(b) for b in bodies)
                                if bodies else render_tool_response(text))
            continue
        pending_tool = _flush(out, "user", pending_tool)
        out.append({"role": "assistant", "content": _hermes_assistant(text)}
                   if role == "assistant" else {"role": role, "content": text.strip()})
    _flush(out, "user", pending_tool)
    return _finalize(out)


#: hf id -> normaliser, for sources whose schema needs one.
NORMALIZERS = {
    "Agent-Ark/Toucan-1.5M": normalize_toucan,
    "NousResearch/hermes-function-calling-v1": normalize_hermes,
}

__all__ = ["NORMALIZERS", "normalize_hermes", "normalize_toucan",
           "render_tool_call", "render_tool_response"]
