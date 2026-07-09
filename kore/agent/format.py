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

from kore.agent.tools import TOOL_SCHEMAS

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FENCED_JSON_RE = re.compile(
    r"```(?:json|tool_call|tool)?\s*\n?(\{.*?\})\s*```", re.DOTALL
)
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_REFLECT_RE = re.compile(r"<reflect>\s*(.*?)\s*</reflect>", re.DOTALL | re.IGNORECASE)

# The three fields of a structured (GEAK/Reflexion) reflection turn.
REFLECT_FIELDS = ("root_cause", "evidence", "planned_fix")


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


# --------------------------------------------------------------------------- #
# Structured reflection (GEAK/Reflexion) — a parsed block, NOT a tool primitive
# --------------------------------------------------------------------------- #
def parse_reflection(text: str) -> Optional[dict]:
    """Extract a structured ``<reflect>{json}</reflect>`` block from ``text``.

    A reflection is the policy's post-failure introspection with three fields:
    ``root_cause`` (why the last attempt failed), ``evidence`` (the concrete
    signal it read — an error line, SNR, counter) and ``planned_fix`` (the next
    concrete change). Returns a dict with all three keys as strings (missing
    ones default to ``""``) or ``None`` when no block is present. Never raises.

    Tolerant of a bare-object form (``{"root_cause": ...}`` with no wrapper) and
    of plain-text ``key: value`` lines inside the block when the JSON is broken.
    """
    if not text:
        return None
    m = _REFLECT_RE.search(text)
    body = m.group(1).strip() if m else None
    if body is None:
        # Bare-object fallback: a top-level JSON object carrying reflect fields.
        obj = _try_json(text.strip())
        if isinstance(obj, dict) and any(k in obj for k in REFLECT_FIELDS):
            return _norm_reflection(obj)
        return None
    obj = _try_json(body)
    if isinstance(obj, dict):
        return _norm_reflection(obj)
    parsed = _parse_reflection_lines(body)
    return parsed if parsed else _norm_reflection({"root_cause": body})


def _norm_reflection(obj: dict) -> dict:
    out = {k: str(obj.get(k, "") or "").strip() for k in REFLECT_FIELDS}
    return out


def _parse_reflection_lines(body: str) -> Optional[dict]:
    """Parse ``root_cause: ...`` style lines when the block isn't valid JSON."""
    found: dict = {}
    for line in body.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().lower().replace(" ", "_")
        if key in REFLECT_FIELDS:
            found[key] = val.strip()
    return _norm_reflection(found) if found else None


def render_reflection(reflection: dict) -> str:
    """Render a reflection dict as a ``<reflect>{json}</reflect>`` block."""
    payload = {k: str(reflection.get(k, "") or "") for k in REFLECT_FIELDS}
    return f"<reflect>\n{json.dumps(payload)}\n</reflect>"


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
# Arch descriptor for the system prompt. Keeps the SFT data honest about the
# ACTUAL target (a gfx950 run must not train the policy to say "gfx942").
_ARCH_DESC: dict[str, str] = {
    "gfx950": "AMD MI355X (gfx950 / CDNA4)",
    "gfx942": "AMD MI325X (gfx942 / CDNA3)",
}
_DEFAULT_ARCH_DESC = "AMD MI325X (gfx942 / CDNA3)"


def arch_desc(arch: Optional[str]) -> str:
    """Human descriptor for an arch slug ('gfx950' -> 'AMD MI355X (gfx950 / CDNA4)').

    Unknown/empty slugs fall back to the historical default so existing callers
    (and their golden strings) are byte-for-byte unchanged when ``arch`` is None.
    """
    if not arch:
        return _DEFAULT_ARCH_DESC
    return _ARCH_DESC.get(str(arch).strip().lower(), str(arch))


def _hermes_header(arch: Optional[str] = None) -> str:
    """The Hermes header, parameterized by target arch (default = legacy gfx942)."""
    desc = arch_desc(arch)
    return (
        f"You are KORE, an expert {desc} kernel engineer operating as "
        "an autonomous agent. You optimize a Triton kernel by ORCHESTRATING your own loop: "
        "plan a change, then CALL a tool to build / test / bench / pmc it, read the result, "
        "and decide to keep or revert. Correctness first, then speed vs the production "
        "baseline. The trajectory is scored by the BEST correct kernel you reach.\n\n"
        "You may call these functions. To call one, emit a Hermes tool call:\n"
        "<tool_call>\n"
        '{"name": <tool-name>, "arguments": <args-json>}\n'
        "</tool_call>\n"
        'Tool results are returned to you as messages with role "tool". Make ONE focused '
        "change per turn, verify it, then keep (commit) or revert (discard) it."
    )


# Backward-compatible module constant (identical to the historical string).
_HERMES_HEADER = _hermes_header()

_REFLECT_GUIDE = """\
After ANY failed build/test/bench, first REFLECT before your next tool call. \
Emit a structured reflection block:
<reflect>
{"root_cause": <why it failed>, "evidence": <the exact error/SNR/counter you \
read>, "planned_fix": <the concrete change you will make next>}
</reflect>
A good reflection names the ACTUAL error you observed (not a generic guess) and \
a specific fix. Do not keep patching a dead kernel — if two or three attempts \
do not improve, abandon that lineage and re-seed a fresh design from the task."""

# Phase-specific guidance (correctness-first, then optimize-for-speed).
_PHASE_CORRECTNESS = """\
CURRENT PHASE — PHASE 1 (CORRECTNESS). Your ONLY goal right now is a NUMERICALLY \
CORRECT kernel that passes the SNR gate on every validation shape. Ignore speed \
for now; use build/test to reach correctness, then keep it."""
_PHASE_OPTIMIZE = """\
CURRENT PHASE — PHASE 2 (OPTIMIZE). You already have a correct kernel. Now \
MAXIMIZE the worst-shape speedup vs the production baseline while STAYING \
correct. Use bench/pmc to find bottlenecks; keep only correct improvements and \
revert any regression."""

_PHASE_PROMPTS = {
    "correctness": _PHASE_CORRECTNESS,
    "phase1": _PHASE_CORRECTNESS,
    "optimize": _PHASE_OPTIMIZE,
    "optimization": _PHASE_OPTIMIZE,
    "phase2": _PHASE_OPTIMIZE,
}


def build_agent_system_prompt(
    tools: Optional[list[dict]] = None, phase: Optional[str] = None,
    arch: Optional[str] = None,
) -> str:
    """Hermes-style system prompt advertising the tool schemas as JSON.

    ``phase`` selects the correctness->optimization sub-phase guidance
    ("correctness" | "optimize"); when ``None`` a neutral both-phases prompt is
    produced. The reflection protocol is always advertised so the policy knows
    how to introspect after a failure. ``arch`` (e.g. "gfx950") sets the target
    descriptor in the header; ``None`` preserves the legacy gfx942 wording.
    """
    tools = tools or TOOL_SCHEMAS
    lines = [_hermes_header(arch), "", _REFLECT_GUIDE]
    phase_block = _PHASE_PROMPTS.get((phase or "").lower())
    if phase_block:
        lines.extend(["", phase_block])
    lines.extend(["", "<tools>"])
    for t in tools:
        lines.append(json.dumps(t))
    lines.append("</tools>")
    lines.append("")
    lines.append(f"Available tools: {', '.join(t['function']['name'] for t in tools)}.")
    return "\n".join(lines)
