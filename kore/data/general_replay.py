"""General-capability replay loaders for the anti-catastrophic-forgetting mix.

KORE specializes a reasoning+code base model on AMD kernels. To keep the ~50%
general-retention backbone (chat / code / math / instruction-following) plus the
new agentic tool-use skill, Stage-1 SFT mixes real general data back in
(Tulu-3-style replay). This module is the loader for that half.

``load_general_replay(kind, n, seed)`` returns HF-style chat rows
``[{"messages": [{"role", "content"}, ...], "_source": kind}]`` for
``kind in {code, math, chat, instruction_following, tool_use}``.

Sourcing is two-tier:
  1. REAL named HF sources via ``datasets`` when explicitly enabled (guarded, so
     the heavy import + network only happen when asked). See ``HF_SOURCES``.
  2. Bundled tiny sample sets under ``kore/data/replay_samples/<kind>.jsonl`` as
     an ALWAYS-available offline fallback (used by tests and dry-runs).

Enable real sources with ``use_hf=True`` (or env ``KORE_GENERAL_REPLAY_HF=1``);
any failure (offline, missing dataset, schema drift) degrades gracefully to the
bundled samples so the pipeline never hard-fails.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Callable, Optional

# The five general-replay capabilities.
REPLAY_KINDS = ("code", "math", "chat", "instruction_following", "tool_use")

_SAMPLES_DIR = Path(__file__).resolve().parent / "replay_samples"


# --------------------------------------------------------------------------- #
# Real HF sources (only touched when use_hf is enabled)
# --------------------------------------------------------------------------- #
# Each kind maps to an ORDERED list of candidate specs (dataset_path, config,
# split, native-schema formatter hint). The loader tries them in order and uses
# the FIRST that yields usable rows, so the SOTA-2026 primary can degrade to a
# proven fallback (offline / gated / schema drift) without breaking the build.
# Verified current (2025-2026) SOTA choices; older sources kept as fallbacks.
HF_SOURCES: dict[str, list[dict]] = {
    "code": [
        {
            # OpenCodeInstruct (NVIDIA, 2025): 5M verified code instruction->solution
            # (Stack-V2 OSS-Instruct + TACO seeds, unit-tested, Qwen/Llama-validated).
            # The current SOTA open code-SFT set; supersedes Magicoder Evol-Instruct.
            "path": "nvidia/OpenCodeInstruct",
            "config": None,
            "split": "train",
            "qa_keys": ("input", "output"),
        },
        {   # fallback: Magicoder Evol-Instruct (2023) - proven, cached offline.
            "path": "ise-uiuc/Magicoder-Evol-Instruct-110K",
            "config": None,
            "split": "train",
            "qa_keys": ("instruction", "response"),
        },
    ],
    "math": [
        {
            # OpenThoughts3-1.2M (2025 SOTA reasoning): 850k math + 250k code + 100k
            # science LONG-CoT traces (QwQ-32B). Primary math+reasoning source - the
            # long chain-of-thought (tiling/indexing/numerics reasoning) transfers to
            # kernels AND closes the reasoning-CoT retention gap so domain SFT/RL
            # doesn't erode the base model's chain-of-thought.
            # max_row_chars caps the CoT so rows FIT the SFT window (16384 tok): the
            # raw QwQ traces are often >16k tok (they were silently dropped by the SFT
            # length filter, wasting the whole slice). ~48k chars ~= 12k tok keeps a
            # LENGTH-DIVERSE tail of traces that survive training.
            "path": "open-thoughts/OpenThoughts3-1.2M",
            "config": None,
            "split": "train",
            "sharegpt_key": "conversations",
            "max_row_chars": 48000,
        },
        {   # fallback: OpenMathInstruct-2 (NVIDIA) 1M CoT math (cached offline).
            "path": "nvidia/OpenMathInstruct-2",
            "config": None,
            "split": "train_1M",
            "qa_keys": ("problem", "generated_solution"),
            "max_row_chars": 48000,
        },
    ],
    "chat": [
        {   # Tulu-3 SFT mixture: already chat-formatted ``messages``.
            "path": "allenai/tulu-3-sft-mixture",
            "config": None,
            "split": "train",
            "messages_key": "messages",
        },
    ],
    "instruction_following": [
        {   # Same mixture; IF-heavy subset (native ``messages``).
            "path": "allenai/tulu-3-sft-mixture",
            "config": None,
            "split": "train",
            "messages_key": "messages",
        },
    ],
    "tool_use": [
        {
            # ToolACE: diverse function-calling trajectories (sharegpt-style).
            "path": "Team-ACE/ToolACE",
            "config": None,
            "split": "train",
            "sharegpt_key": "conversations",
        },
        {   # fallback: Salesforce xLAM function-calling (single-turn).
            "path": "Salesforce/xlam-function-calling-60k",
            "config": None,
            "split": "train",
            "qa_keys": ("query", "answers"),
        },
    ],
}


def _truthy(val: Optional[str]) -> bool:
    return str(val or "").strip().lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------- #
# Row hygiene / coercion
# --------------------------------------------------------------------------- #
def _valid_messages(messages: Any) -> bool:
    if not isinstance(messages, list) or not messages:
        return False
    for m in messages:
        if not isinstance(m, dict):
            return False
        if "role" not in m or "content" not in m:
            return False
        if not isinstance(m.get("content"), str):
            return False
    return True


def _row_chars(row: dict) -> int:
    """Total content length of a chat row (used for the length-diversity cap)."""
    return sum(len(m.get("content", "")) for m in row.get("messages", [])
               if isinstance(m, dict))


def _as_chat_row(obj: Any, kind: str) -> Optional[dict]:
    """Coerce a loaded object into a tagged chat row, or None if malformed."""
    if isinstance(obj, dict) and _valid_messages(obj.get("messages")):
        row = {"messages": [dict(m) for m in obj["messages"]]}
        row["_source"] = obj.get("_source", kind)
        return row
    return None


# --------------------------------------------------------------------------- #
# Bundled offline fallback
# --------------------------------------------------------------------------- #
def _bundled_path(kind: str) -> Path:
    return _SAMPLES_DIR / f"{kind}.jsonl"


def load_bundled_samples(kind: str) -> list[dict]:
    """Load the bundled tiny sample set for ``kind`` (offline, always works)."""
    if kind not in REPLAY_KINDS:
        raise ValueError(f"unknown replay kind {kind!r}; known: {REPLAY_KINDS}")
    path = _bundled_path(kind)
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        row = _as_chat_row(obj, kind)
        if row is not None:
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# HF formatters (native schema -> chat messages)
# --------------------------------------------------------------------------- #
def _fmt_messages_passthrough(ex: dict, spec: dict, kind: str) -> Optional[dict]:
    key = spec.get("messages_key", "messages")
    return _as_chat_row({"messages": ex.get(key)}, kind)


def _fmt_qa(ex: dict, spec: dict, kind: str) -> Optional[dict]:
    qk, ak = spec.get("qa_keys", ("question", "answer"))
    q, a = ex.get(qk), ex.get(ak)
    if not isinstance(q, str):
        return None
    if not isinstance(a, str):
        a = json.dumps(a) if a is not None else ""
    if not q.strip() or not a.strip():
        return None
    return {"messages": [
        {"role": "user", "content": q.strip()},
        {"role": "assistant", "content": a.strip()},
    ], "_source": kind}


def _fmt_code(ex: dict, spec: dict, kind: str) -> Optional[dict]:
    text = None
    for k in spec.get("text_keys", ("content", "text")):
        if isinstance(ex.get(k), str) and ex[k].strip():
            text = ex[k]
            break
    if text is None:
        return None
    text = text.strip()
    if len(text) > 4000:
        text = text[:4000]
    return {"messages": [
        {"role": "user", "content": "Explain what the following code does:\n\n"
                                    f"```python\n{text}\n```"},
        {"role": "assistant", "content": "Here is a walkthrough of the code:\n\n"
                                         f"```python\n{text}\n```"},
    ], "_source": kind}


_SHAREGPT_ROLE = {"human": "user", "user": "user", "gpt": "assistant",
                  "assistant": "assistant", "system": "system", "tool": "tool",
                  "function": "tool", "observation": "tool"}


def _fmt_sharegpt(ex: dict, spec: dict, kind: str) -> Optional[dict]:
    """Convert a sharegpt-style ``conversations`` list to chat messages.

    Handles both the classic ShareGPT ``{from, value}`` turn schema and the
    ``{role, content}`` schema (e.g. OpenThoughts3), so long-CoT reasoning sets
    normalize cleanly."""
    conv = ex.get(spec.get("sharegpt_key", "conversations"))
    if not isinstance(conv, list) or not conv:
        return None
    msgs = []
    for turn in conv:
        if not isinstance(turn, dict):
            return None
        raw_role = turn.get("from", turn.get("role", ""))
        role = _SHAREGPT_ROLE.get(str(raw_role).lower())
        val = turn.get("value", turn.get("content"))
        if role is None or not isinstance(val, str) or not val.strip():
            continue
        msgs.append({"role": role, "content": val.strip()})
    return _as_chat_row({"messages": msgs}, kind) if len(msgs) >= 2 else None


def _formatter_for(kind: str, spec: dict) -> Callable[[dict, dict, str], Optional[dict]]:
    # spec-driven so a "code" source can be real instruction->response (qa) rather
    # than a raw-snippet echo: messages -> passthrough; qa_keys -> qa; text_keys ->
    # code-snippet formatter (last resort).
    if spec.get("messages_key"):
        return _fmt_messages_passthrough
    if spec.get("sharegpt_key"):
        return _fmt_sharegpt
    if spec.get("qa_keys"):
        return _fmt_qa
    if spec.get("text_keys"):
        return _fmt_code
    return _fmt_qa


def _candidate_specs(kind: str) -> list[dict]:
    """Ordered candidate HF specs for a kind (back-compat: wrap a bare dict)."""
    spec = HF_SOURCES[kind]
    return spec if isinstance(spec, list) else [spec]


def _load_one_spec(spec: dict, kind: str, n: int, seed: int) -> list[dict]:
    """Stream ``n`` usable rows from a single HF spec (raises if none)."""
    from datasets import load_dataset  # guarded heavy import

    ds = load_dataset(
        spec["path"], spec.get("config"), split=spec.get("split", "train"),
        streaming=True,
    )
    try:
        ds = ds.shuffle(seed=seed, buffer_size=max(1000, n * 4))
    except Exception:
        pass  # some streaming datasets don't support shuffle; take head order
    fmt = _formatter_for(kind, spec)
    max_row_chars = spec.get("max_row_chars")
    rows: list[dict] = []
    # A char cap needs a bigger scan budget (many long CoT rows are skipped), so
    # widen the scan when a cap is set to still reach ``n`` length-diverse rows.
    budget = max(n * (60 if max_row_chars else 20), 400 if max_row_chars else 200)
    for i, ex in enumerate(ds):
        if len(rows) >= n or i >= budget:
            break
        row = fmt(dict(ex), spec, kind)
        if row is None:
            continue
        if max_row_chars and _row_chars(row) > int(max_row_chars):
            continue  # drop over-length CoT so the row fits the SFT window
        rows.append(row)
    if not rows:
        raise RuntimeError(f"HF source {spec.get('path')!r} for {kind!r} yielded no rows")
    return rows


def _load_from_hf(kind: str, n: int, seed: int) -> list[dict]:
    """Load ``n`` chat rows for ``kind``, trying each candidate spec in order.

    Heavy import is inside. Returns the first spec that yields rows; raises only
    if EVERY candidate fails, so the caller can fall back to bundled samples.
    """
    errors: list[str] = []
    for spec in _candidate_specs(kind):
        try:
            rows = _load_one_spec(spec, kind, n, seed)
            if rows:
                return rows
        except Exception as e:  # noqa: BLE001 - try the next candidate
            errors.append(f"{spec.get('path')}: {type(e).__name__}: {e}")
            continue
    raise RuntimeError(f"all HF sources for {kind!r} failed: {'; '.join(errors)}")


# --------------------------------------------------------------------------- #
# Deterministic sampling to exactly n rows
# --------------------------------------------------------------------------- #
def _resize(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Return exactly ``n`` rows: subsample without replacement when the pool is
    large enough, else deterministically oversample (with replacement) to fill.
    """
    if n <= 0 or not rows:
        return []
    rng = random.Random(seed)
    if n <= len(rows):
        idx = rng.sample(range(len(rows)), n)
        return [rows[i] for i in idx]
    # Oversample: shuffled full passes, then a shuffled remainder.
    out: list[dict] = []
    while len(out) < n:
        order = list(range(len(rows)))
        rng.shuffle(order)
        for i in order:
            out.append(rows[i])
            if len(out) >= n:
                break
    return out


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_general_replay(
    kind: str,
    n: int,
    seed: int = 0,
    use_hf: Optional[bool] = None,
) -> list[dict]:
    """Load ``n`` general-replay chat rows for ``kind``.

    kind in {code, math, chat, instruction_following, tool_use}.

    Returns ``[{"messages": [...], "_source": kind}, ...]`` of length ``n``
    (deterministic given ``seed``).

    Real HF sources (``HF_SOURCES``) are used only when ``use_hf`` is True or, if
    ``use_hf is None``, when env ``KORE_GENERAL_REPLAY_HF`` is truthy. On ANY HF
    failure (offline / missing / schema drift) it falls back to the bundled
    ``replay_samples/<kind>.jsonl`` so it always runs offline and in tests.
    """
    if kind not in REPLAY_KINDS:
        raise ValueError(f"unknown replay kind {kind!r}; known: {REPLAY_KINDS}")
    if n <= 0:
        return []

    if use_hf is None:
        use_hf = _truthy(os.environ.get("KORE_GENERAL_REPLAY_HF"))

    pool: list[dict] = []
    if use_hf:
        try:
            pool = _load_from_hf(kind, n, seed)
        except Exception as e:  # noqa: BLE001 - degrade to offline bundle
            print(f"[general_replay] HF source for {kind!r} unavailable "
                  f"({type(e).__name__}: {e}); using bundled samples")
            pool = []
    if not pool:
        pool = load_bundled_samples(kind)
    return _resize(pool, n, seed)


def load_all_general_replay(
    counts: dict[str, int],
    seed: int = 0,
    use_hf: Optional[bool] = None,
) -> dict[str, list[dict]]:
    """Load several replay kinds at once.

    ``counts`` maps ``kind -> n``. Returns ``{kind: rows}``. Each kind gets a
    decorrelated but deterministic sub-seed.
    """
    out: dict[str, list[dict]] = {}
    for i, kind in enumerate(k for k in REPLAY_KINDS if k in counts):
        out[kind] = load_general_replay(
            kind, counts[kind], seed=seed + 1 + i, use_hf=use_hf
        )
    return out
