"""Policy prompt/response format — PURE, dependency-free, unit-testable.

The policy is a reasoning+code model that iteratively optimizes a ROCm kernel
across turns. This module owns the *contract* between the policy and the env:

  - ``SYSTEM_PROMPT`` — gfx942 kernel discipline, one change per turn, and the
    ANALYSIS / PROPOSED_CHANGE / FULL_KERNEL response contract.
  - ``build_transcript`` — assemble the multi-turn chat (prior kernels + the
    *summarized* verifier feedback) that is fed to the model each turn.
  - ``parse_response`` — pull ``analysis`` / ``proposed_change`` / ``kernel`` out
    of a model response.
  - ``summarize_cot`` — bound the chain-of-thought length so context does not
    grow without limit across turns (Kevin: summarize CoT between turns).
  - ``build_turn_feedback`` — render an ``Observation`` into the compact
    compile / SNR / wall feedback the policy sees on its next turn.

Nothing here imports torch/vllm/transformers; it is safe to import anywhere.
"""

from __future__ import annotations

import re
from typing import Any

# Fenced-code and section markers used by the response contract.
_ANALYSIS = "ANALYSIS"
_PROPOSED = "PROPOSED_CHANGE"
_FULL_KERNEL = "FULL_KERNEL"


SYSTEM_PROMPT = """\
You are KORE, an expert AMD GPU kernel engineer. You optimize a single GPU \
kernel across multiple turns for correctness first and speed second.

TARGET HARDWARE: AMD Instinct MI325X, arch gfx942 (CDNA3), ROCm + Triton.
Do NOT assume NVIDIA specifics (no warp==32, no cp.async, no tensor-core PTX).

GFX942 DISCIPLINE:
  - Wavefront size is 64 (not 32); reason about occupancy in 64-lane wavefronts.
  - Make BLOCK_M/BLOCK_N/BLOCK_K multiples of 64 so tiles map cleanly onto \
wavefronts and the MFMA matrix cores.
  - Use tl.dot for matrix multiply so Triton emits MFMA instructions; never \
hand-roll the inner product with scalar FMAs.
  - Accumulate in fp32 (tl.float32). Inputs/outputs may be bf16/fp16/fp8, but the \
accumulator MUST be fp32 to hold precision across the reduction/K loop.
  - Prefer num_warps in {4, 8}; tune num_stages for software pipelining / LDS \
double-buffering of global loads.
  - Respect the memory hierarchy VGPR -> LDS -> L2 -> HBM; watch for VGPR spills \
and LDS overflow when enlarging tiles, and coalesce global loads.

KERNEL DISCIPLINE:
  - Implement a REAL kernel. Never call a vendor/reference library (rocBLAS, \
hipBLASLt, aiter, ...), never fall back to a framework op (torch.nn.*, \
torch.matmul as the "kernel"), and never wrap the kernel in a try/except that \
silently returns a reference result. Such shortcuts score zero.
  - Preserve the reference numerics: your kernel must pass the SNR correctness \
gate on EVERY validation shape.
  - Keep the public entry-point function signature unchanged.

PER-TURN PROTOCOL:
  - Make exactly ONE focused optimization per turn (e.g. change the block/tile \
size, add LDS staging, vectorize loads, tweak num_warps/num_stages). One change \
per turn makes the speedup attributable and keeps the search stable.
  - Think first, then emit the change and the COMPLETE updated kernel source.

RESPONSE FORMAT — respond with exactly these three sections, in order:

ANALYSIS:
<your reasoning about the previous feedback and what to try next>

PROPOSED_CHANGE:
<one sentence naming the single change you are making this turn>

FULL_KERNEL:
```python
<the ENTIRE kernel source, ready to run — not a diff, not a snippet>
```
"""


# The response-format block, reused verbatim by the data-generation writer
# prompts (kore/data/prompts.py) so the teacher is asked for EXACTLY the contract
# the policy is trained to emit. Single source of truth — do not fork this.
OUTPUT_CONTRACT = """\
## Output Format (required) — respond with EXACTLY these sections, in order:
ANALYSIS:
<2-3 sentences: the current bottleneck, data placement, and the one change you will make>

PROPOSED_CHANGE:
<one sentence naming the single change (imperative, specific)>

FULL_KERNEL:
```python
<the COMPLETE modified kernel source — full file, ready to run, not a diff>
```
"""


def summarize_cot(text: str, max_chars: int = 2000) -> str:
    """Bound a chain-of-thought / analysis blob to ``max_chars`` characters.

    Context grows every turn; Kevin summarizes the CoT so the transcript stays
    within the model's window. We keep the head and tail (the tail usually holds
    the conclusion) and elide the middle. The returned string is guaranteed to
    be at most ``max_chars`` characters long.
    """
    text = (text or "").strip()
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    marker = "\n...[cot summarized]...\n"
    if max_chars <= len(marker):
        return text[:max_chars]
    budget = max_chars - len(marker)
    head = budget - budget // 2
    tail = budget // 2
    out = text[:head] + marker + (text[-tail:] if tail > 0 else "")
    # Guard against any off-by-one from integer splits.
    return out[:max_chars]


def parse_response(text: str) -> dict:
    """Parse a policy response into ``{analysis, proposed_change, kernel}``.

    Robust to missing sections and to models that emit the FULL_KERNEL as a bare
    ```python fenced block without the literal header. ``kernel`` is the raw
    source with no fences.
    """
    text = text or ""

    analysis = _extract_section(text, _ANALYSIS, (_PROPOSED, _FULL_KERNEL))
    proposed = _extract_section(text, _PROPOSED, (_FULL_KERNEL,))
    kernel = _extract_kernel(text)

    return {
        "analysis": analysis.strip(),
        "proposed_change": proposed.strip(),
        "kernel": kernel.strip(),
    }


def _extract_section(text: str, header: str, stops: tuple[str, ...]) -> str:
    """Return the text after ``HEADER:`` up to the next known header/stop."""
    m = re.search(rf"{header}\s*:?", text, flags=re.IGNORECASE)
    if not m:
        return ""
    start = m.end()
    end = len(text)
    for stop in stops:
        sm = re.search(rf"\n\s*{stop}\s*:?", text[start:], flags=re.IGNORECASE)
        if sm:
            end = min(end, start + sm.start())
    return text[start:end]


def _extract_kernel(text: str) -> str:
    """Pull the kernel source out of the FULL_KERNEL block or a fenced block."""
    # Prefer a fenced block that appears after the FULL_KERNEL header.
    fk = re.search(rf"{_FULL_KERNEL}\s*:?", text, flags=re.IGNORECASE)
    scope = text[fk.end():] if fk else text
    fenced = re.search(r"```(?:python|py)?\s*\n(.*?)```", scope, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)
    # No fence: take everything after the header as the kernel.
    if fk:
        return scope
    # Last resort: any fenced block anywhere in the text.
    any_fence = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL)
    return any_fence.group(1) if any_fence else ""


def format_assistant_turn(analysis: str, proposed_change: str, kernel: str) -> str:
    """Render the CANONICAL assistant response (single source of truth).

    Every training row and every inference/transcript turn uses this exact shape:
    ``ANALYSIS: … / PROPOSED_CHANGE: … / FULL_KERNEL: ```python … `````. The
    PROPOSED_CHANGE section is omitted only when empty (e.g. a first-turn / repair
    demo may carry just analysis + kernel). This is what `parse_response` reads back.
    """
    parts: list[str] = [f"{_ANALYSIS}:\n{(analysis or '').strip()}".rstrip()]
    if (proposed_change or "").strip():
        parts.append(f"{_PROPOSED}:\n{proposed_change.strip()}".rstrip())
    parts.append(f"{_FULL_KERNEL}:\n```python\n{(kernel or '').strip()}\n```")
    return "\n\n".join(parts)


def wrap_full_kernel(source: str) -> str:
    """Wrap a kernel body in just the FULL_KERNEL block (DPO/RFT completions).

    Preference completions compare kernels, so they carry only the FULL_KERNEL
    section (a pure, contract-shaped completion) — the canonical single-section form.
    """
    return f"{_FULL_KERNEL}:\n```python\n{(source or '').strip()}\n```\n"


def normalize_assistant(content: str) -> str:
    """Re-render ANY legacy assistant content into the canonical contract.

    Handles every historical shape found in the KORE data (so existing shards can
    be upgraded in place without regeneration):
      * repair ``<think>…</think><answer>FULL_KERNEL:…</answer>`` (LLM-VeriOpt) —
        the ``<think>`` reasoning becomes ANALYSIS;
      * gold-win ``ANALYSIS: … FULL_KERNEL:\\n<src>`` with no PROPOSED_CHANGE and no
        code fence — the fence is added;
      * data-gen ``CHANGE:`` — mapped to PROPOSED_CHANGE;
      * raw teacher text — parsed and re-emitted.
    Idempotent on already-canonical content. Returns the content unchanged if no
    kernel can be extracted (nothing safe to normalize).
    """
    content = content or ""
    kernel = _extract_kernel(content).strip()
    if not kernel:
        return content
    think = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL)
    if think and think.group(1).strip():
        analysis = think.group(1).strip()
    else:
        analysis = _extract_section(
            content, _ANALYSIS, (_PROPOSED, "CHANGE", _FULL_KERNEL)).strip()
    proposed = _extract_section(content, _PROPOSED, (_FULL_KERNEL,)).strip()
    if not proposed:
        proposed = _extract_section(content, "CHANGE", (_FULL_KERNEL,)).strip()
    return format_assistant_turn(analysis, proposed, kernel)


def build_turn_feedback(obs: Any, cfg: Any = None) -> str:
    """Render an ``Observation`` into compact, actionable turn feedback.

    Covers the three failure/success regimes the policy must react to:
    compile failure, correctness (SNR) failure, and a timed/correct kernel with
    its speedup vs. the reference baseline.
    """
    lines: list[str] = []

    compiled = getattr(obs, "compiled", None)
    if compiled is False:
        err = _clip(getattr(obs, "error_text", "") or "", 800)
        lines.append("RESULT: compile/build FAILED.")
        if err:
            lines.append(f"COMPILER ERROR:\n{err}")
        lines.append("Fix the build error before optimizing further.")
        return "\n".join(lines)

    snr = getattr(obs, "snr_db", None)
    validation_passed = bool(getattr(obs, "validation_passed", False))
    snr_by_shape = getattr(obs, "snr_by_shape", {}) or {}

    if not validation_passed:
        lines.append("RESULT: compiled but INCORRECT (failed the SNR gate).")
        if snr is not None:
            lines.append(f"primary SNR: {snr:.2f} dB")
        worst = _worst(snr_by_shape)
        if worst is not None:
            lines.append(f"worst-shape SNR: {worst:.2f} dB")
        err = _clip(getattr(obs, "error_text", "") or "", 400)
        if err:
            lines.append(f"detail: {err}")
        lines.append("Restore numerical correctness; do not sacrifice accuracy for speed.")
        return "\n".join(lines)

    lines.append("RESULT: CORRECT (passed the SNR gate).")
    if snr is not None:
        lines.append(f"primary SNR: {snr:.2f} dB")

    wall = getattr(obs, "wall_ms", None)
    base = getattr(obs, "baseline_ms", None)
    if wall is not None:
        lines.append(f"candidate wall time: {wall:.4f} ms")
    if base is not None:
        lines.append(f"reference baseline: {base:.4f} ms")
    if wall and base and wall > 0 and base > 0:
        speedup = base / wall
        lines.append(f"speedup vs reference: {speedup:.3f}x")

    # Per-shape breakdown helps the model see shape-specific regressions.
    wbs = getattr(obs, "wall_by_shape", {}) or {}
    bbs = getattr(obs, "baseline_by_shape", {}) or {}
    per_shape = []
    for shape, c in wbs.items():
        b = bbs.get(shape)
        if c and b and c > 0 and b > 0:
            per_shape.append(f"  {shape}: {b / c:.3f}x")
    if per_shape:
        lines.append("per-shape speedup:")
        lines.extend(per_shape)

    lines.append("Propose ONE further optimization to improve the speedup while staying correct.")
    return "\n".join(lines)


def build_task_prompt(task: Any) -> str:
    """The canonical INITIAL task prompt (turn-1 user message): seed kernel + contract.

    Single source of truth shared by GRPO rollouts, eval, AND DPO-pair construction,
    so preferences are learned in the SAME context the policy sees at inference (a
    seed kernel to improve + the ANALYSIS/PROPOSED_CHANGE/FULL_KERNEL contract) — not
    a bare "optimize task X" one-shot. ``task`` is any object exposing ``dtype``,
    ``operation``, ``gpu_target``, ``backend``, ``comparison_baseline``, ``seed_source``.
    """
    return (f"Optimize a {task.dtype} {task.operation} kernel for AMD {task.gpu_target} "
            f"(backend: {task.backend}). Baseline to beat: {task.comparison_baseline}. "
            f"Return ANALYSIS, PROPOSED_CHANGE, and a complete FULL_KERNEL.\n\n"
            f"Seed kernel:\n```python\n{task.seed_source}\n```")


def build_transcript(
    task_prompt: str,
    turns: list[dict] | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    max_cot_chars: int = 2000,
) -> list[dict]:
    """Assemble the multi-turn chat messages for the next policy call.

    ``turns`` is the ordered history; each entry may contain:
      - ``response``  : the raw assistant text from that turn (its CoT is
                        summarized to bound context growth), OR
      - ``analysis`` / ``proposed_change`` / ``kernel`` : pre-parsed fields.
      - ``feedback``  : a precomputed verifier-feedback string, OR
      - ``obs`` / ``observation`` : an ``Observation`` rendered via
                        ``build_turn_feedback``.

    Returns a list of ``{"role", "content"}`` dicts (system, user, then the
    alternating assistant/user history).
    """
    turns = turns or []
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_prompt},
    ]

    for turn in turns:
        messages.append({"role": "assistant", "content": _assistant_content(turn, max_cot_chars)})
        fb = _turn_feedback_text(turn)
        if fb:
            messages.append({"role": "user", "content": fb})

    return messages


def _assistant_content(turn: dict, max_cot_chars: int) -> str:
    """Reconstruct a compact assistant message: summarized analysis + full kernel."""
    if "response" in turn and turn["response"]:
        parsed = parse_response(turn["response"])
    else:
        parsed = {
            "analysis": turn.get("analysis", ""),
            "proposed_change": turn.get("proposed_change", ""),
            "kernel": turn.get("kernel", ""),
        }

    analysis = summarize_cot(parsed.get("analysis", ""), max_cot_chars)
    return format_assistant_turn(analysis, parsed.get("proposed_change", ""),
                                 parsed.get("kernel", ""))


def _turn_feedback_text(turn: dict) -> str:
    if turn.get("feedback"):
        return str(turn["feedback"])
    obs = turn.get("obs", turn.get("observation"))
    if obs is not None:
        return build_turn_feedback(obs)
    return ""


def _clip(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + " ...[truncated]"


def _worst(snr_by_shape: dict):
    vals = [v for v in snr_by_shape.values() if v is not None]
    return min(vals) if vals else None
