"""Writer prompts + kernel extraction for KORE data generation.

Pure, side-effect-free string helpers so they can be unit-tested without a GPU
or a teacher model. The writer prompt encodes gfx950 / MI350X (CDNA4) discipline;
``build_turn_prompt`` adapts KernelForge's evolve ``_build_prompt`` into three
modes (exploit / explore / repair) and always ends with the ``FULL_KERNEL:``
output contract. ``extract_kernel`` robustly parses either a ``FULL_KERNEL:``
block or a fenced code block out of a model response.
"""

from __future__ import annotations

import re

# Single source of truth for the prompt/response contract lives in the POLICY
# module (kore.policy.format) - the exact contract the model is trained to emit and
# that the env/eval parse back. Data generation MUST request that same contract so
# the teacher's demonstrations match deployment (no CHANGE-vs-PROPOSED_CHANGE or
# dual-system-prompt drift). We re-export SYSTEM_PROMPT so existing
# ``from kore.data.prompts import SYSTEM_PROMPT`` call-sites transparently get the
# canonical prompt. format.py is pure (re/typing only) so this adds no heavy deps.
from kore.policy.format import (  # noqa: F401  (SYSTEM_PROMPT/gate re-exported)
    OUTPUT_CONTRACT as _OUTPUT_CONTRACT,
    SYSTEM_PROMPT,
    check_change_consistency,
    format_assistant_turn,
    normalize_assistant,
    wrap_full_kernel,
    _split_think,
)

_MODE_INSTRUCTIONS = {
    "exploit": (
        "## Mode: EXPLOIT\n"
        "Find the SINGLE highest-impact small change to this working kernel. "
        "Identify the real bottleneck (compute vs memory vs latency-hiding) and make "
        "exactly one targeted mutation to address it (e.g. tile size, num_warps, "
        "num_stages, load masking). Do NOT rewrite the kernel structure."
    ),
    "explore": (
        "## Mode: EXPLORE (small changes exhausted)\n"
        "Incremental tuning has plateaued. Identify an ARCHITECTURAL pattern that is "
        "fundamentally suboptimal and make one structural change that unlocks a new "
        "region of the performance landscape: different tiling scheme, different LDS "
        "usage, different pipeline (num_stages) structure, or a grouped/streamed launch."
    ),
    "repair": (
        "## Mode: REPAIR\n"
        "The current kernel FAILED. Read the exact verifier error below and fix the "
        "specific bug WITHOUT rewriting the whole kernel. If it failed to compile, fix "
        "the syntax/type/shape error. If correctness failed (low SNR), fix the "
        "numerics (e.g. fp32 accumulator, correct masking/indexing, block alignment)."
    ),
}


def build_turn_prompt(
    parent_source: str,
    feedback: str = "",
    tuning_hints: str = "",
    mode: str = "exploit",
) -> str:
    """Build a single writer-turn user prompt.

    mode in {"exploit", "explore", "repair"}. ``feedback`` is the verifier
    observation / error text from the previous turn; ``tuning_hints`` is optional
    knowledge-base guidance. The prompt always ends with the FULL_KERNEL contract.
    """
    mode = (mode or "exploit").lower()
    mode_block = _MODE_INSTRUCTIONS.get(mode, _MODE_INSTRUCTIONS["exploit"])

    feedback_block = ""
    if feedback:
        label = "Verifier error (fix this)" if mode == "repair" else "Verifier feedback"
        feedback_block = f"\n## {label}\n```\n{feedback.strip()}\n```\n"

    hints_block = ""
    if tuning_hints:
        hints_block = f"\n## Tuning hints (gfx950 / CDNA4)\n{tuning_hints.strip()}\n"

    src_block = (
        "\n## Parent Kernel Source (MODIFY THIS - do NOT write from scratch)\n"
        f"```python\n{parent_source}\n```\n"
    )

    return (
        "You are optimizing a Triton kernel on AMD Instinct MI350X (gfx950 / CDNA4).\n"
        "Reason about the bottleneck FIRST, then make your change.\n\n"
        f"{mode_block}\n"
        f"{feedback_block}{hints_block}{src_block}\n"
        "## HARD CONSTRAINTS\n"
        "1. Keep the public entry-point function signature unchanged.\n"
        "2. BLOCK_* sizes must be multiples of 64; accumulate in fp32; use tl.dot.\n"
        "3. No vendor libs / framework fallbacks. Implement the real kernel.\n"
        "4. Output the COMPLETE kernel source.\n\n"
        f"{_OUTPUT_CONTRACT}"
    )


# --- Kernel extraction (robust; adapted from KernelForge evolve._extract_kernel) ---
_FULL_KERNEL_RE = re.compile(
    r"FULL_KERNEL\s*:\s*\|?\s*\n(.*?)(?=\n[A-Z_]{3,}\s*:|\Z)", re.DOTALL
)
_FENCED_INNER_RE = re.compile(r"```(?:python|py|triton)?\s*\n(.*?)```", re.DOTALL)
_ANY_FENCED_RE = re.compile(
    r"```(?:python|py|triton|cpp|c\+*|cuda|hip)?\s*\n(.*?)```", re.DOTALL
)


def _dedent_block(text: str) -> str:
    lines = text.splitlines()
    # strip leading/trailing blank lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""
    indents = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
    common = min(indents) if indents else 0
    return "\n".join(l[common:] if len(l) >= common else l for l in lines)


def extract_kernel(response: str) -> str:
    """Extract kernel source from a model response.

    The OPTIONAL ``<think>`` deep-reasoning scratchpad is stripped FIRST (shared
    with :func:`kore.policy.format._split_think`) so a scratchpad that quotes
    ``FULL_KERNEL:`` or drafts code in its own fenced block is never mistaken for
    the real kernel. Priority within the think-stripped body: a ``FULL_KERNEL:``
    block (with or without an inner fenced code block), else the first ```python
    (or generic) fenced block, else "". If the body has no kernel but the
    scratchpad did (a think-only response), fall back to the raw response so the
    pre-deep-CoT behavior is preserved.
    """
    if not response:
        return ""
    _, body = _split_think(response)
    src = _extract_kernel_from_body(body)
    if src:
        return src
    if body != response:  # code lived only inside the <think> scratchpad
        return _extract_kernel_from_body(response)
    return ""


def _extract_kernel_from_body(response: str) -> str:
    """FULL_KERNEL/fenced-block extraction on already-scratchpad-stripped text."""
    m = _FULL_KERNEL_RE.search(response)
    if m:
        content = m.group(1)
        inner = _FENCED_INNER_RE.search(content)
        if inner:
            return inner.group(1).strip()
        inner_any = _ANY_FENCED_RE.search(content)
        if inner_any:
            return inner_any.group(1).strip()
        return _dedent_block(content)

    # Fallback: prefer a python-fenced block, else any fenced block.
    inner = _FENCED_INNER_RE.search(response)
    if inner:
        return inner.group(1).strip()
    inner_any = _ANY_FENCED_RE.search(response)
    if inner_any:
        return inner_any.group(1).strip()
    return ""
