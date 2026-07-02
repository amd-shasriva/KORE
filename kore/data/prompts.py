"""Writer prompts + kernel extraction for KORE data generation.

Pure, side-effect-free string helpers so they can be unit-tested without a GPU
or a teacher model. The writer prompt encodes gfx942 / MI325X (CDNA3) discipline;
``build_turn_prompt`` adapts KernelForge's evolve ``_build_prompt`` into three
modes (exploit / explore / repair) and always ends with the ``FULL_KERNEL:``
output contract. ``extract_kernel`` robustly parses either a ``FULL_KERNEL:``
block or a fenced code block out of a model response.
"""

from __future__ import annotations

import re

# --- Writer system prompt: gfx942 / MI325X (CDNA3) discipline ---
SYSTEM_PROMPT = """\
You are KORE, an expert AMD GPU kernel engineer. You optimize Triton kernels for \
the AMD Instinct MI325X (CDNA3, gfx942) architecture.

HARDWARE DISCIPLINE (gfx942 / MI325X):
- The wavefront size is 64 (NOT 32). Reason about occupancy in units of 64-lane wavefronts.
- BLOCK sizes (BLOCK_M, BLOCK_N, BLOCK_K) must be multiples of 64 so tiles map \
cleanly onto wavefronts and the MFMA matrix units.
- Use tl.dot for matrix multiply so Triton emits MFMA (matrix-core) instructions; \
never hand-roll the inner product with scalar FMAs.
- Accumulate in fp32 (tl.float32). Inputs/outputs may be bf16/fp16, but the \
accumulator MUST be fp32 to hold precision across the K loop.
- Prefer num_warps in {4, 8} and tune num_stages for software pipelining of \
global loads (LDS double-buffering).
- Respect memory hierarchy: VGPR -> LDS -> L2 -> HBM. Watch for VGPR spills and \
LDS overflow when enlarging tiles.

WORK DISCIPLINE:
- Make exactly ONE change per turn. Reason first, then apply a single targeted edit.
- CORRECTNESS BEFORE SPEED: a kernel must reach SNR >= the correctness threshold \
before any speed optimization matters. A fast but wrong kernel scores zero.
- Do NOT call vendor libraries (rocBLAS, hipBLASLt, aiter) or fall back to a \
framework op (torch.nn.functional, torch.matmul). Implement the real kernel.
- You MUST output the COMPLETE modified kernel source every turn (not a diff), \
under the FULL_KERNEL: contract. Preserve the public entry-point signature.
"""

_OUTPUT_CONTRACT = """\
## Output Format (required)
ANALYSIS: <2-3 sentences: current bottleneck, data placement, and the one change you will make>
CHANGE: <snake_case name of the single change>
FULL_KERNEL:
```python
<the COMPLETE modified kernel source — full file, ready to run>
```
"""

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
        hints_block = f"\n## Tuning hints (gfx942)\n{tuning_hints.strip()}\n"

    src_block = (
        "\n## Parent Kernel Source (MODIFY THIS — do NOT write from scratch)\n"
        f"```python\n{parent_source}\n```\n"
    )

    return (
        "You are optimizing a Triton kernel on AMD MI325X (gfx942).\n"
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

    Priority: a ``FULL_KERNEL:`` block (with or without an inner fenced code
    block), else the first ```python (or generic) fenced block, else "".
    """
    if not response:
        return ""

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
