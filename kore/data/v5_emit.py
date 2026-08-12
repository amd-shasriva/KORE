"""Emission rules for v5 rows: the system prompt, and what must never be a target.

Two findings drive this module, one from the literature and one from reading the
trainer.

**Knowledge injection is the cheapest large win available.** GEAK's module
ablation on AMD hardware isolates it: putting hardware specifications and
optimization principles in the prompt moved Triton-on-AMD call accuracy from
14.67% to 52.72% and execution accuracy from 8.70% to 20.11%, with no training at
all. On the same benchmark, direct prompting without it scores 0.0%. Nothing else
available to a dataset builder has that leverage, and it costs a few hundred
tokens per row. The one condition is that the same block must be present at
evaluation time -- a model trained to expect these facts and then denied them at
inference is worse off than one that never saw them.

**A kernel that cheats is worse than no kernel.** Every serious kernel-RL paper
independently converged on filtering these: Kevin zeroes the reward for outputs
containing ``torch.nn``, ``try``/``except`` or ``pass``; Kernel-Smith adds a
manual "advanced hacking or trivial optimization" category; Dr. Kernel calls it
"lazy optimization"; GEAK maintains a banned-ops list. Our wins and twins were
mined from agent runs against a harness that accepts anything numerically correct,
so a kernel that quietly calls back into PyTorch passes the gate and looks like a
win. Training on it teaches the fallback, and the fallback scores 20 points at
evaluation -- compiles, wrong -- not 120.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

#: The hardware block. Kept factual and short: these are the numbers a kernel
#: author on this part has to know, and every one of them changes generated code.
GFX950_FACTS = """\
Target: AMD Instinct MI355X, gfx950, CDNA4.
- 256 compute units; wavefront is 64 lanes, so block sizes should be multiples of 64.
- 160 KB LDS per CU. Shared-memory tiles must fit this or the kernel will not launch.
- 32 XCDs: round-robin workgroup dispatch, so grid swizzling matters for locality.
- Matrix cores: MFMA. Prefer 16x16 and 32x32 tiles; keep accumulation in fp32.
- FP8 is OCP e4m3fn (not e4m3fnuz, which was the gfx942 encoding).
- HBM3E, very high bandwidth: most elementwise and normalization kernels are
  bandwidth-bound, so coalesced access and vectorized loads dominate their runtime.
"""

#: Principles that hold across all three dialects.
PRINCIPLES = """\
Rules that apply to every kernel you write:
- Accumulate in fp32 even when inputs and outputs are lower precision.
- Preserve the entry-point name and signature exactly; the harness calls it directly.
- Never fall back to a PyTorch operator for the computation the kernel is meant to
  perform. A wrapper around torch is not a kernel.
- Do not wrap the kernel body in try/except to make it appear to succeed.
"""

BASE_SYSTEM = ("You are KORE, an expert AMD GPU kernel engineer targeting MI355X "
               "(gfx950, CDNA4).")


def system_prompt(dialect: Optional[str] = None, *, facts: bool = True) -> str:
    """The system message for a v5 row.

    ``dialect`` adds the one or two conventions specific to that backend. This
    must match what the evaluation harness sends, or the injection is worse than
    useless.
    """
    parts = [BASE_SYSTEM]
    if facts:
        parts += ["", GFX950_FACTS.rstrip(), "", PRINCIPLES.rstrip()]
    extra = {
        "hip": "Emit complete HIP C++: the kernel, its launcher, and the pybind11 "
               "binding, in one file that compiles as-is.",
        "flydsl": "Emit FlyDSL using the @flyc.jit programming model, with explicit "
                  "layouts and tile shapes.",
        "triton": "Emit a complete Triton kernel plus its Python launcher.",
    }.get((dialect or "").lower())
    if extra:
        parts += ["", extra]
    return "\n".join(parts).strip()


#: Patterns that mean the "kernel" delegates the work it was supposed to do.
#: Deliberately narrow -- ``torch.empty`` for an output buffer is normal and must
#: not be caught, while ``torch.matmul`` inside a matmul kernel is the whole bug.
_FALLBACK_CALLS = re.compile(
    r"\btorch\.(?:nn\.functional\.|nn\.|ops\.aten\.|special\.)\w+"
    r"|\btorch\.(?:matmul|mm|bmm|addmm|einsum|softmax|log_softmax|sigmoid|tanh|relu"
    r"|gelu|silu|layer_norm|group_norm|batch_norm|conv1d|conv2d|conv3d|linear"
    r"|scaled_dot_product_attention|cumsum|sort|topk|argsort)\s*\(",
    re.I)
_SWALLOW = re.compile(r"\btry\s*:|\bexcept\b|#\s*type:\s*ignore", re.I)
_STUB = re.compile(r"\bNotImplementedError\b|\.\.\.\s*$|^\s*pass\s*$", re.M)

#: Buffer allocation and dtype plumbing are legitimate uses of torch in a launcher.
_ALLOWED = re.compile(
    r"\btorch\.(?:empty|empty_like|zeros|zeros_like|ones|ones_like|full|full_like"
    r"|as_tensor|from_numpy|tensor|device|dtype|float16|bfloat16|float32|float8"
    r"|int8|int32|int64|cuda|Tensor|no_grad|inference_mode|compile|library)\b")


def cheats(code: str, *, allow_torch_launcher: bool = True) -> Optional[str]:
    """Why this kernel must not be a training target, or None if it is clean.

    ``allow_torch_launcher`` keeps buffer allocation and dtype handling legal,
    which every real launcher needs; only delegation of the actual computation is
    rejected.
    """
    if not code or not code.strip():
        return "empty"
    body = code
    if allow_torch_launcher:
        body = _ALLOWED.sub("", body)
    m = _FALLBACK_CALLS.search(body)
    if m:
        return f"torch_fallback:{m.group(0)[:40]}"
    if _SWALLOW.search(body):
        return "exception_swallow"
    if _STUB.search(body):
        return "stub"
    return None


def flatten_history(messages: list, *, keep_last_assistant: bool = True) -> list:
    """Collapse a multi-turn row so only the final assistant turn is a target.

    The trainer computes loss on EVERY assistant turn, located by chat-template
    markers, with no per-turn opt-out. A step-centric row ends at the revision
    worth imitating, which means its earlier assistant turns are by construction
    the revisions that were rejected -- broken kernels, trained at full weight.

    Kernel-Smith reaches the same conclusion from the opposite direction: keeping
    all steps lets the model copy superior kernels that leak into later prompts,
    producing a healthy reward curve and marginal real learning.

    So prior assistant content moves into the user turn as quoted context. The
    model still sees the history it needs to make the revision; it is no longer
    asked to reproduce the parts of it that were wrong.
    """
    if not messages:
        return messages
    system = [m for m in messages[:1] if m.get("role") == "system"]
    rest = messages[len(system):]
    assistants = [i for i, m in enumerate(rest) if m.get("role") == "assistant"]
    if len(assistants) <= 1:
        return messages
    last = assistants[-1] if keep_last_assistant else assistants[0]

    chunks = []
    for m in rest[:last]:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            chunks.append(content)
        elif role == "assistant":
            chunks.append("Previous attempt:\n" + content)
        elif role == "tool":
            chunks.append("Result:\n" + content)
    user = {"role": "user", "content": "\n\n".join(chunks)}
    return system + [user, rest[last]]


def assistant_turns(messages: Iterable[dict]) -> int:
    return sum(1 for m in messages
               if isinstance(m, dict) and m.get("role") == "assistant")


__all__ = ["BASE_SYSTEM", "GFX950_FACTS", "PRINCIPLES", "assistant_turns",
           "cheats", "flatten_history", "system_prompt"]
