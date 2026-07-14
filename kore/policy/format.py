"""Policy prompt/response format — PURE, dependency-free, unit-testable.

The policy is a reasoning+code model that iteratively optimizes a ROCm kernel
across turns. This module owns the *contract* between the policy and the env:

  - ``SYSTEM_PROMPT`` — gfx950/CDNA4 kernel discipline, one change per turn, and the
    ANALYSIS / PROPOSED_CHANGE / FULL_KERNEL response contract.
  - ``build_transcript`` — assemble the multi-turn chat (prior kernels + the
    *summarized* verifier feedback) that is fed to the model each turn.
  - ``parse_response`` — pull the optional ``think`` scratchpad plus the
    ``analysis`` / ``proposed_change`` / ``kernel`` out of a model response.
  - ``summarize_cot`` — bound the chain-of-thought length so context does not
    grow without limit across turns (Kevin: summarize CoT between turns). It
    keeps the ANALYSIS/PROPOSED_CHANGE conclusion and drops the verbose
    ``<think>`` scratchpad rather than blindly slicing the middle out.
  - ``check_change_consistency`` — a pure claim<->code gate: does the change the
    ANALYSIS names (num_warps/num_stages/BLOCK_*/tl.dot/LDS/...) actually appear
    in the prev->new kernel diff? (catches "describe but don't implement").
  - ``build_turn_feedback`` — render an ``Observation`` into the compact
    compile / SNR / wall feedback the policy sees on its next turn.

DEEP CoT (additive, backward-compatible): the contract now *invites* an OPTIONAL
``<think>...</think>`` scratchpad (or a longer ANALYSIS) BEFORE the terse
structured sections, so frontier kernel reasoning (roofline math, hypotheses,
counter-citations, hypothesize->measure->revise) is no longer structurally
capped. The scratchpad is preserved verbatim on the *trained* assistant turn
(via ``format_assistant_turn(..., think=...)``) but is summarized/dropped when a
PRIOR turn is re-rendered as cross-turn context (so context cannot explode). The
deep block is NEVER allowed to leak into the extracted kernel, and a response
with NO scratchpad parses exactly as before.

Nothing here imports torch/vllm/transformers; it is safe to import anywhere.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any

# Fenced-code and section markers used by the response contract.
_ANALYSIS = "ANALYSIS"
_PROPOSED = "PROPOSED_CHANGE"
_FULL_KERNEL = "FULL_KERNEL"

# Optional deep-reasoning scratchpad. The block is OPTIONAL and, when present,
# comes BEFORE the structured sections. It is captured separately by
# ``parse_response`` and stripped before kernel/section extraction so it can never
# contaminate the parsed kernel (it may legitimately quote forbidden ops — e.g.
# "do NOT fall back to torch.matmul" — as counter-citations).
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_THINK_RE = re.compile(r"<think\s*>(.*?)</think\s*>", re.DOTALL | re.IGNORECASE)
# Markers that end an UNCLOSED ``<think>`` (a long scratchpad whose closing tag
# the model forgot): the first structured header or a code fence.
_MARKER_RE = re.compile(
    r"(?:^|\n)\s*(?:ANALYSIS|PROPOSED_CHANGE|FULL_KERNEL)\s*:?|```",
    re.IGNORECASE,
)


SYSTEM_PROMPT = """\
You are KORE, an expert AMD GPU kernel engineer. You optimize a single GPU \
kernel across multiple turns for correctness first and speed second.

TARGET HARDWARE: AMD Instinct MI350X, arch gfx950 (CDNA4), ROCm + Triton.
Do NOT assume NVIDIA specifics (no warp==32, no cp.async, no tensor-core PTX).

GFX950 (CDNA4) DISCIPLINE:
  - Wavefront size is 64 (not 32); reason about occupancy in 64-lane wavefronts.
  - Make BLOCK_M/BLOCK_N/BLOCK_K multiples of 64 so tiles map cleanly onto \
wavefronts and the MFMA matrix cores.
  - Use tl.dot for matrix multiply so Triton emits MFMA instructions; never \
hand-roll the inner product with scalar FMAs.
  - Accumulate in fp32 (tl.float32). Inputs/outputs may be bf16/fp16/fp8, but the \
accumulator MUST be fp32 to hold precision across the reduction/K loop.
  - CDNA4 fp8 is OCP e4m3fn/e5m2 (torch.float8_e4m3fn, range +/-448), NOT the \
CDNA3 fnuz variant; CDNA4 also adds native fp6 and fp4/mxfp4 matrix ops at up to \
2x/4x the fp8 rate — prefer the OCP fp8 path (and tl.dot on fp8) for quantized GEMMs.
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

DEEP REASONING (encouraged, optional): before the sections below, you MAY open a \
<think>...</think> scratchpad and reason as long and as branchily as the problem \
deserves — there is NO length limit and it will NOT be counted against you. Do \
real engineering in it: read the profile/feedback, diagnose the TRUE bottleneck \
(compute- vs memory- vs latency-bound), estimate the roofline / arithmetic \
intensity / bytes moved / achieved-vs-peak occupancy, form a hypothesis, predict \
its effect BEFORE you make it, argue the counter-case (why it might regress: VGPR \
spills, LDS overflow, lower occupancy, bank conflicts), and revise. Everything \
inside <think> is ignored by the parser and stripped from later turns' context, \
so it is a free space to think — but it must still converge to the three required \
sections below.

RESPONSE FORMAT — after any optional <think> scratchpad, respond with these \
three sections, in this order (they are what gets parsed, so they are required):

ANALYSIS:
<the decisive synthesis: the true bottleneck WITH its evidence (roofline / \
occupancy / bandwidth), and the single change you will make and why. Be as \
thorough as the problem needs — do not truncate it artificially.>

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
## Reasoning (optional, encouraged) — BEFORE the required sections below:
You MAY open with a <think> ... </think> scratchpad and reason deeply and at
length — there is NO length cap. Do evidence-grounded engineering:
profile -> diagnose the true bottleneck (compute vs memory vs latency) ->
hypothesize -> estimate the effect (roofline / arithmetic intensity / bytes
moved / occupancy) -> transform -> argue the counter-case (VGPR spills, LDS
overflow, occupancy loss, bank conflicts) -> revise. Branch, cite concrete
numbers, and self-correct. Everything inside <think> is ignored by the kernel
parser and is stripped from later-turn context, so think freely.

## Output Format (required) — after any <think>, respond with EXACTLY these sections, in order:
ANALYSIS:
<the decisive synthesis: the current bottleneck WITH its evidence (roofline /
occupancy / bandwidth / data placement) and the one change you will make. Be as
thorough as the problem needs — do NOT truncate it to a fixed length.>

PROPOSED_CHANGE:
<one sentence naming the single change (imperative, specific)>

FULL_KERNEL:
```python
<the COMPLETE modified kernel source — full file, ready to run, not a diff>
```
"""


def _split_think(text: str) -> tuple[str, str]:
    """Split an optional leading ``<think>`` scratchpad from the structured body.

    Returns ``(think, body)`` where ``body`` is ``text`` with the scratchpad
    removed. Handles three shapes, in order:
      * a properly closed ``<think>...</think>`` block (anywhere, but by contract
        it precedes the sections);
      * an UNCLOSED ``<think>`` (a long scratchpad whose ``</think>`` the model
        dropped): everything from ``<think>`` up to the first structured header
        (ANALYSIS/PROPOSED_CHANGE/FULL_KERNEL) or code fence is treated as the
        scratchpad, so the kernel is still recovered;
      * no scratchpad at all -> ``("", text)`` (the back-compat path).
    Only the FIRST scratchpad is split (deep reasoning is a single leading block).
    """
    text = text or ""
    if not text:
        return "", ""
    m = _THINK_RE.search(text)
    if m:
        body = text[:m.start()] + text[m.end():]
        return m.group(1).strip(), body
    lo = text.lower()
    i = lo.find(_THINK_OPEN)
    if i != -1:  # unclosed <think>: cut at the first structured marker/fence.
        rest = text[i + len(_THINK_OPEN):]
        mk = _MARKER_RE.search(rest)
        stop = mk.start() if mk else len(rest)
        think = rest[:stop]
        body = text[:i] + rest[stop:]
        return think.strip(), body
    return "", text


def _structured_conclusion(text: str) -> str:
    """The terse ANALYSIS (+ PROPOSED_CHANGE) conclusion, if the text carries the
    section headers. Empty string when there are no headers to key off of."""
    a = _extract_section(text, _ANALYSIS, (_PROPOSED, _FULL_KERNEL)).strip()
    p = _extract_section(text, _PROPOSED, (_FULL_KERNEL,)).strip()
    parts: list[str] = []
    if a:
        parts.append(f"{_ANALYSIS}:\n{a}")
    if p:
        parts.append(f"{_PROPOSED}:\n{p}")
    return "\n\n".join(parts)


def summarize_cot(text: str, max_chars: int = 2000) -> str:
    """Summarize a chain-of-thought / analysis blob for CROSS-TURN CONTEXT.

    Context grows every turn; we summarize prior-turn reasoning so the transcript
    stays within the model's window. This is *smarter than a head/tail char slice*:

      1. drop the verbose ``<think>`` scratchpad — that is the part that explodes
         context; the terse conclusion is what actually needs to carry over;
      2. if a structured ANALYSIS/PROPOSED_CHANGE conclusion is present, keep
         THAT (dropping only the scratchpad) rather than a blind slice;
      3. otherwise fall back to the original head/tail elision.

    The result is always at most ``max_chars`` characters. NOTE: this is only used
    for PRIOR-turn context — the trained assistant turn keeps its full CoT (see
    ``format_assistant_turn(..., think=...)`` and ``build_transcript``).
    """
    text = (text or "").strip()
    if max_chars <= 0:
        return ""

    # (1) Drop the scratchpad; prefer the remaining conclusion if it now fits.
    _, body = _split_think(text)
    body = body.strip()
    if body and body != text:
        text = body
        if len(text) <= max_chars:
            return text

    if len(text) <= max_chars:
        return text

    # (2) Keep the structured conclusion (ANALYSIS + PROPOSED_CHANGE) if present.
    conclusion = _structured_conclusion(text)
    if conclusion:
        if len(conclusion) <= max_chars:
            return conclusion
        text = conclusion  # still too long -> elide the conclusion itself

    # (3) Head/tail elision fallback (the original, bounded behavior).
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
    """Parse a policy response into ``{analysis, proposed_change, kernel, think}``.

    The OPTIONAL ``<think>`` deep-reasoning scratchpad is split out FIRST and
    returned under the additive ``think`` key; ANALYSIS / PROPOSED_CHANGE / kernel
    are then parsed from the think-stripped body, so the scratchpad can never leak
    into the structured sections or the extracted kernel. Backward compatible:
      * a response with NO scratchpad yields ``think == ""`` and the exact same
        ``analysis`` / ``proposed_change`` / ``kernel`` as before;
      * the three legacy keys are always present, so existing callers (which read
        ``kernel`` / ``analysis`` / ``proposed_change``) are unaffected.

    Robust to missing sections and to models that emit the FULL_KERNEL as a bare
    ```python fenced block without the literal header. ``kernel`` is the raw
    source with no fences.
    """
    text = text or ""
    think, body = _split_think(text)

    analysis = _extract_section(body, _ANALYSIS, (_PROPOSED, _FULL_KERNEL))
    proposed = _extract_section(body, _PROPOSED, (_FULL_KERNEL,))
    kernel = _extract_kernel(body)

    return {
        "analysis": analysis.strip(),
        "proposed_change": proposed.strip(),
        "kernel": kernel.strip(),
        "think": think.strip(),
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
    """Pull the kernel source out of the FULL_KERNEL block or a fenced block.

    The optional ``<think>`` scratchpad is stripped FIRST so a scratchpad that
    quotes ``FULL_KERNEL:`` or contains its own ```python fence (models often
    draft code while reasoning) can never be mistaken for the real kernel. If the
    think-stripped body yields nothing, we fall back to the ORIGINAL text so a
    think-only response that carried code still parses exactly as it used to.
    """
    text = text or ""
    _, body = _split_think(text)

    # Prefer a fenced block that appears after the FULL_KERNEL header.
    fk = re.search(rf"{_FULL_KERNEL}\s*:?", body, flags=re.IGNORECASE)
    scope = body[fk.end():] if fk else body
    fenced = re.search(r"```(?:python|py)?\s*\n(.*?)```", scope, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)
    # No fence: take everything after the header as the kernel.
    if fk:
        return scope
    # Any fenced block in the think-stripped body.
    any_fence = re.search(r"```(?:python|py)?\s*\n(.*?)```", body, flags=re.DOTALL)
    if any_fence:
        return any_fence.group(1)
    # Last resort: a fenced block anywhere (incl. inside the scratchpad) — keeps
    # the pre-deep-CoT behavior for responses that ONLY carried code in <think>.
    orig_fence = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL)
    return orig_fence.group(1) if orig_fence else ""


def format_assistant_turn(analysis: str, proposed_change: str, kernel: str,
                          *, think: str = "") -> str:
    """Render the CANONICAL assistant response (single source of truth).

    Every training row and every inference/transcript turn uses this exact shape:
    ``[<think>…</think>] / ANALYSIS: … / PROPOSED_CHANGE: … / FULL_KERNEL: ```python … `````.
    The PROPOSED_CHANGE section is omitted only when empty (e.g. a first-turn /
    repair demo may carry just analysis + kernel). This is what ``parse_response``
    reads back.

    ``think`` is the OPTIONAL deep-reasoning scratchpad. It is keyword-only and
    defaults to ``""``, so every existing positional call is byte-for-byte
    unchanged (no scratchpad -> the render still starts with ``ANALYSIS:``). Pass
    it to PRESERVE the full CoT on a *trained* assistant turn; leave it empty when
    rendering a PRIOR turn as summarized context.
    """
    parts: list[str] = []
    think = (think or "").strip()
    if think:
        parts.append(f"{_THINK_OPEN}\n{think}\n{_THINK_CLOSE}")
    parts.append(f"{_ANALYSIS}:\n{(analysis or '').strip()}".rstrip())
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


# --------------------------------------------------------------------------- #
# Claim <-> code consistency gate (PURE / CPU-only; any stage can call it).
#
# Frontier CoT is only trustworthy if the code backs the words. This gate answers
# a single question: does the change the ANALYSIS *names* actually appear in the
# prev->new kernel diff? It lets stages drop "describe-but-don't-implement" turns
# (an ANALYSIS that claims "bump num_warps to 8" but ships an unchanged kernel).
# --------------------------------------------------------------------------- #

# Canonical claim -> natural-language keywords that NAME it in the ANALYSIS.
_CONSISTENCY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "num_warps": ("num_warps", "warp", "warps"),
    "num_stages": ("num_stages", "num stages", "stage", "stages", "pipeline",
                   "software pipelin", "double-buffer", "double buffer"),
    "block": ("block_m", "block_n", "block_k", "block size", "block-size",
              "tile size", "tile", "tiling", "blocking", "retile"),
    "group": ("group_m", "group size", "grouping", "grouped", "swizzle",
              "l2 grouping", "supergroup"),
    "vectorize": ("vectoriz", "vector width", "coalesc", "wide load",
                  "128-bit", "multiple_of", "max_contiguous"),
    "tl.dot": ("tl.dot", "mfma", "matrix core", "matrix-core", "mma",
               "tensor core", "dot product", "wmma"),
    "lds": ("lds", "shared memory", "shared-memory", "local data share",
            "smem", "scratchpad"),
    "accumulator": ("fp32 accum", "float32 accum", "accumulat", "tl.float32",
                    "acc dtype", "accumulator"),
    "mask": ("boundary check", "boundary-check", "masking", "mask", "predicat"),
    "cache": ("cache_modifier", "cache modifier", "eviction", "cache hint"),
}
# Canonical claim -> numeric knobs whose VALUE CHANGE proves it was implemented.
_CONSISTENCY_NUMERIC_EVIDENCE: dict[str, set[str]] = {
    "num_warps": {"num_warps"},
    "num_stages": {"num_stages"},
    "lds": {"num_stages"},  # LDS double-buffering on CDNA is driven by num_stages
    "block": {"block_m", "block_n", "block_k", "group_m"},
    "group": {"group_m", "group_size_m"},
    "vectorize": {"block_k"},
}
# Canonical claim -> diff tokens whose ADD/REMOVE proves it (structural changes a
# numeric scan misses: introducing tl.dot, masks, cache hints, LDS staging, ...).
_CONSISTENCY_TOKEN_EVIDENCE: dict[str, set[str]] = {
    "num_warps": {"num_warps"},
    "num_stages": {"num_stages"},
    "block": {"block_m", "block_n", "block_k", "group_m"},
    "group": {"group_m", "group_size_m", "swizzle"},
    "vectorize": {"multiple_of", "max_contiguous", "block_k"},
    "tl.dot": {"dot"},
    "lds": {"lds", "shared", "smem"},
    "accumulator": {"float32"},
    "mask": {"mask", "where"},
    "cache": {"cache_modifier"},
}

_NUMERIC_KNOBS = (
    "num_warps", "num_stages", "block_m", "block_n", "block_k",
    "group_m", "group_size_m", "waves_per_eu", "num_ctas",
    "matrix_instr_nonkdim", "kpack",
)
_NUMERIC_KNOB_RE = re.compile(
    r"\b(num_warps|num_stages|BLOCK_M|BLOCK_N|BLOCK_K|GROUP_M|GROUP_SIZE_M|"
    r"waves_per_eu|num_ctas|matrix_instr_nonkdim|kpack)\b"
    r"\s*(?::\s*tl\.constexpr\s*)?=\s*(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class ChangeConsistency:
    """Result of :func:`check_change_consistency` — truthy iff consistent.

    Usable directly as a gate (``if check_change_consistency(...):``) via
    ``__bool__``, while exposing the per-knob breakdown for stricter callers:
      * ``claimed``  — knobs the ANALYSIS named;
      * ``applied``  — claimed knobs that actually changed in the diff;
      * ``missing``  — claimed knobs that did NOT change (require ``not missing``
                       for an all-knobs-must-land gate).
    """

    consistent: bool
    claimed: tuple[str, ...] = ()
    applied: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.consistent


def _src_tokens(s: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", s or "")


def _changed_token_set(a: str, b: str) -> set[str]:
    """Identifiers/numbers added or removed between two kernel sources."""
    at, bt = _src_tokens(a), _src_tokens(b)
    sm = difflib.SequenceMatcher(a=at, b=bt, autojunk=False)
    out: set[str] = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        out.update(at[i1:i2])
        out.update(bt[j1:j2])
    return out


def _numeric_knob_values(src: str) -> dict[str, int]:
    """Parse tunable numeric-knob values (kwarg, constexpr, and tuple forms)."""
    out: dict[str, int] = {}
    for m in _NUMERIC_KNOB_RE.finditer(src or ""):
        out[m.group(1).lower()] = int(m.group(2))
    for m in re.finditer(r"^[ \t]*([A-Za-z_][\w,\s]*?)=\s*([\d,\s]+)$", src or "",
                         re.MULTILINE):
        names = [n.strip().lower() for n in m.group(1).split(",")]
        vals = [v.strip() for v in m.group(2).split(",")]
        if len(names) == len(vals):
            for n, v in zip(names, vals):
                if n in _NUMERIC_KNOBS and v.isdigit():
                    out[n] = int(v)
    return out


def _changed_numeric_knobs(prev_src: str, new_src: str) -> set[str]:
    """Numeric knobs whose value (or presence) differs between two sources."""
    pv, nv = _numeric_knob_values(prev_src), _numeric_knob_values(new_src)
    return {k for k in set(pv) | set(nv) if pv.get(k) != nv.get(k)}


def check_change_consistency(analysis: str, prev_kernel: str, new_kernel: str,
                             proposed_change: str = "") -> ChangeConsistency:
    """Does the change the ANALYSIS *names* actually appear in the kernel diff?

    A pure, CPU-only claim<->code gate. It scans ``analysis`` (+ optional
    ``proposed_change``) for concrete knobs it recognizes — num_warps, num_stages,
    BLOCK_*/tiling, GROUP_M/swizzle, vectorization, tl.dot/MFMA, LDS/shared-memory,
    fp32 accumulator, masking, cache modifiers — and checks each against the
    ``prev_kernel`` -> ``new_kernel`` diff, using BOTH a numeric value change and
    the identifying token being added/removed (so it catches new-knob /
    structural rewrites a pure value scan would miss).

    Returns a :class:`ChangeConsistency` (truthy iff consistent):
      * no concrete knob named   -> consistent (a vague claim is not falsifiable);
      * >= 1 named knob changed   -> consistent (the claim is supported by code);
      * knob(s) named but NONE changed -> INCONSISTENT (unsupported claim).

    Symmetric in the sense that an EMPTY diff (``prev_kernel == new_kernel``) with
    any named knob is always inconsistent — the textbook "describe but don't
    implement" turn.
    """
    claim = f"{analysis or ''}\n{proposed_change or ''}".lower()
    claimed = [k for k, kws in _CONSISTENCY_KEYWORDS.items()
               if any(w in claim for w in kws)]
    if not claimed:
        return ChangeConsistency(True, (), (), ())

    changed_num = _changed_numeric_knobs(prev_kernel, new_kernel)
    changed_tok = {t.lower() for t in _changed_token_set(prev_kernel, new_kernel)}

    applied: list[str] = []
    missing: list[str] = []
    for k in claimed:
        num_ok = bool(_CONSISTENCY_NUMERIC_EVIDENCE.get(k, set()) & changed_num)
        tok_ok = bool(_CONSISTENCY_TOKEN_EVIDENCE.get(k, set()) & changed_tok)
        (applied if (num_ok or tok_ok) else missing).append(k)

    return ChangeConsistency(bool(applied), tuple(claimed),
                             tuple(applied), tuple(missing))


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
    """Reconstruct a compact PRIOR-turn assistant message for cross-turn context.

    This is the CONTEXT-carryover path: the ``<think>`` scratchpad is dropped and
    the ANALYSIS is summarized (``format_assistant_turn`` is called WITHOUT
    ``think=``), so re-rendering old turns cannot explode the context window. The
    durable artifacts — the PROPOSED_CHANGE and the full FULL_KERNEL source — are
    kept intact. (The freshly generated / trained turn is NOT rendered here, so it
    retains its full CoT.)
    """
    if "response" in turn and turn["response"]:
        parsed = parse_response(turn["response"])  # splits + drops <think>
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
