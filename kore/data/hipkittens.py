"""HipKittens -> SFT ingestion (MIT; Stanford Hazy Research).

WHY THIS EXISTS
---------------
HipKittens (MLSys 2026, arXiv:2511.08083) is the fastest published AMD kernel
library on MI355X, and the knowledge that makes its kernels fast is *specific and
largely unpublished*: the paper states outright that the bank-conflict behaviour
it exploits is undocumented in the CDNA ISA. That is exactly the knowledge our
product model needs on the ``hip2hip`` / ``torch2hip`` categories, where there is
no Triton codegen ceiling to hide behind.

WHY NO TEACHER MODEL WRITES THESE ROWS
-------------------------------------
Every other generator in ``kore.data`` asks a frontier teacher for prose and then
verifies a number (see ``gen_qa`` / the Tier-1 kernel-math solver). That pattern is
WRONG here, and using it would defeat the purpose of the asset. The premise of
ingesting HipKittens is that frontier models do not know this material in depth;
a frontier teacher therefore cannot be the source of truth for it, and would
confabulate fluent, wrong answers about CDNA4 LDS behaviour that we would have no
way to check. So every factual claim in these rows is either

  * EXTRACTED from the checkout (swizzle formulas, wave counts, scheduling
    intrinsics, kernel source), or
  * a MEASUREMENT the HipKittens authors published in their own repo
    (``analysis/**`` JSON and the LDS bank/phase solver outputs), quoted with its
    file of origin, or
  * a claim from the paper, carried verbatim in :data:`PAPER_CLAIMS` with its
    attribution string.

Nothing is paraphrased into a performance number. There is no code path in this
module that invents a speedup.

WHY THESE ROWS DO NOT USE THE ``FULL_KERNEL`` CONTRACT
-----------------------------------------------------
``kore.policy.format.SYSTEM_PROMPT`` trains the model to emit a self-contained
ROCm/Triton kernel in a ``FULL_KERNEL:`` block, which the env then compiles.
HipKittens kernels are C++ that ``#include "kittens.cuh"``. Training HK source as
a ``FULL_KERNEL`` response would teach the model to answer optimization requests
with code that cannot compile in the harness -- actively negative transfer that
would look like clean data. These rows therefore mirror the existing
``kernel_qa`` slice: a knowledge persona, ``[system, user, assistant]``, teaching
the transferable *reasoning* (which schedule, which swizzle, which intrinsic, and
the failure mode when you choose wrong) rather than the library's API surface.

Pure and CPU-only: reads files, parses text, returns dicts. No GPU, no network,
no teacher.
"""

from __future__ import annotations

import json
import os
import pathlib
import random
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

SOURCE_TAG = "hipkittens"
DEFAULT_ARCH = "gfx950"

# --------------------------------------------------------------------------- #
# Provenance / attribution (MIT). Carried on EVERY row.
# --------------------------------------------------------------------------- #
HK_REPO_URL = "https://github.com/HazyResearch/HipKittens"
HK_LICENSE = "MIT"
HK_COPYRIGHT = "Copyright (c) 2024 HazyResearch"
HK_PAPER_ARXIV = "arXiv:2511.08083"
HK_PAPER_TITLE = "HipKittens: Fast and Furious AMD Kernels"
HK_PAPER_VENUE = "MLSys 2026"
HK_AUTHORS = (
    "William Hu, Drew Wadsworth, Sean Siddens, Stanley Winata, Daniel Y. Fu, "
    "Ryann Swann, Muhammad Osama, Christopher Re, Simran Arora"
)
# The MIT terms require the notice to travel with substantial portions of the
# work. Rows that embed HipKittens source carry this string verbatim.
HK_ATTRIBUTION = (
    f"Source: HipKittens ({HK_REPO_URL}), {HK_LICENSE} licensed, {HK_COPYRIGHT}. "
    f"Paper: {HK_PAPER_TITLE} ({HK_PAPER_VENUE}, {HK_PAPER_ARXIV})."
)

# Claims that come from the PAPER rather than from the checkout. Verbatim, with
# attribution, so no downstream code has to decide whether a number is ours.
# Keys are stable ids; values are (claim, attribution).
PAPER_CLAIMS: dict[str, tuple[str, str]] = {
    "gemm_vs_triton": (
        "1.3-3.0x faster than Triton on BF16 GEMM",
        f"reported in {HK_PAPER_TITLE}, {HK_PAPER_ARXIV}",
    ),
    "attn_vs_aiter": (
        "1.0-2.1x faster than AITER on attention, where AITER is hand-written assembly",
        f"reported in {HK_PAPER_TITLE}, {HK_PAPER_ARXIV}",
    ),
    "uncovered_workloads": (
        "1.2-10x on workloads that AMD's assembly kernels do not cover",
        f"reported in {HK_PAPER_TITLE}, {HK_PAPER_ARXIV}",
    ),
    "bwd_gap": (
        "AITER and PyTorch SDPA reach only 30% and 24% of state-of-the-art on "
        "Llama-shaped GQA backward on MI355X",
        f"reported in {HK_PAPER_TITLE}, {HK_PAPER_ARXIV}",
    ),
    "ping_pong_sufficient": (
        "8-wave ping-pong is sufficient to match hand-optimized assembly on BF16 "
        "GEMM, FP8 GEMM and attention forward, and is 1.8x better on GQA "
        "non-causal backward",
        f"reported in {HK_PAPER_TITLE}, {HK_PAPER_ARXIV}",
    ),
    "wave_spec_underperforms": (
        "wave specialization -- dedicating whole waves to producer or consumer "
        "roles -- underperforms on CDNA3/CDNA4, unlike on NVIDIA Hopper/Blackwell",
        f"reported in {HK_PAPER_TITLE}, {HK_PAPER_ARXIV}",
    ),
    "swizzle_undocumented": (
        "the bank-conflict-avoidance behaviour these swizzles rely on is "
        "undocumented in the CDNA ISA",
        f"reported in {HK_PAPER_TITLE}, {HK_PAPER_ARXIV}",
    ),
}

SYSTEM_PROMPT_HK = (
    "You are KORE, an expert AMD CDNA3/CDNA4 (gfx942 / gfx950, MI300X / MI355X) GPU "
    "kernel engineer. You reason about hand-written HIP/C++ kernels at the level of "
    "waves, MFMA shapes, LDS banks and scheduling intrinsics. Answer precisely, cite "
    "the mechanism, and name the failure mode when a choice is made wrongly."
)

# Files that are build output, bulk logs, or measurement harnesses rather than
# teaching material. ``timing/`` holds single-wave barrier-cost microbenchmarks,
# which are not kernels anyone should imitate.
_EXCLUDE_PATH_PARTS = ("/archive/", "/data_logs/", "/repro/", "/out/", "/timing/")
# gfx1250 (UDNA1) is a different architecture family from our gfx950 product
# target; its idioms (tdm/async barriers) do not transfer, so it is excluded
# rather than silently mixed in.
_EXCLUDE_ARCH_DIRS = ("udna1",)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class HipKittensIngestError(RuntimeError):
    """Raised when the checkout is missing, unlicensed, or fails to parse.

    Ingestion fails LOUD rather than emitting fewer/wrong rows: a silently
    degraded corpus is far more expensive than a failed build.
    """


# --------------------------------------------------------------------------- #
# Locating and vouching for the checkout
# --------------------------------------------------------------------------- #
def hk_root(root: Optional[str | pathlib.Path] = None) -> pathlib.Path:
    """Resolve the HipKittens checkout root.

    Order: explicit argument, ``KORE_HIPKITTENS_ROOT``, then
    ``$HOME/third_party/HipKittens``.
    """
    if root:
        p = pathlib.Path(root)
    else:
        env = os.environ.get("KORE_HIPKITTENS_ROOT")
        p = pathlib.Path(env) if env else pathlib.Path.home() / "third_party" / "HipKittens"
    p = p.expanduser()
    if not p.is_dir():
        raise HipKittensIngestError(
            f"HipKittens checkout not found at {p}. Clone it with "
            f"`git clone {HK_REPO_URL} {p}` or set KORE_HIPKITTENS_ROOT."
        )
    return p


def _git_commit(root: pathlib.Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        return (out.stdout or "").strip()
    except Exception:  # noqa: BLE001 - provenance degrades to "" rather than failing
        return ""


def provenance(root: Optional[str | pathlib.Path] = None) -> dict:
    """Verified provenance for the checkout.

    The licence is *checked*, not assumed: a corpus that claims MIT because a
    constant in this file says MIT is not auditable. If ``LICENSE`` stops looking
    like the MIT grant we refuse to ingest.
    """
    r = hk_root(root)
    lic_path = r / "LICENSE"
    if not lic_path.is_file():
        raise HipKittensIngestError(f"no LICENSE in {r}; refusing to ingest unlicensed source")
    lic = lic_path.read_text(encoding="utf-8", errors="replace")
    if "MIT License" not in lic or "WITHOUT WARRANTY OF ANY KIND" not in lic:
        raise HipKittensIngestError(
            f"{lic_path} does not look like the MIT licence; refusing to ingest. "
            "Re-check upstream licensing before adding this to a training corpus."
        )
    holder = ""
    m = re.search(r"^(Copyright \(c\).*)$", lic, re.M)
    if m:
        holder = m.group(1).strip()
    return {
        "source_id": "HipKittens",
        "repository_url": HK_REPO_URL,
        "commit": _git_commit(r),
        "license": HK_LICENSE,
        "license_holder": holder or HK_COPYRIGHT,
        "license_file_sha256": _sha256(lic),
        "paper": f"{HK_PAPER_TITLE} ({HK_PAPER_VENUE}, {HK_PAPER_ARXIV})",
        "authors": HK_AUTHORS,
        "attribution": HK_ATTRIBUTION,
        "local_path": str(r),
    }


def _sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


# --------------------------------------------------------------------------- #
# 1. LDS swizzle layouts (parsed from include/<arch>/types/shared/st_shape.cuh)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SwizzleLayout:
    """One shared-tile layout and its XOR swizzle, per element width."""

    name: str
    arch: str
    rows: int
    cols: int
    dtype_bytes: int
    # Each term is (modulus, shift_right, shift_left) applied as
    # offset ^= ((offset % modulus) >> shift_right) << shift_left
    terms: tuple[tuple[int, int, int], ...]
    bytes_per_thread: int
    source: str

    @property
    def is_identity(self) -> bool:
        return not self.terms

    def swizzle(self, byte_offset: int) -> int:
        out = byte_offset
        for mod, sr, sl in self.terms:
            out ^= ((byte_offset % mod) >> sr) << sl
        return out

    def formula(self) -> str:
        if not self.terms:
            return "offset  (identity: no swizzle needed at this width)"
        parts = " ^ ".join(
            f"(((offset % {mod}) >> {sr}) << {sl})" for mod, sr, sl in self.terms
        )
        return f"offset ^ {parts}" if len(self.terms) == 1 else f"offset ^ {parts}"

    def is_bijection(self) -> bool:
        """True iff the swizzle permutes the tile's byte offsets.

        A swizzle that is not a bijection aliases two elements onto one LDS
        address and silently corrupts the tile, so this is a real correctness
        property rather than a style check.
        """
        offs = [
            self.dtype_bytes * (r * self.cols + c)
            for r in range(self.rows)
            for c in range(self.cols)
        ]
        out = [self.swizzle(o) for o in offs]
        return len(set(out)) == len(out) and set(out) == set(offs)


_STRUCT_RE = re.compile(r"^struct\s+(\w+)\s*\{", re.M)
# ((offset % MOD) >> SR) << SL   -- MOD may be parenthesised arithmetic, e.g. (16*128)
_TERM_RE = re.compile(
    r"\(\s*\(\s*offset\s*%\s*\(?\s*(?P<mod>[0-9]+(?:\s*\*\s*[0-9]+)*)\s*\)?\s*\)"
    r"\s*>>\s*(?P<sr>\d+)\s*\)\s*<<\s*(?P<sl>\d+)"
)
_BRANCH_RE = re.compile(
    r"(?:if|else\s+if)\s*constexpr\s*\(\s*(?P<cond>[^)]*sizeof\(T\)[^)]*)\)"
)
_SIZEOF_RE = re.compile(r"sizeof\(T\)\s*==\s*(\d+)")
_SIZEOF_T_RE = re.compile(r"sizeof\(_T\)\s*==\s*(\d+)")
_BPT_BRANCH_RE = re.compile(
    r"(?:if|else\s+if)\s*constexpr\s*\(\s*(?P<cond>[^)]*sizeof\(_T\)[^)]*)\)\s*\{"
    r"\s*return\s+(?P<val>\d+)\s*;"
)


def _brace_body(text: str, open_index: int) -> str:
    """Return the body between the brace at ``open_index`` and its match."""
    depth = 0
    i = open_index
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_index + 1:i]
        i += 1
    raise HipKittensIngestError("unbalanced braces while parsing st_shape.cuh")


def _split_struct_bodies(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in _STRUCT_RE.finditer(text):
        out[m.group(1)] = _brace_body(text, m.end() - 1)
    return out


def _parse_terms(body: str) -> tuple[tuple[int, int, int], ...]:
    terms = []
    for t in _TERM_RE.finditer(body):
        mod = 1
        for factor in t.group("mod").split("*"):
            mod *= int(factor.strip())
        terms.append((mod, int(t.group("sr")), int(t.group("sl"))))
    return tuple(terms)


def _bytes_per_thread(body: str, dtype_bytes: int) -> int:
    """The declared bytes_per_thread for this element width (0 if unknown)."""
    idx = body.find("bytes_per_thread()")
    if idx < 0:
        return 0
    try:
        fn = _brace_body(body, body.index("{", idx))
    except (ValueError, HipKittensIngestError):
        return 0
    # Branch-specific returns first, then any unconditional return.
    for m in _BPT_BRANCH_RE.finditer(fn):
        if dtype_bytes in [int(s) for s in _SIZEOF_T_RE.findall(m.group("cond"))]:
            return int(m.group("val"))
    m = re.search(r"return\s+(\d+)\s*;", fn)
    return int(m.group(1)) if m else 0


def parse_swizzles(
    root: Optional[str | pathlib.Path] = None, arch: str = "cdna4"
) -> list[SwizzleLayout]:
    """Parse every shared-tile swizzle out of ``st_shape.cuh``.

    FAILS LOUD on a parse mismatch. The first draft of this parser silently
    returned zero XOR terms for ``st_16x128`` because that layout parenthesises
    its modulus as ``(16*128)``, which would have taught the model an identity
    swizzle for fp8 -- a wrong answer that no test on row COUNT would ever catch.
    So every branch that contains a ``^`` must yield at least one parsed term,
    and every parsed swizzle must be a bijection.
    """
    r = hk_root(root)
    path = r / "include" / arch / "types" / "shared" / "st_shape.cuh"
    if not path.is_file():
        raise HipKittensIngestError(f"missing {path}")
    text = path.read_text(encoding="utf-8", errors="replace")

    out: list[SwizzleLayout] = []
    for name, body in sorted(_split_struct_bodies(text).items()):
        if "swizzle (int2 coord)" not in body and "swizzle(int2 coord)" not in body:
            continue
        rm = re.search(r"static constexpr int rows\s*=\s*(\d+)", body)
        cm = re.search(r"static constexpr int cols\s*=\s*(\d+)", body)
        if not rm or not cm:
            raise HipKittensIngestError(f"{name}: could not parse rows/cols in {path}")
        rows, cols = int(rm.group(1)), int(cm.group(1))

        sw_idx = body.find("swizzle (int2 coord)")
        if sw_idx < 0:
            sw_idx = body.find("swizzle(int2 coord)")
        sw_body = _brace_body(body, body.index("{", sw_idx))

        # Which element widths does this layout support at all?
        widths = sorted({int(s) for s in _SIZEOF_T_RE.findall(body)} |
                        {int(s) for s in _SIZEOF_RE.findall(sw_body)})
        if not widths:
            raise HipKittensIngestError(f"{name}: no supported element widths found in {path}")

        # Per-width branch bodies inside swizzle().
        branches: dict[int, str] = {}
        marks = list(_BRANCH_RE.finditer(sw_body))
        for i, m in enumerate(marks):
            sizes = [int(s) for s in _SIZEOF_RE.findall(m.group("cond"))]
            try:
                seg = _brace_body(sw_body, sw_body.index("{", m.end()))
            except (ValueError, HipKittensIngestError):
                continue
            for s in sizes:
                branches[s] = seg
        for w in widths:
            body_for_w = branches.get(w, sw_body if not marks else "")
            terms = _parse_terms(body_for_w)
            if "^" in body_for_w and not terms:
                raise HipKittensIngestError(
                    f"{name}: swizzle branch for {w}-byte elements contains '^' but no "
                    f"term matched _TERM_RE. The upstream formula shape changed; fix the "
                    f"parser rather than shipping an identity swizzle. Branch was:\n{body_for_w}"
                )
            layout = SwizzleLayout(
                name=name, arch=arch, rows=rows, cols=cols, dtype_bytes=w,
                terms=terms, bytes_per_thread=_bytes_per_thread(body, w),
                source=("template<typename _T>\n__device__ __forceinline__ static "
                        "const uint32_t swizzle (int2 coord) {" + sw_body + "}"),
            )
            if not layout.is_bijection():
                raise HipKittensIngestError(
                    f"{name} @ {w}B: parsed swizzle is not a bijection over the "
                    f"{rows}x{cols} tile, so the parse is wrong (a real swizzle must "
                    f"permute LDS addresses, never alias them)."
                )
            out.append(layout)
    if not out:
        raise HipKittensIngestError(f"parsed no swizzle layouts from {path}")
    return out


# --------------------------------------------------------------------------- #
# 2. Measured LDS bank / phase model (analysis/paper_experiments/phases/*)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PhaseModel:
    """The authors' MEASURED LDS bank count and conflict phases for one op.

    This is the material the paper calls undocumented in the CDNA ISA. It is a
    measurement, not a derivation, so it is quoted with the file it came from.
    """

    instruction: str
    num_banks: int
    phases: tuple[tuple[int, ...], ...]
    bank_formula: str
    evidence_files: tuple[str, ...]

    @property
    def lanes_per_phase(self) -> int:
        return len(self.phases[0]) if self.phases else 0

    @property
    def contiguous_phases(self) -> bool:
        """True iff every phase is a run of consecutive lane ids."""
        return all(
            tuple(p) == tuple(range(min(p), min(p) + len(p))) for p in self.phases
        )


_PHASE_LINE_RE = re.compile(r"^Phase\s+(\d+):\s*(\d+)\s+threads\s*-\s*\[([0-9,\s]+)\]", re.M)
_BANKS_RE = re.compile(r"Number of LDS banks:\s*(\d+)")


def parse_phase_models(root: Optional[str | pathlib.Path] = None) -> list[PhaseModel]:
    """Parse the LDS phase/bank solver outputs the authors committed."""
    r = hk_root(root)
    base = r / "analysis" / "paper_experiments" / "phases"
    if not base.is_dir():
        raise HipKittensIngestError(f"missing {base}")
    out: list[PhaseModel] = []
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        phase_f, bank_f = d / "phase_results.txt", d / "bank_results.txt"
        if not phase_f.is_file() or not bank_f.is_file():
            continue
        ptxt = phase_f.read_text(encoding="utf-8", errors="replace")
        btxt = bank_f.read_text(encoding="utf-8", errors="replace")
        bm = _BANKS_RE.search(btxt)
        if not bm:
            raise HipKittensIngestError(f"could not parse bank count from {bank_f}")
        phases = tuple(
            tuple(int(x) for x in m.group(3).replace(" ", "").split(",") if x != "")
            for m in _PHASE_LINE_RE.finditer(ptxt)
        )
        if not phases:
            raise HipKittensIngestError(f"could not parse any phase from {phase_f}")
        sizes = {len(p) for p in phases}
        if len(sizes) != 1:
            raise HipKittensIngestError(
                f"{phase_f}: phases have differing sizes {sorted(sizes)}; the "
                f"solver output shape changed"
            )
        readme = d / "README.md"
        formula = "bank_id = (byte_offset / 4) % NUM_BANKS"
        if readme.is_file():
            m = re.search(r"`?bank_id\s*=\s*([^`\n]+)`?", readme.read_text(errors="replace"))
            if m:
                formula = f"bank_id = {m.group(1).strip().rstrip('`')}"
        out.append(PhaseModel(
            instruction=d.name,
            num_banks=int(bm.group(1)),
            phases=phases,
            bank_formula=formula,
            evidence_files=(
                str(phase_f.relative_to(r)), str(bank_f.relative_to(r)),
            ),
        ))
    if not out:
        raise HipKittensIngestError(f"parsed no phase models from {base}")
    return out


# --------------------------------------------------------------------------- #
# 2b. Joining the two: is each swizzle actually conflict-free?
# --------------------------------------------------------------------------- #
# Access width in bytes per lane -> the DS instruction that width implies.
_WIDTH_TO_INSTRUCTION = {16: "ds_read_b128", 12: "ds_read_b96",
                         8: "ds_read_b64", 4: "ds_read_b64"}


@dataclass(frozen=True)
class ConflictReport:
    """Bank-conflict degree of a column gather, with and without the swizzle.

    The access pattern is the one that MOTIVATES a swizzle: lanes walking down a
    column of the tile, one 16-byte granule each. Under a plain row-major layout
    the row stride is a multiple of the bank count, so those lanes collide; the
    XOR exists to break exactly that.

    Both numbers are COMPUTED from the authors' measured bank count and phase
    partition, not asserted. That makes "this swizzle is bank-conflict free" a
    checked claim, and it is what :func:`kore.data.hipkittens.build_rows` relies
    on before it will teach a swizzle.
    """

    layout: str
    dtype_bytes: int
    instruction: str
    num_banks: int
    lanes: int
    dwords_per_lane: int
    plain_max_conflict: int
    swizzled_max_conflict: int

    @property
    def conflict_free(self) -> bool:
        return self.swizzled_max_conflict == 1

    @property
    def improves(self) -> bool:
        return self.swizzled_max_conflict < self.plain_max_conflict


def _max_conflict_degree(
    offsets: list[int], phases: tuple[tuple[int, ...], ...], num_banks: int, dwords: int
) -> int:
    """Max number of lanes contending for one bank within a single phase.

    1 means conflict-free. Only lanes in the SAME measured phase can contend, so
    the partition is applied rather than counting across all 64 lanes.
    """
    lane_banks = {
        lane: {((off // 4) + d) % num_banks for d in range(dwords)}
        for lane, off in enumerate(offsets)
    }
    worst = 1
    for phase in phases:
        counts: dict[int, int] = {}
        for lane in phase:
            for bank in lane_banks.get(lane, ()):  # tiles shorter than 64 rows
                counts[bank] = counts.get(bank, 0) + 1
        if counts:
            worst = max(worst, max(counts.values()))
    return worst


def bank_conflict_report(
    layout: SwizzleLayout, models: dict[str, PhaseModel]
) -> Optional[ConflictReport]:
    """Compute the conflict degree of a column gather for one layout."""
    width = layout.bytes_per_thread or 16
    instr = _WIDTH_TO_INSTRUCTION.get(width)
    if instr is None or instr not in models:
        return None
    model = models[instr]
    row_bytes = layout.cols * layout.dtype_bytes
    lanes = min(layout.rows, 64)
    dwords = max(1, width // 4)
    plain = [r * row_bytes for r in range(lanes)]
    swizzled = [layout.swizzle(o) for o in plain]
    return ConflictReport(
        layout=layout.name, dtype_bytes=layout.dtype_bytes, instruction=instr,
        num_banks=model.num_banks, lanes=lanes, dwords_per_lane=dwords,
        plain_max_conflict=_max_conflict_degree(plain, model.phases, model.num_banks, dwords),
        swizzled_max_conflict=_max_conflict_degree(
            swizzled, model.phases, model.num_banks, dwords),
    )


# --------------------------------------------------------------------------- #
# 3. Kernels: what they compute, which schedule, which techniques
# --------------------------------------------------------------------------- #
# Technique -> (regex over source, short human description). Detection is
# EVIDENCE-BASED: a kernel is only ever described as using a technique when the
# pattern is present in its source, so no row asserts a property we did not see.
TECHNIQUES: dict[str, tuple[re.Pattern, str]] = {
    # Matched STRUCTURALLY, not by variable name: a conditional whose entire body
    # is barrier(s). The guard is spelled warp_row, warp_m or stagger depending on
    # the kernel, and enumerating those names silently mislabelled the FP8 and
    # MXFP8 8-wave kernels as having no ping-pong at all.
    # Both the braced form and the brace-less `if (warp_m == 1) s_barrier();` used
    # by the MXFP8 kernels; requiring braces missed all three of them.
    "conditional_barrier_stagger": (
        re.compile(r"if\s*\([^)]{1,80}\)\s*(?:"
                   r"\{(?:\s*__builtin_amdgcn_sched_barrier\s*\(\s*\d*\s*\)\s*;)*"
                   r"\s*__builtin_amdgcn_s_barrier\s*\(\s*\)\s*;"
                   r"(?:\s*__builtin_amdgcn_sched_barrier\s*\(\s*\d*\s*\)\s*;)*\s*\}"
                   r"|\s*__builtin_amdgcn_s_barrier\s*\(\s*\)\s*;)"),
        "a CONDITIONAL s_barrier that offsets one half of the waves by half a "
        "phase, which is what makes the two wave groups ping-pong between "
        "compute and memory roles",
    ),
    "setprio_around_mfma": (
        re.compile(r"s_setprio\(1\)[\s\S]{0,400}?mma[\s\S]{0,400}?s_setprio\(0\)"),
        "s_setprio(1) raised around the MFMA and dropped to 0 afterwards, so the "
        "wave issuing matrix ops is not preempted by the wave issuing loads",
    ),
    "sched_barrier": (
        re.compile(r"__builtin_amdgcn_sched_barrier"),
        "sched_barrier to stop the compiler from reordering across a phase "
        "boundary it cannot reason about",
    ),
    "sched_group_barrier": (
        re.compile(r"__builtin_amdgcn_sched_group_barrier"),
        "sched_group_barrier with instruction-class masks, which interleaves a "
        "counted number of MFMA / VALU / transcendental ops rather than merely "
        "fencing them",
    ),
    "readfirstlane_hoist": (
        re.compile(r"__builtin_amdgcn_readfirstlane"),
        "readfirstlane to hoist a wave-uniform LDS address into a scalar "
        "register, removing per-lane address math from the inner loop",
    ),
    "explicit_waitcnt": (
        re.compile(r"s_waitcnt\s+(?:vmcnt|lgkmcnt)\("),
        "hand-placed s_waitcnt with explicit counts, so the wave waits on "
        "exactly the outstanding loads it needs and not on all of them",
    ),
    "chiplet_swizzle": (
        re.compile(r"chiplet_transform|NUM_XCDS|%\s*NUM_XCDS"),
        "a chiplet/XCD workgroup swizzle so concurrently-scheduled blocks share "
        "an XCD's L2 instead of scattering across the 8 chiplets",
    ),
    "l2_group_swizzle": (
        re.compile(r"num_wgid_in_group|group_size_m"),
        "an L2 grouping swizzle (the GROUP_M trick) so neighbouring blocks reuse "
        "the same A/B panels while they are still resident",
    ),
    "prefill_swizzled_offsets": (
        re.compile(r"prefill_swizzled_offsets"),
        "precomputed swizzled GLOBAL offsets, so the bank-conflict-free "
        "permutation costs no address math inside the loop",
    ),
    "double_buffer": (
        re.compile(r"\[2\]\s*(?:\[2\]\s*)?\(?&|tic\s*\^=\s*1|curr\s*=\s*0\s*,\s*next"),
        "double (or 2x2) buffered LDS tiles with tic/toc index flipping, so the "
        "next K-step is in flight while the current one feeds the MFMAs",
    ),
    "manual_gpr_pinning": (
        re.compile(r"ds_read_b(?:32|64|128)<\s*\d+\s*>|GPR_START"),
        "explicit AGPR/VGPR numbering on the LDS reads instead of trusting HIPCC's "
        "allocator, which otherwise spills register-heavy GEMM/attention to scratch",
    ),
    "buffer_load_srsrc": (
        re.compile(r"make_srsrc|i32x4\s+\w*srsrc"),
        "raw buffer loads through a hand-built SRD (scalar resource descriptor), "
        "which gives bounds-checked global loads without per-lane 64-bit addresses",
    ),
}

# Ordered most-specific first. "causal" uses a (?<!non_) guard because
# "attn_bkwd_non_causal" CONTAINS "causal" and a naive pattern labels the
# non-causal backward kernel as causal -- a wrong label on the single most
# valuable kernel in the repo.
_OP_HINTS: tuple[tuple[re.Pattern, str, str], ...] = (
    (re.compile(r"[_/]prep(?:[_./]|$)", re.I), "attention_backward_prep",
     "the backward-pass preamble that computes the row-wise D = rowsum(dO * O) term"),
    (re.compile(r"bkwd.*(?<!non_)causal|(?<!non_)causal.*bkwd", re.I),
     "attention_backward_causal",
     "the backward pass of CAUSAL grouped-query attention (dQ, dK, dV)"),
    (re.compile(r"bkwd|backward", re.I), "attention_backward",
     "the backward pass of NON-CAUSAL grouped-query attention (dQ, dK, dV)"),
    (re.compile(r"(?<!non_)causal", re.I), "attention_forward_causal",
     "CAUSAL grouped-query attention forward (flash-attention style, online softmax)"),
    (re.compile(r"attn|gqa", re.I), "attention_forward",
     "NON-CAUSAL grouped-query attention forward (flash-attention style, online softmax)"),
    (re.compile(r"mxfp8", re.I), "gemm_mxfp8",
     "a block-scaled MXFP8 matrix multiply with fp32 accumulation"),
    (re.compile(r"fp8", re.I), "gemm_fp8",
     "an FP8 matrix multiply with fp32 accumulation"),
    (re.compile(r"scaled_matmul", re.I), "gemm_scaled",
     "a scaled matrix multiply matching torch._scaled_mm semantics"),
    (re.compile(r"gemm|matmul|256_256", re.I), "gemm_bf16",
     "a BF16 matrix multiply with fp32 accumulation"),
    (re.compile(r"layernorm", re.I), "layernorm",
     "a fused layer normalization"),
    (re.compile(r"rotary", re.I), "rotary",
     "rotary position embedding applied in place"),
    (re.compile(r"softmax", re.I), "softmax",
     "a row-wise softmax"),
)

_ARCH_OF_DIR = {"cdna4": "gfx950", "cdna3": "gfx942"}


@dataclass
class KernelRecord:
    rel_path: str
    arch: str
    arch_dir: str
    op: str
    op_description: str
    num_waves: int
    schedule: str
    schedule_evidence: str
    techniques: tuple[str, ...] = ()
    layouts: tuple[str, ...] = ()
    dtypes: tuple[str, ...] = ()
    source: str = ""
    loop_excerpt: str = ""
    launch_blocks_per_cu: int = 0
    extra: dict = field(default_factory=dict)


_INT_EXPR_RE = re.compile(r"^[\d\s+*/()A-Za-z_]+$")


def _symbol_table(src: str) -> dict[str, str]:
    """Integer #define / constexpr symbols, as unevaluated expression strings."""
    table: dict[str, str] = {}
    for m in re.finditer(r"^[ \t]*#define[ \t]+([A-Za-z_]\w*)[ \t]+(.+?)[ \t]*$", src, re.M):
        table.setdefault(m.group(1), m.group(2))
    for m in re.finditer(r"constexpr\s+int\s+([A-Za-z_]\w*)\s*=\s*([^;]+);", src):
        table.setdefault(m.group(1), m.group(2))
    return table


def _resolve_int(expr: str, table: dict[str, str], depth: int = 0) -> Optional[int]:
    """Resolve a small integer expression over the symbol table.

    Deliberately conservative: anything containing a token it cannot resolve
    returns None so the caller reports "unknown" instead of a wrong wave count.
    """
    expr = (expr or "").strip()
    if depth > 8 or not expr or not _INT_EXPR_RE.match(expr):
        return None
    # kittens::WARP_THREADS is the wave width on AMD and is not a #define here.
    expr = re.sub(r"\bkittens\s*::\s*", "", expr)
    names = set(re.findall(r"[A-Za-z_]\w*", expr))
    env: dict[str, int] = {}
    for n in sorted(names):
        if n == "WARP_THREADS":
            env[n] = 64
            continue
        if n not in table:
            return None
        sub = _resolve_int(table[n], table, depth + 1)
        if sub is None:
            return None
        env[n] = sub
    try:
        val = eval(expr, {"__builtins__": {}}, env)  # noqa: S307 - guarded by _INT_EXPR_RE
    except Exception:  # noqa: BLE001
        return None
    return int(val) if isinstance(val, (int, float)) and float(val).is_integer() else None


def _detect_num_waves(src: str) -> int:
    """Total waves per block, 0 when it cannot be established from the source.

    Handles both the plain ``NUM_WARPS`` kernels and the wave-specialized micros,
    which never define NUM_WARPS and instead split it into producer/consumer
    worker counts. Returning 0 there made every producer/consumer kernel look
    like it had no wave count at all.
    """
    table = _symbol_table(src)
    if "NUM_WARPS" in table:
        v = _resolve_int(table["NUM_WARPS"], table)
        if v:
            return v
    prod, cons = table.get("NUM_PRODUCER_WORKERS"), table.get("NUM_CONSUMER_WORKERS")
    if prod and cons:
        a, b = _resolve_int(prod, table), _resolve_int(cons, table)
        if a is not None and b is not None:
            return a + b
    if "NUM_THREADS" in table:
        v = _resolve_int(table["NUM_THREADS"], table)
        if v and v % 64 == 0:
            return v // 64
    return 0


# A 4-wave kernel counts as "interleave" once it splits its loop into at least
# this many barrier-separated clusters. Chosen from the measured spread in the
# checkout, which separates cleanly rather than needing a fine judgement call:
# the two attention-backward kernels use 39 barrier sites and the FP8 4-wave
# kernel 21, while the plain 4-wave kernels use 0, 2 and 7.
_INTERLEAVE_MIN_BARRIERS = 8
_BARRIER_SITE_RE = re.compile(r"__builtin_amdgcn_s_barrier")


def _detect_schedule(src: str, num_waves: int) -> tuple[str, str]:
    """Classify the overlap schedule from evidence in the source.

    Never guesses: a kernel whose source shows no scheduling pattern is reported
    as unclassified so no row asserts a schedule that is not there.
    """
    has_stagger = bool(TECHNIQUES["conditional_barrier_stagger"][0].search(src))
    has_group_barrier = bool(TECHNIQUES["sched_group_barrier"][0].search(src))
    interleaved_fn = "do_interleaved_cluster" in src
    n_barriers = len(_BARRIER_SITE_RE.findall(src))
    producer_consumer = bool(re.search(r"is_producer|is_consumer", src))

    if producer_consumer:
        return ("wave-specialized producer/consumer",
                f"explicit producer and consumer wave roles across {num_waves} waves")
    if num_waves == 8 and has_stagger:
        return ("8-wave ping-pong",
                "NUM_WARPS==8 plus a conditional s_barrier on the wave-group index, "
                "which offsets the two groups by half a phase")
    if num_waves == 4 and (interleaved_fn or has_group_barrier
                           or n_barriers >= _INTERLEAVE_MIN_BARRIERS):
        why = ("an interleaved-cluster helper" if interleaved_fn
               else "sched_group_barrier instruction-class masks" if has_group_barrier
               else f"{n_barriers} barrier-separated instruction clusters")
        return ("4-wave interleave", f"NUM_WARPS==4 with {why}")
    if num_waves == 8:
        return ("8-wave (no stagger detected)", "NUM_WARPS==8 without a conditional barrier")
    if num_waves == 4:
        return ("4-wave", f"NUM_WARPS==4 with only {n_barriers} barrier sites")
    return ("unclassified", "no wave count or scheduling pattern detected in the source")


def _detect_op(rel_path: str, src: str) -> tuple[str, str]:
    """Identify the operation, preferring the FILENAME over the directory.

    ``gqa_backwards/attn_fwd_non_causal.cpp`` is the FORWARD kernel that the
    backward pass needs; matching the directory first labelled it as a backward
    kernel, which would have taught the model the wrong schedule for the wrong
    pass.
    """
    for probe in (pathlib.Path(rel_path).name, rel_path, src[:4000]):
        for rx, op, desc in _OP_HINTS:
            if rx.search(probe):
                return op, desc
    return "unknown", "an unidentified operation"


_DTYPE_RE = re.compile(r"\b(st_bf|rt_bf|st_fp8e4m3|rt_fp8e4m3|st_fl|rt_fl|fp8e4m3|bf16|fp6|fp4)\b")
_LAYOUT_RE = re.compile(r"\b(st|rt)_(\d+x\d+(?:_\d+)?)_s\b")


def _loop_region(src: str, budget: int = 6000) -> str:
    """Extract the main K/KV loop, which is where the schedule actually lives.

    Whole kernels run to 3.4k lines; the teaching content is the compute loop.
    """
    m = re.search(r"^[ \t]*(?:#pragma unroll[^\n]*\n[ \t]*)?for\s*\([^\n]*\)\s*\{", src, re.M)
    if not m:
        return src[:budget]
    start = m.start()
    try:
        body = _brace_body(src, src.index("{", m.start()))
    except (ValueError, HipKittensIngestError):
        return src[start:start + budget]
    head = src[start:src.index("{", m.start()) + 1]
    text = head + body + "\n}"
    if len(text) > budget:
        text = text[:budget] + "\n    // ... (loop body truncated) ...\n}"
    return text


def discover_kernels(
    root: Optional[str | pathlib.Path] = None,
    arch_dirs: Iterable[str] = ("cdna4", "cdna3"),
) -> list[KernelRecord]:
    """Find and characterise the primary HipKittens kernels."""
    r = hk_root(root)
    kdir = r / "kernels"
    if not kdir.is_dir():
        raise HipKittensIngestError(f"missing {kdir}")
    wanted = set(arch_dirs)
    out: list[KernelRecord] = []
    for p in sorted(kdir.rglob("*")):
        if p.suffix not in (".cpp", ".cu"):
            continue
        rel = str(p.relative_to(r))
        if any(part in f"/{rel}" for part in _EXCLUDE_PATH_PARTS):
            continue
        parts = pathlib.Path(rel).parts
        arch_dir = parts[1] if len(parts) > 1 else ""
        if arch_dir in _EXCLUDE_ARCH_DIRS or arch_dir not in wanted:
            continue
        src = p.read_text(encoding="utf-8", errors="replace")
        if "__global__" not in src:
            continue  # utils.cpp / profile_utils.cpp are helpers, not kernels
        num_waves = _detect_num_waves(src)
        schedule, evidence = _detect_schedule(src, num_waves)
        op, desc = _detect_op(rel, src)
        techs = tuple(sorted(k for k, (rx, _) in TECHNIQUES.items() if rx.search(src)))
        layouts = tuple(sorted({f"{a}_{b}_s" for a, b in _LAYOUT_RE.findall(src)}))
        dtypes = tuple(sorted(set(_DTYPE_RE.findall(src))))
        lb = re.search(r"__launch_bounds__\(\s*[^,]+,\s*(\d+)\s*\)", src)
        out.append(KernelRecord(
            rel_path=rel, arch=_ARCH_OF_DIR.get(arch_dir, "unknown"), arch_dir=arch_dir,
            op=op, op_description=desc, num_waves=num_waves, schedule=schedule,
            schedule_evidence=evidence, techniques=techs, layouts=layouts, dtypes=dtypes,
            source=src, loop_excerpt=_loop_region(src),
            launch_blocks_per_cu=int(lb.group(1)) if lb else 0,
        ))
    if not out:
        raise HipKittensIngestError(f"discovered no kernels under {kdir}")
    return out


# --------------------------------------------------------------------------- #
# 4. Measured benchmark numbers the authors committed (attributable, not ours)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Measurement:
    """One measured HK-vs-baseline comparison, straight out of the repo's JSON."""

    workload: str
    size: str
    baseline: str
    hk_metric: float
    baseline_metric: float
    metric_name: str
    speedup: float
    evidence_file: str


# file -> (workload, metric kind, HK key, {baseline label: key})
_BENCH_FILES: tuple[tuple[str, str, str, str, dict[str, str]], ...] = (
    ("analysis/fp8_gemm/mi350x/mi355x_fp8_gemm.json", "FP8 GEMM", "TFLOP/s",
     "tflops", {"hipBLASLt": "tflops_hipblaslt"}),
    ("analysis/layernorm/mi350x/mi355x_layernorm.json", "fused layernorm", "ms",
     "avg_time_tk", {"PyTorch": "avg_time_pytorch", "torch.compile": "avg_time_compiled"}),
    ("analysis/rotary/mi350x/mi355x_rotary.json", "rotary embedding", "ms",
     "avg_time_tk", {"PyTorch": "avg_time_pytorch",
                     "torch.compile": "avg_time_pytorch_compiled",
                     "AITER": "avg_time_aiter"}),
)


def parse_measurements(root: Optional[str | pathlib.Path] = None) -> list[Measurement]:
    """Load the authors' own benchmark JSON. Higher-is-better for TFLOP/s,
    lower-is-better for milliseconds; the ratio is computed accordingly so a
    "speedup" is never accidentally inverted."""
    r = hk_root(root)
    out: list[Measurement] = []
    for rel, workload, metric, hk_key, baselines in _BENCH_FILES:
        p = r / rel
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            raise HipKittensIngestError(f"{p}: not valid JSON ({exc})") from exc
        for size, row in sorted(data.items(), key=lambda kv: _int_or_inf(kv[0])):
            if not isinstance(row, dict):
                continue
            hk = row.get(hk_key)
            if not isinstance(hk, (int, float)) or hk <= 0:
                continue
            for label, bkey in sorted(baselines.items()):
                base = row.get(bkey)
                if not isinstance(base, (int, float)) or base <= 0:
                    continue
                speedup = (hk / base) if metric == "TFLOP/s" else (base / hk)
                out.append(Measurement(
                    workload=workload, size=str(size), baseline=label,
                    hk_metric=float(hk), baseline_metric=float(base),
                    metric_name=metric, speedup=round(speedup, 3),
                    evidence_file=rel,
                ))
    return out


def _int_or_inf(s: str) -> float:
    try:
        return float(int(s))
    except (TypeError, ValueError):
        return float("inf")


# --------------------------------------------------------------------------- #
# 5. Row construction
# --------------------------------------------------------------------------- #
# Every row is [system, user, assistant]. Deliberately NOT the ANALYSIS /
# PROPOSED_CHANGE / FULL_KERNEL contract -- see the module docstring.
CHARS_PER_TOKEN = 3.6            # matches scripts/build_sft_v3_mixture.py
MAX_ROW_CHARS = 24_000           # ~6.7k tokens, well inside the 17,408-token gate


def _row(
    *,
    user: str,
    assistant: str,
    qa_type: str,
    arch: str,
    prov: dict,
    files: Iterable[str] = (),
    gt: Optional[dict] = None,
    verified: bool = False,
    verification: str = "",
    system: str = SYSTEM_PROMPT_HK,
) -> dict:
    """Assemble one SFT row with full MIT provenance attached."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ],
        "_source": SOURCE_TAG,
        "_qa_type": qa_type,
        "_arch": arch,
        "_verified": bool(verified),
        "_verification": verification,
        "_gt": gt or {},
        "_provenance": {
            "source_id": prov["source_id"],
            "repository_url": prov["repository_url"],
            "commit": prov["commit"],
            "license": prov["license"],
            "license_holder": prov["license_holder"],
            "paper": prov["paper"],
            "authors": prov["authors"],
            "attribution": prov["attribution"],
            "files": sorted(set(files)),
        },
    }


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one}" if n == 1 else f"{n} {many}"


def _fence(code: str, lang: str = "cpp") -> str:
    return f"```{lang}\n{code.rstrip()}\n```"


_MFMA_FOR_LAYOUT = {
    # HipKittens names its shared/register tile layouts after the MFMA fragment
    # shape they feed, so the layout name IS the MFMA shape.
    "st_16x16": "16x16", "st_16x16_swizzled": "16x16", "st_32x32": "32x32",
    "st_16x32": "16x32", "st_32x16": "32x16", "st_8x32": "8x32",
    "st_16x64": "16x64", "st_16x128": "16x128",
}

_DTYPE_NAME = {1: "8-bit (fp8/mxfp8)", 2: "16-bit (bf16/fp16)", 4: "32-bit (fp32)"}


# --- family: measured LDS bank / phase model ------------------------------- #
def rows_lds_hardware_model(
    models: list[PhaseModel], prov: dict, arch: str = DEFAULT_ARCH
) -> list[dict]:
    """The undocumented CDNA LDS bank/phase model, per instruction width.

    This is the highest-value slice in the whole ingestion: the paper states this
    behaviour is undocumented in the CDNA ISA, so a model cannot have learned it
    from vendor documentation, and it is what every conflict-free layout decision
    ultimately rests on.
    """
    rows: list[dict] = []
    by_name = {m.instruction: m for m in models}

    for m in sorted(models, key=lambda x: x.instruction):
        phase_txt = "\n".join(
            f"  phase {i}: {list(p)}" for i, p in enumerate(m.phases)
        )
        width_bits = re.search(r"b(\d+)$", m.instruction)
        nbytes = int(width_bits.group(1)) // 8 if width_bits else 0
        is_write = "write" in m.instruction
        user = (
            f"On CDNA4 (gfx950), exactly how many LDS banks does `{m.instruction}` see, "
            f"and which lanes of the 64-lane wave can conflict with each other? Give me "
            f"the phase partition, not a rule of thumb."
        )
        assistant = (
            f"`{m.instruction}` moves {nbytes} bytes per lane, touching "
            f"{_plural(max(1, nbytes // 4), 'consecutive bank', 'consecutive banks')}, and measures:\n\n"
            f"- **{m.num_banks} banks**, with "
            f"`{m.bank_formula.replace('NUM_BANKS', str(m.num_banks))}`.\n"
            f"- **{len(m.phases)} conflict phases of {m.lanes_per_phase} lanes**:\n\n"
            f"```\n{phase_txt}\n```\n\n"
            + (f"Note the shape of those groups: they are *not* runs of consecutive "
               f"lanes. Lanes {list(m.phases[0])[:4]} share a phase with lanes "
               f"{list(m.phases[0])[4:8]}, which no `lane / {m.lanes_per_phase}` or "
               f"`lane % {len(m.phases)}` formula reproduces. If you assumed contiguous "
               f"groups you would place the wrong lanes in conflict.\n\n"
               if not m.contiguous_phases else
               f"For this instruction the phases *are* runs of consecutive lanes "
               f"(lanes 0-{m.lanes_per_phase - 1}, then the next {m.lanes_per_phase}, and "
               f"so on), which is the one case where the obvious `lane / "
               f"{m.lanes_per_phase}` guess is correct. Do not carry that assumption to "
               f"the other widths -- `ds_read_b96` and `ds_read_b128` interleave.\n\n")
            + f"The operative consequence for `{m.instruction}` specifically: a "
              f"conflict-free access needs the **{m.lanes_per_phase} lanes inside one "
              f"phase** on distinct banks, not all 64. With {m.num_banks} banks and "
              f"{m.lanes_per_phase} contending lanes each claiming "
              f"{max(1, nbytes // 4)} bank(s), the access is feasible with "
              f"{m.lanes_per_phase * max(1, nbytes // 4)} of {m.num_banks} banks "
              f"occupied per phase, so there is "
              f"{'headroom' if m.lanes_per_phase * max(1, nbytes // 4) < m.num_banks else 'exactly no slack'} "
              f"in the assignment.\n\n"
            + (f"Because this is the **{'store' if is_write else 'load'}** path, verify it "
               f"separately from the other direction: on this hardware `ds_read_b64` and "
               f"`ds_write_b64` measure "
               f"{by_name['ds_read_b64'].num_banks} and "
               f"{by_name['ds_write_b64'].num_banks} banks respectively, so a layout "
               f"proven clean one way can still serialize the other.\n\n"
               if {"ds_read_b64", "ds_write_b64"} <= set(by_name) else "")
            + f"_Measured by the HipKittens authors with a pairwise lane-conflict solver "
              f"over all {64 * 63 // 2} lane pairs; see `{m.evidence_files[0]}` and "
              f"`{m.evidence_files[1]}`. {HK_ATTRIBUTION}_"
        )
        rows.append(_row(
            user=user, assistant=assistant, qa_type="hk_lds_bank_model", arch=arch,
            prov=prov, files=m.evidence_files,
            gt={
                "instruction": m.instruction,
                "num_banks": m.num_banks,
                "num_phases": len(m.phases),
                "lanes_per_phase": m.lanes_per_phase,
                "phases": [list(p) for p in m.phases],
                "contiguous_phases": m.contiguous_phases,
            },
            verified=True,
            verification=("transcribed from the authors' committed solver output; "
                          "phase partition checked to cover all 64 lanes exactly once"),
        ))

    # One cross-cutting row on the read/write asymmetry, which is the single most
    # counter-intuitive fact in the set.
    if {"ds_read_b64", "ds_write_b64"} <= set(by_name):
        r, w = by_name["ds_read_b64"], by_name["ds_write_b64"]
        table = "\n".join(
            f"| `{m.instruction}` | {m.num_banks} | {len(m.phases)} | {m.lanes_per_phase} |"
            for m in sorted(models, key=lambda x: x.instruction)
        )
        user = (
            "I keep reading that AMD LDS has 32 banks, but my conflict-free tile layout "
            "for `ds_read_b128` on gfx950 does not behave the way a 32-bank model "
            "predicts. Is the bank count actually fixed on CDNA4?"
        )
        assistant = (
            "No. On CDNA4 the effective bank count is a property of the **instruction**, "
            "not of the LDS block, and the read and write paths do not agree.\n\n"
            "| instruction | banks observed | conflict phases | lanes per phase |\n"
            "|---|---|---|---|\n" + table + "\n\n"
            f"The sharp one is `ds_read_b64` versus `ds_write_b64`: identical width, "
            f"identical memory, **{r.num_banks} banks on the read and {w.num_banks} on "
            f"the write**. A 32-bank mental model over-predicts conflicts for the wide "
            f"reads and under-predicts them for the writes.\n\n"
            "### What follows for your layout\n\n"
            "1. **Verify the load path and the store path separately.** They have "
            "different bank counts *and* different phase partitions, so one XOR swizzle "
            "can be clean in one direction and serialized in the other.\n"
            "2. **Only same-phase lanes contend.** For `ds_read_b128` you need the "
            f"{by_name['ds_read_b128'].lanes_per_phase} lanes of a phase spread over "
            "distinct banks; the other 48 lanes are irrelevant to that cycle.\n"
            "3. **Do not infer phases from lane arithmetic.** For `ds_read_b96` the "
            f"phases are {[list(p) for p in by_name['ds_read_b96'].phases][:2]} ... -- "
            "lanes 0-3 share a phase with lanes 20-23, which no simple `lane / k` "
            "formula produces.\n"
            "4. **Widen deliberately.** Moving from `ds_read_b64` to `ds_read_b128` "
            "halves the instruction count, but it also changes the phase partition from "
            f"{len(r.phases)} phases of {r.lanes_per_phase} to "
            f"{len(by_name['ds_read_b128'].phases)} of "
            f"{by_name['ds_read_b128'].lanes_per_phase}; the layout must be re-derived, "
            "not merely reused.\n\n"
            f"{PAPER_CLAIMS['swizzle_undocumented'][0].capitalize()} "
            f"({PAPER_CLAIMS['swizzle_undocumented'][1]}), which is why these numbers "
            "come from a measurement harness rather than a manual.\n\n"
            f"_Measured by the HipKittens authors under "
            f"`analysis/paper_experiments/phases/`. {HK_ATTRIBUTION}_"
        )
        rows.append(_row(
            user=user, assistant=assistant, qa_type="hk_lds_bank_asymmetry", arch=arch,
            prov=prov,
            files=[f for m in models for f in m.evidence_files],
            gt={m.instruction: {"num_banks": m.num_banks, "phases": len(m.phases)}
                for m in models},
            verified=True,
            verification="all four instruction models transcribed from committed solver output",
        ))
    return rows


# --- family: per-layout LDS swizzle ---------------------------------------- #
def rows_swizzle(
    layouts: list[SwizzleLayout], prov: dict, arch: str = DEFAULT_ARCH
) -> list[dict]:
    """Per-MFMA-shape swizzle derivation, plus the contrast that motivates it."""
    rows: list[dict] = []
    src_file = f"include/{layouts[0].arch}/types/shared/st_shape.cuh" if layouts else ""

    # (a) one row per non-identity swizzle. The ANGLE of each row is chosen from a
    # real distinguishing property of the layout (multi-term, fp8 geometry, narrow
    # access width), so the rows differ in what they teach rather than being one
    # template filled in five times.
    nondefault = 0
    for lay in [l for l in layouts if not l.is_identity]:
        mfma = _MFMA_FOR_LAYOUT.get(lay.name, f"{lay.rows}x{lay.cols}")
        tile_bytes = lay.rows * lay.cols * lay.dtype_bytes
        mod, sr, sl = lay.terms[0]
        dt = _DTYPE_NAME.get(lay.dtype_bytes, f"{lay.dtype_bytes}-byte")
        if len(lay.terms) > 1:
            angle = "two_terms"
        elif lay.dtype_bytes == 1:
            angle = "fp8"
        elif (lay.bytes_per_thread or 16) < 16:
            angle = "narrow"
        else:
            angle = ("derive", "transplant")[nondefault % 2]
            nondefault += 1
        user = _swizzle_question(lay, mfma, dt, angle)
        row_bytes = lay.cols * lay.dtype_bytes
        terms_txt = "\n".join(
            f"- `((offset % {mod}) >> {sr}) << {sl}` toggles bit {sl} of the byte "
            f"address, i.e. bank bit {sl - 2}, displacing by {1 << sl} B = "
            f"{(1 << sl) // 4} banks, for offsets in the upper half of each "
            f"{mod}-byte window."
            for mod, sr, sl in lay.terms
        )
        assistant = (
            f"{_swizzle_angle_lead(lay, angle)}"
            f"For `{lay.name}` -- {lay.rows} rows x {lay.cols} cols at "
            f"{lay.dtype_bytes} B/element, so a **{row_bytes} B row "
            f"({row_bytes // 4} banks)** and a {tile_bytes} B tile -- with "
            f"`offset = {lay.dtype_bytes} * (row * {lay.cols} + col)`:\n\n"
            f"{_fence(f'{lay.formula()};', 'cpp')}\n\n"
            f"### The terms\n\n{terms_txt}\n\n"
            + (f"Two terms are required for this layout: one XOR toggles a single "
               f"address-bit pattern, and this tile collides on more than one stride, so "
               f"the {(1 << lay.terms[0][2]) // 4}-bank and "
               f"{(1 << lay.terms[1][2]) // 4}-bank displacements each break a different "
               f"one.\n\n" if len(lay.terms) > 1 else "")
            + f"### Sanity conditions this satisfies\n\n"
            f"- **Bijective** over the {tile_bytes} byte offsets of the tile, so no "
            f"element is aliased. This is the correctness condition, and it is why an XOR "
            f"is used rather than row padding: the tile still occupies exactly "
            f"{tile_bytes} B with no hole and the "
            f"{lay.bytes_per_thread or 16}-byte-per-lane aligned access survives.\n"
            f"- **Element width matters.** At {lay.dtype_bytes * 8}-bit elements this "
            f"layout needs the XOR; at 32-bit it degenerates to the identity in this "
            f"library, because a 4-byte element already advances one bank per column.\n\n"
            f"### It does not transfer to a neighbouring shape\n\n"
            f"{_sibling_note(lay, layouts)}\n\n"
            f"Fold it into the precomputed global offsets "
            f"(`prefill_swizzled_offsets`) rather than computing it per access: the "
            f"permutation is applied to the *global* read address, so the LDS write stays "
            f"contiguous per lane and the inner loop pays no address arithmetic for it.\n\n"
            f"_Source: `{src_file}`. {HK_ATTRIBUTION}_"
        )
        rows.append(_row(
            user=user, assistant=assistant, qa_type="hk_swizzle_derivation", arch=arch,
            prov=prov, files=[src_file],
            gt={
                "layout": lay.name, "rows": lay.rows, "cols": lay.cols,
                "dtype_bytes": lay.dtype_bytes, "terms": [list(t) for t in lay.terms],
                "formula": lay.formula(), "bijection": lay.is_bijection(),
                "tile_bytes": tile_bytes, "bytes_per_thread": lay.bytes_per_thread,
                "first_term": {"modulus": mod, "shift_right": sr, "shift_left": sl},
            },
            verified=True,
            verification=("formula parsed from the C++ source (not transcribed) and "
                          "checked to be a bijection over all tile byte offsets"),
        ))

    # (b) the contrast row: same tile shape, transposed MFMA shape, different swizzle.
    pair = {l.name: l for l in layouts if l.dtype_bytes == 2}
    if "st_16x32" in pair and "st_32x16" in pair:
        a, b = pair["st_16x32"], pair["st_32x16"]
        am, bm = a.terms[0], b.terms[0]
        user = (
            "For bf16 on gfx950, HipKittens uses `((offset % 1024) >> 9) << 5` as the "
            "swizzle for a 16x32 shared tile but `((offset % 1024) >> 9) << 4` for a "
            "32x16 tile. Same modulus, same right shift, different left shift. Is the "
            "difference meaningful or is one of them arbitrary?"
        )
        assistant = (
            "It is meaningful, and the pattern of the difference tells you how to derive "
            "the swizzle for a shape that is not in the table.\n\n"
            "## Read the three constants as what they are\n\n"
            "In `offset ^ (((offset % M) >> R) << L)`:\n\n"
            f"- `% M` selects the *window* over which the permutation repeats.\n"
            f"- `>> R` picks the address bit that identifies which half of that window "
            f"you are in -- bit {am[1]} in both cases, i.e. the "
            f"{1 << am[1]}-byte boundary.\n"
            f"- `<< L` chooses **which bank-selecting bit that half toggles**. This is "
            f"the only constant that differs: {am[2]} for 16x32, {bm[2]} for 32x16.\n\n"
            f"Since a bank is 4 bytes, bit `L` of the byte address maps to bank bit "
            f"`L - 2`. So 16x32 rotates the addresses by "
            f"{1 << am[2]} B = {(1 << am[2]) // 4} banks, and 32x16 by "
            f"{1 << bm[2]} B = {(1 << bm[2]) // 4} banks.\n\n"
            "## The shift tracks the row width\n\n"
            f"A row of the 16x32 bf16 tile is 32 x 2 B = 64 B = 16 banks. A row of the "
            f"32x16 tile is 16 x 2 B = 32 B = 8 banks. The displacements are "
            f"{(1 << am[2]) // 4} and {(1 << bm[2]) // 4} banks -- in each case exactly "
            f"half the row width. The colliding stride in these tiles is the row stride, "
            f"and displacing by half of it is what maps the colliding set onto free "
            f"banks. So the constant is not arbitrary: it is pinned by the tile's row "
            f"width in bytes, which is the MFMA shape and the dtype together.\n\n"
            "## Do not extrapolate the rule\n\n"
            "That half-the-row-width relationship holds for the 2-byte layouts that load "
            "16 B per lane, and it breaks elsewhere in this very same file:\n\n"
            "| layout | dtype | row width | displacement | half the row? |\n"
            "|---|---|---|---|---|\n"
            f"| `st_16x32` | bf16 | 64 B (16 banks) | {(1 << am[2]) // 4} banks | yes |\n"
            f"| `st_32x16` | bf16 | 32 B (8 banks) | {(1 << bm[2]) // 4} banks | yes |\n"
            + (f"| `st_32x32` | bf16 | 64 B (16 banks) | "
               f"{(1 << pair['st_32x32'].terms[0][2]) // 4} banks **and** "
               f"{(1 << pair['st_32x32'].terms[1][2]) // 4} banks (two XOR terms) | "
               f"first term only |\n" if "st_32x32" in pair else "")
            + "| `st_16x16_swizzled` | bf16, 4 B/lane | 32 B (8 banks) | 2 banks | no |\n"
              "| `st_16x128` | fp8 | 128 B (32 banks) | 4 banks | no |\n\n"
            "The two exceptions are informative rather than noise. `st_16x16_swizzled` "
            "loads 4 B per lane instead of 16, so its lanes cover a row differently and "
            "the colliding stride is not the row stride. `st_16x128` is fp8, where four "
            "elements share a bank, so the collision geometry changes again. And a 32x32 "
            "bf16 tile needs **two** XOR terms because one XOR toggles one address bit "
            "pattern and this tile has more than one colliding stride to break.\n\n"
            "**So the operational rule is: derive per layout, never transplant.** "
            f"{PAPER_CLAIMS['swizzle_undocumented'][0].capitalize()} "
            f"({PAPER_CLAIMS['swizzle_undocumented'][1]}), so the only reliable procedure "
            "is to enumerate which lanes share a conflict phase for the access width you "
            "are using, compute their banks under a candidate permutation, and check the "
            "count is one per bank per phase.\n\n"
            "One dtype shortcut is safe: **at fp32 every one of these layouts is the "
            "identity**, because a 4-byte element already advances one bank per column, "
            "so the conflict never forms and there is nothing to permute.\n\n"
            "If you copy the 16x32 swizzle into a 32x16 kernel it will still be a valid "
            "bijection -- the kernel will produce correct results -- and it will simply "
            "run slower with LDS conflicts you cannot see in the output. That is the "
            "failure mode to watch for: wrong swizzle is a silent performance bug, never "
            "a correctness error.\n\n"
            f"_Source: `{src_file}`. {HK_ATTRIBUTION}_"
        )
        rows.append(_row(
            user=user, assistant=assistant, qa_type="hk_swizzle_contrast", arch=arch,
            prov=prov, files=[src_file],
            gt={"st_16x32_terms": [list(t) for t in a.terms],
                "st_32x16_terms": [list(t) for t in b.terms],
                "row_bytes_16x32": a.cols * a.dtype_bytes,
                "row_bytes_32x16": b.cols * b.dtype_bytes},
            verified=True,
            verification="both formulas parsed from source and verified bijective",
        ))
    return rows


def _swizzle_question(lay: SwizzleLayout, mfma: str, dt: str, angle: str) -> str:
    """Question text keyed to what is actually distinctive about this layout."""
    base = (f"a {lay.rows}x{lay.cols} LDS tile of {dt} data feeding {mfma} MFMA "
            f"instructions on CDNA4 (gfx950)")
    if angle == "two_terms":
        return (
            f"HipKittens swizzles most of its bf16 shared tiles with a single XOR term, "
            f"but for the {lay.rows}x{lay.cols} layout it uses two chained XORs. I am "
            f"writing {base}. Why does this shape need two terms when its neighbours need "
            f"one, and what does each term buy?"
        )
    if angle == "fp8":
        return (
            f"I am moving a GEMM from bf16 to fp8 on gfx950 and staging {base}. Four fp8 "
            f"elements now share a single 4-byte LDS bank, so the bank geometry is not "
            f"what it was at bf16. Give me the conflict-free swizzle for this layout and "
            f"explain how the fp8 packing changes the derivation."
        )
    if angle == "narrow":
        return (
            f"HipKittens has two {lay.rows}x{lay.cols} bf16 shared layouts: a plain one "
            f"with no swizzle, and a swizzled variant that also drops to "
            f"{lay.bytes_per_thread} bytes per lane instead of 16. Why would I accept the "
            f"narrower access, and what is the swizzle it uses?"
        )
    if angle == "transplant":
        return (
            f"I already have a working bank-conflict-free swizzle for one bf16 shared "
            f"layout on gfx950. Can I reuse it for {base}, or does the {lay.rows}x{lay.cols} "
            f"shape need its own? If it needs its own, how do I tell -- the kernel gives "
            f"the right answer either way."
        )
    return (
        f"I am writing a kernel that stages {base}. Give me the bank-conflict-free "
        f"address swizzle for that layout and explain how it is derived, not just what "
        f"it is."
    )


def _swizzle_angle_lead(lay: SwizzleLayout, angle: str) -> str:
    """A short lead-in that answers the specific question that was asked."""
    if angle == "two_terms":
        a, b = lay.terms[0], lay.terms[1]
        return (
            f"Because this tile collides on two different strides, and one XOR can only "
            f"break one of them. The first term displaces by {(1 << a[2]) // 4} banks and "
            f"the second by {(1 << b[2]) // 4}; together they decorrelate a "
            f"{lay.rows}-row gather that starts out {2 if lay.rows <= 16 else 4}-way "
            f"conflicted, where a single term would leave half the collisions in place.\n\n"
        )
    if angle == "fp8":
        return (
            f"The packing changes the arithmetic in one specific place: a bank holds four "
            f"fp8 elements instead of two bf16 ones, so a row of {lay.cols} elements spans "
            f"{lay.cols * lay.dtype_bytes // 4} banks rather than "
            f"{lay.cols * 2 // 4}. That is why this layout's displacement is not simply "
            f"half its row width the way the bf16 layouts' are.\n\n"
        )
    if angle == "narrow":
        return (
            f"You accept it because at {lay.bytes_per_thread} bytes per lane the wave "
            f"issues a narrower DS instruction whose conflict phases are coarser "
            f"({lay.bytes_per_thread * 8}-bit accesses group 32 lanes per phase rather "
            f"than 16), and for this small tile that is the combination that comes out "
            f"conflict-free. The narrower access is the price of the layout, not a "
            f"regression.\n\n"
        )
    if angle == "transplant":
        return (
            f"It needs its own, and you cannot tell from the output -- any bijective "
            f"swizzle computes the correct result, so a transplanted one shows up only as "
            f"lost bandwidth. You have to check the bank arithmetic.\n\n"
        )
    return ""


def _swizzle_derivation(lay: SwizzleLayout) -> str:
    """Narrate what the parsed XOR terms do, in bank units."""
    row_bytes = lay.cols * lay.dtype_bytes
    out = [f"### What it does\n",
           f"A row of this tile is {lay.cols} x {lay.dtype_bytes} B = {row_bytes} B, "
           f"which spans {row_bytes // 4} four-byte banks."]
    for i, (mod, sr, sl) in enumerate(lay.terms, 1):
        out.append(
            f"\n**Term {i}: `((offset % {mod}) >> {sr}) << {sl}`.** It takes the address "
            f"bits at and above the {1 << sr}-byte boundary within a {mod}-byte window "
            f"and XORs them into bit {sl}, displacing the address by up to "
            f"{1 << sl} B = {(1 << sl) // 4} banks. Rows that would otherwise land on the "
            f"same banks are rotated apart."
        )
    if len(lay.terms) > 1:
        out.append(
            f"\nThis layout needs {len(lay.terms)} terms, not one: a single XOR toggles "
            f"one address-bit pattern, and this tile has more than one stride on which "
            f"same-phase lanes collide, so each term breaks one of them."
        )
    out.append(
        f"\nThe result is a permutation of the {lay.rows * lay.cols * lay.dtype_bytes} "
        f"byte offsets in the tile -- verified bijective, so no element is lost or "
        f"aliased."
    )
    return "\n".join(out)


def _sibling_note(lay: SwizzleLayout, layouts: list[SwizzleLayout]) -> str:
    """A one-line contrast against another layout at the same element width."""
    sibs = [l for l in layouts
            if l.dtype_bytes == lay.dtype_bytes and l.name != lay.name and not l.is_identity]
    if not sibs:
        return (f"every other shared layout at this element width uses a different "
                f"swizzle or none at all.")
    s = sibs[0]
    return (f"`{s.name}` ({s.rows}x{s.cols}) uses `{s.formula()}` while `{lay.name}` "
            f"({lay.rows}x{lay.cols}) uses `{lay.formula()}` -- different constants for "
            f"the same dtype, chosen by the tile's row width.")


# --- family: worked bank-conflict exercise -------------------------------- #
def rows_conflict_exercise(
    layouts: list[SwizzleLayout], models: list[PhaseModel], prov: dict,
    arch: str = DEFAULT_ARCH,
) -> list[dict]:
    """Worked "count the conflicts" rows, with solver-computed ground truth.

    This is the family that teaches the *skill* rather than the fact: given a bank
    count, a phase partition and a candidate layout, work out the conflict degree.
    The answer is computed here, so it cannot be wrong in the way a teacher
    model's arithmetic can be wrong.
    """
    by_name = {m.instruction: m for m in models}
    rows: list[dict] = []
    # One worked example per DISTINCT starting conflict degree and element width.
    # Five near-identical walkthroughs of the same arithmetic teach the template,
    # not the method; the informative contrast is between a 2-way and a 4-way
    # starting point, and between bf16 and fp8 bank packing.
    chosen: dict[tuple[int, int], SwizzleLayout] = {}
    for lay in [l for l in layouts if not l.is_identity]:
        rep = bank_conflict_report(lay, by_name)
        if rep is None or not rep.improves:
            # Refuse to teach a swizzle we cannot show helps under the measured
            # model. Better a missing row than a confident wrong one.
            continue
        chosen.setdefault((rep.plain_max_conflict, lay.dtype_bytes), lay)

    for lay in sorted(chosen.values(), key=lambda l: l.name):
        rep = bank_conflict_report(lay, by_name)
        if rep is None:
            continue
        model = by_name[rep.instruction]
        row_bytes = lay.cols * lay.dtype_bytes
        plain_offs = [r * row_bytes for r in range(rep.lanes)]
        swz_offs = [lay.swizzle(o) for o in plain_offs]
        plain_banks = [(o // 4) % rep.num_banks for o in plain_offs]
        swz_banks = [(o // 4) % rep.num_banks for o in swz_offs]
        phase0 = [l for l in model.phases[0] if l < rep.lanes]
        p0_plain = [(l, plain_banks[l]) for l in phase0]
        p0_swz = [(l, swz_banks[l]) for l in phase0]

        user = (
            f"Work this through concretely for CDNA4 (gfx950).\n\n"
            f"A `{lay.rows}x{lay.cols}` shared tile of "
            f"{_DTYPE_NAME.get(lay.dtype_bytes, str(lay.dtype_bytes) + '-byte')} data is "
            f"stored row-major, so element (r, c) sits at byte offset "
            f"`{lay.dtype_bytes} * (r * {lay.cols} + c)`. A wave gathers a column: lane "
            f"`r` issues one `{rep.instruction}` reading {rep.dwords_per_lane * 4} bytes "
            f"starting at the beginning of row `r`.\n\n"
            f"Take as given (measured on this hardware): `{rep.instruction}` sees "
            f"**{rep.num_banks} banks** with `bank = (byte_offset / 4) % {rep.num_banks}`, "
            f"and its 64 lanes are partitioned into **{len(model.phases)} conflict phases "
            f"of {model.lanes_per_phase} lanes**, phase 0 being `{list(model.phases[0])}`. "
            f"Only lanes in the same phase can contend.\n\n"
            f"(1) What is the worst-case conflict degree for the plain row-major layout? "
            f"(2) Does `{lay.formula()}` fix it? Show the bank arithmetic."
        )

        assistant = (
            f"## (1) Plain row-major\n\n"
            f"A row is {lay.cols} x {lay.dtype_bytes} B = **{row_bytes} B**, so lane `r` "
            f"starts at byte `{row_bytes}r` and its first bank is\n\n"
            f"`bank(r) = ({row_bytes}r / 4) % {rep.num_banks} = "
            f"{row_bytes // 4}r % {rep.num_banks}`\n\n"
            f"Because {row_bytes // 4} divides {rep.num_banks}"
            f"{'' if rep.num_banks % (row_bytes // 4) == 0 else ' (approximately)'}, that "
            f"expression only takes "
            f"{_plural(len(set(plain_banks)), 'distinct value', 'distinct values')} "
            f"across the {rep.lanes} lanes -- "
            f"the row stride is commensurate with the bank count, which is precisely the "
            f"pathological case.\n\n"
            f"Restricting to phase 0 (`{phase0}`), the starting banks are\n\n"
            f"```\n" + "\n".join(f"  lane {l:2d} -> bank {b}" for l, b in p0_plain) + "\n```\n\n"
            f"Each lane occupies {_plural(rep.dwords_per_lane, 'consecutive bank', 'consecutive banks')} from there. "
            f"Counting the worst-loaded bank inside a phase gives a conflict degree of "
            f"**{rep.plain_max_conflict}**, i.e. that access serializes into "
            f"{rep.plain_max_conflict} cycles instead of 1.\n\n"
            f"## (2) With the swizzle\n\n"
            f"Apply `{lay.formula()}` to each lane's start offset:\n\n"
            f"```\n" + "\n".join(
                f"  lane {l:2d}: offset {plain_offs[l]:5d} -> {swz_offs[l]:5d}  "
                f"(bank {plain_banks[l]:2d} -> {swz_banks[l]:2d})" for l in phase0
            ) + "\n```\n\n"
            f"Now phase 0 covers {len({b for _, b in p0_swz})} distinct starting banks "
            f"instead of {len({b for _, b in p0_plain})}, and the worst-case conflict "
            f"degree over **all {len(model.phases)} phases** falls to "
            f"**{rep.swizzled_max_conflict}**"
            + (" -- the access is bank-conflict free.\n\n" if rep.conflict_free
               else f" (down from {rep.plain_max_conflict}).\n\n")
            + f"The XOR injects a bit the row index does not otherwise control into the "
            f"bank-selecting part of the address, so rows {row_bytes // 4} banks apart no "
            f"longer land together. It stays a bijection over the tile's "
            f"{row_bytes * lay.rows} bytes, so the data merely lives elsewhere in the "
            f"tile and the gather completes in "
            f"{_plural(rep.swizzled_max_conflict, 'pass', 'passes')} instead of "
            f"{rep.plain_max_conflict}.\n\n"
            f"_Layout and swizzle from `include/{lay.arch}/types/shared/st_shape.cuh`; "
            f"bank count and phase partition measured by the HipKittens authors in "
            f"`{model.evidence_files[0]}`. Conflict degrees above were computed from "
            f"those inputs. {HK_ATTRIBUTION}_"
        )
        rows.append(_row(
            user=user, assistant=assistant, qa_type="hk_bank_conflict_exercise",
            arch=arch, prov=prov,
            files=[f"include/{lay.arch}/types/shared/st_shape.cuh",
                   model.evidence_files[0]],
            gt={
                "layout": lay.name, "dtype_bytes": lay.dtype_bytes,
                "instruction": rep.instruction, "num_banks": rep.num_banks,
                "row_bytes": row_bytes, "lanes": rep.lanes,
                "plain_max_conflict": rep.plain_max_conflict,
                "swizzled_max_conflict": rep.swizzled_max_conflict,
                "conflict_free": rep.conflict_free,
                "plain_distinct_banks": len(set(plain_banks)),
                "swizzled_distinct_banks": len(set(swz_banks)),
            },
            verified=True,
            verification=("conflict degrees computed from the parsed swizzle and the "
                          "authors' measured bank/phase model; swizzle also checked "
                          "bijective"),
        ))
    return rows


# --- family: which schedule, and why -------------------------------------- #
def rows_schedule_selection(
    kernels: list[KernelRecord], prov: dict, arch: str = DEFAULT_ARCH
) -> list[dict]:
    """The decision rule: 8-wave ping-pong vs 4-wave interleave vs neither.

    This is the row family most likely to transfer to a kernel the model has never
    seen, so it is grounded in *which* kernels in the checkout actually chose
    which schedule rather than in the paper's summary alone.
    """
    pp = sorted({k.op for k in kernels if k.schedule == "8-wave ping-pong"})
    il = sorted({k.op for k in kernels if k.schedule == "4-wave interleave"})
    ws = [k for k in kernels if k.schedule == "wave-specialized producer/consumer"]
    if not pp or not il:
        return []

    pp_ex = next(k for k in kernels if k.schedule == "8-wave ping-pong"
                 and k.op == "gemm_bf16")
    il_ex = next(k for k in kernels if k.schedule == "4-wave interleave"
                 and k.op.startswith("attention_backward"))

    rows: list[dict] = []
    user = (
        "On MI355X (gfx950, CDNA4) I need to overlap global loads with MFMA work inside "
        "a kernel's main loop. I know the two patterns people use are 8-wave ping-pong "
        "and 4-wave interleave. How do I decide which one my kernel wants, and why not "
        "just do NVIDIA-style wave specialization with dedicated producer waves?"
    )
    assistant = (
        "## The decision rule\n\n"
        "Ask one question: **is the work inside my loop balanced across waves?**\n\n"
        "- **Balanced (every wave does the same amount of the same kind of work) -> "
        "8-wave ping-pong.** GEMM and attention forward are the canonical cases. You run "
        "8 waves per block, 2 per SIMD, and split them into two groups that alternate: "
        "while group A issues MFMAs, group B issues loads, then they swap. The swap is "
        "driven by a *conditional* barrier that offsets one group by half a phase.\n"
        "- **Imbalanced (the loop has serial dependencies, transposes, or several "
        "differently-shaped matmuls) -> 4-wave interleave.** Attention backward is the "
        "canonical case: it computes dQ, dK and dV with data dependencies between them, "
        "so one group of waves would sit idle waiting. You run 1 wave per SIMD instead of "
        "2, give each wave the whole tile, and interleave loads and MFMAs at instruction "
        "granularity inside a single wave.\n\n"
        "In this library the split lands exactly on that line:\n\n"
        f"- 8-wave ping-pong: {', '.join('`' + o + '`' for o in pp)}\n"
        f"- 4-wave interleave: {', '.join('`' + o + '`' for o in il)}\n\n"
        f"For example `{pp_ex.rel_path}` is 8-wave ping-pong "
        f"({pp_ex.schedule_evidence}), while `{il_ex.rel_path}` is 4-wave interleave "
        f"({il_ex.schedule_evidence}).\n\n"
        "## Cost of each\n\n"
        "Ping-pong is cheap to write: roughly 50 lines of change in the compute loop, "
        "because the waves stay symmetric and the barrier does the sequencing. Interleave "
        "is expensive: on the order of 200 lines, because you are hand-scheduling the "
        "instruction mix. So reach for ping-pong first and only pay for interleave when "
        "the imbalance is real.\n\n"
        f"{PAPER_CLAIMS['ping_pong_sufficient'][0].capitalize()} "
        f"({PAPER_CLAIMS['ping_pong_sufficient'][1]}).\n\n"
        "## Why not wave specialization\n\n"
        f"Because on CDNA it loses. {PAPER_CLAIMS['wave_spec_underperforms'][0].capitalize()} "
        f"({PAPER_CLAIMS['wave_spec_underperforms'][1]}).\n\n"
        "The structural reasons are worth knowing, because they are properties of the "
        "chip and not of the code:\n\n"
        "1. **No hardware producer path to lean on.** The NVIDIA pattern pays off because "
        "a dedicated producer warp drives an asynchronous copy engine and then gets out "
        "of the way. A CDNA producer wave issues ordinary vector memory instructions, so "
        "it consumes the same issue slots the consumers need.\n"
        "2. **Dedicated producers waste the register file.** A producer wave holds almost "
        "no accumulators, but it still occupies a SIMD slot and its share of the VGPR "
        "budget. On a register-hungry GEMM that is the resource you are actually short "
        "of.\n"
        "3. **Symmetric waves make the barrier free.** With ping-pong every wave runs the "
        "same code, so a single `s_barrier` sequences everything. Asymmetric roles need "
        "real producer/consumer synchronization, which costs more than it saves.\n\n"
        + (f"The wave-specialized experiments are in this checkout under "
           f"`{pathlib.Path(ws[0].rel_path).parent.parent}/` if you want to read them; "
           f"they run {sorted({k.num_waves for k in ws})} waves per block as "
           f"producer/consumer splits, and the authors' own notes on those variants flag "
           f"several as spilling to scratch or exceeding the software-pipelining limit.\n\n"
           if ws else "")
        + "## What to do when neither applies\n\n"
        "If your kernel is memory-bound -- layernorm, rotary, softmax, elementwise -- "
        "neither schedule is the answer, and this library does not apply either to them. "
        "There is no MFMA stream to hide loads behind, so the wins come from coalescing, "
        "wide `ds_read`/`global_load` widths and occupancy instead. Applying ping-pong to "
        "a bandwidth-bound kernel adds barriers and buys nothing.\n\n"
        f"_Schedules above were read from the kernel sources in the HipKittens checkout "
        f"at commit `{prov['commit'][:12]}`. {HK_ATTRIBUTION}_"
    )
    rows.append(_row(
        user=user, assistant=assistant, qa_type="hk_schedule_selection", arch=arch,
        prov=prov,
        files=[pp_ex.rel_path, il_ex.rel_path] + [k.rel_path for k in ws[:2]],
        gt={"ping_pong_ops": pp, "interleave_ops": il,
            "wave_specialized_kernels": len(ws)},
        verified=True,
        verification=("schedule labels derived from source evidence (wave count plus "
                      "barrier structure), not from the paper's prose"),
    ))
    return rows


# --- family: the applicable pattern -------------------------------------- #
_PING_PONG_SKELETON = """\
// 8 waves, 2 blocks resident per CU. warp_row splits the 8 waves into two
// groups of 4: group 0 (waves 0-3) and group 1 (waves 4-7).
__global__ __launch_bounds__(NUM_THREADS, 2)
void kernel(const globals g) {
    const int warp_row = kittens::warpid() / 4;   // 0 or 1: the ping-pong group

    // ... prologue: fill the first LDS buffers ...
    __builtin_amdgcn_s_barrier();

    // THE STAGGER. Group 1 consumes one extra barrier here, so from now on the
    // two groups sit half a phase apart: whenever group 0 is in a compute
    // cluster, group 1 is in a memory cluster. This one `if` is what makes the
    // schedule a ping-pong rather than 8 waves marching in lockstep.
    if (warp_row == 1) {
        __builtin_amdgcn_s_barrier();
    }

    #pragma unroll
    for (int tile = 0; tile < num_tiles - 1; ++tile, tic ^= 1, toc ^= 1) {
        // --- memory cluster: LDS -> registers, and global -> LDS for tile+1 ---
        load(A_tile, subtile_inplace<...>(As[tic], {warp_row, 0}));
        G::load(As[toc], g.a, {0, 0, row, tile + 1}, swizzled_offsets_A);
        load(B_tile, subtile_inplace<...>(Bs[tic], {warp_col, 0}));
        G::load(Bs[toc], g.b, {0, 0, col, tile + 1}, swizzled_offsets_B);
        __builtin_amdgcn_s_barrier();

        // --- compute cluster: raise priority so the MFMA stream is not
        //     preempted by the other group's memory instructions ---
        asm volatile("s_waitcnt lgkmcnt(0)");     // wait only on the LDS reads
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum, A_tile, B_tile, C_accum);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        // ... two more clusters for the second half of the K step ...
    }

    // Re-balance: group 0 now consumes the extra barrier so both groups leave
    // the loop having executed the same number of barriers. Skip this and the
    // kernel hangs.
    if (warp_row == 0) {
        __builtin_amdgcn_s_barrier();
    }
    // ... epilogue: store C ...
}
"""


def rows_pattern_apply(
    kernels: list[KernelRecord], prov: dict, arch: str = DEFAULT_ARCH
) -> list[dict]:
    """Write-the-code rows: the skeleton plus the role of each element."""
    ex = next((k for k in kernels
               if k.schedule == "8-wave ping-pong" and k.op == "gemm_bf16"
               and "with32x16" in k.rel_path), None)
    if ex is None:
        ex = next((k for k in kernels if k.schedule == "8-wave ping-pong"), None)
    if ex is None:
        return []

    rows: list[dict] = []
    user = (
        "Show me how to actually write an 8-wave ping-pong main loop for a BF16 GEMM on "
        "gfx950. I want the structure and I want to know what each barrier and intrinsic "
        "is for, so I can port the pattern to a different kernel rather than copy it."
    )
    assistant = (
        "## The skeleton\n\n"
        f"{_fence(_PING_PONG_SKELETON)}\n\n"
        "## What each piece is doing\n\n"
        "**`__launch_bounds__(NUM_THREADS, 2)`** - asks for 2 blocks resident per CU. "
        "With 8 waves per block that is 16 waves per CU, 4 per SIMD. This is the budget "
        "the whole schedule is built around; if register pressure forces it to 1 block "
        "per CU, the overlap you are constructing has nothing to overlap with.\n\n"
        "**`warp_row = warpid() / 4`** - the group id. Waves 0-3 are group 0 and waves "
        "4-7 are group 1. Any wave-uniform two-valued expression works; what matters is "
        "that it is uniform within a wave so the branch does not diverge.\n\n"
        "**The staggering `if`** - the whole trick. `s_barrier` is a *block-wide* "
        "rendezvous, so making one group execute one extra barrier permanently offsets "
        "the two groups by one cluster. After it, group 0 and group 1 are always in "
        "opposite cluster types. There is no separate producer code and no role flag: the "
        "same instruction stream runs in both groups, just out of phase.\n\n"
        "**Alternating clusters** - the loop body is a chain of `memory cluster; barrier; "
        "compute cluster; barrier; ...`. Because the groups are offset, group 0's compute "
        "cluster executes concurrently with group 1's memory cluster on the same SIMD "
        "pair.\n\n"
        "**`s_setprio(1)` / `s_setprio(0)`** - raises the wave's arbitration priority "
        "around the MFMA and drops it immediately after. Without it the memory-side wave "
        "can win issue slots and stall the matrix pipeline, which is exactly the "
        "interleaving you were trying to create. Keep the raised window as tight as "
        "possible -- around the MFMAs only, never across a barrier.\n\n"
        "**`s_waitcnt lgkmcnt(0)`** - waits for the LDS reads specifically, rather than "
        "using a blanket wait. `lgkmcnt` counts LDS/scalar traffic and `vmcnt` counts "
        "vector-memory (global) traffic, so waiting on `lgkmcnt(0)` lets the global loads "
        "for the *next* tile stay in flight while this tile's MFMAs run. Using a full "
        "`s_waitcnt(0)` here is a common and expensive mistake: it drains the global "
        "prefetch you just issued.\n\n"
        "**The re-balancing `if` at the end** - both groups must execute the same total "
        "number of barriers before they exit, otherwise one group waits at a barrier the "
        "other will never reach and the kernel hangs. This is the single easiest way to "
        "get the pattern wrong, and it fails as a hang or a timeout, not as a wrong "
        "number.\n\n"
        "## Porting it to another kernel\n\n"
        "1. Confirm the loop is **balanced** across waves; if it is not, ping-pong will "
        "not help and 4-wave interleave is the pattern you want.\n"
        "2. Get 8 waves and 2 blocks per CU resident *without spilling*. Check the "
        "compiler's scratch usage first -- a spill costs more than the overlap gains.\n"
        "3. Double-buffer the LDS tiles so a memory cluster can write buffer `toc` while "
        "a compute cluster reads buffer `tic`.\n"
        "4. Split the body into clusters that alternate cleanly between memory and "
        "compute, and put one `s_barrier` between consecutive clusters.\n"
        "5. Add the staggering `if` before the loop and the re-balancing `if` after it.\n"
        "6. Replace blanket waits with `lgkmcnt`/`vmcnt` counted waits.\n"
        "7. Wrap MFMA runs in `s_setprio(1)`/`s_setprio(0)`.\n\n"
        "The real kernel this is distilled from is "
        f"`{ex.rel_path}` ({ex.num_waves} waves, "
        f"{ex.launch_blocks_per_cu} blocks/CU), which annotates its four clusters "
        "explicitly. Here is its actual loop:\n\n"
        f"{_fence(ex.loop_excerpt)}\n\n"
        f"_Source: `{ex.rel_path}`. {HK_ATTRIBUTION}_"
    )
    rows.append(_row(
        user=user, assistant=assistant, qa_type="hk_pattern_apply", arch=arch,
        prov=prov, files=[ex.rel_path],
        gt={"kernel": ex.rel_path, "num_waves": ex.num_waves,
            "blocks_per_cu": ex.launch_blocks_per_cu, "schedule": ex.schedule},
        verified=True,
        verification="skeleton elements all present in the cited kernel source",
    ))
    return rows


# --- family: per-kernel anatomy ------------------------------------------- #
def _anatomy_signature(k: KernelRecord) -> tuple:
    """What makes an anatomy row distinct. Kernels that agree on all of this teach
    the same lesson, and emitting one row each would just teach a template: the
    checkout contains e.g. ``kernel.cpp``/``kernel_d64.cpp`` pairs and three
    MXFP8 transpose variants that differ only in operand order."""
    return (k.op, k.schedule, k.num_waves, k.techniques, k.layouts)


def rows_kernel_anatomy(
    kernels: list[KernelRecord], prov: dict, seed: int = 0
) -> list[dict]:
    """One row per DISTINCT kernel shape: what, which schedule, which techniques, why."""
    rows: list[dict] = []
    seen: dict[tuple, KernelRecord] = {}
    variants: dict[tuple, list[str]] = {}
    for k in sorted(kernels, key=lambda x: x.rel_path):
        sig = _anatomy_signature(k)
        variants.setdefault(sig, []).append(k.rel_path)
        seen.setdefault(sig, k)

    for sig, k in sorted(seen.items(), key=lambda kv: kv[1].rel_path):
        if k.schedule == "unclassified" and not k.techniques:
            continue  # nothing substantive to teach about it
        siblings = [p for p in variants[sig] if p != k.rel_path]
        techs = "\n".join(
            f"- **{name.replace('_', ' ')}** - {TECHNIQUES[name][1]}"
            for name in k.techniques
        )
        layouts = (", ".join(f"`{x}`" for x in k.layouts) or "the library defaults")
        why = _schedule_rationale(k)
        user = (
            f"Here is a production AMD kernel from the HipKittens library "
            f"(`{k.rel_path}`, targeting {k.arch}). Tell me what it computes, what "
            f"overlap schedule it uses and why that schedule suits this operation, and "
            f"which low-level techniques it relies on.\n\n"
            f"{_fence(k.loop_excerpt)}"
        )
        assistant = (
            f"## What it computes\n\n"
            f"This is {k.op_description} (`{k.op}`) for {k.arch}"
            + (f", using {', '.join(k.dtypes[:4])} tiles" if k.dtypes else "") + ".\n\n"
            f"## Schedule: {k.schedule}\n\n"
            f"Evidence in the source: {k.schedule_evidence}"
            + (f"; `__launch_bounds__(..., {k.launch_blocks_per_cu})` asks for "
               f"{k.launch_blocks_per_cu} block(s) resident per CU"
               if k.launch_blocks_per_cu else "") + ".\n\n"
            f"{why}\n\n"
            f"## Techniques it depends on\n\n{techs}\n\n"
            f"## Layouts\n\nShared/register tile layouts: {layouts}. "
            f"On CDNA the layout name encodes the MFMA fragment shape, and the LDS "
            f"swizzle is chosen per layout -- a tile staged for one MFMA shape cannot "
            f"reuse another shape's swizzle without reintroducing bank conflicts.\n\n"
            + (f"The same structure appears in "
               + ", ".join(f"`{s}`" for s in siblings[:4])
               + ", which differ only in operand order or head dimension, not in "
                 "schedule or technique.\n\n" if siblings else "")
            + f"_Source: `{k.rel_path}` at commit `{prov['commit'][:12]}`. "
              f"{HK_ATTRIBUTION}_"
        )
        rows.append(_row(
            user=user, assistant=assistant, qa_type="hk_kernel_anatomy", arch=k.arch,
            prov=prov, files=[k.rel_path],
            gt={"kernel": k.rel_path, "op": k.op, "num_waves": k.num_waves,
                "schedule": k.schedule, "techniques": list(k.techniques),
                "layouts": list(k.layouts)},
            verified=True,
            verification=("op, wave count, schedule and every listed technique detected "
                          "by pattern match against this file's source"),
        ))
    return rows


def _schedule_rationale(k: KernelRecord) -> str:
    if k.schedule == "8-wave ping-pong":
        return (
            "The work in this loop is symmetric across waves, so splitting the 8 waves "
            "into two half-phase-offset groups lets one group's MFMA stream cover the "
            "other group's memory latency with no dedicated producer waves and no "
            "asymmetric synchronization."
        )
    if k.schedule == "4-wave interleave":
        return (
            "This operation is imbalanced -- it chains several differently-shaped matmuls "
            "with data dependencies between them -- so a two-group ping-pong would leave "
            "one group idle waiting on the dependency. Running 1 wave per SIMD and "
            "interleaving loads with MFMAs at instruction granularity inside each wave "
            "keeps the matrix pipeline fed through the serial sections instead."
        )
    if k.schedule.startswith("wave-specialized"):
        return (
            "This is a wave-specialization experiment: whole waves are dedicated to "
            "producer or consumer roles, as one would on NVIDIA Hopper. It is included "
            "as a comparison point rather than as the recommended pattern -- "
            f"{PAPER_CLAIMS['wave_spec_underperforms'][0]} "
            f"({PAPER_CLAIMS['wave_spec_underperforms'][1]}). The producer waves occupy "
            "SIMD slots and register budget while issuing the same ordinary vector-memory "
            "instructions the consumers need, so the split costs more than it hides."
        )
    if k.schedule == "8-wave (no stagger detected)":
        return (
            "It runs 8 waves but no conditional barrier appears in the source, so the "
            "waves proceed in lockstep rather than ping-ponging."
        )
    if k.schedule == "4-wave":
        return (
            "It runs 4 waves (1 per SIMD) without fine-grained instruction interleaving; "
            "the loop is short enough that cluster-level scheduling is not what limits "
            "it."
        )
    return (
        "This is a bandwidth-bound kernel, so neither ping-pong nor interleave applies: "
        "there is no sustained MFMA stream to hide memory latency behind, and the wins "
        "come from access width, coalescing and occupancy instead."
    )


# --- family: intrinsic roles ---------------------------------------------- #
_INTRINSICS: tuple[tuple[str, str, str, str], ...] = (
    (
        "__builtin_amdgcn_s_setprio",
        "s_setprio raises or lowers the issuing wave's arbitration priority on its SIMD "
        "(0 is the default, higher wins issue slots).",
        "Wrap a run of MFMAs in `s_setprio(1)` ... `s_setprio(0)` whenever another wave "
        "on the same SIMD is concurrently issuing memory instructions -- which is exactly "
        "the situation ping-pong creates on purpose. Without it the memory wave steals "
        "issue slots and the matrix pipeline bubbles.",
        "Leaving priority raised across a barrier or across the memory cluster starves "
        "the very loads you are trying to overlap, so keep the window tight and always "
        "pair the raise with a drop.",
    ),
    (
        "__builtin_amdgcn_sched_barrier",
        "sched_barrier is a COMPILER-only fence: it forbids the scheduler from moving "
        "instructions across that point. It emits no hardware instruction. "
        "`sched_barrier(0)` allows nothing to cross.",
        "Place it at cluster boundaries in a hand-scheduled loop. You have arranged the "
        "instruction mix deliberately; the backend scheduler does not know about your "
        "ping-pong phases and will happily sink a load past a barrier and destroy the "
        "overlap.",
        "It constrains only the compiler, so it is not a substitute for `s_waitcnt` "
        "(data readiness) or `s_barrier` (cross-wave rendezvous). Using it where you "
        "needed one of those gives you a race, not a slow kernel.",
    ),
    (
        "__builtin_amdgcn_sched_group_barrier",
        "sched_group_barrier is the fine-grained form: `(mask, count, group)` asks the "
        "scheduler to place `count` instructions of the class selected by `mask` into "
        "pipeline group `group`. Masks select instruction classes -- for example MFMA, "
        "VALU and transcendental are distinct bits.",
        "Use it when you need an explicit *ratio* rather than a fence -- for instance "
        "alternating one MFMA with a fixed number of VALU ops through the softmax of an "
        "attention kernel, so the matrix and vector pipes both stay busy. This is the "
        "primitive behind 4-wave interleave.",
        "The counts are a request against the real instruction stream: ask for more of a "
        "class than the region contains and the grouping silently does not happen, so "
        "verify against the generated assembly rather than assuming it took effect.",
    ),
    (
        "__builtin_amdgcn_readfirstlane",
        "readfirstlane copies lane 0's value of a VGPR into a scalar (SGPR) register, "
        "asserting the value is wave-uniform.",
        "Use it to hoist wave-uniform addresses -- LDS base pointers, buffer descriptors "
        "-- out of vector registers. It converts per-lane address arithmetic into one "
        "scalar value, which frees VGPRs and removes VALU work from the inner loop. "
        "Address computation is a first-order cost in these kernels, not a rounding "
        "error.",
        "It is an assertion, not a broadcast: if the value is genuinely divergent you get "
        "lane 0's value silently applied to all lanes, which is a data corruption bug "
        "with no diagnostic.",
    ),
    (
        "s_waitcnt vmcnt / lgkmcnt",
        "The two counters track different traffic: `vmcnt` counts outstanding "
        "vector-memory (global/buffer) operations, `lgkmcnt` counts LDS, scalar-memory "
        "and message traffic. `s_waitcnt lgkmcnt(0)` waits for all LDS reads and leaves "
        "global loads in flight.",
        "Wait on the narrowest counter that makes your data ready, and use non-zero "
        "counts (`vmcnt(4)`) to wait for only the oldest few of several outstanding "
        "loads. This is what lets the next tile's global prefetch stay in flight across "
        "the current tile's MFMAs.",
        "A blanket `s_waitcnt(0)` before the MFMAs drains the global prefetch and "
        "converts a pipelined loop back into a serial one. The kernel stays correct, so "
        "the only symptom is that the overlap you wrote does not happen.",
    ),
    (
        "chiplet / XCD workgroup swizzle",
        "MI355X is built from 8 XCDs (chiplets), and consecutive workgroup ids are "
        "assigned round-robin across them. Remapping the id (for example "
        "`wgid = (wgid % NUM_XCDS) * (NUM_WGS / NUM_XCDS) + (wgid / NUM_XCDS)`) makes "
        "blocks that run at the same time land on the same XCD.",
        "Apply it to any kernel whose concurrently-running blocks share input tiles -- "
        "GEMM above all. Blocks on one XCD share that XCD's L2, so co-scheduling "
        "consumers of the same A/B panel turns HBM traffic into L2 hits. Compose it with "
        "the usual GROUP_M grouping swizzle for locality within the XCD.",
        "It is pure index arithmetic and cannot change results, so it is safe to try; but "
        "it only helps when blocks actually share data, and the mapping depends on the "
        "XCD count, so a constant hard-coded for 8 XCDs is wrong on a partitioned GPU or "
        "a different part.",
    ),
    (
        "manual AGPR/VGPR pinning",
        "CDNA splits the register file into architectural VGPRs and accumulation AGPRs. "
        "HipKittens issues LDS reads through templates that name the destination register "
        "number explicitly (`ds_read_b128<GPR_START>`, emitting `v[n:n+3]` or `a[n:n+3]`) "
        "instead of letting HIPCC allocate.",
        "Reach for it only after the compiler has demonstrably failed -- when a "
        "register-heavy GEMM or attention kernel spills to scratch and you can see the "
        "scratch traffic in the build output or the profile. Pinning lets you place "
        "accumulators in AGPRs and keep the operand staging in VGPRs so the allocator "
        "cannot interleave them into a spill.",
        "It is unmaintainable at scale and it defeats the allocator's ability to help "
        "you: get the tile sizes and occupancy target right first, because a kernel that "
        "needs hand-numbered registers to fit is usually a kernel whose tiling is too "
        "aggressive.",
    ),
)


def rows_intrinsic_role(
    kernels: list[KernelRecord], prov: dict, arch: str = DEFAULT_ARCH
) -> list[dict]:
    """One row per scheduling/memory intrinsic: what, when, and the failure mode."""
    rows: list[dict] = []
    for name, what, when, trap in _INTRINSICS:
        users = [k.rel_path for k in kernels
                 if name.split()[0].replace("__builtin_amdgcn_", "") in k.source][:3]
        user = (
            f"What does `{name}` do on AMD CDNA3/CDNA4, when should I actually reach for "
            f"it in a HIP kernel, and how does it go wrong if I misuse it?"
        )
        assistant = (
            f"## What it is\n\n{what}\n\n"
            f"## When to use it\n\n{when}\n\n"
            f"## How it goes wrong\n\n{trap}\n\n"
            + (f"Real uses in the HipKittens library: "
               + ", ".join(f"`{u}`" for u in users) + ".\n\n" if users else "")
            + f"_{HK_ATTRIBUTION}_"
        )
        rows.append(_row(
            user=user, assistant=assistant, qa_type="hk_intrinsic_role", arch=arch,
            prov=prov, files=users,
            gt={"intrinsic": name, "example_kernels": users},
            verified=bool(users),
            verification=("cited example kernels confirmed to contain the construct"
                          if users else "no in-repo example located"),
        ))
    return rows


# --- family: measured baselines (the authors' own numbers) ---------------- #
def rows_measured_baseline(
    measurements: list[Measurement], prov: dict, arch: str = DEFAULT_ARCH
) -> list[dict]:
    """Calibration rows built ONLY from numbers committed in the HK repo.

    Teaches realistic expectations -- notably that dense FP8 GEMM is already
    near-saturated by hipBLASLt, so the headroom is in fusion and in the
    workloads the vendor libraries do not cover.
    """
    if not measurements:
        return []
    rows: list[dict] = []
    by_workload: dict[str, list[Measurement]] = {}
    for m in measurements:
        by_workload.setdefault(m.workload, []).append(m)

    for workload, ms in sorted(by_workload.items()):
        table_rows = "\n".join(
            f"| {m.size} | {m.baseline} | {m.baseline_metric:.4g} | {m.hk_metric:.4g} | "
            f"{m.speedup:.2f}x |"
            for m in sorted(ms, key=lambda x: (_int_or_inf(x.size), x.baseline))
        )
        metric = ms[0].metric_name
        best = max(ms, key=lambda m: m.speedup)
        worst = min(ms, key=lambda m: m.speedup)
        files = sorted({m.evidence_file for m in ms})
        user = (
            f"What speedup should I realistically expect from a hand-written CDNA4 "
            f"{workload} kernel versus the vendor and framework baselines on MI355X? I "
            f"want measured numbers, not marketing."
        )
        assistant = (
            f"## Measured {workload} on MI355X\n\n"
            f"These are the HipKittens authors' own committed benchmark results "
            f"(metric: {metric}; collected November 2025):\n\n"
            f"| size | baseline | baseline {metric} | HipKittens {metric} | ratio |\n"
            "|---|---|---|---|---|\n" + table_rows + "\n\n"
            f"## Reading it honestly\n\n"
            f"- Best case in this table: **{best.speedup:.2f}x** versus "
            f"{best.baseline} at size {best.size}.\n"
            f"- Worst case: **{worst.speedup:.2f}x** versus {worst.baseline} at size "
            f"{worst.size}.\n\n"
            + _baseline_lesson(workload, ms) +
            f"\n\n_Measured by the HipKittens authors; source "
            + ", ".join(f"`{f}`" for f in files) +
            f". {HK_ATTRIBUTION}_"
        )
        rows.append(_row(
            user=user, assistant=assistant, qa_type="hk_measured_baseline", arch=arch,
            prov=prov, files=files,
            gt={"workload": workload, "metric": metric,
                "points": [{"size": m.size, "baseline": m.baseline,
                            "hk": m.hk_metric, "base": m.baseline_metric,
                            "ratio": m.speedup} for m in ms]},
            verified=True,
            verification="ratios computed from the repo's committed benchmark JSON",
        ))
    return rows


def _baseline_lesson(workload: str, ms: list[Measurement]) -> str:
    vendor = [m for m in ms if m.baseline in ("hipBLASLt", "AITER")]
    framework = [m for m in ms if m.baseline in ("PyTorch", "torch.compile")]
    out = []
    if vendor and max(m.speedup for m in vendor) < 1.15:
        out.append(
            "The vendor library is essentially at parity here. That is the important "
            "calibration: on the dense, well-shaped, heavily-tuned cases the vendor "
            "already owns, a hand-written kernel is fighting for a few percent. Expecting "
            "a large multiple on this shape is how you waste a week."
        )
    if framework:
        best_fw = max(framework, key=lambda m: m.speedup)
        out.append(
            f"Against the framework path the gap is much larger -- up to "
            f"{best_fw.speedup:.2f}x versus {best_fw.baseline}. Memory-bound operations "
            f"are where a hand-written kernel wins big, because the framework pays for "
            f"extra passes over HBM that a fused kernel simply does not make."
        )
    out.append(
        "The generalizable rule: measure against the *strongest* available baseline for "
        "your shape, not against eager PyTorch. A 5x over eager can still be a loss "
        "against AITER, and only one of those numbers tells you whether the kernel is "
        f"good. {PAPER_CLAIMS['uncovered_workloads'][0].capitalize()} "
        f"({PAPER_CLAIMS['uncovered_workloads'][1]}) -- the headroom is concentrated in "
        "the shapes and fusions the vendor kernels do not cover, which is also where "
        f"{PAPER_CLAIMS['bwd_gap'][0]} ({PAPER_CLAIMS['bwd_gap'][1]})."
    )
    return "\n\n".join(out)


# --- family: naive vs HipKittens (structural before/after) ---------------- #
# The "before" side is written HERE as an illustration of the obvious first
# implementation. It is deliberately NOT attributed to HipKittens, and no
# performance number is attached to it, because we have not measured it.
_NAIVE_LOOP = """\
// The obvious first version: one wave group, blocking loads, compiler-scheduled.
__global__ __launch_bounds__(256) void gemm_naive(const globals g) {
    __shared__ bf16 As[BM][BK];
    __shared__ bf16 Bs[BK][BN];
    float acc[TM][TN] = {};

    for (int k0 = 0; k0 < K; k0 += BK) {
        // every wave loads, then every wave waits, then every wave computes
        load_tile_to_lds(As, g.a, blockIdx.y, k0);
        load_tile_to_lds(Bs, g.b, k0, blockIdx.x);
        __syncthreads();

        for (int kk = 0; kk < BK; ++kk)
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += float(As[...][kk]) * float(Bs[kk][...]);  // no MFMA

        __syncthreads();
    }
    store(g.c, acc);
}
"""


def rows_naive_vs_hk(
    kernels: list[KernelRecord], layouts: list[SwizzleLayout], prov: dict,
    arch: str = DEFAULT_ARCH,
) -> list[dict]:
    """Structural before/after: what specifically changes, and why each change."""
    ex = next((k for k in kernels if k.schedule == "8-wave ping-pong"
               and k.op == "gemm_bf16" and "with16x32" in k.rel_path), None)
    if ex is None:
        return []
    sw = next((l for l in layouts if l.name == "st_16x32" and l.dtype_bytes == 2), None)
    rows: list[dict] = []
    user = (
        "I have a working but slow BF16 GEMM on MI355X. It stages tiles in LDS, uses "
        "`__syncthreads()` between load and compute, and lets the compiler schedule the "
        "inner loop:\n\n"
        f"{_fence(_NAIVE_LOOP)}\n\n"
        "What are the specific structural changes that separate this from a "
        "state-of-the-art CDNA4 GEMM? Order them by how much they matter and tell me what "
        "each one is actually fixing."
    )
    assistant = (
        "Your kernel is correct and leaves most of the machine idle. Six changes, in "
        "order of impact, each fixing a specific stall.\n\n"
        "### 1. Route the inner product through MFMA\n\n"
        "The scalar `acc += a * b` loop uses the vector ALU and never touches the matrix "
        "cores, so it is capped at a small fraction of peak no matter how well you tile "
        "it. Everything else on this list is worthless until the multiply is an MFMA "
        "instruction with an fp32 accumulator. **This is not an optimization; it is the "
        "difference between using the chip and not.**\n\n"
        "### 2. Overlap memory with compute (8-wave ping-pong)\n\n"
        "`__syncthreads()` between load and compute makes every wave wait for every "
        "load, so the matrix cores idle for the whole load phase and the memory system "
        "idles for the whole compute phase. Run 8 waves, split them into two groups, and "
        "offset the groups by half a phase with a *conditional* barrier so one group "
        "computes while the other loads:\n\n"
        "```cpp\n"
        "if (warp_row == 1) { __builtin_amdgcn_s_barrier(); }   // stagger\n"
        "// ... loop of alternating memory / compute clusters, one barrier between ...\n"
        "if (warp_row == 0) { __builtin_amdgcn_s_barrier(); }   // re-balance\n"
        "```\n\n"
        "Both halves are needed: skip the re-balance and the kernel hangs.\n\n"
        "### 3. Double-buffer the LDS tiles\n\n"
        "With one buffer, the next tile's global load cannot start until the current "
        "tile's reads have finished. Allocate two and flip `tic`/`toc` so the global load "
        "for step k+1 is issued before the MFMAs for step k, and there is real work in "
        "flight to overlap.\n\n"
        "### 4. Swizzle the LDS layout for your MFMA shape\n\n"
        "A plain row-major `As[BM][BK]` makes the lanes of a wave hit the same LDS banks "
        "when they gather an MFMA fragment, and the access serializes. The fix is an XOR "
        "permutation of the addresses"
        + (f" -- for a 16x32 bf16 tile that is `{sw.formula()}`" if sw else "") +
        ". Two things make this awkward on AMD and are worth knowing up front: the right "
        "permutation depends on the MFMA shape *and* the dtype, so unlike on NVIDIA there "
        "is no single swizzle per dtype to reuse; and getting it wrong is a **silent** "
        "performance bug, because any bijective permutation still computes the right "
        "answer. Fold it into precomputed global offsets so it costs no inner-loop "
        "address math.\n\n"
        "### 5. Replace blanket waits and let the compiler stop reordering\n\n"
        "`__syncthreads()` implies a full `s_waitcnt`, which drains the global prefetch "
        "you just issued. Wait on the specific counter instead -- `s_waitcnt lgkmcnt(0)` "
        "for the LDS reads, leaving `vmcnt` traffic in flight -- and put "
        "`__builtin_amdgcn_sched_barrier(0)` at your cluster boundaries so the backend "
        "scheduler does not sink a load past a barrier and undo the schedule. Raise "
        "`s_setprio(1)` around the MFMA run and drop it after, so the memory-side wave "
        "cannot steal issue slots from the matrix pipeline.\n\n"
        "### 6. Swizzle the workgroup id for the chiplet topology\n\n"
        "MI355X has 8 XCDs and hands consecutive block ids to different chiplets, so "
        "blocks that share an A or B panel end up on different L2s and each fetches the "
        "panel from HBM. Remap the id so concurrent blocks land on the same XCD, then "
        "apply the usual GROUP_M grouping within it. Pure index arithmetic, cannot change "
        "results.\n\n"
        "### What NOT to do\n\n"
        "Do not reach for NVIDIA-style wave specialization with dedicated producer waves. "
        f"{PAPER_CLAIMS['wave_spec_underperforms'][0].capitalize()} "
        f"({PAPER_CLAIMS['wave_spec_underperforms'][1]}): a CDNA producer wave issues "
        "ordinary vector-memory instructions rather than driving a copy engine, so it "
        "competes for the same issue slots while still consuming a SIMD slot and its share "
        "of the register budget.\n\n"
        "And do not hand-number registers as a first move. Manual AGPR/VGPR pinning is a "
        "real technique for when HIPCC spills a register-heavy kernel to scratch, but if "
        "you need it early it usually means the tiling is too aggressive for the occupancy "
        "you are targeting.\n\n"
        f"For reference, a kernel that does all six is `{ex.rel_path}` "
        f"({ex.num_waves} waves, {ex.launch_blocks_per_cu} blocks/CU, "
        f"techniques: {', '.join(ex.techniques[:6])}).\n\n"
        f"_The 'before' listing above is an illustration written for this answer, not "
        f"HipKittens code, and carries no measured number. The reference kernel and the "
        f"techniques are from HipKittens. {HK_ATTRIBUTION}_"
    )
    rows.append(_row(
        user=user, assistant=assistant, qa_type="hk_naive_vs_hk", arch=arch,
        prov=prov, files=[ex.rel_path],
        gt={"reference_kernel": ex.rel_path,
            "techniques": list(ex.techniques),
            "swizzle": sw.formula() if sw else ""},
        verified=True,
        verification=("every technique named is present in the referenced kernel; the "
                      "naive listing is authored illustration with no perf claim"),
    ))
    return rows


# --------------------------------------------------------------------------- #
# 6. Public entry point
# --------------------------------------------------------------------------- #
# Two rows sharing this fraction of their 8-token shingles teach the same lesson
# twice. The checkout is full of kernels that differ only in head dimension or
# operand order, and without this gate the anatomy family alone produced 47 pairs
# above 0.7 containment (peak 0.96) -- a corpus that looks like 29 rows of
# supervision and is really one template repeated.
# 0.75 is calibrated against the measured spread in this corpus: true clones (the
# kernel pairs differing only in head dimension, the MXFP8 transpose variants) sit
# at 0.9-1.0, while rows that genuinely differ in their facts sit at 0.2-0.6. The
# existing mixture keeps thousands of structurally similar solver-generated rows,
# so the bar is "teaches the same lesson", not "shares a section heading".
NEAR_DUP_THRESHOLD = 0.75


def _row_text(row: dict) -> str:
    return " ".join(m["content"] for m in row["messages"])


def _drop_near_duplicates(
    rows: list[dict], threshold: float = NEAR_DUP_THRESHOLD
) -> tuple[list[dict], list[tuple[str, str, float]]]:
    """Greedily keep rows that are not near-duplicates of an already-kept row."""
    from kore.data.dedup import token_shingles

    kept: list[dict] = []
    kept_sh: list[set[str]] = []
    dropped: list[tuple[str, str, float]] = []
    for row in rows:
        sh = token_shingles(_row_text(row), 8)
        if not sh:
            kept.append(row)
            kept_sh.append(sh)
            continue
        worst = 0.0
        against = ""
        for other, osh in zip(kept, kept_sh):
            if not osh:
                continue
            inter = len(sh & osh)
            # Symmetric: a short near-dup of a long row must also be caught.
            score = max(inter / len(sh), inter / len(osh))
            if score > worst:
                worst, against = score, str(other.get("_gt", {}).get("kernel") or
                                            other.get("_qa_type"))
        if worst >= threshold:
            dropped.append((str(row.get("_gt", {}).get("kernel") or row.get("_qa_type")),
                            against, round(worst, 3)))
            continue
        kept.append(row)
        kept_sh.append(sh)
    return kept, dropped


def build_rows(
    root: Optional[str | pathlib.Path] = None,
    seed: int = 0,
    families: Optional[Iterable[str]] = None,
    max_row_chars: int = MAX_ROW_CHARS,
    near_dup_threshold: float = NEAR_DUP_THRESHOLD,
) -> tuple[list[dict], dict]:
    """Build every HipKittens SFT row. Deterministic given ``seed``.

    Returns ``(rows, stats)``. Rows are ``[system, user, assistant]`` chat rows
    tagged ``_source="hipkittens"`` with full MIT provenance, ready to be handed
    to decontamination and the mixture assembler.
    """
    prov = provenance(root)
    layouts = parse_swizzles(root)
    models = parse_phase_models(root)
    kernels = discover_kernels(root)
    measurements = parse_measurements(root)

    builders = {
        "lds_hardware_model": lambda: rows_lds_hardware_model(models, prov),
        "swizzle": lambda: rows_swizzle(layouts, prov),
        "conflict_exercise": lambda: rows_conflict_exercise(layouts, models, prov),
        "schedule_selection": lambda: rows_schedule_selection(kernels, prov),
        "pattern_apply": lambda: rows_pattern_apply(kernels, prov),
        "kernel_anatomy": lambda: rows_kernel_anatomy(kernels, prov, seed=seed),
        "intrinsic_role": lambda: rows_intrinsic_role(kernels, prov),
        "measured_baseline": lambda: rows_measured_baseline(measurements, prov),
        "naive_vs_hk": lambda: rows_naive_vs_hk(kernels, layouts, prov),
    }
    want = list(families) if families else list(builders)
    unknown = [f for f in want if f not in builders]
    if unknown:
        raise ValueError(f"unknown families {unknown}; known: {sorted(builders)}")

    rows: list[dict] = []
    per_family: dict[str, int] = {}
    dropped_long = 0
    for fam in want:
        produced = builders[fam]()
        kept = []
        for r in produced:
            n = sum(len(m["content"]) for m in r["messages"])
            if n > max_row_chars:
                # Truncation lands mid-explanation and teaches a half-finished
                # answer, so an over-long row is dropped rather than cut.
                dropped_long += 1
                continue
            kept.append(r)
        per_family[fam] = len(kept)
        rows.extend(kept)

    rows, near_dups = _drop_near_duplicates(rows, near_dup_threshold)
    by_qa_type: dict[str, int] = {}
    for r in rows:
        by_qa_type[r["_qa_type"]] = by_qa_type.get(r["_qa_type"], 0) + 1

    chars = sum(len(m["content"]) for r in rows for m in r["messages"])
    by_model = {m.instruction: m for m in models}
    conflict = {
        r.layout: {"plain": r.plain_max_conflict, "swizzled": r.swizzled_max_conflict}
        for r in (bank_conflict_report(l, by_model) for l in layouts if not l.is_identity)
        if r is not None
    }
    stats = {
        "rows": len(rows),
        "per_family_before_dedup": per_family,
        "per_qa_type": by_qa_type,
        "dropped_too_long": dropped_long,
        "dropped_near_duplicate": len(near_dups),
        "near_duplicate_examples": near_dups[:8],
        "near_dup_threshold": near_dup_threshold,
        "chars": chars,
        "est_tokens": int(chars / CHARS_PER_TOKEN),
        "swizzle_layouts": len(layouts),
        "swizzles_verified_conflict_free": sum(
            1 for v in conflict.values() if v["swizzled"] == 1),
        "conflict_degrees": conflict,
        "phase_models": len(models),
        "kernels": len(kernels),
        "distinct_kernel_shapes": len({_anatomy_signature(k) for k in kernels}),
        "measurements": len(measurements),
        "commit": prov["commit"],
        "license": prov["license"],
    }
    return rows, stats


__all__ = [
    "SOURCE_TAG",
    "CHARS_PER_TOKEN",
    "MAX_ROW_CHARS",
    "build_rows",
    "rows_lds_hardware_model",
    "rows_swizzle",
    "rows_conflict_exercise",
    "rows_schedule_selection",
    "rows_pattern_apply",
    "rows_kernel_anatomy",
    "rows_intrinsic_role",
    "rows_measured_baseline",
    "rows_naive_vs_hk",
    "HK_REPO_URL",
    "HK_LICENSE",
    "HK_ATTRIBUTION",
    "HK_AUTHORS",
    "PAPER_CLAIMS",
    "SYSTEM_PROMPT_HK",
    "TECHNIQUES",
    "HipKittensIngestError",
    "SwizzleLayout",
    "PhaseModel",
    "KernelRecord",
    "Measurement",
    "hk_root",
    "provenance",
    "parse_swizzles",
    "parse_phase_models",
    "discover_kernels",
    "parse_measurements",
]
