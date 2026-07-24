"""Generate repair data (KORE Stage 1: repair-weighted SFT).

Two sources of broken->fixed turns:
  1. INJECTED breakage: take the known-good seed, apply a ``mutate`` breakage,
     confirm via the verifier that it fails, then ask the teacher to repair it
     conditioned on the exact error, and confirm the fix.
  2. NATURAL failures: sample fresh candidates from the teacher; whenever one
     fails the verifier, mine it as a repair opportunity the same way.

Each accepted turn becomes a ``RepairRecord`` whose ``messages`` are the chat
that produced the fix (system + repair-user + assistant), so it can go straight
into SFT via ``build_datasets.build_sft``.
"""

from __future__ import annotations

import difflib
import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from kore.data.amd_knowledge import live_system_prompt
from kore.data.prompts import (
    SYSTEM_PROMPT,
    build_turn_prompt,
    extract_kernel,
    format_assistant_turn,
)
from kore.data.mutate import apply_random_breakage, infer_family
from kore.data.schemas import RepairRecord
from kore.data.teacher import TeacherClient
from kore.env.replay import kernel_hash
from kore.obs import get_logger

log = get_logger("data.gen_repair")

# Per-attempt context (idx / mutator / broke_verified_fail) that the generation
# loops set right before calling ``make_repair_record`` so it can emit a fully
# populated ``repair_attempt`` event without a signature change. Thread-local so
# concurrent datagen never crosses wires. Purely observational.
_ctx = threading.local()


def _failure_class(obs) -> Optional[str]:
    """Map a verifier Observation to a failure bucket, or None if it passed."""
    if not obs.compiled:
        return "compile_fail"
    if not obs.validation_passed:
        return "snr_fail"
    return None


def _error_text(obs) -> str:
    if obs.error_text:
        return obs.error_text
    if not obs.compiled:
        return "kernel failed to compile"
    return f"correctness failed (snr_db={obs.snr_db})"


# --------------------------------------------------------------------------- #
# Evidence-based repair diagnosis (structured diff analyzer)
#
# Problem (audited): the repair chain-of-thought used to be 100% TEMPLATED - it
# emitted one of two fixed strings and never named the actual bug. This block
# replaces that with a DETERMINISTIC analyzer that reads the real difference
# between the BROKEN (parent) source and the VERIFIED FIXED (child) source and
# names the concrete change class + the concrete token that changed. Everything
# it says is grounded ONLY in what the diff shows; when the diff is not a single
# recognizable pattern (e.g. the teacher rewrote the kernel) it falls back to a
# minimal factual statement (the verifier error + the one concrete changed token)
# rather than fabricating a mechanism.
# --------------------------------------------------------------------------- #
@dataclass
class DiffFinding:
    """A recognized broken->fixed change: the class + the concrete tokens."""

    change_class: str
    before: str
    after: str


_CMP_INVERSE = {"<": ">=", ">=": "<", ">": "<=", "<=": ">"}

# A bounds-style comparison (optionally indexed / with a +BLOCK term) on a line
# that concerns masking or offsets - the only place a flipped predicate matters.
_PRED_RE = re.compile(
    r"(\(?\s*[A-Za-z_]\w*(?:\[[^\]]*\])?(?:\s*[+\-]\s*[A-Za-z0-9_]+)?\s*)"
    r"(<=|>=|<|>)"
    r"(\s*\(?\s*[A-Za-z_]\w*(?:\[[^\]]*\])?(?:\s*[+*]\s*[A-Za-z0-9_]+)?\s*\)?)"
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _src_tokens(s: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", s or "")


def _changed_regions(broken: str, fixed: str) -> tuple[str, str, float]:
    """Return (broken_diff_text, fixed_diff_text, similarity_ratio).

    The diff texts are ONLY the lines that differ, so every detector reasons
    strictly about what changed - never about unchanged code."""
    b = broken.splitlines()
    f = fixed.splitlines()
    sm = difflib.SequenceMatcher(a=b, b=f, autojunk=False)
    before, after = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        before.extend(b[i1:i2])
        after.extend(f[j1:j2])
    return "\n".join(before), "\n".join(after), sm.ratio()


def _salient_token_change(broken: str, fixed: str) -> Optional[tuple[str, str]]:
    """The single smallest concrete token edit between the two sources.

    Used for the fallback ("the one concrete token that changed") when the diff
    is not a recognizable single pattern."""
    bt, ft = _src_tokens(broken), _src_tokens(fixed)
    sm = difflib.SequenceMatcher(a=bt, b=ft, autojunk=False)
    best: Optional[tuple[int, str, str]] = None
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        b_tok = " ".join(bt[i1:i2])
        a_tok = " ".join(ft[j1:j2])
        if not (b_tok or a_tok):
            continue
        size = (i2 - i1) + (j2 - j1)
        if best is None or size < best[0]:
            best = (size, b_tok, a_tok)
    return (best[1], best[2]) if best else None


def _predicates(text: str) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for line in text.splitlines():
        if not (re.search(r"\bmask\w*\s*=", line) or "tl.load" in line
                or "tl.store" in line or re.search(r"\boffs\w*\b", line)):
            continue
        for m in _PRED_RE.finditer(line):
            out.append((_norm(m.group(1)), m.group(2), _norm(m.group(3))))
    return out


def _zeros_dtype(text: str) -> Optional[str]:
    m = re.search(r"tl\.zeros\([^)]*dtype\s*=\s*tl\.(\w+)", text)
    if m:
        return m.group(1)
    m = re.search(r"tl\.zeros\(\s*[^,]+,\s*tl\.(\w+)", text)
    return m.group(1) if m else None


def _tile_assignments(text: str) -> dict[str, int]:
    """Map BLOCK_* names to their integer values (tuple + single/constexpr forms)."""
    out: dict[str, int] = {}
    for m in re.finditer(r"^[ \t]*([A-Za-z_][\w,\s]*?)=\s*([\d,\s]+)$", text, re.MULTILINE):
        names = [n.strip() for n in m.group(1).split(",")]
        vals = [v.strip() for v in m.group(2).split(",")]
        if len(names) == len(vals):
            for n, v in zip(names, vals):
                if re.fullmatch(r"BLOCK_\w+", n) and v.isdigit():
                    out[n] = int(v)
    for m in re.finditer(r"\b(BLOCK_\w+)\s*(?::\s*tl\.constexpr\s*)?=\s*(\d+)", text):
        out[m.group(1)] = int(m.group(2))
    return out


def _order(txt: str, a: str, b: str) -> Optional[bool]:
    ia, ib = txt.find(a), txt.find(b)
    if ia >= 0 and ib >= 0:
        return ia < ib
    return None


# --- detectors: each maps (broken_diff, fixed_diff) -> (class, before, after) --
def _arch_is_fnuz(arch: Optional[str]) -> bool:
    """True iff ``arch`` uses the FNUZ fp8 e4m3 encoding (CDNA2/CDNA3: gfx90a/gfx908/
    gfx942). gfx950/CDNA4 (the KORE target) and unknown default to OCP e4m3fn."""
    a = (arch or "").lower()
    return "gfx942" in a or "gfx90a" in a or "gfx908" in a


def _det_fp8(before, after, family):
    # Detect an fp8 e4m3 ENCODING swap in EITHER direction (OCP e4m3fn <-> FNUZ
    # e4m3fnuz). The arch-correct target is resolved at diagnosis time, so the
    # detector only needs to recognize that the fix changed the encoding -- the old
    # one-way "fix ADDED fnuz" rule mis-modeled gfx950, where the fix REMOVES fnuz
    # (moves to OCP e4m3fn) (audit R2 datagen C1).
    if bool(re.search(r"fnuz", before)) == bool(re.search(r"fnuz", after)):
        return None  # no fnuz<->fn swap in the changed region
    bt = re.search(r"float8_?e\d\w*|e[45]m[23]\w*", before)
    at = re.search(r"float8_?e\d\w*|e[45]m[23]\w*", after)
    if bt and at and bt.group(0) != at.group(0):
        return ("fp8_variant", bt.group(0), at.group(0))
    return None


def _det_mask_flip(before, after, family):
    bp, ap = _predicates(before), _predicates(after)
    for lb, ob, rb in bp:
        for la, oa, ra in ap:
            if lb == la and rb == ra and _CMP_INVERSE.get(ob) == oa:
                return ("mask_predicate_flip", f"{lb} {ob} {rb}", f"{la} {oa} {ra}")
    return None


def _det_tail_widen(before, after, family):
    bp, ap = _predicates(before), _predicates(after)
    for lb, ob, rb in bp:
        if ob not in ("<", "<="):
            continue
        for la, oa, ra in ap:
            if lb == la and oa == ob and rb != ra and "+" in rb and "+" not in ra \
                    and re.search(r"\b" + re.escape(ra) + r"\b", rb):
                return ("tail_mask_widened", f"{lb} {ob} {rb}", f"{la} {oa} {ra}")
    return None


def _det_reduction_axis(before, after, family):
    bm = re.search(r"tl\.(?:sum|max|min)\([^)]*axis\s*=\s*(\d)", before)
    am = re.search(r"tl\.(?:sum|max|min)\([^)]*axis\s*=\s*(\d)", after)
    if bm and am and bm.group(1) != am.group(1):
        return ("reduction_axis", f"axis={bm.group(1)}", f"axis={am.group(1)}")
    return None


def _det_acc_dtype(before, after, family):
    bz, az = _zeros_dtype(before), _zeros_dtype(after)
    if bz in ("bfloat16", "float16") and az == "float32":
        return ("accumulator_dtype", f"tl.{bz}", "tl.float32")
    low = re.search(r"tl\.(bfloat16|float16)", before)
    if (low and re.search(r"tl\.float32", after) and "tl.float32" not in before
            and re.search(r"zeros|acc|\+=|tl\.sum|tl\.dot", before + "\n" + after)):
        return ("accumulator_dtype", low.group(0), "tl.float32")
    return None


def _det_off_by_one(before, after, family):
    m = re.search(r"tl\.arange\(\s*0\s*,\s*\w+\s*\)\s*\+\s*1", before)
    if m:
        base = re.search(r"tl\.arange\(\s*0\s*,\s*\w+\s*\)", m.group(0)).group(0)
        if base in after and not re.search(re.escape(base) + r"\s*\+\s*1", after):
            return ("off_by_one_offset", _norm(m.group(0)), _norm(base))
    return None


def _det_transpose(before, after, family):
    if (before.count("[:, None]") != after.count("[:, None]")
            and before.count("[None, :]") != after.count("[None, :]")):
        return ("transpose_operand", "[None, :]", "[:, None]")
    for a, b in (("stride_am", "stride_ak"), ("stride_bn", "stride_bk"),
                 ("stride_xm", "stride_xn")):
        ob, oa = _order(before, a, b), _order(after, a, b)
        if ob is not None and oa is not None and ob != oa:
            return ("transpose_operand", f"{b} … {a}", f"{a} … {b}")
    return None


def _det_block(before, after, family):
    ba, aa = _tile_assignments(before), _tile_assignments(after)
    for name in ("BLOCK_K", "BLOCK_M", "BLOCK_N"):
        bv, av = ba.get(name), aa.get(name)
        if bv is None or av is None or bv == av:
            continue
        if name == "BLOCK_K" and bv % 32 != 0 and av % 32 == 0:
            return ("block_k_multiple", f"{name}={bv}", f"{name}={av}")
        if bv % 64 != 0 and av % 64 == 0:
            return ("block_size_multiple", f"{name}={bv}", f"{name}={av}")
        if name == "BLOCK_M" and bv == 64 and av == 128:
            return ("block_m_guard", f"{name}={bv}", f"{name}={av}")
    return None


def _det_upcast(before, after, family):
    if ".to(tl.float32)" in after and ".to(tl.float32)" not in before:
        return ("dropped_fp32_upcast", "(no fp32 upcast)", ".to(tl.float32)")
    return None


def _det_k_mask(before, after, family):
    am = re.search(r"mask\s*=\s*offs_k[^\n,]*<[^\n]+", after)
    if am and not re.search(r"mask\s*=\s*offs_k[^\n,]*<", before):
        return ("added_k_mask", "(unmasked K load)", _norm(am.group(0)))
    return None


def _det_other_fill(before, after, family):
    am = re.search(r"other\s*=\s*[^,)\n]+", after)
    if am and "tl.load(" in after and "other" not in before:
        return ("dropped_other_fill", "(no other= fill)", _norm(am.group(0)))
    return None


def _det_mask_added(before, after, family):
    am = re.search(r"tl\.(?:load|store)\([^)\n]*mask\s*=\s*[A-Za-z_][^,)\n]*", after)
    if am and not re.search(r"mask\s*=", before):
        m2 = re.search(r"mask\s*=\s*[A-Za-z_][^,)\n]*", am.group(0))
        return ("dropped_mask", "(no bounds mask)", _norm(m2.group(0)))
    return None


def _det_scale(before, after, family):
    for tok in ("sm_scale", "qk_scale", "scale"):
        ac = len(re.findall(rf"\*\s*{tok}\b", after))
        bc = len(re.findall(rf"\*\s*{tok}\b", before))
        if ac > bc:
            return ("dropped_scale", "(scale dropped)", f"* {tok}")
        if bc > ac:
            return ("duplicated_scale", f"(scale applied {bc}x)", f"* {tok}")
    return None


def _det_eps(before, after, family):
    am = re.search(r"\+\s*(?:eps|epsilon|1e-?\d+)\b", after)
    if am and not re.search(r"\+\s*(?:eps|epsilon|1e-?\d+)\b", before):
        return ("dropped_eps", "(no epsilon guard)", _norm(am.group(0)))
    return None


def _det_barrier(before, after, family):
    if re.search(r"tl\.(?:debug_)?barrier\(\)", after) \
            and not re.search(r"tl\.(?:debug_)?barrier\(\)", before):
        return ("missing_barrier", "(no barrier)", "tl.debug_barrier()")
    return None


def _det_normalization(before, after, family):
    norm = r"/\s*(?:N|n_cols|n_rows|M|K|denom|denominator|Z|l_i|_sum|sum_exp)\b"
    am = re.search(norm, after)
    if am and not re.search(norm, before):
        return ("dropped_normalization", "(no normalization divide)", _norm(am.group(0)))
    return None


# "strong" detectors require a PAIRED before/after signature (both sides present,
# clearly inverted/changed) so they are trustworthy even when the fix is a large
# rewrite. "weak" detectors are absent->present signatures, run only when the two
# sources are mostly similar (localized change) to avoid false positives.
_STRONG = (_det_fp8, _det_mask_flip, _det_tail_widen, _det_reduction_axis,
           _det_acc_dtype, _det_off_by_one, _det_transpose, _det_block)
_WEAK = (_det_upcast, _det_k_mask, _det_other_fill, _det_mask_added,
         _det_scale, _det_eps, _det_barrier, _det_normalization)


def classify_repair_diff(broken_src: str, fixed_src: str,
                         failure_class: Optional[str] = None,
                         family: str = "generic") -> Optional[DiffFinding]:
    """Classify the broken->fixed change into a concrete bug class, or None when
    the diff is ambiguous (a rewrite / no recognizable single pattern)."""
    try:
        if not broken_src or not fixed_src or broken_src == fixed_src:
            return None
        before, after, ratio = _changed_regions(broken_src, fixed_src)
        for det in _STRONG:
            hit = det(before, after, family)
            if hit:
                return DiffFinding(*hit)
        if ratio >= 0.6:  # localized change -> trust the weak detectors too
            for det in _WEAK:
                hit = det(before, after, family)
                if hit:
                    return DiffFinding(*hit)
        return None
    except Exception:  # analysis must never crash datagen - degrade to fallback
        return None


def _lead_in(failure_class: Optional[str], error_text: str) -> str:
    et = (error_text or "").strip()
    head = {
        "compile_fail": "The verifier could not build the kernel",
        "snr_fail": "The verifier failed the correctness (SNR) gate",
    }.get(failure_class or "", "The verifier rejected the kernel")
    return f"{head}: {et}" if et else f"{head}."


def _diagnose(finding: DiffFinding, family: str,
              arch: Optional[str] = None) -> tuple[str, str]:
    """Concrete, grounded mechanism + one-line PROPOSED_CHANGE for a finding.

    Op-family aware: MFMA / ``tl.dot`` / multiple-of-64 reasoning is only emitted
    for gemm/attention kernels, never injected into a pointwise/reduction/quant
    diagnosis where it does not apply. ``arch`` (default: the KORE gfx950 target)
    selects the arch-correct fp8 encoding for the fp8_variant diagnosis."""
    cc, b, a = finding.change_class, finding.before, finding.after
    gemmish = family in ("gemm", "attention")
    if cc == "mask_predicate_flip":
        return (f"The bounds mask used `{b}` where the correct predicate is `{a}`. "
                f"Inverting the comparison keeps only the out-of-range lanes and masks "
                f"out every valid element, so the load/store returns the `other` fill "
                f"instead of the real data - the reduction/output collapses toward zero. "
                f"Restoring `{a}` selects exactly the in-range lanes.",
                f"Flip the bounds predicate back to `{a}`.")
    if cc == "tail_mask_widened":
        return (f"The tail guard was widened to `{b}` instead of `{a}`, so the final "
                f"partial tile treated out-of-range lanes as valid and read/wrote past "
                f"the buffer, corrupting the boundary rows/cols. Tightening it to `{a}` "
                f"masks the tail correctly.",
                f"Restore the tail bound to `{a}`.")
    if cc == "accumulator_dtype":
        return (f"The accumulator was declared `{b}` instead of `{a}`. Accumulating the "
                f"reduction in low precision loses mantissa bits on every add, so the "
                f"running sum drifts below the SNR gate; `{a}` holds precision across the "
                f"reduction while inputs/outputs stay low precision.",
                f"Accumulate in `{a}` instead of `{b}`.")
    if cc == "dropped_fp32_upcast":
        return (f"The loaded values were not upcast with `{a}` before the math, so the "
                f"reduction ran in the low input precision and failed the SNR gate. "
                f"Restoring `{a}` upcasts before accumulating.",
                f"Upcast to fp32 with `{a}` before accumulating.")
    if cc == "reduction_axis":
        return (f"The reduction ran over `{b}` instead of `{a}`, collapsing the wrong "
                f"dimension so the per-row/column result was computed along the wrong axis.",
                f"Reduce over `{a}`.")
    if cc == "off_by_one_offset":
        return (f"The offset carried an off-by-one: `{b}` instead of `{a}`. Every lane "
                f"then addressed the neighbouring element, so the loads were shifted by "
                f"one and the result was numerically wrong. Removing the `+ 1` restores "
                f"correct addressing.",
                f"Remove the off-by-one; use `{a}`.")
    if cc == "transpose_operand":
        return (f"The operand was indexed transposed (`{b}` instead of `{a}`), so the "
                f"kernel contracted/broadcast the wrong axis and produced a wrong result. "
                f"Restoring `{a}` fixes the indexing.",
                f"Restore the correct indexing order (`{a}`).")
    if cc == "added_k_mask":
        return (f"The contraction-dimension load was unmasked, so when K is not a "
                f"multiple of BLOCK_K the final tile read past the operand and polluted "
                f"the dot-product. The fix adds the K bound `{a}` so out-of-range K lanes "
                f"are zero-filled.",
                f"Mask the K-loop load with `{a}`.")
    if cc == "dropped_other_fill":
        return (f"The masked load had no `{a}` fill, so masked-off lanes contributed "
                f"undefined memory instead of the identity value. Adding `{a}` makes the "
                f"tail lanes neutral.",
                f"Add `{a}` to the masked load.")
    if cc == "dropped_mask":
        return (f"The load/store dropped its bounds mask (`{a}`), so the partial tile "
                f"read/wrote out of range. Restoring `{a}` bounds the access.",
                f"Restore the bounds mask (`{a}`).")
    if cc == "dropped_eps":
        return (f"The normalization dropped the `{a}` guard, so a zero variance produced "
                f"inf/NaN through the rsqrt/divide. Restoring `{a}` keeps it finite.",
                f"Restore the epsilon guard (`{a}`).")
    if cc == "dropped_scale":
        return (f"The `{a}` scaling factor was dropped, so the output magnitude was wrong "
                f"(softmax/attention/quant scaling). Reapplying `{a}` restores the scale.",
                f"Reapply the scale (`{a}`).")
    if cc == "duplicated_scale":
        return (f"The scale `{a}` was applied more than once ({b}), double-scaling the "
                f"result. Applying `{a}` exactly once restores the correct magnitude.",
                f"Apply the scale (`{a}`) exactly once.")
    if cc == "fp8_variant":
        # Arch-correct fp8 e4m3: OCP e4m3fn on gfx950/CDNA4 (the KORE target), FNUZ
        # e4m3fnuz on gfx942/CDNA3. Teach the fix toward the arch-correct encoding
        # regardless of which direction the raw diff went -- the old text hardcoded
        # "use FNUZ / gfx942" and actively mis-taught gfx950 (audit R2 datagen C1).
        fnuz_arch = _arch_is_fnuz(arch)
        b_is_fnuz = "fnuz" in b.lower()
        correct_tok, wrong_tok = (b, a) if (b_is_fnuz == fnuz_arch) else (a, b)
        correct_name = "FNUZ" if fnuz_arch else "OCP"
        wrong_name = "OCP" if fnuz_arch else "FNUZ"
        archname = "gfx942/CDNA3" if fnuz_arch else "gfx950/CDNA4"
        return (f"The fp8 encoding was `{wrong_tok}` ({wrong_name}) instead of the "
                f"{archname} `{correct_tok}` ({correct_name}); the exponent bias and "
                f"-0/inf handling differ, so the bytes mismatched the production "
                f"reference. Using `{correct_tok}` matches the hardware/AITER layout.",
                f"Use the {correct_name} fp8 encoding (`{correct_tok}`).")
    if cc == "missing_barrier":
        return (f"A synchronization barrier (`{a}`) between the shared-memory write and "
                f"its dependent read was missing, letting wavefronts read stale/partial "
                f"LDS - a nondeterministic correctness failure. Restoring `{a}` orders the "
                f"access.",
                f"Restore the barrier (`{a}`).")
    if cc == "dropped_normalization":
        return (f"The mean/normalization divide `{a}` was dropped after the reduction, so "
                f"the result was left unnormalized (off by the reduction length). Restoring "
                f"`{a}` normalizes it.",
                f"Restore the normalization (`{a}`).")
    if cc == "block_k_multiple":
        return (f"The K tile `{b}` is not a multiple of 32, which is illegal for the "
                f"fp8/MX scale groups (32-element microscaling blocks), so the kernel "
                f"failed to build. `{a}` restores a valid K tile.",
                f"Set the K tile to a multiple of 32 (`{a}`).")
    if cc == "block_size_multiple":
        if gemmish:
            # GEMM/attention: MFMA tiles legitimately want multiples of 64.
            return (f"The tile size `{b}` is not a multiple of 64 (and `tl.arange` needs "
                    f"a power-of-two length) and cannot map onto the MFMA matrix cores, so "
                    f"the kernel failed to build. `{a}` restores a valid tile.",
                    f"Set the tile size to a valid multiple of 64 (`{a}`).")
        # pointwise/reduction/norm/quant: the binding constraint is only that
        # ``tl.arange`` needs a power-of-two length - NOT any MFMA multiple-of-64.
        return (f"The tile size `{b}` is not a power of two, which `tl.arange` requires, "
                f"so the kernel failed to build. `{a}` restores a valid tile.",
                f"Set the tile size to a power of two (`{a}`).")
    if cc == "block_m_guard":
        return (f"BLOCK_M was `{b}`; a 64-row tile causes silent cross-workgroup "
                f"corruption in block-sparse/split-K kernels. Restoring `{a}` avoids the "
                f"race.",
                f"Restore BLOCK_M to `{a}`.")
    return (f"The fix changed `{b}` to `{a}`, which restores correctness.",
            f"Apply the change `{b}` -> `{a}`.")


def analyze_repair_diff(broken_src: str, fixed_src: str,
                        failure_class: Optional[str], error_text: str,
                        family: str = "generic",
                        arch: Optional[str] = None) -> tuple[str, str]:
    """Produce a REAL, evidence-based (ANALYSIS, PROPOSED_CHANGE) for the repair.

    Grounded ONLY in the broken->fixed diff: a specific mechanism when the change
    is a recognizable bug class, otherwise a minimal factual fallback naming the
    verifier error and the one concrete token that changed. ``arch`` (default: the
    KORE gfx950 target) selects the arch-correct fp8 encoding in the diagnosis."""
    lead = _lead_in(failure_class, error_text)
    finding = classify_repair_diff(broken_src, fixed_src, failure_class, family)
    if finding is not None:
        mech, proposed = _diagnose(finding, family, arch)
        return f"{lead}\n{mech}", proposed
    tok = _salient_token_change(broken_src, fixed_src)
    if tok and (tok[0] or tok[1]):
        b = tok[0] or "(absent)"
        a = tok[1] or "(removed)"
        analysis = (f"{lead}\nThe diff is not a single recognizable bug pattern; the "
                    f"smallest concrete change the fix makes is `{b}` -> `{a}`. Applying "
                    f"the verified fix restores correctness.")
        return analysis, f"Apply the verified fix (`{b}` -> `{a}`)."
    return (f"{lead}\nApplying the verified fix restores correctness.",
            "Apply the verified fix that passes the verifier.")


# The GEMM/MFMA hard-constraint line that ``build_turn_prompt`` injects for every
# op; it is inappropriate for pointwise/reduction/norm/activation/quant kernels
# (a relu/row-sum kernel has no ``tl.dot`` and no multiple-of-64 tile), so we swap
# it for an op-appropriate constraint on those families.
_GEMM_CONSTRAINT = ("2. BLOCK_* sizes must be multiples of 64; accumulate in fp32; "
                    "use tl.dot.\n")
_POINTWISE_CONSTRAINT = ("2. Use power-of-two BLOCK sizes; accumulate/reduce in fp32; "
                         "mask the tail.\n")


def _op_appropriate_repair_prompt(prompt: str, family: str) -> str:
    """Strip the MFMA-only constraints (``tl.dot`` / multiple-of-64) from the
    repair prompt for non-GEMM ops. No-op for gemm/attention (where it applies)
    and no-op if the exact constraint line is absent (defensive)."""
    if family in ("gemm", "attention"):
        return prompt
    return prompt.replace(_GEMM_CONSTRAINT, _POINTWISE_CONSTRAINT)


def _diagnostic_assistant(failure_class: str, error_text: str, broken_src: str,
                          fixed_src: str, family: str = "generic",
                          arch: Optional[str] = None) -> str:
    """Self-diagnose-then-fix assistant turn in the CANONICAL contract
    (ANALYSIS / PROPOSED_CHANGE / FULL_KERNEL via :func:`format_assistant_turn`).

    The ANALYSIS is a REAL diagnosis derived from the actual broken->fixed diff
    (see :func:`analyze_repair_diff`) - it names the concrete change class and the
    concrete token that changed - not a templated string. The VERIFIED fixed
    kernel is the FULL_KERNEL, so SFT learns to read the concrete failure and emit
    the fix in the same shape it must produce at inference. ``arch`` selects the
    arch-correct fp8 encoding in the diagnosis (default: the KORE gfx950 target)."""
    analysis, proposed = analyze_repair_diff(
        broken_src, fixed_src, failure_class, error_text, family, arch)
    return format_assistant_turn(analysis, proposed, fixed_src)


def make_repair_record(
    task,
    teacher: TeacherClient,
    env,
    broken_src: str,
    broken_obs,
    diagnostic: bool = True,
) -> Optional[RepairRecord]:
    """Given a known-broken kernel + its observation, get a teacher repair and
    emit a RepairRecord ONLY when the repair actually validates.

    The broken side must genuinely fail (``failure_class`` is not None) and the
    teacher's fix must pass full validation, otherwise we would mislabel a still-
    broken kernel as a correct SFT target. Returns None if the teacher produced
    no kernel, the fix crashed, or the fix did not pass validation.

    When ``diagnostic`` is True (default), the stored assistant turn is rendered in
    the canonical diagnose-then-fix contract (ANALYSIS / PROPOSED_CHANGE / FULL_KERNEL
    via :func:`format_assistant_turn`), folding the verifier's ``error_text`` into the
    ANALYSIS so SFT learns to self-diagnose. The emitted fix is always the VERIFIED
    kernel only - the "only emit verified fixes" rule is unchanged."""
    failure_class = _failure_class(broken_obs)
    if failure_class is None:
        return None

    # Infer the op family so both the prompt constraints and the diagnosis are
    # op-appropriate (no ``tl.dot`` / multiple-of-64 talk for a pointwise op).
    family = infer_family(getattr(task, "operation", None)
                          or getattr(task, "task_id", "") or "")
    error = _error_text(broken_obs)
    user_prompt = _op_appropriate_repair_prompt(
        build_turn_prompt(parent_source=broken_src, feedback=error, mode="repair"),
        family,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    ctx = getattr(_ctx, "attempt", None) or {}
    t_teacher = time.time()
    # Tier 1: give the TEACHER the AMD-Triton playbook for a better fix; the STORED
    # record keeps the canonical SYSTEM_PROMPT (no train/deploy contract drift).
    response = teacher.generate(
        [{"role": "system", "content": live_system_prompt(SYSTEM_PROMPT)}, messages[1]]
    )
    teacher_ms = round((time.time() - t_teacher) * 1000.0, 1)
    fixed_src = extract_kernel(response)

    def _emit(kept: bool, skip_reason: Optional[str], fixed_obs=None,
              child_snr=None) -> None:
        log.event(
            "repair_attempt",
            task=task.task_id, idx=ctx.get("idx"), mutator=ctx.get("mutator"),
            failure_class=failure_class,
            broke_verified_fail=ctx.get("broke_verified_fail"),
            teacher_ms=teacher_ms,
            fixed_compiled=(bool(getattr(fixed_obs, "compiled", False))
                            if fixed_obs is not None else False),
            fixed_correct=(bool(getattr(fixed_obs, "validation_passed", False))
                           if fixed_obs is not None else False),
            child_snr_db=child_snr, kept=kept, skip_reason=skip_reason,
        )

    if not fixed_src:
        _emit(False, "no_kernel")
        return None

    try:
        fixed_obs = env.step(fixed_src, full_validation=True, multi_shape=True)
    except Exception:
        fixed_obs = None

    # Only accept the repair as an SFT target if it genuinely passes validation;
    # otherwise it is a mislabeled still-broken kernel.
    if fixed_obs is None:
        _emit(False, "fix_crashed")
        return None
    if not getattr(fixed_obs, "validation_passed", False):
        _emit(False, "fix_unverified", fixed_obs=fixed_obs)
        return None
    child_snr = fixed_obs.snr_db
    _emit(True, None, fixed_obs=fixed_obs, child_snr=child_snr)

    if diagnostic:
        assistant = _diagnostic_assistant(failure_class, error, broken_src,
                                          fixed_src, family,
                                          getattr(task, "gpu_target", None))
    else:
        assistant = response
    messages = messages + [{"role": "assistant", "content": assistant}]
    return RepairRecord(
        task_id=task.task_id,
        failure_class=failure_class,
        parent_hash=kernel_hash(broken_src),
        error_text=error,
        messages=messages,
        child_snr_db=child_snr,
        gpu=task.gpu_target,
        operation=getattr(task, "operation", None),
        arch=getattr(task, "gpu_target", None),
    )


def generate_repairs(
    task,
    teacher: TeacherClient,
    env,
    n: int,
    seed: int = 0,
    natural_fraction: float = 0.3,
    diagnostic: bool = True,
    on_record: Optional[Callable[[RepairRecord], None]] = None,
) -> list[RepairRecord]:
    """Produce up to ``n`` RepairRecords for ``task``.

    A ``natural_fraction`` of attempts mine naturally-failed teacher generations;
    the rest inject a breakage into the seed and repair that. ``diagnostic``
    selects the diagnose-then-fix assistant format (see ``make_repair_record``).
    """
    with log.stage("generate_repairs", task=task.task_id, n=n,
                   natural_fraction=natural_fraction):
        rng = random.Random(seed)
        records: list[RepairRecord] = []
        seed_src = task.seed_source
        family = infer_family(getattr(task, "operation", "") or task.task_id)
        n_natural = int(round(n * natural_fraction))
        n_injected = n - n_natural

        # (1) Injected breakage repairs.
        t_start = time.time()
        attempts = 0
        while len([r for r in records]) < n_injected and attempts < n_injected * 5:
            attempts += 1
            broken_src, _hint, _name = apply_random_breakage(seed_src, rng, family=family)
            try:
                broken_obs = env.step(broken_src, full_validation=True, multi_shape=False)
            except Exception:
                log.debug("injected breakage crashed verifier",
                          task=task.task_id, idx=attempts, mutator=_name)
                continue
            if _failure_class(broken_obs) is None:
                log.debug("injected breakage did not fail verifier; skipping",
                          task=task.task_id, idx=attempts, mutator=_name)
                continue  # breakage didn't actually break - skip
            _ctx.attempt = {"idx": attempts, "mutator": _name,
                            "broke_verified_fail": True}
            rec = make_repair_record(task, teacher, env, broken_src, broken_obs,
                                     diagnostic=diagnostic)
            if rec is not None:
                records.append(rec)
                if on_record is not None:
                    on_record(rec)
            log.progress(attempts, max(1, n_injected * 5), "repair",
                         t_start=t_start, kept=len(records), target=n_injected)
        _ctx.attempt = {}
        injected_kept = len(records)

        # (2) Naturally-failed teacher turns.
        natural = mine_natural_failures(
            task, teacher, env, n_natural, seed=seed + 1,
            diagnostic=diagnostic, on_record=on_record,
        )
        records += natural
        result = records[:n]
        log.metric(
            "repair_summary", task=task.task_id, attempts=attempts,
            kept=len(result), injected_kept=injected_kept,
            dropped_unverified=attempts - injected_kept,
            natural_mined=len(natural),
        )
        return result


def mine_natural_failures(
    task,
    teacher: TeacherClient,
    env,
    n: int,
    seed: int = 0,
    diagnostic: bool = True,
    on_record: Optional[Callable[[RepairRecord], None]] = None,
) -> list[RepairRecord]:
    """Sample teacher rewrites of the seed; whenever one fails, mine a repair."""
    with log.stage("mine_natural_failures", task=task.task_id, n=n):
        rng = random.Random(seed)
        records: list[RepairRecord] = []
        seed_src = task.seed_source
        t_start = time.time()
        attempts = 0
        budget = max(n * 5, 5)
        while len(records) < n and attempts < budget:
            attempts += 1
            mode = rng.choice(["exploit", "explore"])
            prompt = build_turn_prompt(parent_source=seed_src, mode=mode)
            # Tier 1: playbook-primed teacher yields more realistic candidates to mine
            # (this sampling turn is NOT stored, so it needs no clean-contract copy).
            messages = [
                {"role": "system", "content": live_system_prompt(SYSTEM_PROMPT)},
                {"role": "user", "content": prompt},
            ]
            response = teacher.generate(messages)
            cand_src = extract_kernel(response)
            if not cand_src:
                log.debug("natural sample had no kernel; skipping",
                          task=task.task_id, idx=attempts, mode=mode)
                continue
            try:
                obs = env.step(cand_src, full_validation=True, multi_shape=False)
            except Exception:
                log.debug("natural sample crashed verifier",
                          task=task.task_id, idx=attempts, mode=mode)
                continue
            if _failure_class(obs) is None:
                continue  # it worked - not a repair opportunity
            _ctx.attempt = {"idx": attempts, "mutator": f"natural:{mode}",
                            "broke_verified_fail": True}
            rec = make_repair_record(task, teacher, env, cand_src, obs,
                                     diagnostic=diagnostic)
            if rec is not None:
                records.append(rec)
                if on_record is not None:
                    on_record(rec)
            log.progress(attempts, budget, "natural_mine",
                         t_start=t_start, mined=len(records), target=n)
        _ctx.attempt = {}
        log.metric("natural_mine_summary", task=task.task_id, attempts=attempts,
                   mined=len(records), target=n)
        return records
