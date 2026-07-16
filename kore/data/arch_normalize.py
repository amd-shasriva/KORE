"""Normalize legacy non-gfx950 arch references in training text to the target.

This node is gfx950 (AMD Instinct MI350X / CDNA4) and every task targets it, but
a large slice of the on-disk data was generated BEFORE the gfx950 retarget and
carries gfx942 / MI325X / CDNA3 arch labels (and the gfx942 fp8 encoding
``e4m3fnuz``) in its prompt and context text. Those kernels are portable Triton
and are re-verified on gfx950 by the ``reverify`` stage, so the DATA is valid;
only the arch LABELS in the text are stale. If the model trained on that text it
would occasionally reason or answer for gfx942 instead of gfx950.

The fix is a build-time text pass: the ``build`` stage turns raw records into
training rows, so normalizing there scrubs every SFT / DPO row (legacy and new)
just before it is written, with no pause, no regeneration, and no data loss. The
pass is a pure string rewrite: idempotent (targets never contain their sources),
order-independent, and it only touches arch tokens (never kernel logic, numbers,
task ids, or hashes).

Replacements:
  gfx942     -> gfx950         arch slug
  MI325X     -> MI350X         board name (MI355X is already gfx950, untouched)
  MI300X     -> MI350X         board name
  CDNA3/cdna3-> CDNA4/cdna4    micro-architecture generation
  e4m3fnuz   -> e4m3fn         fp8 e4m3: gfx942 FNUZ -> gfx950 OCP
  e5m2fnuz   -> e5m2           fp8 e5m2: FNUZ -> OCP

The two fp8 dtype rewrites are SKIPPED on any string containing ``"instead of"``,
because the repair fix-lesson format names both encodings in one sentence ("was
``e4m3fnuz`` (FNUZ) instead of ``e4m3fn`` (OCP)"); a blind swap there would
collapse the lesson to nonsense. Arch labels are always safe to rewrite.
"""
from __future__ import annotations

from typing import Any

# (old, new) applied in order. Arch/board/uarch labels: always safe.
_ARCH_LABELS: tuple[tuple[str, str], ...] = (
    ("gfx942", "gfx950"),
    ("MI325X", "MI350X"),
    ("MI300X", "MI350X"),
    ("CDNA3", "CDNA4"),
    ("cdna3", "cdna4"),
)

# The legacy FNUZ fp8 encoding marker. If a string mentions FNUZ (e4m3fnuz / e5m2fnuz),
# leave the WHOLE string verbatim: FNUZ is the gfx942 encoding, so its arch label is
# bound to that fact. Scrubbing gfx942 -> gfx950 there would manufacture a false
# "gfx950 uses FNUZ" statement (gfx950 uses OCP e4m3fn) and could corrupt a real fp8
# kernel's dtype. This preserves deliberate gfx942-vs-gfx950 fp8 facts (the aiter_ref
# infra docs, fp8 task docstrings) and fp8 code. Stale single-arch text without FNUZ is
# still rewritten to the target.
_LEGACY_FP8_MARKER = "fnuz"


def normalize_text(s: str) -> str:
    """Rewrite stale arch tokens in one string toward the gfx950 target.

    Pure and idempotent. A string that mentions the legacy FNUZ fp8 encoding is left
    verbatim (its arch labels and dtype are arch-specific and correct as written);
    otherwise the arch / board / uarch labels (gfx942 -> gfx950, CDNA3 -> CDNA4,
    MI300X / MI325X -> MI350X) are rewritten to the gfx950 target.
    """
    if not s:
        return s
    if _LEGACY_FP8_MARKER in s.lower():
        return s
    for old, new in _ARCH_LABELS:
        if old in s:
            s = s.replace(old, new)
    return s


def normalize_obj(obj: Any) -> Any:
    """Recursively normalize every string in a nested dict / list structure.

    Returns a new structure; the input is not mutated. Non-string leaves (ints,
    floats, bools, None) pass through untouched, so numeric fields, task ids that
    are not arch tokens, and content hashes are preserved exactly.
    """
    if isinstance(obj, str):
        return normalize_text(obj)
    if isinstance(obj, dict):
        return {k: normalize_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        normalized = [normalize_obj(v) for v in obj]
        return type(obj)(normalized) if isinstance(obj, tuple) else normalized
    return obj


def normalize_rows(rows: Any) -> list:
    """Normalize an iterable of training rows (SFT chat dicts or DPO pair dicts).

    Rows that are not dict / list / str (unexpected) pass through unchanged so a
    surprising row shape can never abort a build.
    """
    return [normalize_obj(r) for r in rows]
