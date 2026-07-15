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

# fp8 encoding dtype names: gfx942 FNUZ -> gfx950 OCP. Guarded (see module docstring).
_FP8_DTYPES: tuple[tuple[str, str], ...] = (
    ("e4m3fnuz", "e4m3fn"),
    ("e5m2fnuz", "e5m2"),
)

# Marker of a repair fix-lesson that names both fp8 encodings; skip fp8 rewrites there.
_FIX_LESSON_MARKER = "instead of"


def normalize_text(s: str) -> str:
    """Rewrite stale arch tokens in one string toward the gfx950 target.

    Pure and idempotent. Arch/board/uarch labels are always rewritten; the fp8
    encoding names are rewritten only when the string is not a two-encoding repair
    fix-lesson (guarded by ``"instead of"``).
    """
    if not s:
        return s
    for old, new in _ARCH_LABELS:
        if old in s:
            s = s.replace(old, new)
    if _FIX_LESSON_MARKER not in s:
        for old, new in _FP8_DTYPES:
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
