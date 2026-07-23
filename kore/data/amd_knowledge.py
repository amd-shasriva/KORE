"""AMD-Triton optimization knowledge + cross-attempt experience ledger for datagen.

Two pieces, both aimed at the win-finder (and the rest of datagen):

  * :func:`live_system_prompt` (Tier 1) - augments the TEACHER's *live* system
    prompt with a tight, source-verified AMD-Triton optimization playbook (distilled
    from Hyperloom-Forge's ``triton_levers`` / ``common_methodology`` / hardware /
    ``tuning_db`` knowledge). It is injected into the generation context ONLY, never
    into the STORED SFT trajectory - so the teacher writes better kernels without
    changing the train/deploy prompt contract (the stored turns keep the canonical
    ``SYSTEM_PROMPT``).

  * :class:`ExperienceLedger` (Tier 3) - distills FAILED / regressed attempts into a
    small, deduped "do-NOT-repeat" constraints block that is fed back into later
    turns (and shared across the N ``deepen_wins`` trajectories for a task, so
    trajectory 2 never re-walks trajectory 1's dead-ends). Adapts Hyperloom-Forge's
    ``forge_fusion.FusionExperienceLedger``.

Pure / side-effect-free (only reads one packaged markdown asset, cached), so it is
CPU-unit-testable without a GPU or a teacher.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import Optional

_PLAYBOOK_PATH = Path(__file__).resolve().parent / "knowledge" / "amd_triton_playbook.md"

# Minimal embedded card used iff the full markdown asset is missing (keeps datagen
# working from any checkout / install without a packaging step).
_FALLBACK_PLAYBOOK = (
    "# AMD-Triton quick rules (MI350X gfx950/CDNA4, MI300X gfx942/CDNA3)\n"
    "- num_warps: start at 4. num_warps=8 is the #1 AMD perf bug (VGPR spill to HBM, 3-5x slower).\n"
    "- MFMA: always route inner products through tl.dot; accumulate in fp32; prefer "
    "matrix_instr_nonkdim=16 over 32 (fewer AGPRs, higher occupancy).\n"
    "- Occupancy: VGPR>256 is a hard occupancy=1 cliff - keep <=256 for occupancy>=2; "
    "prefer 2 waves with no spill over 3 waves that spill.\n"
    "- num_stages: single GEMM 1-2, fused flash-attn 1, elementwise/reduction 1. Never 3-4 "
    "(buffers loads in LDS -> occupancy cliff).\n"
    "- Memory: coalesce to 128-bit (global_load_dwordx4, 16B-aligned, contiguous per 64-lane wave); "
    "pad LDS (BK+PAD)%32!=0 or XOR-swizzle to kill bank conflicts; widen LDS reads to ds_read_b128.\n"
    "- BLOCK_* multiples of 64 (min 64; 32 underutilizes MFMA). BLOCK_M=64 SILENTLY CORRUPTS "
    "sparse attention -> use 128.\n"
    "- fp8: gfx950=OCP (e4m3fn); gfx942=FNUZ (tl.float8e4b8). Wrong dialect = 2x SILENT error.\n"
    "- AMD knobs (matrix_instr_nonkdim/waves_per_eu/kpack) only take effect inside triton.Config({...}); "
    "as plain Python vars they are silently ignored.\n"
    "- Triton's real wins are FUSION (epilogue/attention) and skinny split-K decode; plain dense "
    "GEMM usually loses to tuned hipBLASLt/aiter. Peak != achievable (~45-55% of matrix peak).\n"
)


@functools.lru_cache(maxsize=1)
def playbook() -> str:
    """Return the AMD-Triton optimization playbook (cached). Falls back to a compact
    embedded card if the markdown asset is unavailable."""
    try:
        txt = _PLAYBOOK_PATH.read_text(encoding="utf-8").strip()
        return txt or _FALLBACK_PLAYBOOK
    except Exception:  # noqa: BLE001 - never fatal; degrade to the embedded card
        return _FALLBACK_PLAYBOOK


def live_system_prompt(base_system: str) -> str:
    """Augment the teacher's LIVE system prompt with the AMD-Triton playbook.

    Use ONLY for the generation context fed to ``teacher.generate`` - never for the
    STORED SFT trajectory, so the training data keeps the canonical contract while
    the teacher still gets the AMD knowledge that makes it propose good kernels on
    the first move (directly attacks the ~40% zero-win rate)."""
    return f"{base_system}\n\n---\n# Optimization knowledge (reference; do not echo)\n{playbook()}"


# --------------------------------------------------------------------------- #
# Tier 3: cross-attempt experience ledger (distilled "do-NOT-repeat" constraints)
# --------------------------------------------------------------------------- #
# Error / outcome signature -> a crisp, reusable constraint. Mined from the verifier
# error text + the attempt outcome of a FAILED or regressed candidate so the teacher
# does not repeat the same dead-end on the next turn (or the next trajectory).
_CONSTRAINT_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"out of resource|shared memory|\blds\b|local memory", re.I),
     "A prior attempt exceeded LDS/shared memory - lower BLOCK_K or num_stages "
     "(LDS bytes ~= (BM*BK + BK*BN) * dtype_bytes * num_stages; cap 64KB gfx942 / 160KB gfx950)."),
    (re.compile(r"out of memory|OOM|register|spill|scratch", re.I),
     "A prior attempt ran out of registers / spilled - reduce num_warps or tile sizes; "
     "keep VGPR<=256 (>256 forces occupancy=1)."),
    (re.compile(r"f8E4M3FN|Unsupported conversion|float8", re.I),
     "fp8 dialect mismatch - gfx942 MFMA needs FNUZ (tl.float8e4b8); gfx950 uses OCP e4m3fn. "
     "Do not feed the wrong fp8 encoding to tl.dot."),
    (re.compile(r"num_warps\s*=\s*8|num warps", re.I),
     "Do not raise num_warps to 8 (VGPR spill -> 3-5x slower); start at 4."),
    (re.compile(r"nan|\binf\b|snr|correctness|incorrect", re.I),
     "A prior change broke correctness - keep fp32 accumulation, the bounds mask + `other=` "
     "fill, and the scale/normalization factors; do not drop them while optimizing."),
    (re.compile(r"compil|syntax|invalid|not defined|NameError|TypeError|ValueError|assert", re.I),
     "A prior change failed to build - make a smaller, syntactically valid edit and keep the "
     "public entry-point signature unchanged."),
    (re.compile(r"slower|regress|not faster|no.?op|unchanged", re.I),
     "A prior structural change measured SLOWER or made no difference - do not repeat it; "
     "switch dimension (memory-coalescing vs MFMA/tl.dot vs occupancy/num_warps)."),
]


class ExperienceLedger:
    """Deduped 'do-NOT-repeat' constraints distilled from failed/regressed attempts.

    Injected into later turns via ``build_turn_prompt(tuning_hints=...)`` and shared
    across the N ``deepen_wins`` trajectories for a task so no dead-end is re-walked.
    Bounded (``max_constraints``) so the injected block stays small.
    """

    def __init__(self, max_constraints: int = 10):
        self._constraints: list[str] = []
        self._seen: set[str] = set()
        self._max = int(max_constraints)

    def _add(self, c: str) -> None:
        c = (c or "").strip()
        if c and c not in self._seen and len(self._constraints) < self._max:
            self._seen.add(c)
            self._constraints.append(c)

    def record(self, *, error_text: str = "", outcome: str = "", note: str = "") -> None:
        """Distill one failed/regressed attempt into constraint(s)."""
        blob = f"{error_text or ''}\n{outcome or ''}"
        for rx, constraint in _CONSTRAINT_RULES:
            if rx.search(blob):
                self._add(constraint)
        if note:
            self._add(note)

    def render(self) -> str:
        """The constraints block for the ``tuning_hints`` slot (empty if none)."""
        if not self._constraints:
            return ""
        body = "\n".join(f"- {c}" for c in self._constraints)
        return f"Known constraints (do NOT repeat these mistakes):\n{body}"

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._constraints)


__all__ = ["playbook", "live_system_prompt", "ExperienceLedger"]
