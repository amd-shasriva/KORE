"""Data-time verification rigor (Pillar 1).

Datagen (repair / ranked groups / wins) verifies + benchmarks every candidate via
``KoreEnv.step``, whose rigor is controlled by env vars the verifier SUBPROCESS
reads (``KoreEnv._env`` does ``os.environ.copy()``, so anything set in the datagen
process propagates into the child). For the DATA pass we want MAXIMUM rigor — every
accepted kernel adversarially-correct and benchmarked against the STRONGEST
available baseline — even though that is slower than the RL-rollout default (which
deliberately stays fast for throughput).

``set_rigorous_verification()`` enables, for the current process + its children:

  * ``KORE_VERIFIED_CORRECTNESS=1`` — the adversarial correctness battery (zeros,
    +/-1e3, sign_alt, large-magnitude, ...) ON TOP of the 5-seed SNR+allclose gate,
    so no kernel lucky-passes the fixed seeds.
  * ``KORE_SHAPE_AUGMENT=1`` — expand each task to scaled + odd/non-pow2 shapes for
    broader correctness+perf coverage.
  * ``KORE_COMPILE_BASELINE=1`` — benchmark fusion / gemm-fusion tasks against
    ``torch.compile`` (a strong bar) instead of torch-eager, so a measured "speedup"
    is honest (not just "beats eager").
  * ``KORE_BENCH_COLD=1`` — cold-L2 timing (already the default; set explicitly so
    the data pass is unambiguous).

This does NOT force GPU clock-locking: on a shared node that would perturb other
users' jobs, and KORE's 3-5 run median-of-medians + CV<=3% gate + cold-L2 already
controls timing noise. Idempotent; by default respects any value the operator has
already set (pass ``override=True`` to force).
"""

from __future__ import annotations

import os

# The rigor knobs and the value that enables each. Read by the verifier subprocess.
RIGOR_ENV: dict[str, str] = {
    "KORE_VERIFIED_CORRECTNESS": "1",
    "KORE_SHAPE_AUGMENT": "1",
    "KORE_COMPILE_BASELINE": "1",
    "KORE_BENCH_COLD": "1",
}


def set_rigorous_verification(enable: bool = True, override: bool = False) -> dict[str, str]:
    """Enable (or report) data-time verification rigor. Returns the applied vars.

    With ``enable=False`` this is a no-op that returns ``{}`` (leaves the RL-default
    fast path in place). With ``enable=True`` it sets each rigor var unless already
    present (``override=True`` forces). Safe to call multiple times.
    """
    if not enable:
        return {}
    applied: dict[str, str] = {}
    for k, v in RIGOR_ENV.items():
        if override or k not in os.environ:
            os.environ[k] = v
            applied[k] = v
    return applied


def rigor_status() -> dict[str, str | None]:
    """Current value of each rigor var (None if unset) — for logging/audit."""
    return {k: os.environ.get(k) for k in RIGOR_ENV}


__all__ = ["RIGOR_ENV", "set_rigorous_verification", "rigor_status"]
