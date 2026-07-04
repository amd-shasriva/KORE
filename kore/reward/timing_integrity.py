"""Performance-reward-hacking taxonomy + timing-integrity coverage (KORE).

Correctness reward-hacking in code/kernel RL is well studied (verifiers, unit
tests, static scans). *Performance* (latency) reward-hacking is not: when the
reward is a measured speedup, a policy can win the reward WITHOUT producing a
faster correct kernel by attacking the MEASUREMENT. This module names that attack
surface and maps each class to the concrete KORE defense that closes it, so the
timing-integrity guarantee is a stated, auditable contribution rather than a pile
of scattered checks.

Every defense referenced here is implemented and GPU-validated elsewhere in the
codebase (kore.reward.reward scan, kore.env.kore_env verifier, the generated +
hand-authored drivers). This module is the single source of truth for "which
latency-hack does each check stop", and is unit-tested for full coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PerfHackClass(str, Enum):
    """Classes of PERFORMANCE (latency) reward-hacking against a speedup reward."""

    VERDICT_FORGERY = "verdict_forgery"        # print a fake median_ms/wall_ms/SNR line
    MODE_SNIFF = "mode_sniff"                   # detect bench-vs-correctness (argv/env/flags)
    STATEFUL_TIMING = "stateful_timing"        # correct for the checked calls, garbage while timed
    CLOCK_MANIPULATION = "clock_manipulation"  # sleep/sync-tamper/profiler-state to skew the timer
    NONDETERMINISM = "nondeterminism"          # pass the correctness gate by luck (random output)
    CACHE_WARMING = "cache_warming"            # exploit warm L2 so the timed region under-measures
    DELEGATION = "delegation"                  # call the vendor lib/oracle instead of computing
    EXCESSIVE_OUTLIER = "excessive_outlier"    # a measurement-error "speedup" (implausible ratio)


@dataclass(frozen=True)
class Defense:
    hack: PerfHackClass
    where: str          # where the defense lives
    mechanism: str      # how it closes the hack


# The auditable coverage map: exactly one primary defense per hack class. (Several
# classes are additionally covered in depth, but each is CLOSED by the entry here.)
DEFENSES: tuple[Defense, ...] = (
    Defense(PerfHackClass.VERDICT_FORGERY, "reward.scan_for_hacks",
            "reject sources that print SNR:/allclose:/median_ms:/wall_ms: verdict "
            "literals; the driver's verdict is parsed LAST so a forged early line loses"),
    Defense(PerfHackClass.MODE_SNIFF, "reward.scan_for_hacks",
            "reject sys.argv/argparse/getopt/os.environ reads and the driver's "
            "--bench-mode/--impl flag literals (kernel can't tell checked from timed)"),
    Defense(PerfHackClass.STATEFUL_TIMING, "env.KoreEnv + drivers",
            "re-verify correctness AFTER the timed loop on the CACHED candidate module "
            "(counter persists) with a randomized timed window; a False post-timing "
            "verdict flags the eval as a hack (reward floor)"),
    Defense(PerfHackClass.CLOCK_MANIPULATION, "reward.scan_for_hacks",
            "reject time.sleep/asyncio.sleep and GPU sync/profiler-state tampering "
            "(set_sync_debug_mode/cudaProfilerStart/hipDeviceSetLimit)"),
    Defense(PerfHackClass.NONDETERMINISM, "env.KoreEnv determinism re-check",
            "re-run the primary shape; require a stable verdict (SNR within tol) so a "
            "lucky/partly-random pass is dropped to the incorrect tier"),
    Defense(PerfHackClass.CACHE_WARMING, "drivers cold-cache timing",
            "flush the L2/Infinity cache between timed iters (KernelBench-style cold "
            "measurement) so a warmed cache cannot fake a speedup"),
    Defense(PerfHackClass.DELEGATION, "reward.scan_for_hacks",
            "reject vendor/torch/oracle delegation (aiter/hipblaslt, torch matmul family, "
            "the @ operator, sys.modules, getattr handles, reference-oracle imports)"),
    Defense(PerfHackClass.EXCESSIVE_OUTLIER, "reward.compute_reward",
            "cap the scored speedup at excessive_speedup_flag and withhold the fast_p "
            "bonus for measurement-error outliers / high-variance (cv) timings"),
)


def coverage() -> dict[PerfHackClass, Defense]:
    """Map each performance-hack class to its primary defense (one-to-one)."""
    return {d.hack: d for d in DEFENSES}


def uncovered() -> list[PerfHackClass]:
    """Any hack class with no mapped defense (should always be empty)."""
    covered = {d.hack for d in DEFENSES}
    return [h for h in PerfHackClass if h not in covered]


def report() -> str:
    """Human-readable timing-integrity coverage table (for the paper / logs)."""
    cov = coverage()
    lines = ["KORE timing-integrity coverage (performance-reward-hacking):"]
    for h in PerfHackClass:
        d = cov.get(h)
        lines.append(f"  {h.value:20s} <- {d.where}: {d.mechanism}" if d
                     else f"  {h.value:20s} <- UNCOVERED")
    return "\n".join(lines)
