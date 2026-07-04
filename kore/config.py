"""Central KORE configuration.

Single source of truth for arch/target, thresholds, and paths. Everything that
touches the GPU or the verifier reads from here so the gfx942 retarget is done
in exactly one place (the KernelForge sources default to gfx950).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

KORE_ROOT = Path(__file__).resolve().parent.parent          # /root/Kore-rl/kore
WORKSPACE_ROOT = KORE_ROOT.parent                            # /root/Kore-rl
REPOS_DIR = WORKSPACE_ROOT / "repos"
DATA_DIR = KORE_ROOT / "data"
RUNS_DIR = KORE_ROOT / "runs"
CONFIGS_DIR = KORE_ROOT / "configs"


@dataclass
class KoreConfig:
    """Runtime configuration. Override via env vars where noted."""

    gpu_target: str = field(default_factory=lambda: os.environ.get("GPU_TARGET", "gfx942"))
    rocm_path: str = field(default_factory=lambda: os.environ.get("ROCM_PATH", "/opt/rocm"))

    # correctness gate
    snr_threshold_fp32: float = 30.0
    snr_threshold_lowp: float = 25.0
    atol: float = 1e-2
    rtol: float = 1e-2

    # benchmark trust
    warmup_iters: int = 10
    bench_iters: int = 30
    min_variance_runs: int = 3
    max_variance_runs: int = 5
    cv_threshold_pct: float = 3.0
    noise_floor_pct: float = 2.0

    # anti-hack: determinism re-check on the RL correctness path. A reward-hacking
    # kernel that emits (partly) random output can pass the SNR gate by LUCK on a
    # single run. We re-run the primary shape once and require the verdict to be
    # stable: still correct, and SNR within determinism_snr_tol_db of the first run.
    # A kernel whose SNR swings more than the tolerance (or flips to incorrect) is
    # non-deterministic and dropped to the incorrect tier (never rewarded). The
    # tolerance is generous enough to spare legitimate atomic-reduction jitter
    # (which perturbs SNR by <~1 dB) while catching a lucky-pass hack (which swings
    # tens of dB, often to negative SNR).
    verifier_determinism_check: bool = True
    determinism_snr_tol_db: float = 10.0

    # data scale: expand each task's shapes into a diverse (small/medium/large +
    # non-aligned) set so the policy must learn shape-robust kernels, not memorize
    # one tile config. Opt-in (changes eval cost + difficulty). See tasks/augment.py.
    shape_augment: bool = field(
        default_factory=lambda: os.environ.get("KORE_SHAPE_AUGMENT", "0") == "1")
    shape_augment_max: int = 6

    # reward shaping
    correctness_weight: float = 0.3
    excessive_speedup_flag: float = 10.0
    reward_compile_fail: float = -1.0
    reward_incorrect: float = 0.0
    # A flagged reward-hack is punished STRICTLY harder than an honest compile
    # failure: actively cheating is worse than failing to build. This keeps the
    # anti-hack floor as the unique minimum of the tier ladder
    # (hack < compile_fail < incorrect < correct). Must stay < reward_compile_fail.
    reward_hack: float = -1.5

    # --- reward-shaping upgrades (literature review) -----------------------
    # P1 — bounded continuous sub-threshold shaping (LLM-VeriOpt style).
    # A compiled-but-incorrect kernel earns a small credit proportional to how
    # close its worst-shape SNR is to the correctness gate, instead of a flat 0
    # (sparse reward -> early-RL collapse). The credit is bounded in
    # [0, eps_shape] and eps_shape is kept STRICTLY below correctness_weight so a
    # shaped-incorrect kernel can NEVER reach the correct tier (lexicographic
    # dominance holds absolutely). Never applied to a flagged hack / compile-fail.
    subthreshold_shaping: bool = True
    eps_shape: float = 0.05  # invariant: 0 < eps_shape + format_weight < correctness_weight

    # P2 — format-compliance term (Compiler-R1 style). Small bonus for emitting a
    # valid FULL_KERNEL contract (parses to a kernel), small penalty for malformed
    # output. Symmetric magnitude, applied only to the incorrect/correct tiers and
    # bounded so tiny (<< every inter-tier gap) it can never flip tier ordering.
    format_weight: float = 0.02  # invariant: 2*format_weight < smallest inter-tier gap

    # P3 — correctness->latency curriculum phase for compute_reward:
    #   "full"        : correctness_weight + speedup (default; current behavior)
    #   "correctness" : zero the speed term -> every correct kernel == correctness_weight
    #                   (run a correctness-only GRPO phase first)
    #   "latency"     : full correctness_weight + speedup (same as "full")
    reward_phase: str = "full"

    # --- P4: speedup shaping to break the "correct-but-slow" (lazy-optimization)
    # plateau and give real GROUP-RELATIVE gradient at the >1x crossover ----------
    # Diagnosis (Dr.Kernel 2026 "lazy optimization"): a purely LINEAR speed term
    # (reward = correctness_weight + su) gives almost no reward *contrast* in the
    # 0.7-1.1x band the policy stalls in, so GRPO's group-relative advantage barely
    # distinguishes a 0.95x kernel from a 1.05x one — the model learns "be correct"
    # and stops. Two fixes (both preserve lexicographic dominance: every correct
    # kernel still scores >= correctness_weight > any incorrect kernel):
    #   1. speedup_log: shape the speed term as w*su for su<=1 (linear, non-negative)
    #      and w*(1+ln(su)) for su>1 — continuous at su=1, monotonic, but with a
    #      steeper effective slope right at the baseline crossover.
    #   2. fast_p_bonus: significance-gated DISCRETE bonuses for actually BEATING the
    #      baseline at 1.0x / 1.2x / 1.5x. This is the strong signal that makes ">1x"
    #      a distinct, high-value outcome (large positive group-relative advantage
    #      for the kernels that cross the baseline), instead of a marginal linear
    #      increment. Only awarded when the timing is statistically trustworthy
    #      (cv <= cv_threshold_pct) and not flagged as a measurement-error outlier,
    #      so the policy cannot farm the bonus with noisy/lucky timings.
    speedup_weight: float = 1.0
    speedup_log: bool = True
    # cumulative (threshold, bonus) pairs; awarded for every threshold the (worst-
    # shape) speedup meets when significant. Sum kept modest vs correctness_weight
    # so it never inverts the correctness gate.
    fast_p_bonus: tuple = ((1.0, 0.30), (1.2, 0.30), (1.5, 0.40))
    fast_p_significant_only: bool = True

    # --- P5: hardware-counter-grounded DENSE reward (flagship novelty) ----------
    # A bounded, baseline-relative roofline-attainment bonus (see
    # kore.reward.profile_reward) added ONLY on the correct tier. It gives gradient
    # in the correct-but-slow band where wall-clock speedup is flat, by rewarding
    # the causes of speed (fewer pipeline stalls / less memory traffic than the
    # vendor baseline). Kept STRICTLY below the fast_p bonuses so actually beating
    # the baseline always dominates a merely counter-efficient kernel. Weight 0.0
    # => fully inert (feature-flagged); enabled via --profile-reward once the
    # rocprofv3 path is GPU-validated, then ablated as the novel contribution.
    # Env-overridable so it propagates to the accelerate-launched training subprocs.
    profile_reward_weight: float = field(
        default_factory=lambda: float(os.environ.get("KORE_PROFILE_REWARD_WEIGHT", "0.0")))

    # multi-turn credit
    gamma: float = 0.4

    data_dir: Path = DATA_DIR
    runs_dir: Path = RUNS_DIR

    def snr_threshold_for(self, dtype: str) -> float:
        d = (dtype or "").lower()
        if "fp8" in d or "mxfp4" in d or "fp4" in d or "mxfp8" in d:
            return self.snr_threshold_lowp
        if "fp16" in d or "bf16" in d or "float16" in d or "bfloat16" in d:
            return self.snr_threshold_lowp
        return self.snr_threshold_fp32

    def fp8_dtype(self):
        """Arch-correct fp8 e4m3: FNUZ on gfx942 (CDNA3), OCP on gfx950 (CDNA4)."""
        import torch
        if self.gpu_target == "gfx942":
            return torch.float8_e4m3fnuz
        return torch.float8_e4m3fn

    def __post_init__(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._check_reward_invariants()

    def _check_reward_invariants(self) -> None:
        """Fail fast if a (possibly env-overridden) config would break the
        lexicographic reward ladder or let a shaping term lead the objective.

        These are the invariants the reward code documents but previously never
        enforced — a bad KORE_PROFILE_REWARD_WEIGHT or an edited weight could
        silently invert tiers or let the profiler bonus outweigh a real speed win.
        """
        # anti-hack floor is the unique minimum: hack < compile_fail < incorrect.
        assert self.reward_hack < self.reward_compile_fail < self.reward_incorrect, (
            "reward tiers must satisfy reward_hack < reward_compile_fail < reward_incorrect")
        # a shaped-incorrect kernel (<= eps_shape + format_weight) can never reach
        # the correct tier (>= correctness_weight).
        assert self.eps_shape + self.format_weight < self.correctness_weight, (
            "eps_shape + format_weight must stay below correctness_weight "
            "(else an incorrect kernel could reach the correct tier)")
        # the profiler dense bonus SHAPES, never LEADS: it must be strictly below the
        # smallest fast_p threshold bonus so a genuinely faster kernel always wins.
        if self.profile_reward_weight and self.fast_p_bonus:
            min_bonus = min(b for _, b in self.fast_p_bonus)
            assert self.profile_reward_weight < min_bonus, (
                f"profile_reward_weight ({self.profile_reward_weight}) must be < the "
                f"smallest fast_p bonus ({min_bonus}) so the profiler never leads")


CONFIG = KoreConfig()
