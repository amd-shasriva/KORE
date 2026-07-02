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

    # reward shaping
    correctness_weight: float = 0.3
    excessive_speedup_flag: float = 10.0
    reward_compile_fail: float = -1.0
    reward_incorrect: float = 0.0

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


CONFIG = KoreConfig()
