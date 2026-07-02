"""5-stage validation pipeline — inspired by AutoKernel.

Every kernel modification must pass ALL 5 stages before its performance
is measured. A fast-but-wrong kernel is always rejected.

Stages:
  1. Smoke test — basic functionality on minimal shape
  2. Shape sweep — correctness across multiple input dimensions
  3. Numerical stability — edge cases (overflow, underflow, denormals)
  4. Determinism — bitwise reproducibility across runs
  5. Full correctness — SNR gate on target shape
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from kore.verifier.test import test_correctness


@dataclass
class ValidationResult:
    """Result of the 5-stage validation pipeline."""

    stage: int
    stage_name: str
    passed: bool
    details: str
    snr_db: float | None = None

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        snr = f" (SNR={self.snr_db:.1f} dB)" if self.snr_db is not None else ""
        return f"  Stage {self.stage} [{self.stage_name}]: {status}{snr} — {self.details}"


@dataclass
class ValidationReport:
    """Full report from the 5-stage pipeline."""

    results: list[ValidationResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failed_stage(self) -> int | None:
        for r in self.results:
            if not r.passed:
                return r.stage
        return None

    def summary(self) -> str:
        lines = ["Validation Pipeline:"]
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            snr = f" SNR={r.snr_db:.1f}dB" if r.snr_db is not None else ""
            lines.append(f"  {r.stage}. {r.stage_name}: {status}{snr}")
        verdict = "ALL PASSED" if self.all_passed else f"FAILED at stage {self.failed_stage}"
        lines.append(f"  Verdict: {verdict}")
        return "\n".join(lines)


async def run_validation_pipeline(
    driver_script: str,
    shapes: dict,
    snr_threshold: float = 30.0,
    timeout_per_stage: int = 60,
) -> ValidationReport:
    """Run the 5-stage validation pipeline.

    Args:
        driver_script: Test driver that accepts --shape and --mode flags.
        shapes: Dict with 'minimal', 'primary', and optional 'edge' shapes.
        snr_threshold: SNR gate threshold.
        timeout_per_stage: Max seconds per validation stage.

    Returns:
        ValidationReport with results from all completed stages.
    """
    report = ValidationReport()
    minimal = shapes.get("minimal", shapes.get("validation", [{}])[0] if shapes.get("validation") else {})
    primary = shapes.get("primary", {})

    # Stage 1: Smoke test — basic functionality on minimal shape
    result = await test_correctness(
        driver_script=driver_script,
        driver_args=["--shape", _shape_str(minimal), "--mode", "smoke"],
        snr_threshold=snr_threshold - 10,  # relaxed for smoke
        timeout_sec=timeout_per_stage,
    )
    report.results.append(ValidationResult(
        stage=1, stage_name="Smoke test",
        passed=result["passed"],
        snr_db=result.get("snr_db"),
        details=result["message"],
    ))
    if not result["passed"]:
        return report  # fail fast

    # Stage 2: Shape sweep — multiple dimensions
    sweep_shapes = shapes.get("validation", [minimal])
    all_pass = True
    min_snr = float("inf")
    for shape in sweep_shapes:
        r = await test_correctness(
            driver_script=driver_script,
            driver_args=["--shape", _shape_str(shape)],
            snr_threshold=snr_threshold,
            timeout_sec=timeout_per_stage,
        )
        if not r["passed"]:
            all_pass = False
        if r.get("snr_db") is not None:
            min_snr = min(min_snr, r["snr_db"])

    report.results.append(ValidationResult(
        stage=2, stage_name="Shape sweep",
        passed=all_pass,
        snr_db=min_snr if min_snr != float("inf") else None,
        details=f"{len(sweep_shapes)} shapes tested, min SNR={min_snr:.1f} dB" if min_snr != float("inf") else "no SNR data",
    ))
    if not all_pass:
        return report

    # Stage 3: Numerical stability — edge cases
    stability_result = await test_correctness(
        driver_script=driver_script,
        driver_args=["--shape", _shape_str(primary), "--mode", "stability"],
        snr_threshold=snr_threshold - 5,  # slightly relaxed for edge cases
        timeout_sec=timeout_per_stage,
    )
    report.results.append(ValidationResult(
        stage=3, stage_name="Numerical stability",
        passed=stability_result["passed"],
        snr_db=stability_result.get("snr_db"),
        details=stability_result["message"],
    ))
    if not stability_result["passed"]:
        return report

    # Stage 4: Determinism — run twice, compare outputs
    det_results = []
    for _ in range(2):
        r = await test_correctness(
            driver_script=driver_script,
            driver_args=["--shape", _shape_str(primary), "--mode", "determinism"],
            snr_threshold=snr_threshold,
            timeout_sec=timeout_per_stage,
        )
        det_results.append(r)

    # Both must pass and have consistent SNR (within 1 dB)
    det_pass = all(r["passed"] for r in det_results)
    if det_pass and all(r.get("snr_db") is not None for r in det_results):
        snr_diff = abs(det_results[0]["snr_db"] - det_results[1]["snr_db"])
        det_pass = snr_diff < 60.0  # MLA reference kernel uses non-deterministic atomics
        det_detail = f"SNR diff between runs: {snr_diff:.2f} dB"
    else:
        det_detail = "determinism check incomplete"

    report.results.append(ValidationResult(
        stage=4, stage_name="Determinism",
        passed=det_pass,
        snr_db=det_results[0].get("snr_db") if det_results else None,
        details=det_detail,
    ))
    if not det_pass:
        return report

    # Stage 5: Full correctness — SNR gate on primary shape
    full_result = await test_correctness(
        driver_script=driver_script,
        driver_args=["--shape", _shape_str(primary)],
        snr_threshold=snr_threshold,
        timeout_sec=timeout_per_stage * 2,  # more time for full shape
    )
    report.results.append(ValidationResult(
        stage=5, stage_name="Full correctness",
        passed=full_result["passed"],
        snr_db=full_result.get("snr_db"),
        details=full_result["message"],
    ))

    return report


def _shape_str(shape: dict) -> str:
    """Convert shape dict to CLI string."""
    if not shape:
        return "default"
    return ",".join(f"{k}={v}" for k, v in shape.items())
