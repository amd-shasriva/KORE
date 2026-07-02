"""Test tool — run kernel correctness checks with SNR gating."""

from __future__ import annotations

import asyncio
import re
import sys


async def test_correctness(
    driver_script: str,
    driver_args: list[str] | None = None,
    snr_threshold: float = 30.0,
    timeout_sec: int = 120,
) -> dict:
    """Run a kernel test driver and extract SNR for correctness gating.

    The driver script MUST print at least one of:
      - "SNR: XX.XX dB"  (preferred)
      - "allclose: True/False"
      - "max_diff: X.XXe-XX"

    Args:
        driver_script: Path to Python test driver.
        driver_args: Additional arguments to pass to the driver.
        snr_threshold: Minimum SNR in dB to pass (default 30.0).
        timeout_sec: Maximum runtime before killing (default 120s).

    Returns:
        Dict with: passed, snr_db, max_diff, allclose, output.
    """
    cmd = [sys.executable, driver_script] + (driver_args or [])

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        proc.kill()
        return {
            "passed": False,
            "message": f"TIMEOUT after {timeout_sec}s",
            "output": "",
        }

    stdout_text = stdout.decode(errors="replace")
    stderr_text = stderr.decode(errors="replace")
    full_output = stdout_text + "\n" + stderr_text

    if proc.returncode != 0:
        return {
            "passed": False,
            "message": f"DRIVER CRASHED (exit {proc.returncode})",
            "output": full_output[-2000:],
        }

    # Parse SNR
    snr_match = re.search(r"SNR:\s*([-\d.]+)\s*dB", full_output)
    snr_db = float(snr_match.group(1)) if snr_match else None

    # Parse allclose
    allclose_match = re.search(r"allclose:\s*(True|False)", full_output, re.IGNORECASE)
    allclose = allclose_match.group(1).lower() == "true" if allclose_match else None

    # Parse max_diff
    diff_match = re.search(r"max_diff:\s*([\d.eE+-]+)", full_output)
    max_diff = float(diff_match.group(1)) if diff_match else None

    # Determine pass/fail
    if snr_db is not None:
        passed = snr_db >= snr_threshold
        verdict = f"SNR={snr_db:.2f} dB (threshold={snr_threshold})"
    elif allclose is not None:
        passed = allclose
        verdict = f"allclose={allclose}"
    else:
        passed = False
        verdict = "NO CORRECTNESS METRIC FOUND in output"

    result = {
        "passed": passed,
        "snr_db": snr_db,
        "max_diff": max_diff,
        "allclose": allclose,
        "message": f"{'PASS' if passed else 'FAIL'}: {verdict}",
    }
    # On PASS, SNR/max_diff/allclose already carry the signal — the raw tail
    # is dead weight against the next turn's input budget. Keep on FAIL so
    # the agent can inspect warnings / numerical context.
    if not passed:
        result["output"] = full_output[-1500:]
    return result
