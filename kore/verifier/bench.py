"""Bench tool — wall-clock GPU kernel benchmarks with proper synchronization."""

from __future__ import annotations

import asyncio
import re
import sys
from typing import Any, Callable


async def bench_wallclock(
    driver_script: str,
    driver_args: list[str] | None = None,
    warmup_iters: int = 10,
    bench_iters: int = 30,
    timeout_sec: int = 300,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> dict:
    """Run a wall-clock benchmark with GPU synchronization.

    The driver script should accept --warmup, --iters, --bench-mode flags
    and print "wall_ms: XX.XX" for each measured iteration or a single
    summary line.

    Alternatively, the driver can print "median_ms: XX.XX" directly.

    Args:
        driver_script: Path to Python benchmark driver.
        driver_args: Additional arguments.
        warmup_iters: Warmup iterations (not timed).
        bench_iters: Timed iterations.
        timeout_sec: Maximum runtime.

    Returns:
        Dict with: median_ms, min_ms, max_ms, all_times, message.
    """
    args = (driver_args or []) + [
        "--warmup", str(warmup_iters),
        "--iters", str(bench_iters),
        "--bench-mode",
    ]
    cmd = [sys.executable, driver_script] + args

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        proc.kill()
        return {"success": False, "message": f"TIMEOUT after {timeout_sec}s"}

    stdout_text = stdout.decode(errors="replace")
    stderr_text = stderr.decode(errors="replace")
    full_output = stdout_text + "\n" + stderr_text

    if proc.returncode != 0:
        return {
            "success": False,
            "message": f"BENCH CRASHED (exit {proc.returncode})",
            "output": full_output[-2000:],
        }

    # Parse individual wall_ms values
    times = [float(m) for m in re.findall(r"wall_ms:\s*([\d.]+)", full_output)]

    # Or parse a single median_ms
    median_match = re.search(r"median_ms:\s*([\d.]+)", full_output)

    if times:
        times_sorted = sorted(times)
        median = times_sorted[len(times_sorted) // 2]
        result = {
            "success": True,
            "median_ms": round(median, 4),
            "min_ms": round(min(times), 4),
            "max_ms": round(max(times), 4),
            "n_samples": len(times),
            "all_times_ms": [round(t, 4) for t in times],
            "message": (
                f"BENCH: median={median:.4f} ms "
                f"(min={min(times):.4f}, max={max(times):.4f}, n={len(times)})"
            ),
        }
        if on_result:
            on_result(result)
        return result
    elif median_match:
        median = float(median_match.group(1))
        result = {
            "success": True,
            "median_ms": round(median, 4),
            "message": f"BENCH: median={median:.4f} ms",
        }
        if on_result:
            on_result(result)
        return result
    else:
        return {
            "success": False,
            "message": "NO TIMING DATA in output. Driver must print 'wall_ms: X.XX' or 'median_ms: X.XX'.",
            "output": full_output[-1500:],
        }
