"""PMC tool — hardware performance counter collection via rocprofv3."""

from __future__ import annotations

import asyncio
import glob
import os
import sys
import tempfile
import time

from kore.verifier.parsers.rocprofv3 import KernelPMC, parse_rocprofv3_csv


# Standard counter sets for common analysis patterns
COUNTER_SETS = {
    "standard": [
        "SQ_INSTS_VALU_MFMA_BF16",
        "SQ_INSTS_VMEM",
        "SQ_WAIT_INST_LDS",
        "SQ_WAIT_INST_ANY",
    ],
    "full": [
        "SQ_INSTS_VALU",
        "SQ_INSTS_VALU_MFMA_BF16",
        "SQ_INSTS_VALU_MFMA_F16",
        "SQ_INSTS_VMEM",
        "SQ_INSTS_SALU",
        "SQ_WAIT_INST_LDS",
        "SQ_WAIT_INST_VMEM",
        "SQ_WAIT_INST_ANY",
    ],
    "memory": [
        "SQ_INSTS_VMEM",
        "SQ_WAIT_INST_VMEM",
        "TCP_TCC_READ_REQ_sum",
        "TCP_TCC_WRITE_REQ_sum",
    ],
    "compute": [
        "SQ_INSTS_VALU",
        "SQ_INSTS_VALU_MFMA_BF16",
        "SQ_INSTS_VALU_MFMA_F16",
        "SQ_INSTS_VALU_MFMA_F32",
        "SQ_INSTS_SALU",
    ],
}


async def profile_pmc(
    driver_script: str,
    driver_args: list[str] | None = None,
    counters: list[str] | None = None,
    counter_set: str = "standard",
    kernel_filter: str | None = None,
    timeout_sec: int = 300,
) -> dict:
    """Collect hardware performance counters via rocprofv3.

    Args:
        driver_script: Path to Python driver that invokes the kernel.
        driver_args: Additional arguments for the driver.
        counters: Explicit counter list. Overrides counter_set if provided.
        counter_set: Named counter set ("standard", "full", "memory", "compute").
        kernel_filter: If set, only return data for kernels matching this substring.
        timeout_sec: Maximum runtime.

    Returns:
        Dict with: kernels (list of KernelPMC summaries), raw_csv_path, message.
    """
    if counters is None:
        counters = COUNTER_SETS.get(counter_set, COUNTER_SETS["standard"])

    outdir = tempfile.mkdtemp(prefix=f"pmc_{int(time.time())}_")

    cmd = [
        "rocprofv3",
        "--pmc", *counters,
        "-d", outdir,
        "--output-format", "csv",
        "--",
        sys.executable, driver_script,
    ] + (driver_args or [])

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

    stderr_text = stderr.decode(errors="replace")

    if proc.returncode != 0:
        return {
            "success": False,
            "message": f"rocprofv3 FAILED (exit {proc.returncode})",
            "output": stderr_text[-2000:],
        }

    # Find CSV output
    csv_files = glob.glob(os.path.join(outdir, "*.csv"))
    if not csv_files:
        # Try pftrace format
        csv_files = glob.glob(os.path.join(outdir, "**/*.csv"), recursive=True)

    if not csv_files:
        return {
            "success": False,
            "message": f"No CSV output in {outdir}",
            "output": stderr_text[-1000:],
        }

    # Parse all CSV files
    all_kernels: list[KernelPMC] = []
    for csv_path in csv_files:
        try:
            all_kernels.extend(parse_rocprofv3_csv(csv_path))
        except Exception as e:
            return {
                "success": False,
                "message": f"CSV parse error: {e}",
            }

    # Filter if requested
    if kernel_filter:
        all_kernels = [k for k in all_kernels if kernel_filter in k.kernel_name]

    if not all_kernels:
        return {
            "success": False,
            "message": f"No kernel dispatches found{' matching ' + kernel_filter if kernel_filter else ''}",
        }

    # Build summary
    summaries = [k.summary() for k in all_kernels]
    summary_text = "\n\n".join(summaries)

    return {
        "success": True,
        "n_kernels": len(all_kernels),
        "kernels": [
            {
                "name": k.kernel_name,
                "counters": k.counters,
                "wait_mfma_ratio": round(k.wait_mfma_ratio, 2),
                "diagnosis": k.diagnosis,
            }
            for k in all_kernels
        ],
        "raw_csv_path": csv_files[0],
        "message": f"PMC collected for {len(all_kernels)} kernel dispatch(es):\n\n{summary_text}",
    }
