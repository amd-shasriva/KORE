"""Build tool — compile GPU kernels with artifact verification.

Build output is routed through RTK when available for 80-90% token savings.
"""

from __future__ import annotations

import asyncio
import glob
import os
import shutil
from pathlib import Path

from kore.verifier.rtk import smart_wrap


async def ck_build(
    build_dir: str,
    deploy_dir: str | None = None,
    clean_stale: bool = True,
    jobs: int = 4,
) -> dict:
    """Build a CK kernel via ninja and verify .so deployment.

    Args:
        build_dir: Path to the ninja build directory.
        deploy_dir: Where to copy the final .so. If None, uses parent of build_dir.
        clean_stale: Remove stale .cuda.o files first (header dep tracking workaround).
        jobs: Parallel ninja jobs.

    Returns:
        Dict with keys: success, message, so_path, build_output.
    """
    build_path = Path(build_dir)
    if deploy_dir is None:
        deploy_dir = str(build_path.parent.parent)
    deploy_path = Path(deploy_dir)

    # Step 1: Clean stale objects (CK header deps not tracked by ninja)
    if clean_stale:
        stale = glob.glob(str(build_path / "*.cuda.o"))
        for f in stale:
            os.remove(f)
        stale_msg = f"Cleaned {len(stale)} stale .cuda.o files. " if stale else ""
    else:
        stale_msg = ""

    # Step 2: Build (RTK filters ninja output: ~80% token savings)
    build_cmd = smart_wrap(["ninja", f"-j{jobs}"])
    proc = await asyncio.create_subprocess_exec(
        *build_cmd,
        cwd=str(build_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)
    stdout_text = stdout.decode(errors="replace")
    stderr_text = stderr.decode(errors="replace")

    if proc.returncode != 0:
        return {
            "success": False,
            "message": f"BUILD FAILED (exit {proc.returncode})",
            "build_output": stderr_text[-3000:],
        }

    # Step 3: Find and deploy .so
    so_files = glob.glob(str(build_path / "*.so"))
    if not so_files:
        return {
            "success": False,
            "message": "BUILD SUCCEEDED but no .so produced — check CMakeLists.txt",
            "build_output": stdout_text[-1000:],
        }

    deployed = []
    for so in so_files:
        dest = deploy_path / os.path.basename(so)
        shutil.copy2(so, str(dest))
        deployed.append(str(dest))

    return {
        "success": True,
        "message": f"{stale_msg}BUILD OK. Deployed: {', '.join(deployed)}",
        "so_path": deployed[0] if len(deployed) == 1 else deployed,
        "build_output": stdout_text[-500:],
    }


async def triton_build(
    kernel_path: str,
    gpu_target: str = "gfx950",
) -> dict:
    """Verify a Triton kernel compiles by importing it.

    Triton kernels compile on first call, so we do a dry-run import
    and check for syntax/type errors.
    """
    proc = await asyncio.create_subprocess_exec(
        "python", "-c", f"import importlib.util; "
        f"spec = importlib.util.spec_from_file_location('k', '{kernel_path}'); "
        f"mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)

    if proc.returncode != 0:
        return {
            "success": False,
            "message": f"IMPORT FAILED: {stderr.decode(errors='replace')[-2000:]}",
        }

    return {
        "success": True,
        "message": f"Triton kernel at {kernel_path} imports successfully.",
    }


async def generic_build(
    command: list[str],
    cwd: str,
    deploy_so: str | None = None,
    deploy_dest: str | None = None,
) -> dict:
    """Generic build via arbitrary command (cmake, make, pip install -e, etc.)."""
    cmd = smart_wrap(command)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)
    stdout_text = stdout.decode(errors="replace")
    stderr_text = stderr.decode(errors="replace")

    if proc.returncode != 0:
        return {
            "success": False,
            "message": f"BUILD FAILED (exit {proc.returncode})",
            "build_output": (stdout_text + "\n" + stderr_text)[-3000:],
        }

    result = {
        "success": True,
        "message": "BUILD OK.",
        "build_output": stdout_text[-500:],
    }

    # Optional .so deployment
    if deploy_so and deploy_dest:
        if os.path.exists(deploy_so):
            shutil.copy2(deploy_so, deploy_dest)
            result["message"] += f" Deployed {deploy_so} → {deploy_dest}"
            result["so_path"] = deploy_dest

    return result
