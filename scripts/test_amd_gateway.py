"""Verify the AMD LLM gateway teacher (Claude Opus) is reachable and can write a
kernel. Reads creds from .env.local. If the gateway base URL is missing, prints
exactly what to add.

    PYTHONPATH=. python scripts/test_amd_gateway.py
"""

from __future__ import annotations

import os
import sys

from kore.data.prompts import extract_kernel
from kore.data.teacher import ClaudeTeacher, load_env_local


def main() -> int:
    load_env_local()
    if not os.environ.get("AMD_LLM_API_KEY"):
        print("Missing AMD_LLM_API_KEY in .env.local")
        return 2
    if not os.environ.get("AMD_LLM_GATEWAY_URL"):
        print("AMD_LLM_GATEWAY_URL not set. Add the gateway base URL to .env.local, e.g.:")
        print("  AMD_LLM_GATEWAY_URL=https://<amd-slai-gateway-host>/<path>")
        print("(the key is present; only the endpoint is needed).")
        return 3
    model = os.environ.get("KORE_TEACHER_MODEL", "claude-opus-4.8")
    try:
        t = ClaudeTeacher(model=model, max_tokens=1024, temperature=0.2)
        out = t.generate([
            {"role": "user", "content":
             "Write a minimal compiling Triton RMSNorm kernel for AMD gfx942. "
             "Respond with ANALYSIS, PROPOSED_CHANGE, and FULL_KERNEL."}])
    except Exception as e:  # noqa: BLE001
        print(f"Gateway call failed: {e}")
        return 1
    print(f"model={model} response_chars={len(out)}")
    print("kernel_extracted:", bool(extract_kernel(out)))
    print(out[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
