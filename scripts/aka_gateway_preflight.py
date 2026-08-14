#!/usr/bin/env python
"""Prove an API arm can actually reach its model, before the sweep starts.

An unreachable gateway does not look like an error to the arena. `_api_generate`
raises per call, the attempt is caught, the task is scored with no kernel, and
the ledger fills with rows that are indistinguishable from "the model wrote
nothing usable". That is how a network outage becomes a published capability
number. The 2026-08-10 opus arm had the mirror-image failure -- it ran to
completion and every speedup was null -- and cost a full allocation.

So: one real generation, on the compute node that will do the work, using the
exact resolution path `_api_generate` uses (KORE_TEACHER_MODEL forced to the
requested model, then asserted), before a single task is discovered.

Exit codes are distinct because the sbatch treats them differently:
  0  the arm is live
  3  configuration is wrong (no key, model resolves to something else)
  4  the gateway is unreachable or refused the request
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    try:
        from kore.data.teacher import ClaudeTeacher, load_env_local
    except Exception as exc:  # noqa: BLE001 - a broken import is a config fault
        print(f"PREFLIGHT FAIL: cannot import the teacher: "
              f"{type(exc).__name__}: {exc}", flush=True)
        return 3

    load_env_local()
    if not os.environ.get("AMD_LLM_API_KEY"):
        print("PREFLIGHT FAIL: AMD_LLM_API_KEY is unset. The gateway needs it in "
              ".env.local at the repo root; a compute node inherits nothing from "
              "your login shell.", flush=True)
        return 3

    # Mirror _api_generate exactly. ClaudeTeacher resolves its model from
    # KORE_TEACHER_MODEL, so .env.local would otherwise silently win over the
    # argument and we would preflight a different model than the arm runs.
    os.environ["KORE_TEACHER_MODEL"] = args.model
    try:
        teacher = ClaudeTeacher(model=args.model, temperature=0.0,
                                max_tokens=args.max_tokens)
    except Exception as exc:  # noqa: BLE001
        print(f"PREFLIGHT FAIL: cannot construct the teacher for {args.model}: "
              f"{type(exc).__name__}: {exc}", flush=True)
        return 3
    if teacher.model != args.model:
        print(f"PREFLIGHT FAIL: asked for {args.model}, teacher resolved "
              f"{teacher.model}. A ledger naming the wrong model is worse than "
              f"no ledger.", flush=True)
        return 3

    try:
        reply = teacher.generate([{"role": "user",
                                   "content": "Reply with the single word: ready"}])
    except Exception as exc:  # noqa: BLE001 - this is the case we exist to catch
        print(f"PREFLIGHT FAIL: {args.model} is unreachable from "
              f"{os.uname().nodename}: {type(exc).__name__}: {exc}", flush=True)
        return 4

    if not (reply or "").strip():
        print(f"PREFLIGHT FAIL: {args.model} returned an empty reply. Every task "
              f"would score zero and look like a model failure.", flush=True)
        return 4

    print(f"preflight ok: {args.model} live from {os.uname().nodename} "
          f"({len(reply)} chars)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
