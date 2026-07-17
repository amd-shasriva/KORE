#!/usr/bin/env python3
"""Supervise + monitor the KORE fold-in campaign.

Keeps the multi-day run alive across transient deaths (OOM-kill, node hiccups) and
emits sparse ALERT lines so a tail -F (with notify-on-output on "ALERT ") pings the
operator on the events that matter.

Each attempt: reap our own stale workers, (re)launch the campaign with --force (which
resumes via shard_done + the manifest), then poll its log while it runs, alerting on:
  - stage transitions (start/done)
  - real errors (Traceback / ERROR level / OOM / CUDA|HIP error)
  - retention-gate failures
On a non-completion exit it relaunches (bounded retries + cooldown); on completion or
exhausted retries it stops. Only shasriva-owned processes are reaped (root's shared
workers are never touched).
"""
from __future__ import annotations

import datetime
import os
import re
import subprocess
import time

REPO = "/home/shasriva/Kore-RL/KORE"
LOGDIR = os.path.join(REPO, "runs/full/logs")
VENV = "/home/shasriva/kore-venv/bin/python"

# Rigorous verification gates -- MUST match the datagen + the wrapper launch scripts
# (run_full_14b.sh etc.), otherwise GRPO trains on speedups vs an UNFUSED-eager
# baseline (inflated) and a weaker correctness oracle, inconsistent with the verified
# data. Set here so the campaign AND every training subprocess (GRPO rollouts) inherit
# them: honest compiler-fused baseline, enumerated adversarial+metamorphic correctness
# battery, shape augmentation, cold-cache timing. setdefault so an explicit env wins.
for _gk, _gv in {"KORE_VERIFIED_CORRECTNESS": "1", "KORE_COMPILE_BASELINE": "1",
                 "KORE_SHAPE_AUGMENT": "1", "KORE_BENCH_COLD": "1"}.items():
    os.environ.setdefault(_gk, _gv)
WORKERS = os.environ.get("KORE_DATAGEN_WORKERS", "64")
POLL_S = int(os.environ.get("KORE_SUP_POLL_S", "120"))
MAX_RETRIES = int(os.environ.get("KORE_SUP_MAX_RETRIES", "12"))
COOLDOWN_S = int(os.environ.get("KORE_SUP_COOLDOWN_S", "90"))
LOGPATH_FILE = "/tmp/kore_foldin_logpath.txt"

# Stages + --force are env-configurable so the supervisor serves BOTH the datagen
# phase and the post-datagen training chain. Default = the training chain
# (build->eval) with NO --force, so a relaunch RESUMES via the manifest + on-disk
# artifacts (completed stages skip) instead of re-running from scratch.
STAGES = os.environ.get("KORE_SUP_STAGES", "build,sft,dpo,grpo,soup,eval")
FORCE = os.environ.get("KORE_SUP_FORCE", "0") == "1"
# SFT mix cap -- lower than the 20k default so the verified-kernel slice (~4k rows)
# actually reaches its 0.28 target fraction instead of being water-filled by general.
SFT_TOTAL = os.environ.get("KORE_SUP_SFT_TOTAL", "13000")
CMD = [
    VENV, "scripts/run_campaign.py", "--model", "Qwen/Qwen3-14B", "--full-ft", "--use-hf",
    "--teacher", "claude", "--adaptive-steps",
]
if FORCE:
    CMD.append("--force")
CMD += [
    "--stages", STAGES, "--sft-total", SFT_TOTAL,
    "--gpu-ids", "0,1,2,3,4,5,6,7", "--datagen-workers", WORKERS, "--ground-reasoning",
    "--profile-reward", "0.15", "--data-root", "data/full14b",
    "--midtrain-out", "runs/full/midtrain", "--sft-out", "runs/full/sft",
    "--dpo-out", "runs/full/dpo", "--grpo-out", "runs/full/grpo", "--soup-out", "runs/full/soup",
]

STAGE_RE = re.compile(r"STAGE  campaign \(\w+\): stage (start|done): (\w+)")
ERR_RE = re.compile(
    r"Traceback \(most recent call last\)| ERROR |out of memory|HIP out of memory|"
    r"CUDA error|HIP error|OutOfMemoryError|CalledProcessError"
)
GATE_RE = re.compile(r"retention[^\n]*(FAIL|regress|below)|GATE[^\n]*FAIL|hard-stop|hard stop")


def reap_orphans():
    """Kill ONLY our own stale campaign processes (never root's shared workers)."""
    for pat in ("scripts/run_campaign.py", "from multiprocessing.spawn", "/tmp/kore_"):
        subprocess.run(["pkill", "-9", "-u", "shasriva", "-f", pat], capture_output=True)
    time.sleep(4)


def run_once(log_path: str):
    last_stage = None
    last_err = 0
    last_gate = 0
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(CMD, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL)
    while proc.poll() is None:
        time.sleep(POLL_S)
        try:
            txt = open(log_path, errors="ignore").read()
        except OSError:
            txt = ""
        stages = STAGE_RE.findall(txt)
        cur = stages[-1] if stages else None
        errs = len(ERR_RE.findall(txt))
        gates = len(GATE_RE.findall(txt))
        if cur is not None and cur != last_stage:
            print(f"ALERT STAGE {cur[1]}:{cur[0]}", flush=True)
            last_stage = cur
        if errs > last_err:
            print(f"ALERT ERROR_DETECTED total={errs}", flush=True)
            last_err = errs
        if gates > last_gate:
            print(f"ALERT RETENTION_GATE total={gates}", flush=True)
            last_gate = gates
        stage_str = f"{cur[1]}:{cur[0]}" if cur else "none"
        print(f"HEARTBEAT stage={stage_str} errs={errs}", flush=True)
    try:
        txt = open(log_path, errors="ignore").read()
    except OSError:
        txt = ""
    return ("campaign complete" in txt), proc.returncode


def main():
    print("SUPERVISOR start", flush=True)
    for attempt in range(1, MAX_RETRIES + 1):
        reap_orphans()
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(LOGDIR, f"campaign_foldin_{ts}.log")
        try:
            open(LOGPATH_FILE, "w").write(log_path)
        except OSError:
            pass
        print(f"ALERT LAUNCH attempt={attempt}/{MAX_RETRIES} workers={WORKERS} "
              f"log={os.path.basename(log_path)}", flush=True)
        done, rc = run_once(log_path)
        if done:
            print(f"ALERT CAMPAIGN_COMPLETE attempt={attempt} rc={rc}", flush=True)
            return
        print(f"ALERT CAMPAIGN_DIED attempt={attempt} rc={rc} (no completion); "
              f"relaunch in {COOLDOWN_S}s", flush=True)
        time.sleep(COOLDOWN_S)
    print("ALERT SUPERVISOR_GIVEUP (max retries exhausted)", flush=True)


if __name__ == "__main__":
    main()
