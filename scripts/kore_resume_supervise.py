#!/usr/bin/env python3
"""Auto-resume + supervise the pipeline on the paradigm-v2 code after the pause.

The pause guard (``kore_pause_after_datagen.py``) halts the campaign at the
datagen->build boundary and writes ``/tmp/kore_paused_after_datagen``. This watcher:

  1. waits for that sentinel (datagen fully done, campaign stopped);
  2. reaps any stale campaign/supervisor/worker processes we own;
  3. relaunches the campaign with ``--stages build,midtrain,sft,dpo,grpo,soup,eval``
     (datagen is DONE and skipped; ``--force`` re-runs build on the NEW code, and a
     fresh process picks up the paradigm-v2 code via the editable install);
  4. supervises it across transient deaths (bounded retries + cooldown), emitting
     sparse ALERT lines (stage transitions / errors / retention-gate) for a tail-F.

Only shasriva-owned processes are reaped; root's shared workers are never touched.
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
WORKERS = os.environ.get("KORE_DATAGEN_WORKERS", "64")
POLL_S = int(os.environ.get("KORE_SUP_POLL_S", "120"))
MAX_RETRIES = int(os.environ.get("KORE_SUP_MAX_RETRIES", "12"))
COOLDOWN_S = int(os.environ.get("KORE_SUP_COOLDOWN_S", "90"))
SENTINEL = "/tmp/kore_paused_after_datagen"
LOGPATH_FILE = "/tmp/kore_resume_logpath.txt"

# RESUME stage list: datagen is complete + on disk, so skip it; run everything else
# on the new paradigm-v2 code. --force re-runs build (regenerates SFT/DPO from the
# verified shards using the new assembly) and never re-runs the (skipped) datagen.
CMD = [
    VENV, "scripts/run_campaign.py", "--model", "Qwen/Qwen3-14B", "--full-ft", "--use-hf",
    "--teacher", "claude", "--adaptive-steps", "--force",
    "--stages", "build,midtrain,sft,dpo,grpo,soup,eval",
    "--gpu-ids", "0,1,2,3,4,5,6,7", "--datagen-workers", WORKERS, "--ground-reasoning",
    "--profile-reward", "0.15", "--data-root", "data/full14b",
    "--midtrain-out", "runs/full/midtrain", "--sft-out", "runs/full/sft",
    "--dpo-out", "runs/full/dpo", "--grpo-out", "runs/full/grpo", "--soup-out", "runs/full/soup",
]

STAGE_RE = re.compile(r"STAGE  campaign \(\w+\): stage (start|done): (\w+)")
ERR_RE = re.compile(
    r"Traceback \(most recent call last\)| ERROR |out of memory|HIP out of memory|"
    r"CUDA error|HIP error|OutOfMemoryError|CalledProcessError")
GATE_RE = re.compile(r"retention[^\n]*(FAIL|regress|below)|GATE[^\n]*FAIL|hard-stop|hard stop")


def reap_orphans() -> None:
    for pat in ("scripts/run_campaign.py", "from multiprocessing.spawn",
                "kore_supervise.py"):
        subprocess.run(["pkill", "-9", "-u", "shasriva", "-f", pat], capture_output=True)
    time.sleep(4)


def wait_for_pause() -> None:
    print("RESUME_SUP waiting for datagen->build pause sentinel...", flush=True)
    while not os.path.exists(SENTINEL):
        time.sleep(POLL_S)
    print(f"RESUME_SUP sentinel found ({open(SENTINEL).read().strip()}) -> resuming build..eval",
          flush=True)


def run_once(log_path: str):
    last_stage = None
    last_err = last_gate = 0
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
        errs, gates = len(ERR_RE.findall(txt)), len(GATE_RE.findall(txt))
        if cur is not None and cur != last_stage:
            print(f"ALERT STAGE {cur[1]}:{cur[0]}", flush=True)
            last_stage = cur
        if errs > last_err:
            print(f"ALERT ERROR_DETECTED total={errs}", flush=True)
            last_err = errs
        if gates > last_gate:
            print(f"ALERT RETENTION_GATE total={gates}", flush=True)
            last_gate = gates
        print(f"HEARTBEAT stage={cur[1] + ':' + cur[0] if cur else 'none'} errs={errs}",
              flush=True)
    try:
        txt = open(log_path, errors="ignore").read()
    except OSError:
        txt = ""
    return ("campaign complete" in txt), proc.returncode


def main() -> None:
    print("RESUME_SUP start", flush=True)
    wait_for_pause()
    for attempt in range(1, MAX_RETRIES + 1):
        reap_orphans()
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(LOGDIR, f"resume_paradigmv2_{ts}.log")
        try:
            open(LOGPATH_FILE, "w").write(log_path)
        except OSError:
            pass
        print(f"ALERT RESUME_LAUNCH attempt={attempt}/{MAX_RETRIES} "
              f"stages=build..eval log={os.path.basename(log_path)}", flush=True)
        done, rc = run_once(log_path)
        if done:
            print(f"ALERT RESUME_COMPLETE attempt={attempt} rc={rc}", flush=True)
            return
        print(f"ALERT RESUME_DIED attempt={attempt} rc={rc}; relaunch in {COOLDOWN_S}s",
              flush=True)
        time.sleep(COOLDOWN_S)
    print("ALERT RESUME_GIVEUP (max retries exhausted)", flush=True)


if __name__ == "__main__":
    main()
