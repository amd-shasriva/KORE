#!/usr/bin/env python3
"""Campaign stage/health monitor for the KORE fold-in run.

Polls the live campaign log + process every POLL_S seconds and emits sparse ALERT
lines on the events that matter, so a tail -F of this monitor's output (with
notify-on-output on "ALERT ") pings the operator on:
  - stage transitions (start/done of datagen/build/midtrain/sft/dpo/grpo/soup/eval)
  - real errors (Traceback / ERROR level / OOM / CUDA|HIP error)
  - retention-gate failures / hard-stops
  - a rate-limit (429) STORM (a big jump, not the odd retry)
  - campaign completion or death

HEARTBEAT lines (every poll) carry pid/stage/error-count for reference and are NOT
alerts. The monitor self-terminates on completion or a confirmed death. It reads the
current log path from LOGPATH_FILE (updated on each (re)launch) so it follows restarts.
"""
from __future__ import annotations

import glob
import os
import re
import subprocess
import time

REPO = "/home/shasriva/Kore-RL/KORE"
LOGPATH_FILE = "/tmp/kore_foldin_logpath.txt"
POLL_S = int(os.environ.get("KORE_MONITOR_POLL_S", "180"))
R429_STORM_STEP = 40  # only alert on a big jump in retries, not the occasional one

STAGE_RE = re.compile(r"STAGE  campaign \(\w+\): stage (start|done): (\w+)")
ERR_RE = re.compile(
    r"Traceback \(most recent call last\)| ERROR |OutOfMemoryError|out of memory|"
    r"HIP out of memory|CUDA error|HIP error|torch\.cuda\.OutOfMemory|CalledProcessError"
)
GATE_RE = re.compile(r"retention[^\n]*(FAIL|regress|below)|GATE[^\n]*FAIL|hard-stop|hard stop")
R429_RE = re.compile(r"429|rate.?limit|RateLimit|overloaded|retries=[1-9]")


def newest_log():
    try:
        p = open(LOGPATH_FILE).read().strip()
        if p and os.path.exists(p):
            return p
    except OSError:
        pass
    cands = sorted(glob.glob(os.path.join(REPO, "runs/full/logs/campaign_foldin_*.log")))
    return cands[-1] if cands else None


def campaign_pid():
    try:
        out = subprocess.run(["pgrep", "-f", "run_campaign.py --model"],
                             capture_output=True, text=True)
        pids = [int(x) for x in out.stdout.split() if x.strip()]
        return pids[0] if pids else None
    except Exception:  # noqa: BLE001
        return None


def main():
    last_stage = None
    last_err = 0
    last_gate = 0
    last_429 = 0
    dead = 0
    n = 0
    print("MONITOR start", flush=True)
    while True:
        log = newest_log()
        pid = campaign_pid()
        txt = ""
        if log and os.path.exists(log):
            try:
                txt = open(log, errors="ignore").read()
            except OSError:
                txt = ""

        stages = STAGE_RE.findall(txt)
        cur = stages[-1] if stages else None
        errs = len(ERR_RE.findall(txt))
        gates = len(GATE_RE.findall(txt))
        r429 = len(R429_RE.findall(txt))

        if cur is not None and cur != last_stage:
            print(f"ALERT STAGE {cur[1]}:{cur[0]}", flush=True)
            last_stage = cur
        if errs > last_err:
            print(f"ALERT ERROR_DETECTED total={errs} (log={os.path.basename(log or '?')})",
                  flush=True)
            last_err = errs
        if gates > last_gate:
            print(f"ALERT RETENTION_GATE total={gates}", flush=True)
            last_gate = gates
        if r429 >= last_429 + R429_STORM_STEP:
            print(f"ALERT RATE_LIMIT_STORM total={r429}", flush=True)
            last_429 = r429

        if "campaign complete" in txt:
            print("ALERT CAMPAIGN_COMPLETE", flush=True)
            break
        if pid is None:
            dead += 1
            if dead >= 2:
                print("ALERT CAMPAIGN_DIED (pid gone 2 consecutive polls)", flush=True)
                break
        else:
            dead = 0

        stage_str = f"{cur[1]}:{cur[0]}" if cur else "none"
        print(f"HEARTBEAT n={n} pid={pid} stage={stage_str} errs={errs} r429={r429}",
              flush=True)
        n += 1
        time.sleep(POLL_S)
    print("MONITOR exit", flush=True)


if __name__ == "__main__":
    main()
