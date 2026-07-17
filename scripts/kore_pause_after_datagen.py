#!/usr/bin/env python3
"""Pause the running fold-in campaign cleanly at the datagen->build boundary.

The paradigm-v2 build must land BEFORE the build/midtrain/sft/dpo/grpo stages run,
and the training stages shell out to fresh processes that re-import the (about to
change) code. So we let datagen fully finish, then stop the SUPERVISOR first (so it
cannot relaunch into build) and the CAMPAIGN second. Datagen shards are on disk, so
nothing is lost; build is re-run fresh from the new code on the manual resume.

Trigger = (all 282 group shards present) OR (the campaign log shows 'stage start:
build'). Writes /tmp/kore_paused_after_datagen on success so the operator/agent knows
the window is open. Only shasriva-owned campaign/supervisor procs are touched.
"""
from __future__ import annotations

import glob
import os
import subprocess
import time

REPO = "/home/shasriva/Kore-RL/KORE"
GROUPS_DIR = os.path.join(REPO, "data/full14b/groups")
LOGPATH_FILE = "/tmp/kore_foldin_logpath.txt"
SENTINEL = "/tmp/kore_paused_after_datagen"
POLL_S = 30
TARGET_GROUPS = 282


def _n_groups() -> int:
    fs = [f for f in glob.glob(os.path.join(GROUPS_DIR, "*.jsonl"))
          if not os.path.basename(f).startswith("_") and os.path.getsize(f) > 0]
    return len(fs)


def _build_started() -> bool:
    try:
        log = open(LOGPATH_FILE).read().strip()
        txt = open(log, errors="ignore").read()
    except OSError:
        return False
    return "stage start: build" in txt


def main() -> None:
    print("PAUSE_GUARD start (waiting for datagen->build boundary)", flush=True)
    while True:
        n = _n_groups()
        built = _build_started()
        print(f"PAUSE_GUARD groups={n}/{TARGET_GROUPS} build_started={built}", flush=True)
        if n >= TARGET_GROUPS or built:
            print("PAUSE_GUARD TRIGGER: datagen complete -> pausing campaign", flush=True)
            # 1) stop the supervisor FIRST so it cannot relaunch into build.
            subprocess.run(["pkill", "-9", "-u", "shasriva", "-f", "kore_supervise.py"],
                           capture_output=True)
            time.sleep(3)
            # 2) stop the campaign + any datagen spawn workers.
            for pat in ("scripts/run_campaign.py", "from multiprocessing.spawn"):
                subprocess.run(["pkill", "-9", "-u", "shasriva", "-f", pat],
                               capture_output=True)
            time.sleep(2)
            open(SENTINEL, "w").write(f"paused at groups={n} build_started={built}\n")
            print("PAUSE_GUARD ALERT PAUSED_AFTER_DATAGEN (window open for paradigm-v2)",
                  flush=True)
            return
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
