#!/bin/bash
# Keep exactly one verified FlyDSL seeding job alive until the list is seeded.
#
# Written because spurctld went down for half an hour with the fix ready and
# nothing able to submit it. The scavenger keeps the reservation warm across an
# outage but only ever staffs miners, so without this the one dialect missing
# from the mixture waits for a human.
#
# It supervises rather than submits once: the submission that prompted this was
# accepted seconds before the controller dropped again, and a job the controller
# forgets looks exactly like a job that finished. Checking for the job instead of
# remembering that we sent one covers both.
set -uo pipefail

REPO="${KORE_REPO:-/home/shasriva/Kore-RL/KORE}"
PY="${KORE_PY:-/home/shasriva/kore-venv/bin/python}"
LOG="$REPO/runs/flyseed_watch.log"
TASK_LIST="${1:-$REPO/runs/flydsl_retry.txt}"
ATTEMPTS="${2:-3}"
OUT_ROOT="$REPO/data/registry_flydsl_frontier"
CAP="${GPU_JOB_CAP:-8}"
POLL="${POLL:-30}"

[ -z "${SPUR_CONTROLLER_ADDR:-}" ] && [ -r /etc/profile.d/spur.sh ] && . /etc/profile.d/spur.sh
# mine_res_arg: a job without --reservation cannot use the nodes the scavenger
# is holding, however idle they are. Submitted without it, this job sat on
# "(Resources)" while the scavenger logged a free held node on the same second.
# shellcheck source=/dev/null
. "$REPO/scripts/gpu_slots.sh" 2>/dev/null || true

say() { echo "[$(date -u '+%H:%M:%S')] $*" >> "$LOG"; }

cd "$REPO" || exit 1
say "flyseed supervisor up: list=$TASK_LIST attempts=$ATTEMPTS cap=$CAP"

# Every task on the list has either passed or been given its attempts. Asking the
# ledger rather than counting submissions means a job that died mid-list still
# leaves the remainder visible, and a finished list stops the loop for good.
work_remains() {
    "$PY" - "$TASK_LIST" "$OUT_ROOT/verified_seed_attempts.jsonl" <<'PY'
import json, sys
from pathlib import Path
ids = [l.split("#", 1)[0].strip() for l in Path(sys.argv[1]).read_text().splitlines()]
ids = {i for i in ids if i}
seen = set()
p = Path(sys.argv[2])
if p.is_file():
    for line in p.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("task_id"):
            seen.add(row["task_id"])
print(len(ids - seen))
PY
}

while :; do
    if ! q=$(squeue -u "$USER" -h -o '%i %j %T' 2>/dev/null); then
        sleep "$POLL"; continue
    fi

    if grep -q 'kore-flyseed' <<<"$q"; then
        sleep "$POLL"; continue
    fi

    left=$(work_remains 2>/dev/null || echo 0)
    if ! [[ "$left" =~ ^[0-9]+$ ]] || [ "$left" -eq 0 ]; then
        say "no unattempted tasks left on the list; supervisor exiting"
        exit 0
    fi

    n=$(grep -c . <<<"$q")
    if [ "$n" -ge "$CAP" ]; then
        sleep "$POLL"; continue
    fi

    resv="$(mine_res_arg 2>/dev/null || true)"
    say "no flyseed job and $left task(s) left, $n/$CAP held -> submitting ${resv:-(no free held node)}"
    # shellcheck disable=SC2086
    out=$(sbatch $resv --job-name=kore-flyseed scripts/spur_verified_flydsl.sbatch \
          "$TASK_LIST" "$OUT_ROOT" "$ATTEMPTS" 8 2>&1)
    say "sbatch rc=$?: $out"
    sleep 60
done
