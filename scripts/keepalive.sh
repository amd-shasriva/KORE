#!/bin/bash
# Restart a long-running unattended command whenever it stops.
#
#   scripts/keepalive.sh NAME -- command args...
#
# The seeding sweeps and the pipeline loops are hours-to-days of work on a login
# node, and they stop for reasons that have nothing to do with their own logic:
# the node reboots, or a process dies mid-sweep leaving no traceback. Both
# happened here -- a functionalized seeding run stopped after five minutes with a
# clean log, and a later reboot took out all four background processes at once,
# after which 745 gated tasks sat unharvested for six hours because nothing was
# left alive to promote them.
#
# Every stage is idempotent and ledgered, so the recovery for all of these is the
# same: start it again. This wrapper does that, records why it restarted, and
# backs off if the command is failing instantly rather than spinning on it.
set -uo pipefail

NAME="${1:?usage: keepalive.sh NAME -- cmd...}"; shift
[ "${1:-}" = "--" ] && shift

REPO=/home/shasriva/Kore-RL/KORE
LOG="$REPO/runs/keepalive_$NAME.log"
MIN_RUN="${MIN_RUN:-120}"   # a run shorter than this counts as a fast failure

say() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" >> "$LOG"; }

say "=== keepalive $NAME starting (pid $$): $* ==="
fails=0
while :; do
    t0=$(date +%s)
    "$@" >> "$REPO/runs/${NAME}.log" 2>&1
    rc=$?
    dur=$(( $(date +%s) - t0 ))

    # A command that exits 0 after real work has finished its sweep; restarting
    # it would just re-scan a ledger that says everything is done.
    if [ "$rc" -eq 0 ] && [ "$dur" -ge "$MIN_RUN" ]; then
        say "$NAME finished cleanly after ${dur}s; not restarting"
        exit 0
    fi

    if [ "$dur" -lt "$MIN_RUN" ]; then
        fails=$(( fails + 1 ))
    else
        fails=0   # it ran a while, so this is a fresh problem, not a loop
    fi

    # Back off only on repeated fast failures. A crash hours into a sweep should
    # be retried at once; a command that cannot start should not be retried in a
    # tight loop.
    delay=$(( fails > 4 ? 600 : fails * 30 ))
    [ "$delay" -lt 10 ] && delay=10
    say "$NAME exited rc=$rc after ${dur}s (fast failures: $fails); restarting in ${delay}s"
    sleep "$delay"
done
