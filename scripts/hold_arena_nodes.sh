#!/bin/bash
# Hold the nodes the arena arms are already on, so a rollover does not lose them.
#
# The arena cannot finish inside one 8h allocation -- 413 tasks at up to 3600s a
# gate -- so it crosses several, resuming from its ledger each time. On a cluster
# reporting zero idle nodes, the seconds between one job exiting and the
# supervisor submitting its successor are enough for another tenant to take the
# machine, and the sweep then waits hours for a node instead of minutes. That is
# the single largest source of wall-clock loss in a long arena run.
#
# This reserves ONLY nodes our own arena jobs currently occupy. It never takes a
# node from anyone: if we are not running on it, it is not eligible here. The
# reservation is what supervise.sh's res_arg() then submits against, so the same
# node comes back to us after each wall-clock rollover.
#
# Safe to run repeatedly -- it recreates the hold from wherever the arenas are
# now, which is also how you renew it before the 24h lapses. If the hold is
# absent res_arg() returns nothing and submissions go back to ordinary
# scheduling, so a lapse costs throughput but never blocks the queue.
#
#   scripts/hold_arena_nodes.sh            # 24h hold on the current arena nodes
#   KORE_HOLD_MINUTES=480 scripts/hold_arena_nodes.sh
set -uo pipefail

REPO=/home/shasriva/Kore-RL/KORE
LOG="$REPO/runs/hold_arena_nodes.log"
NAME="${KORE_RESERVATION:-kore_hold}"
MINUTES="${KORE_HOLD_MINUTES:-1440}"
KORE_USER="${USER:-${LOGNAME:-$(id -un)}}"

[ -z "${SPUR_CONTROLLER_ADDR:-}" ] && [ -r /etc/profile.d/spur.sh ] && . /etc/profile.d/spur.sh
say() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$LOG"; }

# Only the arena arms. Mining is deliberately excluded: a reserved node is
# unusable by jobs that do not request the reservation, and mining should stay
# free to land on whatever the scheduler has, wherever that is.
nodes="$(squeue -u "$KORE_USER" -h -t R -o "%j %N" 2>/dev/null |
         awk '$1 ~ /^kore-aka/ && $2 != "" {print $2}' |
         sort -u | tr '\n' ',' | sed 's/,$//')"

if [ -z "$nodes" ]; then
    say "no arena job is running, so there is no node to hold; leaving any existing $NAME alone"
    exit 0
fi

say "holding $nodes for ${MINUTES}m as $NAME"

# Recreate rather than extend: the arms move between nodes across rollovers, and
# a hold on a node we no longer occupy would fence off someone else's machine.
if scontrol show reservation 2>/dev/null | grep -q "ReservationName=${NAME}$"; then
    scontrol delete-reservation "$NAME" >>"$LOG" 2>&1 &&
        say "removed the previous $NAME"
fi

# ignore_jobs because our own arena job is already running on these nodes; the
# reservation is meant to cover them, not to evict what is there.
if scontrol create-reservation --name "$NAME" --duration "$MINUTES" \
        --nodes "$nodes" --users "$KORE_USER" --flags ignore_jobs >>"$LOG" 2>&1; then
    say "created $NAME"
    scontrol show reservation 2>/dev/null |
        grep -A3 "ReservationName=${NAME}$" | tee -a "$LOG"
else
    say "FAILED to create $NAME; submissions fall back to ordinary scheduling"
    exit 1
fi
