#!/bin/bash
# Drive the DATA phase to completion, unattended, and stop before training.
#
# The product model is trained once, on whatever mixture exists at that moment,
# so the mixture is the thing worth getting right. This driver owns that phase
# and deliberately does not launch SFT: runs/DATA_NOT_FINAL stays up until a
# human has seen the final counts.
#
# Order is forced by a dependency, not by preference. Task diversity is the
# ceiling on how much NON-REDUNDANT trajectory data can exist -- generating many
# episodes over few tasks yields near-duplicates -- so the task pool has to land
# before datagen is worth saturating.
#
# Capacity is the other constraint and it is not the 26 idle nodes on the
# cluster: the QoS caps us at MAX_NODES concurrent, so this backfills datagen
# into whatever slots the measurement jobs are not using, and never preempts
# them.
set -uo pipefail

REPO="/home/shasriva/Kore-RL/KORE"
export SPUR_CONTROLLER_ADDR="${SPUR_CONTROLLER_ADDR:-http://crs-m2m-cpu-spur-005:6817}"
export KORE_SPUR_CONTROLLER_ADDR="$SPUR_CONTROLLER_ADDR"
cd "$REPO" || exit 1

LOG="$REPO/runs/data_driver.log"
STATE="$REPO/runs/data_state.json"
MAX_NODES="${KORE_MAX_NODES:-6}"          # QoS group node limit
TARGET_EPISODES="${KORE_TARGET_EPISODES:-50000}"
MT_DIR="$REPO/data/b05factory/agentic_mt"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

# Distinguish "the controller answered" from "it did not". On failure squeue
# prints nothing, and every caller below would read that as "no jobs running"
# and submit duplicates on top of live work.
squeue_snapshot() {
  local out rc
  out="$(squeue -u "$USER" -h -o "$1" 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] || printf '%s' "$out" \
       | grep -qiE 'failed to connect|transport error|connection refused|^error:'; then
    return 2
  fi
  printf '%s' "$out"
}

nodes_in_use() {
  local snap
  snap="$(squeue_snapshot "%T %D")" || return 2
  printf '%s\n' "$snap" | awk '$1=="RUNNING"{n+=$2} END{print n+0}'
}

running_named() {
  local snap
  snap="$(squeue_snapshot "%j %T")" || return 2
  printf '%s\n' "$snap" | awk -v n="$1" '$1 ~ n && $2=="RUNNING"' | wc -l
}

# Clear shards the controller wedged in JobHoldMaxRequeue. These occupy a queue
# slot while never running, so left alone they starve the campaign.
clear_held_shards() {
  local snap j
  snap="$(squeue_snapshot "%i %j %R")" || return 0
  for j in $(printf '%s\n' "$snap" | awk '/kore-agentic/ && /JobHoldMaxRequeue/ {print $1}'); do
    log "datagen: cancelling held shard $j"
    scancel "$j" 2>/dev/null
  done
}

episodes_so_far() {
  # Count trajectory records actually on disk, which is the only number that
  # matters -- job logs report attempts, not what survived filtering.
  find "$MT_DIR" -name '*.jsonl' -type f 2>/dev/null \
    | xargs cat 2>/dev/null | wc -l
}

log "================ data driver start (HEAD $(git rev-parse --short HEAD)) ================"
log "capacity: ${MAX_NODES} nodes (QoS group limit)   target: ${TARGET_EPISODES} episodes"

# ---------------------------------------------------------------- stage 1 ----
# Wait for the task pool. Datagen submitted before this lands would sample the
# old ~1k tasks and produce exactly the redundancy the expansion exists to fix.
for i in $(seq 1 480); do
  n="$(running_named kore-taskpool)"; rc=$?
  if [ "$rc" = "2" ]; then log "taskpool: controller unreachable, holding"; sleep 120; continue; fi
  [ "$n" = "0" ] && break
  [ $(((i - 1) % 15)) -eq 0 ] && log "taskpool: still building"
  sleep 120
done
log "taskpool: finished (or absent)"

# ---------------------------------------------------------------- stage 2 ----
# Backfill datagen into unused capacity until the episode target is met.
for i in $(seq 1 2000); do
  have="$(episodes_so_far)"
  if [ "$have" -ge "$TARGET_EPISODES" ]; then
    log "datagen: TARGET MET -- ${have} episodes on disk"
    break
  fi
  used="$(nodes_in_use)"; rc=$?
  if [ "$rc" = "2" ]; then log "datagen: controller unreachable, holding"; sleep 120; continue; fi
  clear_held_shards
  free_slots=$(( MAX_NODES - used ))
  if [ "$free_slots" -gt 0 ]; then
    log "datagen: ${have}/${TARGET_EPISODES} episodes, ${used}/${MAX_NODES} nodes busy; filling ${free_slots}"
    # Reuse the campaign's own partitioner rather than submitting raw shards:
    # it derives the task split and the per-shard manifest that the sbatch
    # preflight checks, so hand-rolled submissions would not line up.
    # WORKERS=32 is where the measured throughput curve flattened
    # (66 -> 164 -> ~200+ episodes/hour as workers went 8 -> 16 -> 32).
    if bash scripts/spur_submit_agentic.sh "$free_slots" \
         "${KORE_EPISODES_PER_TASK:-6}" "${KORE_WORKERS:-32}" >>"$LOG" 2>&1; then
      log "datagen: submitted ${free_slots} shard(s)"
    else
      log "datagen: partitioner returned non-zero; will retry next cycle"
    fi
  else
    [ $(((i - 1) % 10)) -eq 0 ] && log "datagen: ${have}/${TARGET_EPISODES} episodes, all ${used} nodes busy"
  fi
  sleep 180
done

log "================ data phase complete: $(episodes_so_far) episodes ================"
log "runs/DATA_NOT_FINAL is intentionally still up. Review the counts, build v3,"
log "then remove it to release training."
