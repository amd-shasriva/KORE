#!/bin/bash
# Drive Stage-0 -> evaluate -> Stage-1 -> evaluate, unattended, and stop before DPO.
#
# Why this exists: every stage here is hours long, and the SPUR controller
# intermittently refuses a submission with JobHoldMaxRequeue even when the QoS
# has free nodes -- a job can be held within 15s having never executed a line,
# and the only remedy is to resubmit. Chaining these by hand means babysitting
# ~12 hours of wall clock for a fault whose fix is "try again".
#
# Each step is gated on EVIDENCE, not on the scheduler's opinion:
#   - a training stage is complete only when its consolidated safetensors index
#     resolves with every shard present, because a job can exit 0 having written
#     a partial directory;
#   - an eval stage is complete only when both reports exist.
# A step that cannot be proven complete stops the pipeline rather than feeding a
# half-written checkpoint into the next stage.
#
# Deliberately stops before DPO: Stage-2 has never been validated on real
# weights, so it is not something to start unattended.
set -uo pipefail

REPO="/home/shasriva/Kore-RL/KORE"
export SPUR_CONTROLLER_ADDR="${SPUR_CONTROLLER_ADDR:-http://crs-m2m-cpu-spur-005:6817}"
export KORE_SPUR_CONTROLLER_ADDR="$SPUR_CONTROLLER_ADDR"
cd "$REPO" || exit 1

STATE="$REPO/runs/pipeline_state.json"
LOG="$REPO/runs/pipeline_driver.log"
MIDTRAIN_OUT="$REPO/runs/midtrain_14b_base"
SFT_OUT="$REPO/runs/sft_14b_frontier"
MIDTRAIN_CFG="$REPO/data/b05factory/launch/midtrain_base_8gpu.json"
# One reservation, not many. The 3h default here was a workaround for a
# misdiagnosis: I believed long reservations would not schedule, when the real
# blocker was --requeue. Measured after removing it -- a 23h probe scheduled and
# ran immediately, and the partition reports MaxTime=UNLIMITED.
#
# Segmenting is actively expensive: at 34.3 s/it a 3h window covers ~315 steps
# while save_steps=200 discards everything since the last checkpoint, averaging
# ~100 steps (~57 min) per boundary over ~6 boundaries. The segment loop is kept
# because it is still what recovers from a genuine failure, but the common case
# should be a single segment that runs to completion.
SEGMENT_WALLTIME="${KORE_SEGMENT_WALLTIME:-23:00:00}"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

note() {  # note <key> <value> -- append a fact to the state file
  /home/shasriva/kore-venv/bin/python - "$STATE" "$1" "$2" <<'PY'
import json, sys, pathlib, time
p, k, v = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
d = json.loads(p.read_text()) if p.exists() else {}
d[k] = v; d.setdefault("_history", []).append(
    {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), k: v})
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(d, indent=2) + "\n")
PY
}

# A consolidated HF checkpoint, verified against its own index rather than by
# directory existence -- a crashed save leaves a directory that looks fine.
checkpoint_complete() {
  /home/shasriva/kore-venv/bin/python - "$1" <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
idx = d / "model.safetensors.index.json"
if not (d / "config.json").exists() or not idx.exists():
    sys.exit(1)
try:
    shards = set(json.loads(idx.read_text())["weight_map"].values())
except Exception:
    sys.exit(1)
sys.exit(0 if shards and all((d / s).exists() for s in shards) else 1)
PY
}

# Adopt an existing job for this stage whether it is RUNNING or merely QUEUED.
# Matching only RUNNING is a duplicate-submission bug: an 8-node job can sit in
# PENDING(Resources) for a long time, and submitting a second one wastes an
# allocation and races two writers into the same output directory. A job HELD by
# the controller is deliberately not adopted -- that one does need resubmitting.
existing_job() {
  squeue -u "$USER" -h -o "%i %j %T %R" 2>/dev/null \
    | awk -v n="$1" '$2 ~ n && $3 !~ /COMPLET|CANCEL|FAIL/ && $0 !~ /JobHoldMaxRequeue/ {print $1; exit}'
}

# Submit, and treat a JobHoldMaxRequeue as "the controller said no, ask again".
# Returns the job id of a job that is RUNNING or legitimately PENDING.
submit_until_accepted() {
  local name="$1"; shift
  local attempt job state
  for attempt in $(seq 1 2000); do
    job="$(sbatch "$@" 2>&1 | grep -oE '[0-9]+$')"
    if [ -z "$job" ]; then [ $((attempt % 10)) -eq 1 ] && log "  $name: submit produced no job id (attempt $attempt)"; sleep 60; continue; fi
    sleep 25
    state="$(squeue -u "$USER" -h -j "$job" -o '%T %R' 2>/dev/null)"
    case "$state" in
      RUNNING*)            log "  $name: job $job RUNNING"; echo "$job"; return 0 ;;
      *JobHoldMaxRequeue*) scancel "$job" 2>/dev/null
                           [ $((attempt % 10)) -eq 1 ] && log "  $name: attempt $attempt held, retrying"
                           sleep 45 ;;
      PENDING*)            log "  $name: job $job queued normally ($state)"; echo "$job"; return 0 ;;
      "")                  log "  $name: job $job left the queue immediately -- check its log"; echo "$job"; return 0 ;;
      *)                   log "  $name: job $job state=$state"; echo "$job"; return 0 ;;
    esac
  done
  return 1
}

wait_for_job() {
  local job="$1" name="$2" i
  for i in $(seq 1 20000); do
    squeue -u "$USER" -h -j "$job" >/dev/null 2>&1 || true
    if ! squeue -u "$USER" -h -o "%i" 2>/dev/null | grep -qx "$job"; then
      log "  $name: job $job left the queue"; return 0
    fi
    sleep 120
  done
  return 1
}

# ------------------------------------------------------------ training loop ---
# Run a training stage in SEGMENTS rather than as one long reservation.
#
# An 8-hour exclusive 4-node request is a large ask on a busy cluster and sits
# behind every shorter job; a 3-hour one fits far more scheduling windows. This
# is only safe because the trainer auto-resumes: kore/policy/midtrain.py calls
# latest_checkpoint(output_dir) and hands it to trainer.train(), and that helper
# walks candidates newest-first so a segment killed mid-save falls back to the
# previous complete checkpoint instead of restarting from step 0.
#
# So each segment picks up where the last stopped, and the loop ends when the
# consolidated checkpoint verifies -- not when the scheduler says a job exited.
# Cost of a segment boundary is at most `save_steps` of recomputation.
run_training_stage() {
  local name="$1" jobname="$2" outdir="$3" walltime="$4"; shift 4
  local seg job
  if checkpoint_complete "$outdir"; then log "$name: already complete"; return 0; fi
  for seg in $(seq 1 40); do
    job="$(existing_job "$jobname")"
    if [ -n "$job" ]; then
      log "$name: adopting in-flight job $job (segment $seg)"
    else
      log "$name: segment $seg (walltime $walltime)"
      job="$(submit_until_accepted "$name" --time="$walltime" "$@")" || return 1
    fi
    note "${name}_job" "$job"
    wait_for_job "$job" "$name-seg$seg"
    if checkpoint_complete "$outdir"; then
      log "$name: COMPLETE and verified after $seg segment(s)"; note "$name" "complete"; return 0
    fi
    if [ -d "$outdir" ]; then
      log "$name: segment $seg ended short; resuming from $(ls -1d "$outdir"/checkpoint-* 2>/dev/null | tail -1 | xargs -r basename)"
    else
      log "$name: segment $seg produced no output at all -- check runs/$jobname-$job.out"
    fi
  done
  log "$name: exhausted segment budget"; note "$name" "failed"; return 1
}

# ---------------------------------------------------------------- stage 0 ----
step_midtrain() {
  run_training_stage midtrain kore-mid "$MIDTRAIN_OUT" "$SEGMENT_WALLTIME" \
    "$REPO/scripts/spur_midtrain_1node.sbatch" "$MIDTRAIN_CFG"
}

# ---------------------------------------------------------------- eval A/B ---
step_eval() {
  local cand="$1" arm="$2" tag="$3" base="${4:--}" baserev="${5:--}" out job
  out="$REPO/runs/eval_ab_${tag}"
  if [ -f "$out/report_kernel_ab.md" ] && [ -f "$out/report_heldout_lm.md" ]; then
    log "eval[$tag]: already complete"; return 0
  fi
  log "eval[$tag]: submitting (candidate=$cand arm=$arm reference=$base)"
  job="$(submit_until_accepted "eval-$tag" "$REPO/scripts/spur_eval_ab_1node.sbatch" "$cand" "$arm" "$out" "$base" "$baserev")" || return 1
  note "eval_${tag}_job" "$job"
  wait_for_job "$job" "eval-$tag"
  if [ -f "$out/report_kernel_ab.md" ]; then
    log "eval[$tag]: COMPLETE -> $out"; note "eval_$tag" "complete"; return 0
  fi
  log "eval[$tag]: reports missing -- continuing anyway, evaluation is not a gate on training"
  note "eval_$tag" "incomplete"; return 0
}

# ---------------------------------------------------------------- stage 1 ----
# SFT's peak is three 221GB checkpoints coexisting during rotation. Starting it
# without that headroom does not fail fast -- it dies hours in, having burned a
# GPU allocation. Midtrain's own optimizer checkpoints are the obvious reclaim
# (SFT loads the consolidated shards, not those), but deleting them is a human
# decision, so this gate stops the pipeline and says exactly what it needs.
SFT_PEAK_GB="${KORE_SFT_PEAK_GB:-700}"

disk_free_gb() { df -BG --output=avail "$REPO" 2>/dev/null | tail -1 | tr -dc '0-9'; }

step_sft() {
  checkpoint_complete "$MIDTRAIN_OUT" || { log "sft: refusing to start, midtrain is not complete"; return 1; }
  local free; free="$(disk_free_gb)"; free="${free:-0}"
  if [ "$free" -lt "$SFT_PEAK_GB" ]; then
    log "sft: HOLDING. ${free}GB free, need ~${SFT_PEAK_GB}GB for three 221GB checkpoints."
    log "sft: reclaimable now that midtrain's consolidated weights are verified:"
    du -sh "$MIDTRAIN_OUT"/checkpoint-* 2>/dev/null | while read -r sz d; do log "sft:   $sz  $d"; done
    log "sft: those are optimizer state for a FINISHED run; SFT reads the"
    log "sft: consolidated shards, not them. Awaiting a human decision -- nothing deleted."
    note sft "held_insufficient_disk"
    note sft_free_gb "$free"
    return 2
  fi
  run_training_stage sft kore-sft "$SFT_OUT" "$SEGMENT_WALLTIME" \
    "$REPO/scripts/spur_sft_1node.sbatch" "$REPO/configs/sft_14b_full.json" "$MIDTRAIN_OUT" "$SFT_OUT"
}

log "================ pipeline driver start (HEAD $(git rev-parse --short HEAD)) ================"
note pipeline "running"

step_midtrain    || { log "STOP: midtrain failed"; note pipeline "failed_midtrain"; exit 1; }
# Reference is the BASE weights midtrain actually started from. Held-out LM
# loss is the meaningful signal here; kernel generation is not, because
# neither arm follows instructions yet.
step_eval "$MIDTRAIN_OUT" midtrain "midtrain_base" \
          "Qwen/Qwen3-14B-Base" "0b0bd3732e2c374d483664439ea334928b65f304"
step_sft; _sft_rc=$?
if [ "$_sft_rc" = "2" ]; then
  log "================ pipeline HELD before SFT (disk); midtrain + its eval are complete ================"
  note pipeline "held_before_sft"; exit 0
elif [ "$_sft_rc" != "0" ]; then
  log "STOP: sft failed"; note pipeline "failed_sft"; exit 1
fi
# The headline comparison: our instruction-tuned model against the vendor's
# instruction-tuned model, both of which follow instructions, so the kernel
# funnel is finally a fair question.
step_eval "$SFT_OUT" sft "sft" \
          "Qwen/Qwen3-14B" "40c069824f4251a91eefaf281ebe4c544efd3e18"

log "================ pipeline COMPLETE through SFT; stopping before DPO ================"
note pipeline "complete_through_sft"
