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
CHATVEC_OUT="$REPO/runs/midtrain_14b_chatvec"
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

# The model SFT is compared against. In direct mode this must be the very
# checkpoint SFT started from, or the A/B measures the vendor's model quality
# instead of our training. Defaults to the production target.
SFT_REFERENCE_MODEL="${KORE_SFT_REFERENCE:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
SFT_REFERENCE_REV="${KORE_SFT_REFERENCE_REV:--}"

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

# Read the queue, distinguishing "the controller answered and the job is not
# there" from "the controller did not answer". Collapsing those is dangerous:
# spurctld does refuse connections intermittently (observed: "failed to connect
# to spurctld ... Connection refused"), and on failure squeue prints nothing,
# which every caller below would otherwise read as a job that has finished.
# Prints the listing and returns 0 on success; returns 2 if unreachable.
squeue_snapshot() {
  local out rc
  out="$(squeue -u "$USER" -h -o "$1" 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] || printf '%s' "$out" \
       | grep -qiE 'failed to connect|transport error|connection refused|^error:'; then
    return 2
  fi
  printf '%s' "$out"
}

# Adopt an existing job for this stage whether it is RUNNING or merely QUEUED.
# Matching only RUNNING is a duplicate-submission bug: an 8-node job can sit in
# PENDING(Resources) for a long time, and submitting a second one wastes an
# allocation and races two writers into the same output directory. A job HELD by
# the controller is deliberately not adopted -- that one does need resubmitting.
# Returns 2 if the controller is unreachable, so callers do not mistake silence
# for "no job is running" and submit a duplicate on top of a live one.
existing_job() {
  local snap
  snap="$(squeue_snapshot "%i %j %T %R")" || return 2
  printf '%s\n' "$snap" \
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

# Block until the job genuinely leaves the queue. A controller outage must not
# be read as completion, and a single clean read that omits the job is not quite
# enough either, so require two consecutive good reads before believing it.
wait_for_job() {
  local job="$1" name="$2" i snap misses=0 fails=0
  for i in $(seq 1 20000); do
    if snap="$(squeue_snapshot "%i")"; then
      fails=0
      if printf '%s\n' "$snap" | grep -qx "$job"; then
        misses=0
      else
        misses=$((misses + 1))
        if [ "$misses" -ge 2 ]; then log "  $name: job $job left the queue"; return 0; fi
      fi
    else
      fails=$((fails + 1)); misses=0
      [ $(((fails - 1) % 15)) -eq 0 ] && log "  $name: controller unreachable (${fails}x); holding position, not assuming the job ended"
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
  local seg job rc
  if checkpoint_complete "$outdir"; then log "$name: already complete"; return 0; fi
  for seg in $(seq 1 40); do
    # Never submit while the controller is unreachable. existing_job cannot see
    # an in-flight job then, so a blind submit would put a second 8-GPU writer
    # into the same output directory as the live one.
    while :; do
      job="$(existing_job "$jobname")"; rc=$?
      [ "$rc" -ne 2 ] && break
      log "$name: controller unreachable; waiting rather than risking a duplicate submission"
      sleep 120
    done
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
# ------------------------------------------------- stage 0.5: chat vector ----
# Midtrain holds the Triton domain knowledge but cannot follow an instruction,
# because continued pretraining had to run on the BASE model -- running it on
# the instruct model destroyed instruction-following outright.
#
# Rather than spend the SFT budget teaching chat back from scratch (Tulu 3
# needed 939k samples to reach that from a Llama base; we have 56k), transplant
# the vendor's own post-training as a delta:
#
#   theta_chatvec = theta_midtrain + (theta_instruct - theta_base)
#
# This is the general-SFT stage of AceMath's general-then-domain recipe, which
# spent ~3.9M samples on it; here it costs a tensor subtraction.
#
# Verification runs FIRST and has to pass, because a sign or dtype error still
# produces a model that loads and generates plausibly -- nothing downstream
# would catch it. Failure here is NOT fatal to the pipeline: SFT falls back to
# midtrain, which is the previously planned path.
step_residual() {
  if checkpoint_complete "$CHATVEC_OUT"; then
    log "residual: already built and verified"; return 0
  fi
  checkpoint_complete "$MIDTRAIN_OUT" || { log "residual: midtrain not complete, skipping"; return 1; }
  local job

  log "residual: verifying the arithmetic against the real 14B checkpoints"
  job="$(submit_until_accepted residual-verify "$REPO/scripts/spur_residual_1node.sbatch" verify)" || return 1
  note residual_verify_job "$job"
  wait_for_job "$job" residual-verify
  if ! grep -q "VERDICT: PASS" "$REPO/runs/residual-$job.out" 2>/dev/null; then
    log "residual: verification did NOT pass -- see runs/residual-$job.out"
    note residual "verify_failed"; return 1
  fi
  log "residual: verified -- base + delta reproduces instruct exactly"

  log "residual: building $CHATVEC_OUT"
  job="$(submit_until_accepted residual-build "$REPO/scripts/spur_residual_1node.sbatch" \
         build "$MIDTRAIN_OUT" "$CHATVEC_OUT")" || return 1
  note residual_build_job "$job"
  wait_for_job "$job" residual-build
  if checkpoint_complete "$CHATVEC_OUT"; then
    log "residual: COMPLETE and verified"; note residual "complete"; return 0
  fi
  log "residual: build produced no verifiable checkpoint"; note residual "build_failed"; return 1
}

# Reclaim a finished stage's optimizer state, and ONLY that.
#
# A checkpoint-N directory holds Adam moments, RNG and scheduler state. It
# exists to resume the run that wrote it. Once that run has emitted consolidated
# model-*.safetensors, the next stage loads those and never reads checkpoint-N
# again -- so for a FINISHED stage it is pure dead weight, at 221GB apiece.
#
# Deleting weights would be unrecoverable without a ~20h rerun, so every
# precondition below must hold, and the function refuses rather than guesses:
#   1. the consolidated index resolves and EVERY shard it names exists,
#   2. no shard is zero-length (a truncated write passes an existence check),
#   3. the thing being removed matches checkpoint-* under that exact directory.
# It also does nothing at all unless space is actually short: keeping resumable
# state costs nothing when there is room for it.
reclaim_optimizer_state() {
  local outdir="$1" need="$2" free before after
  free="$(disk_free_gb)"; free="${free:-0}"
  if [ "$free" -ge "$need" ]; then
    log "reclaim: ${free}GB free >= ${need}GB needed; keeping resumable state in $(basename "$outdir")"
    return 0
  fi
  if ! checkpoint_complete "$outdir"; then
    log "reclaim: REFUSING -- $(basename "$outdir") has no verified consolidated checkpoint"
    return 1
  fi
  # A shard can exist and still be truncated, which an existence check passes.
  # Size must be given in BYTES: `-size -1M` rounds up to whole units, so a 1KB
  # file counts as 1 and never matches "less than 1M" -- a unit test caught this
  # guard silently passing a deliberately truncated shard. Real shards here are
  # ~4.6GB, so anything under 100MiB is corrupt rather than merely small.
  if find "$outdir" -maxdepth 1 -name 'model-*.safetensors' -size -104857600c 2>/dev/null | grep -q .; then
    log "reclaim: REFUSING -- a consolidated shard in $(basename "$outdir") is under 100MiB (truncated?)"
    find "$outdir" -maxdepth 1 -name 'model-*.safetensors' -size -104857600c -printf '  reclaim:   %s bytes  %p\n' 2>/dev/null
    return 1
  fi
  local n; n=$(ls -1d "$outdir"/checkpoint-* 2>/dev/null | wc -l)
  [ "$n" -eq 0 ] && { log "reclaim: nothing to reclaim in $(basename "$outdir")"; return 0; }
  before="$free"
  log "reclaim: ${free}GB free < ${need}GB needed; consolidated weights verified, removing $n optimizer checkpoint(s):"
  for ck in "$outdir"/checkpoint-*; do
    [ -d "$ck" ] || continue
    log "reclaim:   $(du -sh "$ck" 2>/dev/null | cut -f1)  $ck"
    rm -rf "$ck"
  done
  sleep 10
  after="$(disk_free_gb)"
  log "reclaim: ${before}GB -> ${after}GB free"
  checkpoint_complete "$outdir" \
    && log "reclaim: consolidated weights re-verified intact after reclaim" \
    || log "reclaim: WARNING consolidated weights no longer verify -- investigate before SFT"
  note reclaimed_gb "$(( after - before ))"
}

# SFT's peak is three 221GB checkpoints coexisting during rotation. Starting it
# without that headroom does not fail fast -- it dies hours in, having burned a
# GPU allocation. reclaim_optimizer_state above frees midtrain's own checkpoints
# to make that headroom automatically; this gate is the backstop for when even
# that is not enough, or when reclaim refused because something looked wrong.
SFT_PEAK_GB="${KORE_SFT_PEAK_GB:-700}"

disk_free_gb() { df -BG --output=avail "$REPO" 2>/dev/null | tail -1 | tr -dc '0-9'; }

step_sft() {
  # Only the cpt recipe has a midtrain to wait on. In direct mode the base is
  # the vendor's instruct checkpoint named by the config, which the launcher's
  # resolver validates, so there is nothing local to verify here.
  if [ "${KORE_RECIPE:-direct}" = "cpt" ]; then
    checkpoint_complete "$MIDTRAIN_OUT" || { log "sft: refusing to start, midtrain is not complete"; return 1; }
  fi
  # Hold while the training mixture is still being built. Two reasons, and the
  # second is the one that bites: an 11h SFT on superseded data wastes the
  # allocation, but worse, Path A and Path B only answer whether midtrain earns
  # its keep if both arms train on IDENTICAL data. Path B starts hours after
  # Path A, so letting Path A run on v2 while v3 lands would silently invalidate
  # the comparison rather than fail it.
  if [ -f "$REPO/runs/DATA_NOT_FINAL" ]; then
    log "sft: HOLDING -- runs/DATA_NOT_FINAL: $(head -1 "$REPO/runs/DATA_NOT_FINAL" 2>/dev/null)"
    log "sft: remove that file to release both SFT arms onto the final mixture."
    note sft "held_data_not_final"
    return 2
  fi
  # In direct mode the base is whatever the config names -- the vendor instruct
  # checkpoint -- so pass '-' and let the launcher's resolver keep model_id.
  local sft_base="-"
  if [ "${KORE_RECIPE:-direct}" = "cpt" ]; then
    # Prefer the chat-vector model: it already follows instructions, so the SFT
    # budget buys kernel skill instead of re-learning chat from scratch. Fall
    # back to raw midtrain if the residual did not verify.
    sft_base="$MIDTRAIN_OUT"
    if checkpoint_complete "$CHATVEC_OUT"; then
      sft_base="$CHATVEC_OUT"
      log "sft: starting from the chat-vector model (instruction-following already present)"
    else
      log "sft: chat-vector model unavailable; starting from raw midtrain"
    fi
    # Midtrain is finished and its A/B eval has run, so its optimizer state is
    # now dead weight. Reclaim it if -- and only if -- space is short.
    reclaim_optimizer_state "$MIDTRAIN_OUT" "$SFT_PEAK_GB"
  else
    log "sft: starting from the config's model_id (direct recipe, no midtrain)"
  fi
  note sft_base "$sft_base"
  local free; free="$(disk_free_gb)"; free="${free:-0}"
  if [ "$free" -lt "$SFT_PEAK_GB" ]; then
    log "sft: HOLDING. ${free}GB free after reclaim, need ~${SFT_PEAK_GB}GB for three 221GB checkpoints."
    log "sft: reclaim already ran; either it refused (see its REFUSING line above)"
    log "sft: or the space it freed was not enough. Remaining large items:"
    du -sh "$REPO"/runs/* 2>/dev/null | sort -rh | head -5 | while read -r sz d; do log "sft:   $sz  $d"; done
    log "sft: not starting SFT into a disk wall -- this needs a look."
    note sft "held_insufficient_disk"
    note sft_free_gb "$free"
    return 2
  fi
  run_training_stage sft kore-sft "$SFT_OUT" "$SEGMENT_WALLTIME" \
    "$REPO/scripts/spur_sft_1node.sbatch" "$REPO/configs/sft_14b_full.json" "$sft_base" "$SFT_OUT"
}

log "================ pipeline driver start (HEAD $(git rev-parse --short HEAD)) ================"
note pipeline "running"

# Two recipes, and which one is available is decided by the model, not by us.
#
#   cpt     midtrain -> A/B -> chat vector -> SFT. Requires a BASE checkpoint to
#           continue-pretrain, because doing it on an instruct model destroys
#           instruction-following (docs/EVAL_RESULTS.md), and requires a
#           base/instruct PAIR for the residual transfer.
#   direct  instruct -> SFT. What Dr. Kernel and Kernel-Smith actually do.
#
# The production target is Qwen3-Coder-30B-A3B-Instruct, and neither it nor any
# other 30B-class candidate (Qwen3-32B, Qwen3.6-35B-A3B) ships a Base variant.
# So the cpt recipe cannot reach production at all: it is a 14B-only experiment.
# direct is the default for that reason, not as a fallback.
KORE_RECIPE="${KORE_RECIPE:-direct}"
log "recipe: $KORE_RECIPE"

if [ "$KORE_RECIPE" = "cpt" ]; then
  step_midtrain  || { log "STOP: midtrain failed"; note pipeline "failed_midtrain"; exit 1; }
  # Reference is the BASE weights midtrain actually started from. Held-out LM
  # loss is the meaningful signal here; kernel generation is not, because
  # neither arm follows instructions yet.
  step_eval "$MIDTRAIN_OUT" midtrain "midtrain_base" \
            "Qwen/Qwen3-14B-Base" "0b0bd3732e2c374d483664439ea334928b65f304"
  # Transplant Qwen's post-training onto our midtrained weights. Non-fatal: if
  # it does not verify, step_sft falls back to training from raw midtrain.
  step_residual || log "residual: unavailable; SFT will start from raw midtrain"
fi

step_sft; _sft_rc=$?
if [ "$_sft_rc" = "2" ]; then
  log "================ pipeline HELD before SFT (disk); midtrain + its eval are complete ================"
  note pipeline "held_before_sft"; exit 0
elif [ "$_sft_rc" != "0" ]; then
  log "STOP: sft failed"; note pipeline "failed_sft"; exit 1
fi
# The headline comparison: our tuned model against the untuned vendor model it
# started from. Both follow instructions, so the kernel funnel is finally a fair
# question, and using the exact starting checkpoint as reference isolates what
# SFT contributed rather than conflating it with the base model's own strength.
step_eval "$SFT_OUT" sft "sft" "$SFT_REFERENCE_MODEL" "$SFT_REFERENCE_REV"

# Third arm, and the one that keeps us honest. The chat-vector model is our
# domain delta on top of Qwen's instruct weights with NO SFT at all, so it
# isolates what the SFT stage actually bought. If SFT cannot beat it, the
# mixture is too narrow rather than the recipe being wrong -- a conclusion the
# two-arm comparison could not have reached.
if checkpoint_complete "$CHATVEC_OUT"; then
  step_eval "$CHATVEC_OUT" chatvec "chatvec" \
            "Qwen/Qwen3-14B" "40c069824f4251a91eefaf281ebe4c544efd3e18"
fi

log "================ pipeline COMPLETE through SFT; stopping before DPO ================"
note pipeline "complete_through_sft"
