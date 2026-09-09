#!/bin/bash
# Run shasriva's KORE GRPO training inside an allocation owned by SOMEBODY ELSE.
#
# WHY THIS EXISTS: amd-spur hands out whole nodes and every job sits at the same
# priority 1000, so a freed node goes to the oldest pending job, not to the
# person who wants it. Lending a live allocation is therefore the only way to
# hand a specific node to a specific person. This script is the lender's side of
# that: one command, no scheduler surgery, nothing to undo.
#
# WHAT THE LENDER RUNS (from the SPUR login node):
#
#   nohup srun --overlap --jobid=<THEIR_JOBID> --nodes=1 --ntasks=1 \
#       bash /home/shasriva/Kore-RL/KORE/scripts/borrow_run_grpo.sh \
#       > ~/kore-borrow.log 2>&1 &
#
# --overlap is required: without it the step waits for the job's own resources
# instead of sharing them. nohup + & so closing the terminal does not kill it.
#
# WHAT IT DOES NOT DO, by construction (see KORE_BORROWED_ALLOCATION in
# scripts/spur_grpo_1node.sbatch): it never calls scontrol, never requeues, never
# arms a drain timer, and never signals anything outside its own process group.
# It cannot affect the lender's job. Ending the loan is just scancel/ctrl-C of
# the step, or letting the job end - the trainer checkpoints every step, so the
# work is not lost.
#
# It also never reads .env.local, so no credential of the borrower's is exposed.
set -uo pipefail

REPO=/home/shasriva/Kore-RL/KORE
CFG=configs/grpo_coder30b_a3b_trloo_burst.json
MODEL=/shared_nfs/shasriva/kore/runs/sft_coder30b_a3b_v5
DATA_ROOT=data/b05factory

# Checkpoints go somewhere the LENDER owns, because the borrower's tree is not
# writable by them and asking them to chmod it would be the wrong direction of
# trust. Overridable in case /shared_nfs is not the roomy filesystem here.
OUT_DIR="${KORE_BORROW_OUT:-/shared_nfs/${USER}/kore-borrow/grpo_v5_trloo}"

echo "=== KORE borrowed-allocation launch ==="
echo "host=$(hostname -s) user=$(id -un) job=${SLURM_JOB_ID:-none}"
echo "out_dir=$OUT_DIR"

fail() { echo "FATAL: $*" >&2; exit 2; }

# Fail fast and legibly on the two things that are genuinely outside this
# script's control, so the lender gets an answer instead of a stack trace.
[[ -d "$REPO" ]] || fail "cannot see $REPO (is /home shared on this node?)"
[[ -r "$MODEL" && -x "$MODEL" ]] || fail \
    "cannot read the model at $MODEL. shasriva must widen it: chmod -R g+rX (needs a compute node, /shared_nfs is not on the login node)."

mkdir -p "$OUT_DIR" || fail "cannot create $OUT_DIR"
# setgid + group write so the borrower can resume from these checkpoints later
# without another hand-off; both accounts share group ubuntu.
chmod 2775 "$OUT_DIR" 2>/dev/null || true
# Same reason, for every file the run is about to create.
umask 002

echo "=== GPUs visible to this step ==="
rocm-smi --showid 2>/dev/null | grep -ciE "^GPU|card" || true

export KORE_BORROWED_ALLOCATION=1
cd "$REPO" || fail "cannot cd $REPO"

echo "=== launching (log also at $OUT_DIR/borrow-launch.log) ==="
bash scripts/spur_grpo_1node.sbatch \
    "$CFG" "$MODEL" "$OUT_DIR" "$DATA_ROOT" \
    2>&1 | tee -a "$OUT_DIR/borrow-launch.log"
exit "${PIPESTATUS[0]}"
