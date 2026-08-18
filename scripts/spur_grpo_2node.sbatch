#!/bin/bash
# KORE 30B A3B STAGE 3 (multi-turn GRPO) - TWO nodes, FSDP HYBRID_SHARD, 16x MI355X.
#
#   sbatch scripts/spur_grpo_2node.sbatch [CONFIG] [FROM_STAGE] [OUTPUT_DIR] [DATA_ROOT]
#
#   CONFIG      default configs/grpo_coder30b_a3b_trloo_2node.json
#   FROM_STAGE  checkpoint to train from
#   OUTPUT_DIR  where this stage writes
#   DATA_ROOT   co-evolution archive root  default data/b05factory
#
# WHY TWO NODES HELPS HERE, AND ONLY WITH HYBRID_SHARD
#   The GRPO step is rollout-dominated (~90 min single-node), and the rollout is
#   embarrassingly parallel: tasks_per_step 4 x num_trajectories 16 = 64 episodes,
#   sliced per rank. Doubling ranks 8 -> 16 halves the rollout, and 64 divides
#   evenly by 16 so no rank goes ragged (ragged per-rank counts deadlock
#   generation).
#
#   The catch is the network. These nodes have NO RDMA: one Ethernet interface,
#   which is why every single-node log is full of "NCCL WARN Could not find any
#   local path from gpu N to net". A 16-rank inter-node all_reduce measures
#   ~4.5 GB/s against hundreds of GB/s for intra-node xGMI. FULL_SHARD across
#   nodes re-gathers parameters over that link on EVERY forward and backward, so
#   it would be slower than one node. HYBRID_SHARD shards inside a node and
#   replicates across, leaving one gradient all-reduce per step as the only
#   inter-node traffic - a sub-1% tax against a 90-minute step.
#
# WHY THIS IS A BATCH JOB WITH AN srun INSIDE
#   An earlier attempt concluded SPUR could not create job steps from a batch
#   script ("job N is not running (state: Pending)") and moved the launcher to an
#   srun issued from the login node. That was a transient race, not a rule:
#   re-tested directly (job 24248), `srun --nodes=2 --ntasks-per-node=1` inside an
#   sbatch dispatched to both nodes and JobState read RUNNING from inside the
#   script. Being a real batch job is worth the re-test: it keeps the timed
#   self-requeue, append-mode logs, and survives an ssh drop.
#SBATCH --job-name=kore-grpo-2n
#SBATCH --account=amd-burst
#SBATCH --partition=amd-spur
#SBATCH --qos=amd-burst-qos
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
# gpu:8, not gpu:mi355x:8 -- the typed form does not dispatch on this controller.
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=12:00:00
# --requeue is deliberately ABSENT: on this controller a requeued job trips
# JobHoldMaxRequeue on its FIRST requeue and is held permanently, so a transient
# NODE_FAIL became an unrecoverable hold. Recovery is resubmission (the watchdog),
# plus the timed self-requeue below.
#SBATCH --exclusive
#SBATCH --open-mode=append
#SBATCH --output=/home/shasriva/Kore-RL/KORE/runs/grpo2n-%j.out
#SBATCH --error=/home/shasriva/Kore-RL/KORE/runs/grpo2n-%j.err
set -uo pipefail

REPO="${KORE_REPO:-/home/shasriva/Kore-RL/KORE}"
PY="${KORE_PY:-/home/shasriva/kore-venv/bin/python}"
# Positional, not --export: SPUR does not reliably propagate arbitrary
# --export KEY=VALUE into a requeued child; positional args survive requeue.
CFG="${1:-$REPO/configs/grpo_coder30b_a3b_trloo_2node.json}"
FROM_STAGE="${2:-/shared_nfs/shasriva/kore/runs/sft_coder30b_a3b_v5}"
OUT_DIR="${3:-/shared_nfs/shasriva/kore/runs/grpo_v5_trloo_2node}"
DATA_ROOT="${4:-data/b05factory}"

cd "$REPO" || exit 2
mkdir -p runs

echo "[grpo-2node] START job=${SLURM_JOB_ID:-none} batch_host=$(hostname)"
echo "[grpo-2node] nodes=${SLURM_JOB_NODELIST:-?} num_nodes=${SLURM_JOB_NUM_NODES:-?}"
echo "[grpo-2node] HEAD=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
echo "[grpo-2node] cfg=$CFG from=$FROM_STAGE out=$OUT_DIR data_root=$DATA_ROOT"

# ---- rendezvous address ---- #
# The batch script runs on the FIRST node of the allocation, which is also rank 0,
# so the master address is simply THIS node's own. Read it locally: these compute
# nodes cannot resolve each other by name (`hostname -f` returns
# localhost.localdomain) and `scontrol show hostnames` is not implemented here, so
# resolving a peer name is not an option.
IFACE="${KORE_NCCL_IFNAME:-ens3}"
MASTER_IP="$(ip -4 -o addr show "$IFACE" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)"
if [ -z "$MASTER_IP" ]; then
    read -r IFACE MASTER_IP <<<"$(ip -4 -o addr show | awk '$4 ~ /^10\.245\./ {print $2, $4}' \
                                   | cut -d/ -f1 | head -1)"
fi
if [ -z "${MASTER_IP:-}" ]; then
    echo "[grpo-2node] FATAL could not determine this node's IPv4 address" >&2
    ip -4 -o addr show >&2
    exit 2
fi
export KORE_MAIN_IP="$MASTER_IP"
export KORE_NCCL_IFNAME="$IFACE"
# A fixed port would collide with a sibling 2-node job on a shared node; derive it
# from the job id instead, in the ephemeral-safe 29500-30499 band.
export KORE_MAIN_PORT="${KORE_MAIN_PORT:-$(( 29500 + (${SLURM_JOB_ID:-0} % 1000) ))}"
echo "[grpo-2node] rendezvous iface=$IFACE master=$MASTER_IP port=$KORE_MAIN_PORT"

export KORE_NUM_MACHINES="${SLURM_JOB_NUM_NODES:-2}"
export KORE_ACCEL_CONFIG="$REPO/configs/accelerate_fsdp_grpo_2node.yaml"
[ -f "$KORE_ACCEL_CONFIG" ] || { echo "[grpo-2node] FATAL missing $KORE_ACCEL_CONFIG" >&2; exit 2; }

# ---- resolve the launch config ON THE BATCH NODE, once ---- #
# Doing this per-task would have 2 writers racing on one path.
source /home/shasriva/kore-venv/bin/activate
export PATH="/home/shasriva/kore-venv/bin:$PATH"
export PYTHONPATH="$REPO:$REPO/repos/aiter"
RESOLVED="$REPO/runs/grpo2n-${SLURM_JOB_ID:-manual}.resolved.json"
"$PY" scripts/spur_resolve_launch_config.py \
    --stage grpo --config "$CFG" --out "$RESOLVED" \
    --from-stage "$FROM_STAGE" --output-dir "$OUT_DIR" --data-root "$DATA_ROOT" || exit 2

# Refuse to burn a 2-node allocation on a config that is not actually hybrid: the
# whole reason for the second node is the topology, and under FSDP1 a mismatched
# accelerate YAML can silently substitute a different one.
"$PY" - "$RESOLVED" <<'PY' || exit 2
import json, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text())
strat = str(cfg.get("fsdp_sharding_strategy", "")).lower()
if not strat.startswith("hybrid"):
    sys.exit(f"[grpo-2node] FATAL fsdp_sharding_strategy={strat!r}, but a 2-node run "
             "must be hybrid_shard; FULL_SHARD/SHARD_GRAD_OP across a 4.5 GB/s link "
             "is slower than a single node")
print(f"[grpo-2node] sharding={strat} output_dir={cfg.get('output_dir')}")
PY

# Report the resume point BEFORE 16 ranks load a 30B model.
"$PY" - "$RESOLVED" <<'PY'
import json, sys
from pathlib import Path
from kore.policy.grpo import _find_grpo_resume_checkpoint
cfg = json.loads(Path(sys.argv[1]).read_text())
found = _find_grpo_resume_checkpoint(type("C", (), {"output_dir": cfg["output_dir"]}))
if found is None:
    print(f"[grpo-2node] resume_from=None output_dir={cfg['output_dir']} (fresh run)")
else:
    print(f"[grpo-2node] resume_from={found[0]} start_step={found[1].get('global_step')} "
          f"total_steps={cfg.get('total_steps')}")
PY

# ---- drain + self-requeue, same contract as the 1-node launcher ---- #
CHILD_PID=""
REQUEUE_TIMER_PID=""
terminate_child() {
    if [[ -n "$CHILD_PID" ]]; then
        kill -TERM -- "-$CHILD_PID" 2>/dev/null || true
        wait "$CHILD_PID" 2>/dev/null || true
    fi
}
handle_termination() {
    trap - USR1 TERM INT
    terminate_child
    echo "[grpo-2node] signal received; newest complete checkpoint is safe; exiting for requeue"
    exit 143
}
# SPUR's scontrol exits 0 even when it cannot reach the controller, so the exit
# status alone cannot confirm a requeue landed.
_requeue_attempt() {
    local out rc
    out="$(scontrol requeue "$SLURM_JOB_ID" 2>&1)"; rc=$?
    if ((rc == 0)) && ! grep -qiE 'error|failed|unavailable|no leader|refused' <<<"$out"; then
        return 0
    fi
    printf '%s\n' "$out" >&2
    return 1
}
requeue_self() {
    _requeue_attempt && return 0
    if [[ -n "${KORE_SPUR_CONTROLLER_ADDR:-}" ]]; then
        ( export SPUR_CONTROLLER_ADDR="$KORE_SPUR_CONTROLLER_ADDR"; _requeue_attempt ) && return 0
    fi
    return 1
}
handle_requeue_warning() {
    trap - USR1 TERM INT
    echo "[grpo-2node] USR1 time-limit warning; draining and requesting requeue"
    terminate_child
    if requeue_self; then exit 0; fi
    echo "[grpo-2node] FATAL explicit requeue failed" >&2
    exit 143
}
trap handle_termination TERM INT
trap handle_requeue_warning USR1

# SPUR's sbatch does not implement --signal, and Slurm does NOT auto-requeue a job
# that merely hits its time limit. burst's ceiling is 12h, so warn at 11h.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    ( sleep "${KORE_REQUEUE_AFTER_SECONDS:-39600}"; kill -USR1 "$$" ) &
    REQUEUE_TIMER_PID="$!"
fi
cleanup() { [[ -n "$REQUEUE_TIMER_PID" ]] && kill "$REQUEUE_TIMER_PID" 2>/dev/null || true; }
trap cleanup EXIT

# ---- wait for the controller to agree we are RUNNING ---- #
# THIS IS THE STEP THAT DECIDES WHETHER A 2-NODE JOB STARTS AT ALL.
#
# SPUR begins executing the batch script before the controller has necessarily
# marked the job RUNNING, and `srun` refuses to create a job step until it has:
#
#   Error: failed to create job step
#   Caused by: ... "job 24409 is not running (state: Pending)"
#
# It is a RACE, not a rule, which is why it is so easy to misdiagnose. A bare
# probe (job 24248) read JobState=RUNNING from inside the batch script and its
# srun dispatched to both nodes fine; jobs 24099 and 24409 lost the same race and
# died instantly. An earlier investigation saw only the failures and concluded
# batch jobs could not create steps at all, which cost the launcher its
# self-requeue by moving it to a login-node srun.
#
# So wait for the state to settle, then retry the step a few times anyway: the
# window is short, and burning a 2-node allocation on a two-second race is the
# most expensive possible way to lose.
wait_for_running() {
    local i st
    for i in $(seq 1 "${KORE_STEP_WAIT_TRIES:-90}"); do
        st="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
              | tr ' ' '\n' | grep -m1 '^JobState=' | cut -d= -f2)"
        if [[ "$st" == "RUNNING" ]]; then
            echo "[grpo-2node] controller reports RUNNING after ${i} check(s)"
            return 0
        fi
        sleep 2
    done
    echo "[grpo-2node] WARNING controller still reports '${st:-unknown}' after the" \
         "wait; attempting the job step anyway" >&2
    return 1
}
if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null; then
    wait_for_running || true
fi

# One task per node; each derives its own machine_rank from SLURM_NODEID.
# setsid so the drain above can signal the whole tree.
RC=0
ATTEMPTS="${KORE_STEP_RETRIES:-5}"
for attempt in $(seq 1 "$ATTEMPTS"); do
    setsid srun --nodes="${SLURM_JOB_NUM_NODES:-2}" --ntasks-per-node=1 \
        bash "$REPO/scripts/spur_grpo_2node_inner.sh" "$RESOLVED" &
    CHILD_PID="$!"
    if wait "$CHILD_PID"; then
        RC=0
        echo "[grpo-2node] grpo completed"
        break
    fi
    RC=$?
    # Only the step-creation race is worth retrying, and the per-node logs are how
    # to tell the two apart: the inner script's first act is to create
    # runs/grpo2n-$JOBID/node$N.log, so if none exists then no task ever started.
    # A TRAINING failure must not be retried here - it would re-enter a 30B load
    # that already failed, and the watchdog plus checkpoint resume is the right
    # recovery for that.
    if [[ "$attempt" -lt "$ATTEMPTS" ]] && ! ls "$REPO/runs/grpo2n-${SLURM_JOB_ID:-manual}"/node*.log >/dev/null 2>&1; then
        # No per-node log exists, so no task ever started: this is the step race,
        # not a training failure. Back off and try again.
        echo "[grpo-2node] job step did not start (attempt $attempt/$ATTEMPTS, rc=$RC); retrying in 10s"
        sleep 10
        continue
    fi
    echo "[grpo-2node] grpo failed rc=$RC"
    break
done

echo "GRPO_RC=$RC"
echo "[grpo-2node] END job=${SLURM_JOB_ID:-none} status=$RC"
exit "$RC"
