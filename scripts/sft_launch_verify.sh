#!/usr/bin/env bash
# Answer "is the SFT run actually healthy?" from the job log, not from the fact that
# a job id exists.
#
# A 30B run can be in the queue, or loading, or training, or silently training on
# garbage, and `squeue` says RUNNING for all of them. These are the checks that
# distinguish those states, in the order the run passes through them. Anything marked
# FAIL is a reason to kill the job rather than let it burn a 26-36 hour allocation.
#
# Usage: sft_launch_verify.sh [JOBID]   (defaults to the newest kore-sft job)
set -uo pipefail
cd "${KORE_REPO:-/home/shasriva/Kore-RL/KORE}" || exit 1
export SPUR_CONTROLLER_ADDR="${SPUR_CONTROLLER_ADDR:-http://crs-m2m-cpu-spur-005:6817}"

JOB="${1:-}"
if [[ -z "$JOB" ]]; then
    JOB="$(squeue -u "$USER" -h -o "%i %j" 2>/dev/null | awk '$2=="kore-sft"{print $1}' | tail -1)"
fi
[[ -z "$JOB" ]] && { echo "no kore-sft job found (pass a JOBID explicitly)"; exit 1; }

state="$(squeue -j "$JOB" -h -o "%T" 2>/dev/null)"
node="$(squeue -j "$JOB" -h -o "%N" 2>/dev/null)"
elapsed="$(squeue -j "$JOB" -h -o "%M" 2>/dev/null)"
LOG="runs/sft-${JOB}.out"
ERR="runs/sft-${JOB}.err"

echo "=============================================================="
echo " SFT job $JOB  state=${state:-<not in queue>}  node=${node:-?}  elapsed=${elapsed:-?}"
echo "=============================================================="

if [[ "$state" == "PENDING" ]]; then
    echo "  PENDING: $(squeue -j "$JOB" -h -o '%R')"
    echo "  Nothing to verify yet. amd-general-qos is capped at 8 nodes team-wide,"
    echo "  so this waits for another job to exit. Remaining times:"
    squeue -t R -h -o "%.8i %.10u %.18q %.12l %.10M %N" 2>/dev/null \
        | awk '$3=="amd-general-qos"{printf "    %s %s limit=%s elapsed=%s\n", $1, $2, $4, $5}'
    exit 0
fi

[[ -f "$LOG" ]] || { echo "  no log at $LOG yet"; exit 0; }

pass() { printf "  PASS  %s\n" "$1"; }
fail() { printf "  FAIL  %s\n" "$1"; FAILED=1; }
wait_() { printf "  ....  %s\n" "$1"; }
FAILED=0

echo
echo "-- startup ------------------------------------------------------"
# Eight ranks must all reach the model. A partial launch trains a shard of the model
# on a shard of the data and looks superficially fine.
ranks="$(grep -c "sft: model loaded" "$LOG" 2>/dev/null || echo 0)"
if grep -q "Traceback\|CUDA out of memory\|HIP out of memory" "$LOG" "$ERR" 2>/dev/null; then
    fail "a traceback or OOM is present in the log"
    grep -m3 -A2 "Traceback\|out of memory" "$LOG" "$ERR" 2>/dev/null | sed 's/^/        /'
else
    pass "no traceback or OOM"
fi
grep -q "sft: assistant-only loss" "$LOG" 2>/dev/null \
    && pass "assistant-only loss masking verified (only completions are targets)" \
    || wait_ "assistant-only masking not yet reported"
grep -qE "resolved|dataset rows|sft.dataset" "$LOG" 2>/dev/null \
    && pass "dataset loaded" || wait_ "dataset load not yet reported"

echo
echo "-- MoE routing (is the router collapsing?) ----------------------"
if grep -q "MoE routing hooks installed" "$LOG" 2>/dev/null; then
    pass "$(grep -m1 -o "MoE routing hooks installed.*" "$LOG")"
else
    wait_ "routing hooks not yet installed"
fi
if grep -q "router load entropy collapsing" "$LOG" 2>/dev/null; then
    fail "ROUTER COLLAPSE WARNING -- experts are concentrating"
    grep -m2 "router load entropy collapsing" "$LOG" | sed 's/^/        /'
else
    pass "no router-collapse warning"
fi

echo
echo "-- divergence guards -------------------------------------------"
if grep -q "NON-FINITE loss" "$LOG" 2>/dev/null; then
    fail "NaN/Inf loss detected -- the run is diverging, kill it"
else
    pass "no NaN/Inf loss"
fi
if grep -q "grad_norm spike" "$LOG" 2>/dev/null; then
    printf "  WARN  %s\n" "grad-norm spikes seen ($(grep -c 'grad_norm spike' "$LOG")); watch, do not necessarily kill"
else
    pass "no grad-norm spikes"
fi

echo
echo "-- retention (the risk this run was built to detect) -----------"
if grep -q "RETENTION REGRESSION" "$LOG" 2>/dev/null; then
    fail "RETENTION REGRESSION -- kernel loss falling while retained capabilities climb"
    grep -m2 "RETENTION REGRESSION" "$LOG" | sed 's/^/        /'
else
    pass "no retention regression"
fi
evals="$(grep -c "sft_eval" "$LOG" 2>/dev/null || echo 0)"
if (( evals > 0 )); then
    pass "$evals eval(s) recorded; most recent per-capability losses:"
    grep "sft_eval" "$LOG" | tail -1 | tr ',' '\n' \
        | grep -E "_loss|_delta|step" | sed 's/^ */        /'
else
    wait_ "no eval yet (eval_on_start should produce one before step 1)"
fi

echo
echo "-- progress ----------------------------------------------------"
steps="$(grep -c "sft_step" "$LOG" 2>/dev/null || echo 0)"
if (( steps > 0 )); then
    last="$(grep "sft_step" "$LOG" | tail -1)"
    first="$(grep "sft_step" "$LOG" | head -1)"
    pass "$steps logged step events"
    echo "        first: $(echo "$first" | tr ',' '\n' | grep -E 'step=|loss=' | tr '\n' ' ')"
    echo "        last : $(echo "$last"  | tr ',' '\n' | grep -E 'step=|loss=|lr=' | tr '\n' ' ')"
    echo "        (1,609 steps is one epoch; loss should trend down, lr should peak at step 241)"
else
    wait_ "no optimizer steps logged yet"
fi

echo
echo "-- checkpoints -------------------------------------------------"
OUT="$(/home/shasriva/kore-venv/bin/python -c "
import json;print(json.load(open('configs/sft_coder30b_a3b.json'))['output_dir'])" 2>/dev/null)"
if [[ -n "$OUT" && -d "$OUT" ]]; then
    n="$(find "$OUT" -maxdepth 1 -name 'checkpoint-*' 2>/dev/null | wc -l)"
    pass "$n checkpoint(s) in $OUT"
    find "$OUT" -maxdepth 1 -name 'checkpoint-*' -printf '        %f\n' 2>/dev/null | sort | tail -3
else
    wait_ "output_dir not created yet (${OUT:-unknown})"
fi

echo
echo "=============================================================="
if (( FAILED )); then
    echo " VERDICT: PROBLEM DETECTED -- see FAIL lines above"
    exit 1
fi
echo " VERDICT: healthy so far"
echo "=============================================================="
