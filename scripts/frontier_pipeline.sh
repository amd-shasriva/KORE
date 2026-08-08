#!/bin/bash
# Keep every frontier dialect moving from seed to training rows, unattended.
#
# The old hip_pipeline_loop drove exactly one dialect through one path: seed HIP,
# gate HIP, mine HIP. Everything else was a separate thing somebody had to
# remember to start, which is how pool-Triton and registry-HIP each died twice in
# a night and how FlyDSL sat at zero rows while 45 arena tasks scored 1%.
#
# The three dialects do not actually need three pipelines. A twin is a task
# directory whose seed is in a different language; verify_pool_hip_seeds already
# gates ``*__hip``, ``*__hipf`` and ``*__flydsl`` on one path because the
# candidate filename comes from the backend the task declares. So the stages are
# shared and only the roots differ:
#
#   materialize   teacher writes a twin seed        (CPU + gateway, NO GPU slot)
#   gate          gfx950 says it compiles and is    (GPU, one job per root)
#                 numerically correct
#   harvest       promote passers, re-partition     (CPU)
#   mine          datagen optimizes the gated twin  (GPU, staff_datagen)
#
# Materialization deliberately runs on the login node. It is gateway-bound -- 8
# workers at ~1 CPU-second per 5 minutes -- so putting it in an allocation would
# hold a node hostage to network latency while the arena queues behind it.
#
# Everything here is idempotent: seeds skip what exists, the gate resumes, harvest
# only promotes what passed, and datagen skips finished shards. Safe to restart at
# any point, which is what keepalive does when it dies.
#
#   scripts/frontier_pipeline.sh
set -uo pipefail

REPO=/home/shasriva/Kore-RL/KORE
PY=/home/shasriva/kore-venv/bin/python
LOG="$REPO/runs/frontier_pipeline.log"
GATE_EVERY="${GATE_EVERY:-30}"       # gate once this many new seeds have landed
SLEEP="${FRONTIER_SLEEP:-300}"

#: root | seed-glob | materializer | extra args. The materializers are the slow,
#: gateway-bound half and are kept alive here rather than being run by hand.
#: --families is the whole point: without it these regenerate the launch-bound
#: bulk whose median baseline is 17us.
FAMILIES="${FRONTIER_FAMILIES:-attention gemm quantization}"
HIP_ROOT="${HIP_ROOT:-data/pool_hip_frontier}"
FLYDSL_ROOT="${FLYDSL_ROOT:-data/pool_flydsl}"
#: HIP twins of the *registry's* frontier tasks -- flash attention, fused MoE,
#: fp8 GEMM -- rather than of the pool. This is the only root whose difficulty
#: comes from the task rather than from the dialect: primary scales run from
#: 16.7M to 68.7B elements against the pool's uniform 1M, and the baselines are
#: AITER and hipBLASLt rather than eager torch.
#:
#: It takes --source-root instead of --families, because the registry is already
#: the curated set: select_frontier_tasks ranks it and everything above the
#: histogram break is a frontier family by construction.
REG_HIP_ROOT="${REG_HIP_ROOT:-data/registry_hip_frontier}"
#: The same twin, in the other dialect the arena scores. FlyDSL is 25% of the
#: arena and could not read a registry task at all until its spec adapter was
#: shared with the HIP path, so it had been porting only the pool. 480 of the
#: 482 frontier registry tasks ship a working seed_triton.py, which is exactly
#: what the port prompt needs.
REG_FLYDSL_ROOT="${REG_FLYDSL_ROOT:-data/registry_flydsl_frontier}"

#: The FlyDSL port is written by a different teacher than the HIP seeds.
#:
#: .env.local points KORE_TEACHER_MODEL at claude-opus-5, which writes HIP fine
#: -- 248 registry seeds, 4% truncated -- and cannot write FlyDSL at all. On the
#: port prompt it runs to the token ceiling and returns no text whatsoever: 439s
#: to consume 32,768 output tokens and hand back zero characters, on every
#: single call. That is why this root had one seed and 14 of 14 truncations.
#:
#: Measured on the same prompt and task, claude-opus-4.8 returns a valid kernel
#: -- @flyc.jit present, entry point defined -- in 36s and 3,417 tokens, and
#: sonnet-4.5 in 42s. The failure is specific to opus-5 on this prompt, so only
#: this stream moves.
FLYDSL_TEACHER_MODEL="${FLYDSL_TEACHER_MODEL:-claude-opus-4.8}"

cd "$REPO" || exit 1
[ -z "${SPUR_CONTROLLER_ADDR:-}" ] && [ -r /etc/profile.d/spur.sh ] && . /etc/profile.d/spur.sh
# shellcheck disable=SC1091
. "$REPO/scripts/gpu_slots.sh"

say() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$LOG"; }

n_seeds()    { ls -d "$REPO/$1"/tasks/*__* 2>/dev/null | wc -l; }
n_promoted() { ls -d "$REPO/data/pool_hip_ok"/tasks/*__* 2>/dev/null | wc -l; }
queued()     { squeue -u "${USER:-$(id -un)}" -h -n "$1" 2>/dev/null | wc -l; }

#: A materializer is alive if its process is; it holds no allocation, so this is
#: a plain pgrep rather than a queue question.
#:
#: Matched on --out, not on the script name. Two of these roots run the same
#: script against different sources, so a name match reports the registry
#: materializer as the pool one and the pool root never restarts.
materializer_alive() { pgrep -f -- "--out $1" >/dev/null 2>&1; }

#: A root with nothing left to seed says so, and is left alone until the marker
#: ages out. Restarting one is not free: deciding a pool task is HIP-eligible
#: means running the module to see whether its weights can be supplied from
#: outside, so a sweep that selects nothing still costs ~90s of CPU to find
#: that out, and on a 5-minute loop that is a third of a core spent on a
#: settled question. The pool-sourced HIP root reached this state at 4
#: remaining tasks, with 783 of its 787 untwinned frontier-family tasks not
#: functionalizable.
#:
#: It ages out rather than being permanent because the source roots grow.
EXHAUSTED_TTL="${EXHAUSTED_TTL:-21600}"   # 6h

root_exhausted() {
    local marker="$REPO/$1/.exhausted"
    [ -f "$marker" ] || return 1
    local age=$(( $(date +%s) - $(stat -c %Y "$marker" 2>/dev/null || echo 0) ))
    [ "$age" -lt "$EXHAUSTED_TTL" ]
}

#: Trailing KEY=VALUE arguments are passed to the materializer's environment.
#: load_env_local uses setdefault, so anything set here beats .env.local -- which
#: is how a per-stream teacher is chosen without editing a secrets file.
start_materializer() {
    local script="$1" out="$2" limit="$3" log="$4"
    shift 4
    materializer_alive "$out" && return 0
    root_exhausted "$out" && return 0
    say "materializer $script absent -> starting (out=$out, families=$FAMILIES)"
    # shellcheck disable=SC2086
    setsid nohup env PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 "$@" \
        "$PY" "$REPO/scripts/$script" \
        --families $FAMILIES --limit "$limit" --workers 8 --out "$out" \
        >> "$REPO/runs/$log" 2>&1 < /dev/null &
    sleep 2
}

#: A twin of the *registry* rather than of the pool.
#:
#: These take --source-root instead of --families, because the registry is
#: already the curated set: everything above the histogram break in
#: select_frontier_tasks is a frontier family by construction. It is also the
#: only source whose difficulty comes from the task rather than the dialect --
#: primary scales from 16.7M to 68.7B elements against the pool's uniform 1M,
#: and baselines of AITER and hipBLASLt rather than eager torch.
#:
#: Liveness keys on --out like the pool ones do. Matching "source-root
#: kore/tasks" instead would have made the two registry dialects answer for
#: each other, and whichever started second would never run.
start_registry_materializer() {
    local script="$1" out="$2" limit="$3" log="$4"
    shift 4
    materializer_alive "$out" && return 0
    root_exhausted "$out" && return 0
    say "registry materializer $script absent -> starting (out=$out)"
    setsid nohup env PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 "$@" \
        "$PY" "$REPO/scripts/$script" \
        --source-root kore/tasks --limit "$limit" --workers 8 --out "$out" \
        >> "$REPO/runs/$log" 2>&1 < /dev/null &
    sleep 2
}

say "=== frontier pipeline start (pid $$) families='$FAMILIES' ==="
declare -A LAST_GATED=()

while :; do
    purge_held | while read -r l; do say "$l"; done

    # --- 1. keep both materializers alive (no GPU slot consumed) -------------
    start_materializer materialize_pool_hip.py    "$HIP_ROOT"    600 hip_frontier_materialize.log
    start_materializer materialize_pool_flydsl.py "$FLYDSL_ROOT" 400 \
        flydsl_materialize.log KORE_TEACHER_MODEL="$FLYDSL_TEACHER_MODEL"
    # Matched on its --source-root so it is distinguishable from the pool-sourced
    # HIP materializer, which runs the same script against a different root.
    start_registry_materializer materialize_pool_hip.py "$REG_HIP_ROOT" 500 \
        registry_hip_materialize.log
    start_registry_materializer materialize_pool_flydsl.py "$REG_FLYDSL_ROOT" 400 \
        registry_flydsl_materialize.log KORE_TEACHER_MODEL="$FLYDSL_TEACHER_MODEL"

    # --- 2. gate whichever root has accumulated enough new seeds ------------
    # One gate job per root, named after it, so a slow HIP gate never blocks the
    # FlyDSL one -- the mistake that left the functionalized root undecided for a
    # night while the parameter-free root finished.
    for root in "$REG_HIP_ROOT" "$REG_FLYDSL_ROOT" "$HIP_ROOT" "$FLYDSL_ROOT"; do
        [ -d "$REPO/$root/tasks" ] || continue
        tag=$(basename "$root")
        seeds=$(n_seeds "$root")
        last=${LAST_GATED[$tag]:-0}
        [ "$seeds" -eq 0 ] && continue
        [ "$(queued "kore-gate-$tag")" -gt 0 ] && continue
        if [ "$((seeds - last))" -ge "$GATE_EVERY" ]; then
            have_slot || { say "  no slot for gate-$tag; next pass"; continue; }
            say "gating $root: $seeds seed(s), $((seeds - last)) new"
            # shellcheck disable=SC2086
            GATE_ROOT="$root" sbatch $QOS_ARG --job-name="kore-gate-$tag" \
                scripts/spur_gate_pool_hip.sbatch 2>&1 | tee -a "$LOG"
            LAST_GATED[$tag]=$seeds
            sleep 5
        fi
    done

    # --- 3. harvest anything newly gated, then re-partition -----------------
    # Passed twins land in data/pool_hip_ok regardless of dialect, because the
    # gate writes there for whatever it admitted.
    promoted=$(n_promoted)
    if [ "$promoted" -gt 0 ] && [ ! -f "$REPO/runs/frontier_harvest.lock" ]; then
        : # harvest is owned by hip_pool_harvest.sh; only re-partition below
    fi

    # Frontier registry shards are static (the 482 ids do not change), so only
    # refresh their commit stamp -- a stale manifest kills every worker on the
    # preflight check with NonZeroExitCode and reads as a queue problem.
    for d in runs/shards_frontier runs/shards_hippool runs/shards_pooltriton; do
        [ -d "$REPO/$d" ] && "$PY" "$REPO/scripts/refresh_shards.py" "$d" >> "$LOG" 2>&1
    done

    # --- 4. staff mining across every declared stream -----------------------
    bash "$REPO/scripts/staff_datagen.sh" 2>&1 | tail -2 | tee -a "$LOG"

    sleep "$SLEEP"
done
