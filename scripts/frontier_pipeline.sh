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
#: How many gates may hold a general-QoS slot at once. Gating is the step that
#: turns a seed into something mineable, so it is worth one of the eight shared
#: nodes rather than a place in the burst queue behind 35 other jobs.
GENERAL_GATE_MAX="${GENERAL_GATE_MAX:-1}"
#: Gates in flight across every root.
#:
#: One was right while the cap was 8 and a second gate could only sit in burst
#: holding a slot. The cap is 10 now and the backlog changed shape: the repair
#: loop has rewritten 9,466 kernels, and every one of them needs a verdict
#: before it can be mined. Serialised at one gate that queue drains a root at a
#: time -- 1,034 tasks were waiting behind a single running gate. The second
#: gate takes a burst slot rather than a general one, so it costs nothing that
#: mining or the arena wanted.
MAX_GATES_IN_FLIGHT="${MAX_GATES_IN_FLIGHT:-2}"

#: How many failing twins to hand back to the teacher per repair pass, and how
#: many times one task may be retried before it is written off. Bounded because
#: a kernel that cannot be fixed would otherwise cost a teacher call every pass
#: forever.
REPAIR_LIMIT="${REPAIR_LIMIT:-150}"
#: Frontier twins are worth more attempts than the default. 364 of the 482
#: frontier tasks have no working twin in either dialect, and the corpus is
#: 11,884 Triton rows against 738 HIP and 229 FlyDSL -- so a frontier task
#: rescued into HIP or FlyDSL is the scarcest row available, while a fourth
#: attempt at an elementwise op is not.
REPAIR_MAX_ATTEMPTS="${REPAIR_MAX_ATTEMPTS:-4}"

#: Which roots the repair budget goes to.
#:
#: Measured over roughly 9,500 repairs: 121 HIP kernels rescued from failure to
#: passing, and zero FlyDSL. The model can act on a HIP error and cannot act on
#: a FlyDSL one -- it does not know the language well enough for the message to
#: mean anything. So the FlyDSL roots come out and the whole budget goes to HIP,
#: where a repair converts into a gated, mineable task about a fifth of the time.
# The default has to repeat the literal rather than read $REG_HIP_ROOT, which
# is not defined until 26 lines below this. Under `set -u` that made the script
# abort on its own first line of work whenever it was run without ensure_loops
# to supply the variable from the environment -- so it worked under the loop
# and died by hand, which is the worst way round for something you debug.
REPAIR_ROOTS="${REPAIR_ROOTS:-${REG_HIP_ROOT:-data/registry_hip_frontier}}"

#: Which roots the gate spends GPU slots on. Gating a root nothing is mining
#: buys a verdict that will not be read: FlyDSL mining is paused, so a FlyDSL
#: gate holds one of the eight shared general nodes to produce passers that sit
#: unconsumed. The already-gated FlyDSL set stays promoted and sharded by the
#: harvest, so resuming it later costs nothing.
GATE_ROOTS="${GATE_ROOTS:-$REG_HIP_ROOT $REG_FLYDSL_ROOT}"

#: The pool-sourced roots: seeded, gated and harvested, but no longer mined.
#:
#: Their streams are retired to 0 because neither is frontier -- pool_flydsl is
#: kbk_actor and kbk_mlp at fp32, 0% of the frontier list, and pool_hip_frontier
#: is not gated at all, so every seed it has ever produced was dead on arrival.
#: Keeping them alive costs the one resource the frontier set is actually short
#: of: there are 224 mineable frontier tasks and the teacher is what makes more,
#: so a gateway call spent on a scraped MLP is a frontier twin not seeded.
#:
#: Set to 1 to revive both. The ledgers and shards are untouched.
POOL_STREAMS="${POOL_STREAMS:-0}"
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

#: The 482 ids select_frontier_tasks ranked above the histogram break.
#:
#: Pointing a registry stream at the source root alone was wrong: kore/tasks is
#: 1,549 dirs, and the other 1,067 are generated elementwise and reduction ops
#: that sort before the interesting ones. Both registry streams therefore spent
#: their first thousand teacher calls on gen_abs and gelu_tanh -- 66% of the
#: twins they had produced were off-target -- while flash attention and fused
#: MoE waited behind them in name order.
FRONTIER_TASK_LIST="${FRONTIER_TASK_LIST:-$REPO/runs/frontier_tasks.txt}"

#: Where gated frontier twins live once promoted, and the shard set mined from
#: them. Separate from data/pool_hip_ok on purpose: that root holds 6,457 pool
#: twins, and merging would make the frontier ones a rounding error in the
#: sampling. This is the set that gives the HIP and FlyDSL halves of the arena
#: -- 22% and 25% of it -- training signal at frontier difficulty.
TWIN_OK_ROOT="${TWIN_OK_ROOT:-data/frontier_twins_ok}"
TWIN_DATA_ROOT="${TWIN_DATA_ROOT:-data/v5frontier_twins}"
TWIN_SHARD_DIR="${TWIN_SHARD_DIR:-runs/shards_frontier_twins}"
TWIN_SHARDS="${TWIN_SHARDS:-3}"

#: The pool-sourced FlyDSL twins, promoted and mined separately from the
#: frontier set so difficulty is not silently mixed. Currently the only FlyDSL
#: signal that clears the gate in quantity.
POOL_FLYDSL_OK_ROOT="${POOL_FLYDSL_OK_ROOT:-data/pool_flydsl_ok}"
POOL_FLYDSL_DATA_ROOT="${POOL_FLYDSL_DATA_ROOT:-data/v5pool_flydsl}"
POOL_FLYDSL_SHARD_DIR="${POOL_FLYDSL_SHARD_DIR:-runs/shards_pool_flydsl}"
POOL_FLYDSL_SHARDS="${POOL_FLYDSL_SHARDS:-4}"

#: The hard half of the external pool, already gated, selected by the same
#: scorer that picks the registry frontier. It is a static set -- the ids come
#: from data/pool_hip_ok, which the retired pool sweep already filled -- so it
#: needs its commit stamp kept fresh and nothing else.
HARDPOOL_SHARD_DIR="${HARDPOOL_SHARD_DIR:-runs/shards_hardpool}"

#: Nodes the mining hold should carry. Six, because the job cap is eight and
#: the two arena arms hold their own nodes under kore_hold. Claiming more would
#: idle nodes nobody else can then use, which is the thing the holds on this
#: cluster are already too good at.
MINE_HOLD_NODES="${MINE_HOLD_NODES:-6}"

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

#: How many of a root's twins already have a verdict. The gate resumes from this
#: file, so seeds minus verdicts is exactly the work it would do.
n_verdicts() {
    local f="$REPO/runs/$(basename "$1")_gate.json"
    [ -f "$f" ] || { echo 0; return; }
    "$PY" -c 'import json,sys
try: print(len(json.load(open(sys.argv[1])).get("rows", [])))
except Exception: print(0)' "$f" 2>/dev/null || echo 0
}

#: Gates queued or running for any root. They compete with each other for the
#: one general slot and with mining for the job cap, so the pipeline runs them
#: one at a time rather than one per root.
gates_in_flight() {
    squeue -u "${USER:-$(id -un)}" -h -o '%j' 2>/dev/null \
        | grep -c '^kore-gate-'
}

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
    local marker="$REPO/$1/.exhausted" tasks="$REPO/$1/tasks"
    [ -f "$marker" ] || return 1
    # A marker older than the root it describes is answering a question about a
    # different root. Twins get deleted -- a batch written against a broken
    # prompt is worth less than the slot it occupies -- and after one such
    # cleanup two registry streams sat idle holding a verdict recorded before
    # the deletion, with hours left on the TTL and 480 tasks waiting.
    # -nt, not stat: stat's %Y is whole seconds, and a cleanup that finishes in
    # the same second the marker was written then compares equal and is missed.
    if [ -d "$tasks" ] && [ "$tasks" -nt "$marker" ]; then
        rm -f "$marker"
        return 1
    fi
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
#: Rebuild the selection if it is missing. Without it these streams would fall
#: back to the whole registry, which is the failure this guards -- two thirds of
#: the teacher calls spent on generated elementwise ops. Skipping the stream is
#: the better failure: seeding the wrong thing costs more than seeding nothing.
ensure_task_list() {
    [ -s "$FRONTIER_TASK_LIST" ] && return 0
    say "frontier task list missing -> rebuilding $FRONTIER_TASK_LIST"
    "$PY" "$REPO/scripts/select_frontier_tasks.py" --out "$FRONTIER_TASK_LIST" \
        >> "$LOG" 2>&1
    [ -s "$FRONTIER_TASK_LIST" ] && return 0
    say "  could not build the frontier list; skipping registry streams"
    return 1
}

start_registry_materializer() {
    local script="$1" out="$2" limit="$3" log="$4"
    shift 4
    materializer_alive "$out" && return 0
    root_exhausted "$out" && return 0
    ensure_task_list || return 0
    say "registry materializer $script absent -> starting (out=$out)"
    setsid nohup env PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 "$@" \
        "$PY" "$REPO/scripts/$script" \
        --source-root kore/tasks --task-list "$FRONTIER_TASK_LIST" \
        --limit "$limit" --workers 8 --out "$out" \
        >> "$REPO/runs/$log" 2>&1 < /dev/null &
    sleep 2
}

say "=== frontier pipeline start (pid $$) families='$FAMILIES' ==="

while :; do
    purge_held | while read -r l; do say "$l"; done

    # Keep the mining hold pointed at nodes that are about to free. The cluster
    # runs at 137 allocated and 2 idle, and both of those idle nodes take a
    # five-minute hello-world straight to JobLaunchFailure -- so new capacity
    # only ever arrives as somebody else's job ends, and the only way to be
    # first to it is to have claimed the node beforehand. Picking without
    # checking the clock is how this hold came to sit on two nodes with 26 days
    # left while nodes freeing in two hours went elsewhere.
    "$PY" "$REPO/scripts/claim_soonest_nodes.py" --want "$MINE_HOLD_NODES" \
        2>&1 | grep -E 'claimed|released' | while read -r l; do say "  $l"; done

    # Nudge. A miner queued against the hold binds to the node set as it stood
    # when it was submitted, so one queued before a swap waits forever on nodes
    # we no longer hold while a reserved node sits idle beside it -- four of
    # them did exactly that overnight. When the hold has a free node and miners
    # are queued, drop the queued ones; the staffing pass at the bottom of this
    # loop resubmits them against the set we hold now.
    if [ "$(mine_res_free)" -gt 0 ]; then
        stale=$(_squeue -t PD -o "%i %j" | awk '$2 ~ /^kore-mine-/ {print $1}')
        if [ -n "$stale" ]; then
            say "  hold has a free node with miners queued; rebinding them"
            # shellcheck disable=SC2086
            scancel $stale 2>/dev/null
        fi
    fi

    # --- 1. keep both materializers alive (no GPU slot consumed) -------------
    if [ "$POOL_STREAMS" = "1" ]; then
        start_materializer materialize_pool_hip.py    "$HIP_ROOT"    600 hip_frontier_materialize.log
        start_materializer materialize_pool_flydsl.py "$FLYDSL_ROOT" 400 \
            flydsl_materialize.log KORE_TEACHER_MODEL="$FLYDSL_TEACHER_MODEL"
    fi
    # Matched on its --source-root so it is distinguishable from the pool-sourced
    # HIP materializer, which runs the same script against a different root.
    start_registry_materializer materialize_pool_hip.py "$REG_HIP_ROOT" 500 \
        registry_hip_materialize.log
    start_registry_materializer materialize_pool_flydsl.py "$REG_FLYDSL_ROOT" 400 \
        registry_flydsl_materialize.log KORE_TEACHER_MODEL="$FLYDSL_TEACHER_MODEL"

    # --- 1b. show failing twins their own error ------------------------------
    # Seeding is one-shot and the gate's verdict was a dead end: a kernel that
    # failed was discarded with the reason unread. For FlyDSL that is nearly all
    # of the work -- 173 of 3,974 ports passed, and 3,109 of the failures crash
    # before producing a number, on API misuse the gate names in one line. This
    # hands that line back to the teacher. Teacher-bound like the materializers,
    # so it holds no allocation; the repaired kernels are re-gated on a later
    # pass because the pass drops their stale verdicts.
    for spec in $REPAIR_ROOTS; do
        [ -d "$REPO/$spec/tasks" ] || continue
        pgrep -f -- "repair_twin_seeds.py --root $spec" >/dev/null 2>&1 && continue
        setsid nohup env PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 \
            KORE_TEACHER_MODEL="$FLYDSL_TEACHER_MODEL" \
            "$PY" "$REPO/scripts/repair_twin_seeds.py" --root "$spec" \
            --task-list "$FRONTIER_TASK_LIST" \
            --limit "$REPAIR_LIMIT" --workers 8 --max-attempts "$REPAIR_MAX_ATTEMPTS" \
            >> "$REPO/runs/repair_$(basename "$spec").log" 2>&1 < /dev/null &
        sleep 2
    done

    # --- 2. gate whichever root has accumulated enough new seeds ------------
    # One gate job per root, named after it, so a slow HIP gate never blocks the
    # FlyDSL one -- the mistake that left the functionalized root undecided for a
    # night while the parameter-free root finished.
    for root in $GATE_ROOTS; do
        [ -d "$REPO/$root/tasks" ] || continue
        tag=$(basename "$root")
        seeds=$(n_seeds "$root")
        # Ungated work is seeds minus verdicts, not seeds minus what this loop
        # remembers gating. Two things broke the remembered version. It is held
        # in memory, so a restart re-gated everything; and a repaired kernel
        # replaces a seed in place, leaving the seed count unchanged, so 628
        # repairs produced no gate at all and the whole repair loop was a no-op.
        # Verdicts are on disk and the repair pass deletes the ones it
        # invalidates, which makes this the same question in both cases.
        ungated=$(( seeds - $(n_verdicts "$root") ))
        [ "$seeds" -eq 0 ] && continue
        [ "$(queued "kore-gate-$tag")" -gt 0 ] && continue
        # Across all roots, not just this one. Only one gate can hold a general
        # slot, so submitting a gate per root put three of them in burst -- a
        # queue 35 deep that they never came out of -- and each still counted
        # against the job cap. Four roots of gates plus two arenas filled all
        # eight slots and mining could not be staffed at all. Gating is
        # sequential work anyway; one in flight loses nothing.
        if [ "$(gates_in_flight)" -ge "$MAX_GATES_IN_FLIGHT" ]; then
            say "  gate in flight already; $tag waits its turn"
            continue
        fi
        if [ "$ungated" -ge "$GATE_EVERY" ]; then
            # An idle node on the hold is a slot even when the cap says there is
            # none: the cap counts jobs, and four miners wedged in burst spent
            # the whole allowance without running. A gate blocked on that is
            # worse than a miner blocked on it -- gating is what turns seeds
            # into mineable tasks at all -- and one sat pending for five hours
            # while a node I hold stood idle.
            have_slot || [ "$(mine_res_free)" -gt 0 ] || {
                say "  no slot for gate-$tag; next pass"; continue; }
            say "gating $root: $seeds seed(s), $ungated without a verdict"
            # The hold first: a free general slot on this cluster is usually a
            # node that cannot launch, and a gate that wedges there stops the
            # supply of mineable tasks entirely.
            gate_qos="$(mine_res_arg)"
            [ -z "$gate_qos" ] && gate_qos="$(pick_qos kore-gate- "$GENERAL_GATE_MAX")"
            # shellcheck disable=SC2086
            GATE_ROOT="$root" sbatch $gate_qos \
                --job-name="kore-gate-$tag" \
                scripts/spur_gate_pool_hip.sbatch 2>&1 | tee -a "$LOG"
            sleep 5
        fi
    done

    # A twin's driver.py is copied out of its source task once, when the seed is
    # written, so a later fix to that driver never reaches twins already on
    # disk. That is how 34 twins went on failing in a loader that had already
    # been fixed. Cheap to re-check every pass; it only writes on drift.
    "$PY" "$REPO/scripts/refresh_twin_drivers.py" 2>&1 \
        | grep -v "^twin drivers: 0 stale" | tee -a "$LOG"

    # --- 3. harvest anything newly gated, then re-partition -----------------
    # Passed twins land in data/pool_hip_ok regardless of dialect, because the
    # gate writes there for whatever it admitted.
    # This step used to be a comment and a no-op, and that is where the whole
    # pipeline stopped. The gate only writes a verdict file; promoting the
    # passers into a resolvable task root and sharding them is the harvester's
    # job, and nothing was calling it. So every twin ever gated -- 1,104
    # registry-HIP, 309 registry-FlyDSL -- got as far as a verdict and no
    # further, and no mining stream contained a single one of them.
    #
    # The frontier twins get their own promoted root and their own shard set.
    # Pooling them with the 6,457 already in data/pool_hip_ok would leave them
    # a few percent of it and mine the launch-bound majority instead.
    HIP_ROOTS="$REG_HIP_ROOT $REG_FLYDSL_ROOT" \
    HIP_PROMOTED="$REPO/$TWIN_OK_ROOT" \
    HIP_DATA_ROOT="$REPO/$TWIN_DATA_ROOT" \
    HIP_SHARD_DIR="$REPO/$TWIN_SHARD_DIR" \
    HIP_TASK_IDS_OUT="$REPO/runs/frontier_twin_tasks.txt" \
    HIP_TWIN_GLOBS='*__hip *__hipf *__flydsl' \
    HIP_TASK_LIST="$FRONTIER_TASK_LIST" \
    NO_SUBMIT=1 \
        bash "$REPO/scripts/hip_pool_harvest.sh" "$TWIN_SHARDS" 2>&1 \
        | grep -E "promoted|PARTITION|nothing to do" | tee -a "$LOG"

    # The pool-sourced FlyDSL twins, kept separate from the frontier set.
    #
    # They are the launch-bound half of the corpus and do not belong in a set
    # named for its difficulty. They are also, right now, the only FlyDSL
    # signal there is: 172 of them clear the gate against 3 from the registry,
    # and FlyDSL is a quarter of the arena. Leaving them unharvested was the
    # same mistake as before -- gated twins that no stage reads -- so they get
    # their own promoted root, shard set and stream, and the mixture can weight
    # them later rather than being silently deprived of them now.
    #
    # No task list: these ids are the external pool's and were already narrowed
    # by --families when they were seeded; frontier_tasks.txt is registry ids
    # and would reject every one of them.
    if [ "$POOL_STREAMS" = "1" ]; then
        HIP_ROOTS="$FLYDSL_ROOT" \
        HIP_PROMOTED="$REPO/$POOL_FLYDSL_OK_ROOT" \
        HIP_DATA_ROOT="$REPO/$POOL_FLYDSL_DATA_ROOT" \
        HIP_SHARD_DIR="$REPO/$POOL_FLYDSL_SHARD_DIR" \
        HIP_TASK_IDS_OUT="$REPO/runs/pool_flydsl_tasks.txt" \
        HIP_TWIN_GLOBS='*__flydsl' \
        NO_SUBMIT=1 \
            bash "$REPO/scripts/hip_pool_harvest.sh" "$POOL_FLYDSL_SHARDS" 2>&1 \
            | grep -E "promoted|PARTITION|nothing to do" | tee -a "$LOG"
    fi

    # Frontier registry shards are static (the 482 ids do not change), so only
    # refresh their commit stamp -- a stale manifest kills every worker on the
    # preflight check with NonZeroExitCode and reads as a queue problem.
    for d in "$TWIN_SHARD_DIR" "$HARDPOOL_SHARD_DIR"; do
        [ -d "$REPO/$d" ] && "$PY" "$REPO/scripts/refresh_shards.py" "$d" >> "$LOG" 2>&1
    done

    # --- 4. staff mining across every declared stream -----------------------
    bash "$REPO/scripts/staff_datagen.sh" 2>&1 | tail -2 | tee -a "$LOG"

    sleep "$SLEEP"
done
