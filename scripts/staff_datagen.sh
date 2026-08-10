#!/bin/bash
# Keep several datagen streams staffed at once, within the QoS job cap.
#
# One stream is not enough for the dataset we need. Mining only pool-HIP grows HIP
# volume but leaves two gaps that no amount of it closes:
#
#   pool-Triton   the 13,570 KernelBook tasks have never been mined for Triton, so
#                 all Triton data is registry ops. Mining the same task ids we mine
#                 for HIP is also the only way translation pairs ever appear: the
#                 reshaper emits a dialect-to-dialect row only where one op won in
#                 both backends.
#   registry-HIP  the 171 unmined registry tasks. HIP wins today are 7 elementwise
#                 ops while hip2hip scores 38%, so this is the quality gap rather
#                 than the volume one.
#
# Streams are declared with a share of the available slots rather than a fixed
# count, so the split holds whether the cap grants us three slots or eight.
#
#   scripts/staff_datagen.sh
set -uo pipefail

REPO=/home/shasriva/Kore-RL/KORE
cd "$REPO" || exit 1
[ -z "${SPUR_CONTROLLER_ADDR:-}" ] && [ -r /etc/profile.d/spur.sh ] && . /etc/profile.d/spur.sh
# shellcheck disable=SC1091
. "$REPO/scripts/gpu_slots.sh"

LOG="$REPO/runs/staff_datagen.log"
say() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" >> "$LOG"; }

# One staffing pass at a time, across every caller.
#
# Deciding to submit is a read-then-write against the queue: gpu_free() and
# covered_shards() ask the scheduler what exists, and the submission happens
# afterwards. Nothing makes that atomic, so concurrent passes all read the same
# answer and all act on it. Ten copies of the pipeline loop did exactly that --
# each logged "1/3 staffed, 2 free -> adding shard(s) 0 2" in the same second,
# each submitted shard 0, slept 3s, submitted shard 2 -- and twenty jobs landed
# against an eight-job cap, on two shards, in three seconds.
#
# Holding the lock for the whole pass makes a second pass observe the first
# one's submissions rather than race them. Skipping rather than waiting is
# deliberate: a queued pass would submit against a queue snapshot that is
# already stale by the time it wakes, which is the bug again with extra steps.
exec 9>"$REPO/runs/.staff_datagen.lock"
if ! flock -n 9; then
    say "another staffing pass holds the lock; skipping this one"
    exit 0
fi

#: name | shard dir | data root | wanted elements | job names that serve it
#:
#: The name list matters: jobs submitted before per-stream naming existed are still
#: called kore-factory and are still mining pool-HIP. Not counting them made the
#: stream look understaffed, and the top-up landed a second worker on a shard
#: another job was already grinding through.
#: Wanted counts sum to one less than the job cap, leaving a slot for the arena --
#: which the supervisor owns, not this script. Sizing them to the cap instead left
#: the arena unable to launch; sizing them below it left slots idle, which on a pool
#: this contended is the more expensive mistake.
#: The default must match what ensure_loops.sh configures, because this script
#: is also run by hand -- and when it was left pointing at pool-HIP, a manual
#: pass silently staffed four miners onto the stream that had just been retired,
#: against shards whose ledger we were deliberately no longer growing.
#: hardpool is the second HIP stream, and it is where token parity comes from.
#:
#: frontiertwins is the best data there is -- vendor baselines, fp8 and bf16,
#: attention and MoE -- and it is also nearly finite: 250 tasks at ~103k tokens
#: each is a ceiling around 25M, against a Triton side of 370M. Mining it
#: harder does not close that; there is nothing left to mine.
#:
#: The pool does have the volume, and the reason it was retired was never the
#: pool as such -- it was mining all 6,457 of it, where the median baseline is
#: 17us and a kernel that finishes in 17us has no tiling or MFMA scheduling to
#: demonstrate. select_frontier_tasks already scores those tasks; it was only
#: ever asked for its registry half. Asking for the pool half at --min-score 2
#: keeps attention, MoE, quant and GEMM at a million elements or more -- 1,857
#: tasks, all already gated, none of them yet mined -- and drops the
#: launch-bound remainder. At the 152k tokens/task the earlier pool-HIP mining
#: actually produced, that is ~283M tokens, which is what parity needs.
#:
#: 2 miners stay on frontier and 4 go here: frontier is the higher-value data
#: per task and must not stall, but it cannot absorb more than a couple of
#: workers without them fighting over the same 250 tasks.
#:
#: hipreg and poolflydsl are retired to 0, measured against the frontier list:
#:
#:   frontiertwins  224 tasks  185 HIP + 39 FlyDSL  100% frontier
#:   hipreg         171 tasks                         1% frontier
#:   poolflydsl     226 tasks                         0% frontier
#:
#: hipreg reads runs/unmined_hip.txt, which is not the registry frontier its
#: name suggests -- it is hip_abs_fp16, hip_add_relu_bf16, hip_div_fp32, the
#: generated elementwise set. It had mined 3,250 rows, more than any other
#: stream, which made the largest part of the corpus its least difficult part.
#: poolflydsl is the right dialect and the wrong difficulty: kbk_actor,
#: kbk_mlp, kbk_classifier at fp32, scraped modules whose baseline is eager
#: torch rather than AITER or hipBLASLt.
#:
#: Both stay listed at 0 rather than being deleted so their ledgers survive and
#: either can be revived by changing one number.
#: Two balanced streams, three miners, and the arena is the fourth job.
#:
#: frontierhip merges both HIP sources -- the 211 registry frontier twins and
#: the 1,857 hard pool tasks -- and interleaves them across attention, MoE,
#: quant and GEMM so any prefix is balanced. Mining them as separate streams
#: gave 45% attention on one and 94% GEMM on the other, because that is what
#: each source happens to hold.
#:
#: frontiertriton finishes the registry frontier in Triton: 128 of its 482
#: tasks are mined, and triton2triton is 38% of the arena with a 15-point gap
#: to Opus, so the remaining 354 are worth one miner until they run out.
#:
#: frontiertwins and hardpool go to 0 because their tasks are inside
#: frontierhip now; mining both would double-cover them.
STREAMS="${DATAGEN_STREAMS:-\
frontierhip:runs/shards_frontierhip:data/v5frontierhip:2:kore-mine-frontierhip \
frontiertriton:runs/shards_frontiertriton:data/v5frontier:1:kore-mine-frontiertriton \
frontiertwins:runs/shards_frontier_twins:data/v5frontier_twins:0:kore-mine-frontiertwins \
hardpool:runs/shards_hardpool:data/v5hardpool:0:kore-mine-hardpool \
poolflydsl:runs/shards_pool_flydsl:data/v5pool_flydsl:0:kore-mine-poolflydsl \
hipreg:runs/shards_hipreg:data/v5hip:0:kore-mine-hipreg \
poolhip:runs/shards_hippool:data/v5hippool:0:kore-mine-poolhip+kore-factory \
frontier:runs/shards_frontier:data/v5frontier:0:kore-mine-frontier \
pooltriton:runs/shards_pooltriton:data/v5pooltriton:0:kore-mine-pooltriton}"

#: Arena arms this script must not crowd out.
#:
#: "One less than the cap" above assumed a single arm. There are two, plus a
#: gate, so six miners filled all eight slots -- and the arena cannot staff
#: itself out of that: the supervisor submits it, and only when a slot is free.
#: An arm's allocation ends every 8 hours, and its slot is then free for exactly
#: as long as it takes the next staffing pass to take it. After that the
#: supervisor finds the cap full and, having no branch for it, silently does
#: nothing. The v4 arm sat dead for 50 minutes that way while its node stood
#: reserved for it.
#:
#: The hold-back counts only the arms that are *absent*, so a running arena
#: costs nothing here and the slots go to mining as before.
#:
#: An arm that has *finished* is not absent, it is done, and holding a slot for
#: it would idle one of eight forever. Completion is the summary file, which is
#: the same signal the supervisor stops on.
ARENA_JOB_NAMES="${ARENA_JOB_NAMES:-kore-aka:runs/aka_full_v4/results_v4.json \
kore-aka-base:runs/aka_base/results_base.json \
kore-aka-opus:runs/aka_opus/results_opus.json}"

arena_reserve() {
    local n=0 spec name done_marker
    for spec in $ARENA_JOB_NAMES; do
        name="${spec%%:*}"; done_marker="${spec#*:}"
        [ "$name" = "$done_marker" ] && done_marker=""
        [ -n "$done_marker" ] && [ -f "$REPO/$done_marker" ] && continue
        [ "$(_squeue -t R,PD -n "$name" -o '%i' | wc -l)" -eq 0 ] && n=$(( n + 1 ))
    done
    echo "$n"
}

#: The gate needs a slot too, and it is the only route by which a new seed
#: becomes mineable at all. Mining will otherwise expand to the cap and the gate
#: waits behind it -- which is how 1,500 seeds sat ungated. Held only while
#: there is something to gate and no gate already in flight, so a caught-up
#: pipeline still gives every slot to mining.
GATE_JOB_PREFIX="${GATE_JOB_PREFIX:-kore-gate-}"

gate_reserve() {
    [ "$(_squeue -t R,PD -o '%j' | grep -c "^$GATE_JOB_PREFIX")" -gt 0 ] && { echo 0; return; }
    local root
    for root in ${GATE_ROOTS:-data/registry_hip_frontier data/registry_flydsl_frontier data/pool_flydsl}; do
        [ -d "$REPO/$root/tasks" ] || continue
        # Anything seeded but not yet in this root's verdicts is gateable.
        if [ "$(ls "$REPO/$root"/tasks 2>/dev/null | wc -l)" -gt 0 ]; then
            echo 1; return
        fi
    done
    echo 0
}

# Count by job name, which the scheduler knows for a job the moment it is
# submitted. Counting by reading each job's log missed every pending job, because
# a job that has not started has written nothing -- so a stream already fully
# queued looked empty and got submitted again on the next pass.
#
# Queue position is expensive here: burst runs 35+ jobs deep, so a job cancelled
# and resubmitted goes to the back. Nothing in this script cancels; it only tops a
# stream up, and only when the scheduler says that stream is genuinely short.
staffed_for() { _squeue -t R,PD -n "kore-mine-$1" -o "%i" | wc -l; }

# Submit to the pool that can actually start the job.
#
# Every mining submission went to burst, because that is the default in
# gpu_slots.sh and nothing here overrode it. Burst is the large pool but it is
# persistently saturated -- 125 nodes running against 55 queued -- so a
# replacement worker for a stream that had just lost its node sat queued for
# hours, and the stream stayed dead the whole time. Both pool-Triton and
# registry-HIP died that way twice in one night.
#
# amd-general-qos is small, 8 nodes shared with every other user, and it is the
# only other pool this account may submit to (amd-primus-qos is refused
# outright: "not permitted for user under account amd-general"). So prefer it
# when it has headroom, and fall back to burst when it does not -- a queued
# burst job is still better than no job.
# Two limits, not one. Free capacity says whether a job *can* start there; the
# self-imposed cap says whether it *should*. General is 8 nodes for all 363
# users on this filesystem, and the two arena arms already sit there because the
# eval is the one thing whose progress has to be observable. Letting mining take
# every slot that happens to be free would leave nothing for anyone else, so
# mining is held to three and the remainder falls back to burst.
# pick_qos lives in gpu_slots.sh so the gate can use it too -- it could not
# reach this copy, was submitted to burst by default, and sat there for an hour.
#
# Two miners, not three. general is 8 nodes for everyone who uses it, and both
# arena arms have to live there as well, so three miners made five of the eight
# mine and left the gate to fall back to burst -- where it waited behind 35
# other jobs and never ran. A third miner buys a third more rows from tasks
# that are already gated; the slot it costs is the only way any *new* seed
# becomes mineable at all, and 1,500 of them were waiting on it.
GENERAL_MINE_MAX="${GENERAL_MINE_MAX:-4}"

# A shard manifest records the commit it was partitioned at, and the worker refuses
# to mine a shard whose code has moved -- a deliberate guard, but it means every
# commit invalidates every manifest. Left unhandled, submissions die instantly with
# NonZeroExitCode and the stream looks like it is merely waiting in the queue: that
# is exactly how pool-Triton produced nothing for an entire afternoon while three
# fixes landed on top of its manifest.
# Delegate to Python. This was shell, and the shell was the bug: reading the
# manifest with a here-document nested inside a command substitution mis-parsed
# twice -- once shifting fields so n_shards came back wrong, once failing outright
# with "here-document delimited by end-of-file" -- and both times rebuilt a
# seven-shard layout as one shard, making every array index above 0 illegal.
# scripts/refresh_shards.py parses JSON with a JSON parser and refuses to
# re-partition on values that cannot be right.
refresh_if_stale() {
    "${KORE_PY:-/home/shasriva/kore-venv/bin/python}" \
        "$REPO/scripts/refresh_shards.py" "$1" >> "$LOG" 2>&1
}

#: Shard indices a stream already has covered.
#:
#: Two sources, because neither alone is complete. A running worker prints the shard
#: it took, which is authoritative -- the scheduler cannot answer this at all, it
#: reports "?" for the array index and exposes neither ArrayTaskId nor the submit
#: command. But a job submitted seconds ago has printed nothing yet, and treating it
#: as uncovered puts the next pass on the same shard: that is exactly how two
#: pool-Triton workers both ended up on shard 000. So the index is also recorded at
#: submit time and believed for any job still in the queue.
_CLAIMS="${_CLAIMS:-$REPO/runs/shard_claims.tsv}"

claim_shard() { printf '%s\t%s\n' "$1" "$2" >> "$_CLAIMS"; }

covered_shards() {
    local names="$1" j nm live
    live=" $(_squeue -t R,PD -o "%i" | tr '\n' ' ') "
    {
        for j in $(_squeue -t R,PD -o "%i"); do
            nm=$(_squeue -j "$j" -o "%j" | head -1)
            case " $names " in *" $nm "*) ;; *) continue ;; esac
            grep -aoE "START job=$j array=[0-9]+ .*shard=[0-9]+" \
                "$REPO/runs/spur-$j.out" 2>/dev/null | tail -1 |
                grep -oE "shard=[0-9]+" | cut -d= -f2 | sed 's/^0*//;s/^$/0/'
        done
        # Claims made at submit time, kept only while the job is still queued so a
        # finished job stops reserving its shard.
        if [ -f "$_CLAIMS" ]; then
            while IFS=$'\t' read -r cj cs; do
                case "$live" in *" $cj "*) ;; *) continue ;; esac
                nm=$(_squeue -j "$cj" -o "%j" | head -1)
                case " $names " in *" $nm "*) echo "$cs" ;; esac
            done < "$_CLAIMS"
        fi
    } | sort -un
}

for spec in $STREAMS; do
    IFS=: read -r name dir root want names <<< "$spec"
    names="${names//+/ }"
    [ -d "$REPO/$dir" ] || { say "$name: no shard dir $dir; skipping"; continue; }
    # A retired stream still had its manifest rebuilt on every commit, which
    # re-partitioned 13,570 pool-Triton tasks and 6,457 pool-HIP ones each pass
    # to keep shards nobody was going to mine current.
    [ "$want" -le 0 ] && continue
    refresh_if_stale "$dir"

    nsh=$("${KORE_PY:-/home/shasriva/kore-venv/bin/python}" -c "
import json;print(json.load(open('$REPO/$dir/manifest.json')).get('n_shards',0))" 2>/dev/null)
    case "$nsh" in ''|*[!0-9]*) say "$name: n_shards unreadable; skipping"; continue ;; esac
    [ "$nsh" -lt 1 ] && { say "$name: n_shards=$nsh; skipping"; continue; }

    have=$(_squeue -t R,PD -n "${names%% *}" -o "%i" | wc -l)
    for extra in ${names#* }; do
        [ "$extra" = "$names" ] && break
        have=$(( have + $(_squeue -t R,PD -n "$extra" -o "%i" | wc -l) ))
    done
    free=$(gpu_free)
    reserve=$(( $(arena_reserve) + $(gate_reserve) ))
    if [ "$reserve" -gt 0 ]; then
        free=$(( free - reserve )); [ "$free" -lt 0 ] && free=0
        say "$name: holding $reserve slot(s) for the arena/gate; $free usable"
    fi
    # An idle reserved node is capacity the cap does not describe. GPU_JOB_CAP
    # counts every job I hold, so four miners stuck in burst for eight hours
    # spent the whole allowance without running anything, and the loop then
    # declined to submit to nodes that were sitting idle under my own name.
    res_free=$(mine_res_free)
    if [ "$res_free" -gt 0 ]; then
        free=$(( free + res_free ))
        say "$name: $res_free idle node(s) on $KORE_MINE_RESERVATION; $free usable"
    fi
    need=$(( want - have )); [ "$want" -gt "$nsh" ] && need=$(( nsh - have ))
    [ "$need" -gt "$free" ] && need=$free
    [ "$need" -le 0 ] && continue

    # Pick shard indices nobody is on. Submitting by count assumed shards 0..have-1
    # were covered, which is false as soon as any job sits on a different index --
    # two workers then raced through the same 922 tasks while four shards went
    # untouched.
    busy=" $(covered_shards "$names" | tr '\n' ' ') "
    picked=""
    i=0
    while [ "$i" -lt "$nsh" ] && [ "$(printf '%s' "$picked" | wc -w)" -lt "$need" ]; do
        case "$busy" in *" $i "*) ;; *) picked="$picked $i" ;; esac
        i=$(( i + 1 ))
    done
    [ -z "${picked// /}" ] && { say "$name: all $nsh shard(s) already covered"; continue; }

    say "$name: $have/$want staffed (shards busy:${busy%  }), $free free -> adding shard(s)$picked"
    for idx in $picked; do
        # One element per submission: a range would re-queue indices that are
        # already covered, and there is no way to express a gap in an array range.
        # The hold first, then general, then burst.
        #
        # This was the other way round on the theory that general is shared and
        # use-it-or-lose-it. The theory was wrong about what a free general slot
        # is: on this cluster a node sits idle precisely because it cannot
        # launch -- 36 burst jobs were queued against 18 free nodes, so anything
        # healthy is taken within seconds and the remainder is broken. Sending a
        # job to "free" general capacity put it on one of those, and it burned
        # three submissions in a row on JobLaunchFailure while a node I hold sat
        # idle. Reserved nodes are the ones already proven to launch.
        qos_arg="$(mine_res_arg)"
        [ -z "$qos_arg" ] && qos_arg="$(pick_qos kore-mine- "$GENERAL_MINE_MAX")"
        # shellcheck disable=SC2086
        out=$(sbatch $qos_arg --job-name="kore-mine-$name" --array="$idx-$idx" \
            scripts/spur_datagen_array.sbatch \
            "$REPO/$dir" "$REPO/$root" 3 run 2>&1)
        say "  shard $idx via ${qos_arg#--qos=}"
        echo "$out" >> "$LOG"
        jid=$(printf '%s' "$out" | grep -oE '[0-9]+$' | tail -1)
        # Claim the index immediately: the job will not print its own shard for a
        # minute or more, and until then the next pass would pick it again.
        [ -n "$jid" ] && claim_shard "$jid" "$idx"
        sleep 3
    done
done
exit 0
