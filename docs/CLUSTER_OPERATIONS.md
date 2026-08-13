# Cluster operations (SPUR)

This document is for launching and supervising a training job on the SPUR
Slurm cluster, and for debugging it at 2am without rediscovering the lessons
below. It complements [`DISTRIBUTED.md`](DISTRIBUTED.md) (what the run trains
and why) with how the scheduler actually behaves. SPUR is a real-but-modified
Slurm controller (`Version=0.8.0`); several defaults and even some field
availability differ from stock Slurm, and every gotcha in this document was
found by hitting it, not by reading a manual.

## 1. The launch chain

| Script | Role |
| --- | --- |
| [`scripts/spur_sft_1node.sbatch`](../scripts/spur_sft_1node.sbatch) | The job itself: resolves the launch config, checks the node, trains, self-resubmits at the walltime boundary. |
| [`scripts/spur_resolve_launch_config.py`](../scripts/spur_resolve_launch_config.py) | Rewrites the shipped config's `model_id`/`output_dir` for the stage handoff; called by the sbatch script, not run standalone. |
| [`scripts/launch_distributed.sh`](../scripts/launch_distributed.sh) | Builds and execs the `accelerate launch` command; supports `--dry-run` to validate paths without touching the scheduler. |
| [`scripts/watch_and_resume.sh`](../scripts/watch_and_resume.sh) | The supervisor: submits, polls `squeue`, resubmits on preemption/dispatch failure/dirty node, decides completion from the run's own output. |
| [`scripts/sft_supervise_v5.sh`](../scripts/sft_supervise_v5.sh) | The entry point: wraps the supervisor around the correct `sbatch` command for the production 30B run (account, QoS, thresholds, output dir). |
| [`scripts/sft_launch_verify.sh`](../scripts/sft_launch_verify.sh) | Read-only health check against the job's log: startup, MoE routing, divergence, retention, progress, checkpoints. |

The chain is: `sft_supervise_v5.sh` execs `watch_and_resume.sh`, which calls
`sbatch` against `spur_sft_1node.sbatch`. Each submitted job resolves its own
config, trains, and either finishes, self-resubmits at its walltime boundary,
or exits for the supervisor to resubmit. `sft_launch_verify.sh` is a separate,
read-only tool you run against the log whenever you want a status snapshot.

### Current run (reference)

| Field | Value |
| --- | --- |
| Job | 9229 |
| Node | crsuse2-m2m-037 |
| Account / QoS | `amd-primus` / `amd-primus-qos` (guaranteed, non-preemptible) |
| Walltime | 7 days |
| Measured step time | ~60 s/step |
| Expected duration | ~29 h |

## 2. Launching a run

### Controller address, every time

`spurctld` does not run on the login node, and its client library defaults to
`http://localhost:6817`, which is wrong everywhere except a node that runs the
controller itself. `SPUR_CONTROLLER_ADDR` is normally exported by
`/etc/profile.d/spur.sh`, but that file is sourced only by an interactive
login shell. This account's default shell on SPUR is `csh`, which does not
source a POSIX-`sh` profile fragment at all even on login, and a
non-interactive `ssh host command` (cron, a detached supervisor, a service
manager) does not run a login shell in the first place. In both cases
`SPUR_CONTROLLER_ADDR` stays unset and every `scontrol`/`squeue`/`sbatch` call
fails against `localhost:6817`. Export it explicitly rather than trusting
shell startup:

```bash
export SPUR_CONTROLLER_ADDR="http://crs-m2m-cpu-spur-005:6817"
```

Every script in the launch chain exports this itself (with the same default),
but any manual `squeue`/`scontrol`/`sbatch` you type by hand needs it too.

### Recommended: supervised launch

```bash
nohup bash scripts/sft_supervise_v5.sh >/dev/null 2>&1 &
```

This is safe to re-run: `watch_and_resume.sh` looks for an in-flight
`kore-sft` job under your user before submitting, and adopts it instead of
starting a duplicate. Two concurrent trainers would write into the same
`output_dir` and corrupt each other's checkpoints (see "The single-trainer
lock" below), so this adoption check is what makes restarting the supervisor
after a disconnect safe rather than dangerous.

`sft_supervise_v5.sh` hardcodes the launch parameters that are not
guessable: account `amd-primus` with QoS `amd-primus-qos` (the pairing that
demonstrably schedules for this user; see §6), `KORE_OUTPUT_DIR` read from the
shipped config so the supervisor's completion check targets the right
directory, and the resubmission thresholds tuned for this run (`STALL_SECS`,
`MIN_PROGRESS_SECS`, `MAX_FAST_FAILURES`, `STUCK_PENDING_SECS=0`; see §5 and
§8 for why each is set the way it is). Its own log is
`runs/supervise_v5_<timestamp>.log`.

### Manual, unsupervised launch

For a one-off or a smoke test, submit the sbatch script directly:

```bash
sbatch --account=amd-primus --qos=amd-primus-qos \
    scripts/spur_sft_1node.sbatch configs/sft_coder30b_a3b.json - -
```

Positional arguments are `CONFIG FROM_STAGE OUTPUT_DIR [GENERATION]`. They are
positional rather than `--export` because SPUR's `sbatch` does not reliably
propagate `--export` into a job the launcher resubmits; positional arguments
are part of the job specification and survive that. `-` for `FROM_STAGE` or
`OUTPUT_DIR` keeps the config's own value; for the production run this means
training the pinned `Qwen/Qwen3-Coder-30B-A3B-Instruct` checkpoint into the
config's own `output_dir` rather than rewriting either. `GENERATION` is
internal bookkeeping (see below) and should not be passed by hand.

A job launched this way is not supervised: if it is preempted, dispatch-fails,
or lands on a dirty node, nothing resubmits it. Prefer the supervised path
unless you specifically want a single, unrecovered attempt.

### The single-trainer lock

A job can end up scheduled under more than one account/QoS pool for the same
`output_dir`: the pattern that motivated this lock is queuing the same run
under both `amd-general` and `amd-primus` so whichever pool frees first is
used. That means two jobs can legitimately start in the same scheduling
cycle. `spur_sft_1node.sbatch` takes an `mkdir`-based lock
(`<output_dir>/.kore_train.lock`) before training: `mkdir` is atomic even on
NFS, where test-then-write is not. The first job to create the directory
writes its job id inside and trains; a later starter that finds the lock held
prints `KORE_LOCK_HELD=<jobid>` to its log and exits 0 without touching
anything else.

The lock is judged stale, and taken over, only when its recorded holder is
genuinely gone from the queue or in a terminal state, not merely
non-`RUNNING`. That distinction exists because of a real incident: jobs 9201
(general) and 9229 (primus) started in the same cycle, and the original check
treated only `RUNNING` as "the holder is alive," so each job saw the other as
`CONFIGURING`/`PENDING`, judged the lock stale, and took it. Both then trained
into the same `output_dir` for about an hour, roughly 10 steps from writing
`checkpoint-50` on top of each other. Release is holder-checked (a job can
only remove a lock that records its own id), so a loser exiting cannot free
the winner's lock.

### Resume is automatic and keyed on `output_dir`

`sft.py` calls `latest_checkpoint(output_dir)` itself and walks candidates
newest-first, so a checkpoint left half-written by a mid-save kill is skipped
in favor of the newest complete one. Resuming a stage therefore needs only a
stable `output_dir` across attempts, which every path above preserves. There
is nothing to pass to resume a run; launching against the same `output_dir`
is resuming it.

### Two resubmission mechanisms, not one

Two independent things add a job to the queue under a preserved `output_dir`,
and their logs look similar but mean different things:

- **`resubmitted as generation N`** comes from inside the running job itself
  (`spur_sft_1node.sbatch`), fired only when its own walltime is about to
  expire (§4). `GENERATION` is carried as a positional argument and capped at
  `KORE_MAX_GENERATIONS` (default 12), so a job that somehow fails at startup
  every time cannot resubmit itself forever.
- **`submitted job=<id> (attempt N)`** comes from `watch_and_resume.sh`
  outside the job, fired whenever it sees the job leave the queue without the
  work being done (preemption, dispatch failure, a dirty node, a crash). This
  counter is `MAX_RESUBMITS` (300 for the production run) and is unrelated to
  the generation counter: an externally-resubmitted job starts back at
  generation 0, since the supervisor's `sbatch` call never passes a
  `GENERATION` argument.

A run that is both preempted several times and long enough to cross its own
walltime will show both counters advancing independently in the logs; neither
one is a proxy for the other.

## 3. Checking health

Quick queue status:

```bash
squeue -u "$USER" -o "%.8i %.10j %.8T %.10M %.6D %R"
```

For an actual verdict on whether the run is healthy, not just whether it is
`RUNNING`:

```bash
bash scripts/sft_launch_verify.sh [JOBID]     # defaults to your newest kore-sft job
```

`squeue` reports `RUNNING` identically for a job that is still loading the
model, one training normally, and one silently training on garbage.
`sft_launch_verify.sh` reads the job's own log
(`runs/sft-<jobid>.out`/`.err`) and works through the phases a healthy run
actually passes through, failing closed on the checks that matter:

| Section | What a FAIL means |
| --- | --- |
| Startup | A traceback or OOM anywhere in the log. |
| MoE routing | Router load entropy is collapsing (experts concentrating). |
| Divergence guards | Non-finite (NaN/Inf) loss. |
| Retention | Kernel loss falling while retained-capability eval losses climb: catastrophic forgetting, the risk this run was built to catch. |
| Progress / checkpoints | Informational only (step count, latest checkpoints); not a pass/fail gate. |

Anything the script marks `FAIL` is a reason to kill the job (§4) rather than
let it burn a multi-hour or multi-day allocation on a run that has already
gone wrong.

Other places to look:

- `runs/sft-<jobid>.out` / `.err`: the job's own stdout/stderr. `--open-mode=append` matters for the `scontrol requeue` fallback specifically (§2, §6): a requeue restarts the *same* job id, and append mode preserves the earlier attempt's output instead of truncating it. A normal `resubmit_self` generation is a *new* job id with its own log file; `grep 'resubmitted as generation' runs/sft-<jobid>.out` against the old log tells you which new id to look at next.
- `runs/sft-<jobid>.resolved.json`: the exact config the job trained, written by `spur_resolve_launch_config.py` before the model loads. This is the audit trail for what actually ran, independent of what you intended to pass.
- `runs/supervise_v5_<timestamp>.log`: the supervisor's own decision log, recording every state observed, every resubmission and why, and every node exclusion.

## 4. Stopping a run

**Kill the supervisor before you `scancel` the job.** `watch_and_resume.sh`
cannot distinguish an operator-initiated cancel from a preemption: both make
the job leave the queue without `run_completed` being true, and its default
response to that is "treating as interruption; resubmitting to resume" within
about 20 seconds. A `scancel` issued while the supervisor is still running
will be undone almost immediately by a fresh submission.

```bash
pkill -f sft_supervise_v5.sh          # or: pkill -f watch_and_resume.sh
scancel 9229
```

Killing the supervisor process does not touch the running job; it only stops
the loop that would resubmit after you cancel it. If you only want to stop
supervision and let the current job keep training unattended, killing the
supervisor alone is sufficient and `scancel` is not needed at all.

A clean `scancel` sends the job script `TERM`. The launcher's own handler
(`handle_termination`) forwards `TERM` to the whole process group, waits for
it to exit, and exits `143` itself; it does not attempt to save anything
extra, because the newest *complete* checkpoint is already on disk from the
last `save_steps` boundary (every 50 steps for the production config) and
that is what the next launch resumes from. Training since that checkpoint is
lost; nothing before it is.

## 5. Failure modes

Symptom first. Find what you are looking at, then read the cause and fix.

| Symptom | Cause | Fix |
| --- | --- | --- |
| Job state `JobLaunchFailure`, reason contains `dispatch confirmation failed: 0 of 1 nodes confirmed` | Random dispatch failure on this controller; not correlated with any particular flag combination. Measured: `--gres=gpu:mi355x:8` 0/3 dispatched head-to-head against `--gres=gpu:8` 2/3. | Under supervision this self-heals: `watch_and_resume.sh` cancels and resubmits within 30-120s. Unsupervised, `scancel` and resubmit yourself immediately; do not `scontrol release` (nothing is held) and do not wait (job 6530 sat unchanged across three poll cycles). |
| Job `PENDING`, `Reason=None`, `StartTime=N/A`, unchanged for hours while later jobs run | Usually a phantom account/QoS pairing: accepted by the controller, never a real scheduling candidate. Verify against the table in §6. | If the pairing is one of the rejected/unverified ones, cancel and resubmit under a verified pairing. If the pairing is verified, this is likely a slow cold start; the supervisor holds queue position by design (§8) rather than resubmitting a job that would lose its place. |
| Job `PENDING`, `Reason=QOSGrpNodeLimit` | The QoS's node cap is full team-wide. This is a real capacity answer, not a wedge. | Wait. The supervisor explicitly holds position on this reason. |
| Job `PENDING`, reason contains `JobHoldMaxRequeue` or `held`, `Priority=0` | SPUR parks some newly submitted jobs in a requeue hold before they ever run, independent of whether `--requeue` was used. | `scontrol release <jobid>`, then wait a full ~60s scheduler cycle before concluding it failed: a 45s check has reported still-held while the release was in fact landing. |
| Script dies within seconds, empty or near-empty stderr, log looks otherwise clean | `set -euo pipefail` plus a bare `grep` inside a command substitution: SPUR's `scontrol` omits fields like `TimeLimit`, `grep` finds nothing and returns 1, and that makes the *assignment* fail, which exits the script silently. This killed job 6520 in 12s. | Every such read in the current scripts already ends `|| true`. If you add a new `scontrol`/`squeue` field read inside `set -euo pipefail`, add `|| true` to it too. |
| Job exits 2, log says `FATAL expected 8 GPUs, found N` | A partial GPU allocation was granted (fewer than 8 visible devices), most likely from a GPU request that did not resolve to a full node. | Check the job's `--gres`; it should be `gpu:8`, not `gpu:mi355x:8` (see §6). |
| Job exits 3, log contains `KORE_BAD_NODE=<host>` | A previous tenant left GPU memory allocated on a card (measured once at ~270GB); the node passes a device-count check but fails the free-memory check. | Nothing to do manually: the supervisor adds the node to `--exclude` and resubmits immediately, and does **not** count this against the crash-loop limit. Confirm with `grep KORE_BAD_NODE runs/sft-<jobid>.err`. |
| Job `RUNNING`, `squeue` looks normal, but the log has not grown in tens of minutes | Training wedged, most often after the model load on a node with a hardware or driver issue the free-memory check did not catch, or a genuinely slow first dataset map. | The supervisor cancels and excludes the node once the log is stale past `STALL_SECS` (45 min for the production run). If checking by hand, compare log mtime against the current time before assuming a hang. |
| Log contains `KORE_LOCK_HELD=<jobid>`, job then exits 0 | This job started while another job (queued in the paired pool) already held the single-trainer lock on the same `output_dir`. This is by design, not a failure. | No action. The supervisor detects this and follows the holder job id instead of resubmitting. |
| Two jobs both training into the same `output_dir` at once | The single-trainer lock's staleness check used to treat only `RUNNING` as "the holder is alive," so a competing job starting in the same scheduling cycle saw the other as `CONFIGURING`/`PENDING`, judged the lock stale, and took it. Fixed: any state that means the job still exists now counts as live. | Should not recur. If it does, treat it as the lock logic regressing, not as ordinary preemption: stop both jobs and inspect `output_dir/.kore_train.lock/jobid` against `squeue`. |
| Supervisor logs `"<stage> completed; done"` and exits, but `output_dir` has no consolidated model | The supervisor's completion check has two layers: the authoritative one (`run_completed`: a consolidated model, or the launcher's `SFT_RC=0` sentinel) and a fallback that treats a Slurm `JobState=COMPLETED` as done. A job that self-resubmits at its own walltime boundary (§2, "Two resubmission mechanisms") exits `0` (a clean exit from Slurm's point of view) without ever printing `SFT_RC=0`. The fallback check would read that exit as "the stage is done" for a job that only rolled over into a new generation. | Grep the finished job's log for `resubmitted as generation`; if present, a new job id exists and needs its own supervision (relaunch `sft_supervise_v5.sh`, or supervise the new id directly). This has not been observed on the current run because a ~29h run under a 7-day limit never reaches its own walltime boundary; it is a latent risk if the walltime is ever shortened relative to the true run length. |

## 6. SPUR-specific gotchas

**Typed GPU requests do not dispatch.** `--gres=gpu:mi355x:8` is accepted by
the controller and then never confirmed by a node: the job sits in
`JobLaunchFailure`. Measured head-to-head, three trials each, otherwise
identical: `--gres=gpu:mi355x:8` 0/3 dispatched, `--gres=gpu:8` 2/3,
`--gpus-per-node=8` 0/3. Other users' running jobs show `TresPerNode=gpu:8/node`,
not the typed form; the likely reason is that SPUR advertises each node's
GPUs as eight separate `gpu:mi355x:1` entries rather than one `gpu:mi355x:8`.
All single-node launchers in this repo request `--gres=gpu:8`. The multi-node
launchers (`spur_sft_4node.sbatch` and the `spur_midtrain_*node*.sbatch`
family) still request the typed `--gpus-per-node=mi355x:8` and have not been
re-verified against this finding; treat a multi-node launch as unverified for
dispatch reliability until it is.

**Account and QoS pairings are not interchangeable, and there is no
`sacctmgr` to check them with.** A pairing can be *accepted* by `sbatch` and
never actually schedule. The only way to tell is to watch what really runs.

| Account + QoS | Status for this user | Evidence |
| --- | --- | --- |
| `amd-general` + `amd-general-qos` | Works | In production use (general pool). |
| `amd-primus` + `amd-primus-qos` | Works | Current run (job 9229). |
| `amd-primus` + `amd-general-qos` | Works | Verified directly. |
| `amd-general` + `amd-primus-qos` | Rejected | Controller refuses it outright for this account. |
| `amd-general` + `amd-burst-qos` | Accepted, never scheduled | `Reason=None`, `StartTime=N/A` for 9 hours while 42 later-submitted jobs ran. Of 60 running burst jobs sampled, zero used the `amd-general` account; the accounts actually running under burst were `amd-burst`, `amd-hyperloom`, `amd-aifw-dev`, `amd-collectives`, `amd-silo-tiger`, `amd-primus`. |
| `amd-burst` + `amd-burst-qos` | Added, still did not place | Even after ops added this user to the `amd-burst` account, a minimal 1-GPU/1-CPU/5-minute probe would not schedule. |

The distinguishing signal is `Reason=None` with `StartTime=N/A` holding for
hours while jobs submitted *after* yours start and run: that means the job is
not a real scheduling candidate at all, as opposed to `QOSGrpNodeLimit`, which
means the pool is genuinely full and you are legitimately queued (§7, §8).

**A requeued job is a trap, not a recovery mechanism.** On this controller, a
requeued job trips `JobHoldMaxRequeue` on its *first* requeue and is held
permanently: observed live in other users' jobs (guanchen, genesu12, benle,
suranjan, jingaiyu, zhuang12). `#SBATCH --requeue` is therefore deliberately
absent from every launcher in this repo, and `scontrol requeue` is never
issued as a recovery action. Recovery is always a fresh `sbatch` submission
against the same `output_dir`, which resumes from the newest complete
checkpoint (§2). The launcher keeps `scontrol requeue` only as a last-resort
fallback if a fresh submission itself fails (e.g. a full queue), on the
reasoning that a held job is still manually recoverable and a lost run is
not.

**`scontrol` omits fields stock Slurm scripts assume exist**, most notably
`TimeLimit` on `scontrol show job`. Any script that reads it must fall back
(the drain timer here tries `squeue -o %l` first, then `scontrol`, then a
config default) and must guard the read with `|| true` under
`set -euo pipefail` (see the job-6520 row in §5). `scontrol ping` also exits
`0` even when it cannot reach the controller at all (it prints `failed to
connect to spurctld` to its own output and still returns success), so exit
status alone never confirms an `scontrol`/`sbatch` call actually reached the
controller; check the printed text.

**No `sacctmgr`, no `--test-only`.** There is no tool on this cluster to list
valid account/QoS associations or their caps ahead of time, which is why the
table above exists and must be re-verified by observation (a real submission,
or a scan of who is actually running) rather than by a lookup command whenever
a new pairing is needed. `sbatch --test-only`, which on stock Slurm previews
placement without submitting, is not available either; the closest thing to a
dry run is `scripts/launch_distributed.sh --dry-run`, which validates config
and accelerate paths but cannot tell you anything about scheduling.

**Multi-node specifics** (relevant only if you use `spur_sft_4node.sbatch` or
a `spur_midtrain_*node*` launcher, not the current single-node run):
`scontrol` has no `hostnames` subcommand, but `SLURM_JOB_NODELIST` is already
a comma-separated list of real hostnames, so the rendezvous master is just its
first field. And an `sbatch` submitted with `--ntasks=N` runs the batch
*body* on all `N` tasks rather than once on the batch host, so each node's
copy must act as its own per-node worker (keyed off `SLURM_NODEID`) rather
than trying to `srun` a single driver: `srun` from inside a batch-body copy
fails with "job is not running" on this controller.

## 7. Interpreting pending reasons

| Reason (from `squeue -o %R`) | Meaning | What to do |
| --- | --- | --- |
| `Resources` | Waiting on hardware that legitimately is not free yet (e.g. a whole-node request with no fully idle node). | Wait. |
| `QOSGrpNodeLimit` | The QoS's node cap is full team-wide (§8). | Wait; the supervisor holds position on this reason specifically. |
| `None` / blank, `StartTime=N/A`, persisting for hours while later-submitted jobs start | The job is not a real scheduling candidate; check the account/QoS pairing against §6 first. | If the pairing is wrong, cancel and resubmit correctly. If the pairing is verified-good, this is most likely a long but legitimate queue for a whole-node request; the supervisor holds position by default (`STUCK_PENDING_SECS=0`) because resubmitting sends a merely-waiting job to the back of the queue. |
| Job state `JobLaunchFailure` (not a pending *reason*, but shows up the same way) | Random dispatch failure (§5, §6). | Cancel and resubmit immediately; do not wait, do not release. |
| Contains `JobHoldMaxRequeue` or `held` | Parked in a requeue hold, `Priority=0`. | `scontrol release <jobid>`, then wait a full ~60s cycle. |

## 8. QoS pools and capacity

| Pool | Node cap (team-wide) | Observed non-rotating holders | Measured churn |
| --- | --- | --- | --- |
| `amd-general-qos` | 8 nodes | 3 of 8 slots held by jobs with no near-term end (two `UNLIMITED`, one 15-day) | ~1 opening every 4h |
| `amd-primus-qos` | 16 nodes | 7 of 16 slots held similarly (three 14-day, two 30-day, two `UNLIMITED`) | ~1 opening every 4h |

The cap is in nodes and is independent of overall cluster load: idle nodes
elsewhere do not help once a pool's cap is full, because the cap governs how
many nodes *that QoS* may occupy, not how many nodes exist. A node's Slurm
state of `mix` means only that it is partially allocated (e.g. some but not
all GPUs/CPUs in use); it does not mean the node is reserved for some other
QoS, and filtering candidate nodes on state has previously discarded usable
nodes for no reason. With both pools' rotation this slow (roughly one opening
every four hours against several competitors ahead in each), holding a
position in a pool and waiting is usually cheaper than searching for
alternatives; see §6 for why cancelling and resubmitting into a "better"
queue can lose more time than it saves.

## 9. Node hygiene

Counting eight visible GPUs is not sufficient: a previous tenant's job can
end while still holding memory on a card (measured once at ~270GB), and the
only symptom used to be training wedging after the model load, caught only by
a 45-minute stall timer. `spur_sft_1node.sbatch` now checks free/total memory
on all eight devices before any weights load, and refuses the node (exit 3,
printing `KORE_BAD_NODE=<host>`) if any card is under 90% free. The 90%
threshold leaves room for driver/context overhead while still catching a leak
measured in the tens of gigabytes. `watch_and_resume.sh` adds that host to
`--exclude` and resubmits immediately, and, importantly, does not count a
bad-node exit against the crash-loop limit that stops a genuinely broken run
(§5, §10): a dirty node is a cluster-hygiene problem, not a bug in this
project's code.

## 10. Supervisor behavior summary

For reference, the rules `watch_and_resume.sh` follows, all covered in more
detail above:

- Adopts an in-flight `kore-sft` job under your user rather than submitting a
  duplicate.
- Holds queue position on any pending reason by default
  (`STUCK_PENDING_SECS=0`): because queue position is ordered by submit time,
  cancelling and resubmitting a merely-waiting job sends it to the back.
- Resubmits immediately (30-120s backoff) on `JobLaunchFailure`, since that
  failure is random rather than persistent and such a job holds no queue
  position worth preserving.
- Decides completion from the work itself (a consolidated model in
  `output_dir`, or the launcher's `SFT_RC=0` sentinel, checked only against
  job ids this supervisor actually started or adopted) rather than from
  Slurm accounting, because an empty `sacct` lookup would otherwise make a
  finished run look preempted and trigger a pointless resubmission (see the
  `resubmit_self`/`COMPLETED` caveat in §5's table).
- Stops after 3 consecutive fast failures (a job dying within
  `MIN_PROGRESS_SECS`, 600s for the production run) so a reproducible crash is
  not retried up to the resubmit cap.
