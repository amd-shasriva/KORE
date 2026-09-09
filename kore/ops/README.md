# `kore/ops` — the operational control plane

Launchers and supervisors need to start a long job, watch it, decide whether it
finished, and sometimes kill it. Every one of those steps has a way of going
wrong that a training bug cannot cause and a training test cannot catch: a
signal delivered to the wrong process, a state file swapped for a symlink, a
half-written checkpoint read as a completed one. This package is where those
concerns live, and it is deliberately separated from everything that trains.

Stdlib only — `os`, `signal`, `fcntl`, `subprocess`, `hashlib`, `json`,
`pathlib`. No torch, no accelerate, no network. A supervisor has to keep working
when the training environment is exactly what is broken.

## A PID never authorizes a signal

This is the rule the package exists to enforce. PIDs are reused. A recorded pid
that has been recycled since the record was written names somebody else's
process, and `os.kill` on it is indistinguishable from `os.kill` on ours.

So `capture_process_identity` records five things beyond the pid, and
`identity_matches` re-derives all of them from `/proc` immediately before any
signal is sent:

| Checked | Why it is not redundant |
| --- | --- |
| start time (field 22 of `/proc/<pid>/stat`) | The one field a recycled pid cannot reproduce. |
| uid | Refuses to signal another user even if everything else lines up. |
| pgid | The unit actually signalled; `terminate_owned` uses `killpg`, not `kill`. |
| cgroup | Catches a process that moved scope — a different container or slice. |
| `KORE_RUN_ID` | Inherited from the environment `OwnedProcess.spawn` set. Marks the process as *this run's*, not merely ours. |

`OwnedProcess.spawn` sets `KORE_RUN_ID` in the child environment and
`start_new_session=True`, then polls until the child owns its own process group
and carries the marker, so the identity it persists describes a process that is
already isolated. Termination is `SIGTERM` to the group, a bounded wait, then
`SIGKILL`, and it re-checks group membership on every poll: if a foreign uid, a
foreign run marker, or a changed cgroup appears inside the group, it stops and
returns `refused=True` rather than escalating. Refusing to kill is a valid
outcome here, and `TerminationResult` reports it distinctly from "stopped".

## Filesystem state that cannot be swapped underneath you

`SecureRuntime` roots private state below `$KORE_RUNTIME_DIR`,
`$XDG_RUNTIME_DIR/kore-ops`, or `<tmp>/kore-ops-<uid>`. Directories are created
mode 0700 and re-checked for ownership after the `chmod`, because an unusual
umask can otherwise leave them group-readable. Every state file is opened with
`O_NOFOLLOW`, verified as an owned regular file of mode 0600 through the *file
descriptor* rather than the path, and written by `mkstemp` + `fsync` +
`os.replace` + a directory `fsync`, so a reader never observes a partial write.
Path components are validated against a conservative name pattern and `..` is
rejected outright.

Two operations are worth naming because they encode a decision:

- `consume_sentinel` renames the sentinel to a per-pid claim *before* reading
  it, so two supervisors racing for the same signal cannot both act on it.
- `store_task_set` writes a task set once and raises `SecurityError` if a later
  call disagrees. A campaign's train/eval split must not drift mid-run, and
  silently rewriting it is how it would.

`IncrementalLogReader` reads only bytes appended since the previous call,
resetting on rotation or truncation and buffering a partial trailing line. That
last detail is what stops a supervisor counting the same `ERROR` twice because
it polled mid-write.

## "Done" is an artifact question, not an exit code

`verify.py` is side-effect-free and answers one question: did the thing that was
supposed to be produced actually get produced? A zero exit code does not answer
it — a job that self-resubmits at its walltime boundary also exits zero.

`verify_model_artifact` requires a real directory (not a symlink) holding a
config and final weights, and fails on any leftover `*.inprogress` or `*.tmp`
marker. `verify_task_shards` walks a task set, validates every shard as JSONL of
objects, and counts *distinct* `final_source` values against a target, so a
shard full of one repeated win does not read as coverage. `verify_campaign`
dispatches per stage and refuses stages it has no strict verifier for, rather
than passing them by default. `verify_sft_gate` additionally checks that the
manifest's recorded checkpoint is the one being handed to the next stage.

`SupervisorStateMachine` is the small piece that ties an exit code to that
verdict: `RUNNING → VERIFYING → SUCCEEDED` only when the return code is zero
*and* the artifact status is ok; otherwise `WAITING` while attempts remain, then
`GAVE_UP`. Illegal transitions raise instead of being absorbed.

## Modules

| Module | Lines | What it holds |
| --- | ---: | --- |
| [`runtime.py`](runtime.py) | 945 | Process identity and ownership, `SecureRuntime`, `SecureFileLock`, `OwnedProcess`, `IncrementalLogReader`, `SupervisorStateMachine`, `deprecated_entrypoint`. |
| [`__main__.py`](__main__.py) | 336 | `python -m kore.ops run / status / stop / verify ...` — the same primitives for shell callers, JSON on stdout, meaningful exit codes. |
| [`verify.py`](verify.py) | 326 | The strict artifact verifiers above. |
| [`campaign.py`](campaign.py) | 308 | `CampaignSupervisor`: bounded attempts around one owned child, log-driven stage/error/gate alerts, refuses to start beside a previous run that is still alive. |
| [`__init__.py`](__init__.py) | 40 | Re-exports the runtime surface. |

`deprecated_entrypoint` lives in `runtime.py` because it is the enforcement half
of `scripts/operations_registry.json`: a script the registry marks
`lifecycle: deprecated` exits 64 unless `KORE_ALLOW_DEPRECATED_DEV=1` is set
exactly. `tests/test_docs_contract.py` checks the other half — that no document
presents such a script as the way to run something.

## Nothing in `kore/` imports this, on purpose

Zero `kore/` source files import `kore.ops`. Its callers are 13 scripts —
`scripts/kore_supervise.py`, `scripts/run_sft_gate.py`,
`scripts/lib/ops_runtime.sh` and the rest of the supervisor and factory
wrappers — plus `tests/test_ops_runtime.py` and `tests/test_ops_verify.py`.

The separation is the design, not an accident of layering.
`kore/env/kore_env.py` needs a lock and says so in its own comment: it
deliberately inlines one rather than importing `SecureFileLock`, because this
package is fail-*closed* and raises `SecurityError` when something looks wrong,
which is right for a supervisor deciding whether to send a signal and wrong for
an environment that should degrade rather than abort a training step. Importing
`kore.ops` into the training path would put that fail-closed behaviour on the
hot path where it does not belong, and would make a torch-free control plane
depend on torch by transitivity.

See [`../policy/README.md`](../policy/README.md) for what the training stages
themselves do.
