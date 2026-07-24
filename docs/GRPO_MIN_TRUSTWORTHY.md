# Minimum trustworthy GRPO profile

`configs/grpo_32b_min_trustworthy.json` is a semantic safety profile for the
first bounded 32B GRPO canary. It is not a launch manifest, a hardware sizing
claim, or evidence that the current trainer topology fits a 32B model.

The profile fails before model allocation unless:

- the explicit task list is non-empty, unique, registered, and contains no
  registry-held-out task;
- GRPO LoRA is off;
- every numeric sampling/search/accounting budget is valid;
- every requested feature has an active consumer in the selected runtime; and
- `requested_features` is exactly equal to `effective_features`.

## Deliberately disabled

The six audited research levers are disabled: AlphaKernel search, value
prefiltering, task minting, branch-and-bound, Opus regret, and same-run
“distillation.” The profile also disables coevolution, RC-GRPO, SC-GRPO, GTPO,
transform discovery/tools, incorrect-turn credit, physics shaping, live
counters, and adversarial coevolution. Strict validation rejects attempts to
turn on an audited lever before its accepted artifact and consumer contracts
exist.

StarPO-S, dynamic refill, and the AVSPO variance floor remain provisional. They
are not promoted by configuration alone: each has a required invocation canary
at the first rollout/update. A canary failure aborts the run.

## Registered curriculum

`RegisteredStratifiedScheduler` validates the immutable training set, computes a
SHA-256 task-set digest, and stratifies by `(operator_family, dtype)`. It serves
sorted strata round-robin and uses SHA-256 over the seed, stratum, epoch, and
task ID to permute each stratum without Python hash or rank-local RNG state.

In distributed mode only rank 0 selects. The selected task and post-draw
`CurriculumStateV1` are broadcast; follower ranks never make an independent
selection. The state contains the global draw index, per-stratum counts, seed,
scheduler version, and immutable task digest. Restoring it reproduces the exact
uninterrupted task suffix. Strict checkpoints emit `curriculum_state.json`.

## Budget and feature receipts

`BudgetLedgerV1` records these dimensions separately:

- generated assistant tokens and optimizer tokens;
- correctness calls, fresh timed calls, and replay hits;
- verifier and profiler GPU seconds;
- task groups attempted and kept; and
- per-feature invocation counts.

The ledger never treats a replay hit as a physical call and never assumes that
a timed call, correctness call, or profiler call implies another counter.
Strict runs emit stable `feature_manifest.json` and resumable
`budget_ledger.json` receipts. Distributed finalization sums physical rank-local
token/time counts while rank 0 owns logical attempted/kept group counts.

The shipped profile leaves `budget_limits` empty because safe 32B limits must
come from measured resource preflight, not model-name guesses. Empty limits
still produce complete accounting; launch materialization must add approved
hard caps before production.

## Remaining launch blockers

This foundation does not implement full optimizer/model/RNG checkpoint resume,
the versioned replay/evaluation contract, sandboxed verifier execution, or a
32B resource topology. Evaluation code must expose replay/fresh-call and GPU
time metadata before those ledger dimensions can be wired end to end. A launch
must also pass the separate model identity, resource, sandbox, verifier, and
trainer-resume gates. Do not infer 32B fit or launch readiness from this config.
