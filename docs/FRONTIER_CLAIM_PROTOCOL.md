# Frontier model-versus-system claim protocol

`kore.eval.frontier_protocol` is the offline adjudication layer for a frontier
claim. It does not run a benchmark. It accepts a signed, preregistered artifact
from benchmark adapters and returns a versioned report.

The claim is always relative to the frozen task manifest, comparator revisions,
hardware, verifier, and budgets in that artifact. “Best in class” must not be
shortened to an unrestricted “best” claim.

## Two required tracks

Every profile requires both tracks. One cannot substitute for the other.

1. `closed-model-harness`
   - Compares two model checkpoints.
   - Prompt, tool, verifier, hardware, context policy, and every budget dimension
     must have identical fingerprints.
   - This isolates the model contribution under a common harness.
2. `open-system-harness`
   - Compares two fully disclosed systems.
   - Verifier, hardware, context policy, and every budget dimension must match.
   - Prompt and tool fingerprints may differ because they are part of the system
     under test; both remain immutable and report-visible.

A skipped track produces missing confidence intervals and is a categorical
failure, including for the `development` profile.

## Frozen identities and budgets

`Fingerprint` stores both canonical JSON and its SHA-256. Validation recomputes
the digest. Each arm freezes these identities:

- artifact and checkpoint;
- tool and prompt;
- verifier and hardware;
- budget.

`BudgetEnvelope` is a per-task ceiling with exact matching across arms for:

- output tokens;
- context-policy fingerprint;
- tool calls;
- correctness calls;
- fresh timed calls;
- profiler calls;
- GPU-seconds;
- wall-time seconds;
- cost in USD.

Each sample also records `ResourceUsage`. Exceeding any ceiling fails the
artifact. Giving the candidate a larger envelope fails even if it did not use
the extra capacity.

## Denominator and canonical metric

The preregistered claim unit is every `task × run` cell. Every run must contain
the complete ordered task manifest. A missing cell:

1. contributes zero to both arms rather than disappearing from the denominator;
2. creates a coverage integrity error, so no profile can pass.

The canonical point estimate is the existing KernelBench-style metric against
`vendor-production`:

```text
fast_p = count(hidden_correct AND vendor_time / candidate_time > p)
         / count(all preregistered task × run cells)
```

The report deliberately has two different fields:

- `canonical_fast_p`: the point estimate;
- `certified_lower_ci_fast_p`: the lower endpoint of the preregistered
  hierarchical bootstrap interval.

The lower endpoint is not substituted into the point-estimate field. Claims
should quote both.

Hidden correctness is the scoring verdict. Public and hidden correctness rates
are reported separately, and disagreements produce warnings. Incorrect,
untimed, invalid, or omitted outputs remain in the full denominator as zeros.

## Baselines and the SoL ceiling

Every sample must report four independently identified, hidden-correct
baselines:

- `vendor-production` — the canonical fast_p denominator;
- `best-vendor`;
- `compiler`;
- `eager`.

They remain separate in `baseline_fast_p`; they are not collapsed into a
best-of-baselines number after seeing outcomes.

`sol_time_ms` is different. It is a physical lower bound sourced and hashed per
sample. Any candidate, comparator, or baseline timing below SoL fails integrity
and cannot score. The report exposes `sol_attainment = sol_time / observed_time`,
capped at one, as a physical-ceiling attainment metric. SoL is intentionally
absent from `BaselineKind` and can never be framed as a comparator that a system
“beats.”

## Inference and multiplicity

The primary paired outcome for each cell is whether the arm is hidden-correct
and clears canonical `fast_p` at the profile threshold. The bootstrap samples:

1. runs with replacement;
2. families within selected runs with replacement;
3. tasks within selected families with replacement.

The candidate/comparator pair always travels together. The resulting report
contains arm intervals, a paired-delta interval, standard error, and one-sided
bootstrap p-values for preregistered non-inferiority and superiority boundaries.

Primary non-inferiority and superiority hypotheses are corrected together with
Holm’s step-down family-wise correction. Passing a gate requires both:

- the paired lower confidence bound to clear its preregistered margin; and
- the Holm-adjusted p-value to clear alpha.

The following are reported as secondary sensitivity tests and receive their own
Holm correction family:

- exact paired McNemar;
- exact or deterministic Monte Carlo paired permutation;
- Wilcoxon signed-rank from `kore.eval.paired_stats`.

They do not replace the preregistered primary gate.

The geometric-mean candidate/comparator speed ratio on both-correct cells is
descriptive only. It is marked `gate_eligible: false`; whenever cells are
excluded, the report emits a survivor-bias warning. No claim gate is conditioned
on both-correct survivors.

## Preregistered profiles

The immutable schemas live in `CLAIM_PROFILES`; artifacts store the selected
schema fingerprint.

- `development`
  - 90% CI, alpha 0.10, 2,000 bootstrap draws.
  - At least 4 tasks, 2 families, and 2 complete runs per track.
  - Both tracks must be non-inferior within an absolute fast_p margin of 0.10.
- `frontier-competitive`
  - 95% CI, alpha 0.05, 10,000 draws.
  - At least 50 tasks, 5 families, and 3 complete runs per track.
  - Both tracks non-inferior within 0.02; at least one track superior.
- `best-in-class-model`
  - 95% CI, alpha 0.05, 20,000 draws.
  - At least 100 tasks, 8 families, and 5 complete runs per track.
  - Closed-model track superior; open-system track non-inferior within 0.01.
- `best-in-class-system`
  - Same sample and inferential requirements as `best-in-class-model`.
  - Open-system track superior; closed-model track non-inferior within 0.01.

All use canonical `fast_1`. A new threshold or margin requires a new,
fingerprinted profile schema rather than a runtime override.

No profile can pass with smoke or fallback data, contamination, an unsigned or
unverified artifact, a skipped comparator track, incomplete evidence, non-finite
inputs, budget mismatch, an SoL violation, or missing confidence intervals.

## Artifact and report evidence

`FrontierArtifact` (`kore.frontier-claim/v1`) contains:

- the ordered task manifest and its preregistered hash;
- both track specifications and run IDs;
- all paired task samples;
- exact source URLs and immutable revisions;
- raw trace and normalized sample SHA-256 hashes;
- per-baseline and SoL source evidence;
- a detached artifact signature binding the full payload.

`FrontierReport` (`kore.frontier-report/v1`) repeats the artifact payload hash,
profile, fingerprints, raw evidence manifest, baseline-specific scores,
inference, corrected tests, gates, warnings, and integrity errors.
`ArtifactSignature.verified` is an ingestion trust boundary: this pure module
checks the binding and status but does not perform key discovery or
cryptographic verification itself.

## Benchmark adapters still required

This module intentionally makes no live model, network, GPU, or profiler calls.
A real publication run still needs adapters that:

1. execute the closed model harness and export exact token/tool/cost accounting;
2. execute and package both disclosed open systems;
3. convert KORE-Bench/KernelBench task manifests into `TaskManifestEntry`;
4. run the hidden verifier and retain public/hidden verdict traces;
5. collect fresh candidate, vendor-production, best-vendor, compiler, and eager
   timings on fingerprinted hardware;
6. derive and source task-specific physical SoL bounds;
7. resolve source URLs to immutable revisions and hash raw/normalized samples;
8. cryptographically sign the final payload and set `verified` only after trust
   policy succeeds.

Adapters must emit data; they must not make statistical or denominator choices.
Those decisions belong to the preregistered profile and this adjudicator.
