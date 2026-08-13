# Source provenance and decontamination

This document covers two questions about the data specified in
`docs/DATASET_SPEC.md`: where its content originates, and how it is screened so
that a kernel the model is evaluated on never reaches training in another form.
The build process that consumes these sources is `DATAGEN_OVERVIEW.md`. A short
historical note on the retired 14B midtrain corpus closes this document.

## Where a v5 kernel's underlying task comes from

Every kernel row traces to one of three task sources.

| Source | What it is | Scale |
| --- | --- | --- |
| Task registry | hand-authored and mechanically generated tasks, each with its own `reference.py` oracle, driver, and shapes | 1,546 tasks |
| Mined pool | `torch.nn.Module` definitions mined from `GPUMODE/KernelBook` and a synthetic operator-composition pipeline | 13,570 admitted |
| FlyDSL ecosystem | human-written FlyDSL kernels from the DSL's own tree and 21 further public/internal repositories | 1,582 distinct kernels, up from 130 |

**The mined pool** is admitted by `kore/data/task_mining.py` through five ordered
gates, each independently sufficient to drop a candidate: the module must import
only from an allowlist and never call out of process (safety); it must classify
into exactly one canonical operator family from the torch operators it actually
calls, not its class name (classification); it must clear the decontamination
check described below; it must run deterministically at a shape large enough to
be an optimization target rather than a launch-overhead measurement, and its
oracle must be a function of its declared inputs alone, not of hidden module
state such as an `nn.Linear`'s randomly-initialized weights (execution); and it
must not structurally or semantically duplicate another admitted module or a
registry task source (dedup). Only the PyTorch reference side of a mined module
is used; any accompanying Inductor-generated Triton is NVIDIA-targeted and is
never ingested.

**The FlyDSL ecosystem** exists because FlyDSL is roughly a quarter of the
evaluation benchmark and there is no public FlyDSL dataset (a search across five
terms on Hugging Face returned nothing). `scripts/v5_stage2b_flydsl.py` mines
the DSL's own test suite, examples, and documentation, plus 21 further
repositories where FlyDSL kernels turned out to live inside unrelated production
codebases (AMD's AI Tensor Engine, an inference runtime, a profiling library)
rather than in dedicated FlyDSL projects, which is why a GitHub repository search
found roughly a tenth of the eventual supply. AMD's own production FlyDSL
library is deliberately excluded wholesale, because it is the corpus the
benchmark's FlyDSL tasks are drawn from (below); the ecosystem repositories are
screened by filename and by normalized kernel body against every named FlyDSL
benchmark task so a vendored copy under a different name still gets caught.

## Screening against the evaluation benchmark

Training on the corpus described above without checking it against
AgentKernelArena, the benchmark the project's results are reported on, would
mean training on the answer key. Before this project's `kore/data/arena_index.py`
existed, no such check ran anywhere in the repository: the existing
decontaminator indexed KernelBench and the project's own held-out task sources,
and the arena runner scored a candidate with no reference to contamination of
any kind. The exposure was not hypothetical: the mined pool and the arena's own
`gpumode` sub-suites both descend from the same `GPUMODE/KernelBook` scrape.

`scripts/v5_build_arena_index.py` builds a frozen table, scoring every mined
pool task against every arena task that ships a parseable PyTorch source, and
records each pool task's single best match. As measured against the current
arena checkout: 355 of the arena's tasks have a parseable PyTorch source (68 do
not, mostly `instruction2triton` and `image_kernel` tasks with no reference
module to score against, and are reported as unscreened rather than treated as
clear); scored against the 13,570-task mined pool, 6 pairs are byte-identical
after normalization and 24 pool tasks score at or above the blocking threshold
and are excluded from training.

Two comparisons are combined, because they fail in different directions.
**Exact normalized-AST identity** catches byte-identical modules regardless of
length and cannot be defeated by renaming or reformatting. **Document-frequency-filtered
shingle Jaccard** catches near-duplicates: every `nn.Module` shares a
`class X / __init__ / super().__init__()` skeleton, which on a small module is
most of the document, so unfiltered similarity rates unrelated three-line
wrappers at 0.44. Shingles that appear in more than 0.5% of the pool are
structural boilerplate rather than content and are dropped before scoring, and
the blocking threshold, 0.30, sits in the empirical gap between the lowest true
match (0.333) and the next-highest coincidence.

Operator identity is deliberately not treated as contamination. The arena's
KernelBench suite is textually clean but asks for GELU, softmax, and LayerNorm,
and the pool holds 27 distinct GELU modules; a model cannot be trained to write
kernels without training on GELU. What this screen catches is a shared *source
document*, not a shared *operation*.

## The backend-suffix bug

A twin directory carries a suffix naming its backend, for example
`genb_attn2_cross_gqa_step_fp16__hip`. The held-out classifier that decides
whether a record may enter training matches its near-generalization probe set
and its contaminated-task list by exact task id, and a record not found in the
registry defaults its own lineage root to its own task id. Neither lookup
recognizes a suffixed id, so a suffixed twin of a held-out probe classified as
trainable: 44 twin directories covered 39 of the 43 near-generalization probes,
and all 11 tasks already marked contaminated, all reachable through their
suffixed form. This is a strictly worse failure than ordinary contamination,
because the held-out probe set is the yardstick the project's zero-shot claims
are measured against.

The fix, in `kore/data/v5_policy.py`, canonicalizes a task id to its
base (suffix-stripped) form before any held-out or contamination lookup, and
runs that check unconditionally rather than after the classifier's own verdict,
so a `train` answer for a suffixed id can never be reached without first
clearing the same guard the base id would face. `scripts/v5_verify.py`'s
`heldout_probe_leak` and `contaminated_task_leak` gates re-check both forms on
every build so a regression here fails the pipeline rather than the benchmark.

## Cross-root and near-duplicate provenance

Two independent fingerprinting mechanisms, both pure and stdlib-only
(`kore/data/dedup.py`), are used at different points in the pipeline.
**Structural fingerprinting** parses a kernel as Python, strips docstrings,
alpha-renames local identifiers by first occurrence (while keeping module
attribute paths like `tl.dot` distinct from `tl.load`), and hashes the result,
catching a kernel that recurs with a renamed variable or a reflowed comment.
**Semantic-graph fingerprinting** additionally omits constants and local names
and hashes call targets, operators, and control-flow structure, catching a
near-duplicate that differs only by a tuned constant. `kore/data/task_mining.py`
seeds both fingerprint sets with every registry task's own source before
admitting a single mined module, so a mined duplicate of a hand-authored task is
caught at admission rather than later. Within the v5 build itself, exact-hash
deduplication across the 13 gathered data roots removes 81,550 rows that were
independent re-generations of the same task under different roots, a
consequence of the datagen resume ledger being scoped to a single root rather
than of any content actually differing.

## Reward-hacking removal

A kernel that quietly delegates to a torch operator instead of performing the
computation itself will pass a harness that grades on numerical agreement alone,
and training on it teaches the fallback. Every serious kernel-RL paper
independently converged on filtering this class of output: Kevin zeroes the
reward for `torch.nn`, `try`/`except`, or `pass` in the output; Kernel-Smith adds
a dedicated hacking category; Dr. Kernel calls it lazy optimization; GEAK
maintains a banned-operator list. `kore/data/v5_emit.py`'s `cheats` function
implements the same idea for v5: it flags a torch call that performs the
operation the kernel exists to perform, an exception-swallowing pattern, or a
stub body, while explicitly permitting the torch calls a legitimate launcher
needs (buffer allocation, dtype and device plumbing) so those are never
mistaken for a hack. Applied during stage 4's sanitize pass, this removes 2,806
targets that would otherwise have taught the fallback.

A separate, deliberately manufactured mechanism (`kore/data/hard_negatives.py`)
generates nine labeled reward-hack variants of a known-correct kernel (copying
the oracle, calling a vendor library, delegating to torch, silently swallowing
a failure, recycling an unwritten output buffer, computing only one tile,
special-casing a hardcoded shape, skipping all compute, and a weak accumulator
dtype) for DPO preference training. Those pairs are a preference-learning
artifact of the legacy v3/v4 recipe, not a v5 SFT source, and the 2,806 figure
above does not include them.

## Decontamination against KORE's own held-out tasks and public benchmarks

`kore/data/decontam.py` provides the general-purpose held-out and benchmark
decontamination framework that `task_mining.py` calls during pool admission
(above) and that the legacy midtrain corpus builder (below) uses directly. Its
checks run in a fixed order, each with a stable, auditable reason: held-out
family, task, source, lineage, or time-partition membership; declared ancestry
from a held-out root; exact SHA-256; normalized Python AST; normalized semantic
graph; MinHash near-duplicate; and directional containment within one held-out
lineage cluster. Containment divides shared signal by the size of the held-out
reference, never the candidate, so a held-out kernel pasted into an otherwise
unrelated long document is not diluted into passing. Common Triton import,
decorator, coordinate-setup, load, and store idioms are removed from this fuzzy
evidence beforehand, so those idioms alone can never manufacture a false
contamination hit against a legitimate training kernel.

## General-replay source provenance

The replay half of v5 (`docs/DATASET_SPEC.md`) is loaded by
`kore/data/general_replay.py` from named, versioned Hugging Face sources, each
with an explicit fallback if the primary is unavailable offline: code from
`nvidia/OpenCodeInstruct` (falling back to `ise-uiuc/Magicoder-Evol-Instruct-110K`),
math and reasoning from `open-thoughts/OpenThoughts3-1.2M` (falling back to
`nvidia/OpenMathInstruct-2`), chat and instruction-following from
`allenai/tulu-3-sft-mixture`, and tool-use from `Team-ACE/ToolACE` (falling back
to `Salesforce/xlam-function-calling-60k`). Two further tool-use corpora,
`Agent-Ark/Toucan-1.5M` and `NousResearch/hermes-function-calling-v1`, are
re-rendered by `kore/data/tooluse_normalize.py` into the model's own nested-XML
tool-call surface rather than passed through in their native schema; an earlier
pass over `Toucan-1.5M` that skipped this step kept zero of 12,000 rows, because
the dataset's `messages` field is a JSON string rather than a list, its tool-call
bodies are Python `repr` output rather than JSON, and its role vocabulary
differs by config. Every replay slice is passed through the same held-out
n-gram decontamination as the kernel side, plus the retention evaluation
benchmarks (MMLU, HumanEval, LiveCodeBench, IFEval, BFCL, MT-Bench) themselves,
so a mined replay row cannot carry a held-out kernel or a retention benchmark
question into training.

This repository's own license terms for each named dataset and repository are
not re-derived in this document; consult `THIRD_PARTY.md` for the project's
third-party attribution ledger.

## Historical note: the retired 14B midtrain corpus

`kore/data/midtrain_corpus.py` and `kore/data/decontam.py`'s frozen-benchmark
machinery were built for a continued-pretraining (CPT) corpus for a 14B model.
Production does not build or consume this corpus: the 30B target has no Base
sibling for CPT to run against, and the recorded 14B experiment
(`docs/EVAL_RESULTS.md`) showed that CPT on an instruct checkpoint destroys
instruction-following. A production build of that corpus required a full frozen
benchmark-text artifact, the actual pinned model tokenizer, and verified source
roots with an immutable commit and license for every input
(`docs/source_metadata.schema.json` is the schema that catalog was validated
against); none of that machinery runs as part of the v5 build described in
`DATAGEN_OVERVIEW.md`. It remains in the repository, unused by production,
because the decontamination primitives it introduced (`kore/data/decontam.py`)
are the ones `task_mining.py` now calls for pool admission.
