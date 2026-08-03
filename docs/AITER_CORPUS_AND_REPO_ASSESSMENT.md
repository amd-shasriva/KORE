# AITER as a corpus, and as a repository-scale target

Companion to [`AITER_WEAKNESS_SURFACE.md`](AITER_WEAKNESS_SURFACE.md), which
covers AITER as a *baseline*. This document answers two different questions:
is AITER worth ingesting as training data, and what would it take for our
pipeline to optimize inside the real upstream tree.

Everything here is [source]: read off the AITER checkouts at the commits pinned
in the companion document. No performance claims are made.

License: **MIT**, © Advanced Micro Devices, Inc. Permissive for use,
modification and redistribution; the copyright notice must travel with any
substantial portion. Anything derived from AITER carries that attribution.

## Verdict on AITER as a corpus

**Worth ingesting, but only a specific slice, and not the slice the framing
suggests.** The value is *not* in the hand-written assembly, and the reason is
categorical rather than a judgement call.

### The assembly is not ingestible, because it is not in the repository

AITER contains **zero** `.s`, `.S`, or `.asm` files. The hand-written assembly
ships as **2,867 pre-assembled ELF code objects** (`.co`, 117 MB) under `hsa/`,
alongside a `codegen.py` and CSV tuning tables. There is no assembly source
text to read, and therefore none to train on.

One could disassemble the objects with `llvm-objdump`, but that would produce
machine output, not the engineers' reasoning: no register-allocation intent, no
comments, no scheduling rationale, no names. Training on it would teach a model
to imitate a disassembler's formatting. The premise "ingest AMD's hand-written
assembly" cannot be executed against this repository at all.

### What is actually readable, and what each part is worth

| Slice | Size | Verdict |
|---|---|---|
| Triton kernels (`aiter/ops/triton`) | 348 files | **Ingest.** See below. |
| Tuning tables (`aiter/configs/*.csv`) | 10,806 rows, **86% gfx950** | **Ingest, transformed.** |
| `op_tests` | 344 files, 49/127 Triton tests carry an eager reference | **Ingest as pairs.** |
| CK C++ templates (`.cuh`, `.cu`) | ~139k lines | **Skip.** Template-heavy, low signal per token, and not our output language. |
| Prebuilt asm (`hsa/**.co`) | 2,867 objects | **Cannot ingest.** Binary. |

### Why the Triton slice is the one that matters, and why it is not duplicative

Another agent already ingested HipKittens (`kore/data/hipkittens.py`). That
module made a deliberate and correct call: HipKittens kernels are C++ that
`#include "kittens.cuh"`, so training them as `FULL_KERNEL` responses would
teach the model to answer with code the harness cannot compile — negative
transfer that looks like clean data. It therefore emits knowledge-QA rows
teaching transferable reasoning, not kernel bodies.

AITER's Triton slice is the complementary case, and this is the whole argument
for ingesting it: **it is written in the language our harness actually
compiles.** Our tasks declare `backend: triton` and the policy contract expects
a self-contained Triton kernel. AITER's Triton kernels are the vendor's own
Triton, tuned for our exact target arch.

Self-containment is the obvious objection, so it was measured rather than
assumed: 264 of 348 Triton files import something from `aiter`. But the
dependencies are thin infrastructure, not framework entanglement:

| Import | Files | What it is |
|---|---|---|
| `utils.logger.AiterTritonLogger` | 73 | logging; delete |
| `utils._triton.kernel_repr.make_kernel_repr` | 66 | a `__repr__` helper; delete |
| `utils._triton.arch_info` | 40 | arch string lookup; inline |
| `utils._triton.pid_preprocessing` (`pid_grid`, `remap_xcd`) | 40 | **the XCD swizzle** — small, substantive, inlinable |
| `utils.gemm_config_utils.get_gemm_config` | 22 | reads the tuning CSVs |

19 files depend on `aiter` *only* for the repr helper. So unlike HipKittens
C++, a large fraction of these kernels can be made self-contained by deleting
logging and inlining two small helpers. That is a mechanical transform, not a
rewrite.

`remap_xcd` deserves a specific mention: it is the XCD swizzle, the same
technique AITER's `opus_gemm` credits to HipKittens. It is a real, transferable
gfx950 optimization that fits in a few lines — exactly the kind of pattern that
is worth teaching and cheap to teach.

### The form: pairs, not dumps

Raw kernel bodies dumped into a chat corpus teach little, and the useful signal
is the *relationship* between a naive implementation and the tuned one. AITER
supplies both halves for a meaningful subset: **49 of 127 Triton `op_tests`
contain an eager torch reference next to the AITER call**, because that is how
the tests assert correctness. That is a naive→optimized pair with a built-in
correctness oracle, extracted rather than authored.

The tuning CSVs should be transformed, not dumped. `gfx,cu_num,B,M,N,K,
kernelId,splitK,us,kernelName,tflops,bw` is AMD's own measured autotuning
result on our target arch — the schema teaches *which tile shape wins at which
problem shape*, which is a real skill. Emitted as raw table rows it would teach
memorization of a table that goes stale; emitted as selection-reasoning it
teaches the skill. Note the timings in those CSVs are **AMD's measurements on
their machines**, not ours, and must never be relabelled as our numbers.

### The blocking precondition: decontamination

This is the part that would sink the effort if it were skipped. **63 of our
1,334 tasks use an `aiter*` comparison baseline.** For any task whose baseline
resolves to an AITER *Triton* kernel, training on that kernel's source is
training on the answer, and a subsequent win over that baseline would be
contaminated rather than earned.

So AITER Triton ingestion must be decontaminated against the aiter-baselined
task set specifically, not just against the usual benchmark holdouts. The
repository already has `kore/data/decontam.py` and a holdout-family mechanism,
so this is a wiring requirement rather than new machinery — but it is a
precondition, not a follow-up.

### Provenance, and a catalog discrepancy worth knowing

`data/release/meta/source_metadata.json` already lists an `aiter` source at
commit `028756633e4192785217838f4924dc16516f5780`, MIT, `verified: true`. Two
things follow.

First, that commit is **not** the installed build we measure
(`7e0d11626`) nor the upstream clone (`702aacd6`). Three AITER commits are in
play on this machine and any ingestion must state which one it read.

Second, **27 of the 28 source roots in that catalog no longer exist on disk**
(`repos/` has been removed entirely; only `kore-repo` resolves). This is a
pre-existing, catalog-wide condition rather than anything specific to AITER,
but it means the recorded commit is a historical assertion that cannot
currently be re-verified against a tree. Any new AITER ingestion should record
a path that actually exists.

## Repository-scale capability: the gap

AgentKernelArena ships five repository-scope AITER tasks under
`tasks/repository/aiter`: `mla_decode_rope`, `pa_decode`, `pa_prefill`,
`unified_attention`, and `moe_routing_sigmoid_top1_fused`. Each is a
`config.yaml` plus a per-task `task_runner.py` adapter (inside the AKA
checkout, not this repo) exposing `compile`, `correctness`, and `performance`,
writing a `performance_report.json` with `execution_time_ms` entries.

Two properties make these more tractable than they first look. They target
**pure-Triton** AITER paths and run with `ENABLE_CK=0`, so there is no
Composable Kernel submodule build; and each runner creates its own
`--system-site-packages` venv, so dependency setup is the task's problem rather
than ours.

### What we have

`kore/eval/agent_kernel_arena.py` already parses `config.yaml`, reads
`task_type`, and runs compile → correctness → performance **gated in that
order**, scoring with AKA's own policy. The gating is the important part and it
is already right: timing a kernel that failed correctness would reward the
shortcut we filter out.

### What is missing

The gap is narrower than "build repository-scale optimization" suggests, and it
is concentrated in one place: **workspace construction**.
`evaluate_task(task, workspace, ...)` takes an already-populated workspace as an
argument. Nothing in our tree performs AKA's
`src/preprocessing.py::setup_workspace()` role, which for a repository task
must clone `https://github.com/ROCm/aiter.git` into the task directory and then
duplicate the fully-populated folder into a per-run workspace so each attempt
starts clean.

Concretely, to attempt one of these tasks we would need:

1. **A workspace builder**: clone-or-reuse upstream aiter at a *pinned* commit,
   copy the task folder in, duplicate per attempt. Pinning matters more here
   than anywhere else — an unpinned clone makes the baseline drift between the
   attempt and the measurement.
2. **A disk and time budget**: a full aiter clone is ~650 MB before submodules,
   per task and potentially per attempt. `/home` is shared and at 79%.
3. **An agent loop that edits a file in a tree** rather than emitting a
   self-contained `FULL_KERNEL` block. This is the real conceptual gap: our
   policy contract produces one kernel body, while a repository task requires
   locating and modifying `aiter/ops/triton/attention/unified_attention.py`
   in place. That is a different action space, not a bigger version of the
   same one.

Items (1) and (2) are plumbing, on the order of a day. Item (3) is the
substantive one and should not be estimated from this document.

### Why it is still worth doing

Kernel-Smith's credibility did not come from a benchmark row; it came from
merged upstream PRs. An accepted PR into `ROCm/aiter` would outweigh a
leaderboard result, and the weakness surface gives a specific, defensible
candidate that needs no new kernel at all: the dense/varlen `deterministic`
default asymmetry documented in the companion file. If the measurement confirms
that a default `flash_attn_func` backward is landing on the CK fallback while a
shipped asm kernel sits unused, that is a dispatch bug with a small fix and a
clear argument — the kind of change an upstream maintainer can evaluate
quickly.

That PR should not be opened until the behaviour is measured on hardware. The
argument depends on a claim about performance, and right now we have a claim
about code.
