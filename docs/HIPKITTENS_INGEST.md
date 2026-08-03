# HipKittens ingestion: CDNA4 kernel knowledge as SFT

## Attribution and licence

This slice is derived from **HipKittens**, © 2024 HazyResearch (Stanford), released
under the **MIT Licence**.

- Repository: <https://github.com/HazyResearch/HipKittens>
- Paper: *HipKittens: Fast and Furious AMD Kernels*, MLSys 2026,
  [arXiv:2511.08083](https://arxiv.org/abs/2511.08083)
- Authors: William Hu, Drew Wadsworth, Sean Siddens, Stanley Winata,
  Daniel Y. Fu, Ryann Swann, Muhammad Osama, Christopher Ré, Simran Arora

The MIT terms require the copyright notice and permission notice to travel with
substantial portions of the work. Every generated row therefore carries a
`_provenance` block with the repository URL, the exact commit, the licence, the
licence holder, the paper, the author list, and the specific source files the row
draws on. `kore/data/hipkittens.py` **verifies** the licence at ingest time rather
than asserting it: if `LICENSE` stops looking like the MIT grant, ingestion raises
instead of producing rows, because a corpus that claims MIT because one of our own
constants says MIT is not auditable.

The checkout itself is read-only and lives outside the repo (default
`$HOME/third_party/HipKittens`, overridable with `KORE_HIPKITTENS_ROOT`). Nothing
here vendors HipKittens source into this repository.

## Why this asset, and why it is shaped this way

HipKittens is the fastest published AMD kernel library on MI355X, and the reason
it matters to us is not the code — it is that the knowledge making the code fast
is largely unpublished. The paper states that the bank-conflict-avoidance
behaviour the library relies on is *undocumented in the CDNA ISA*. That is
precisely the knowledge our product model lacks on the `hip2hip` and `torch2hip`
categories, where there is no Triton codegen ceiling to hide behind.

Two design decisions follow from that, and both are deliberate.

### No teacher model writes these rows

Every other generator in `kore/data/` asks a frontier teacher for prose and then
verifies a number against a solver (see `scripts/build_sft_v3_mixture.py`'s inputs
and the Tier-1 kernel-math path). That pattern is **backwards for this asset**.
The premise of ingesting HipKittens is that frontier models do not know this
material in depth, so a frontier teacher cannot be the source of truth for it and
would produce fluent, confident, wrong statements about CDNA4 LDS behaviour that
we would have no way to check.

Every factual claim in these rows is therefore one of:

1. **Extracted** from the checkout — swizzle formulas, wave counts, scheduling
   intrinsics, kernel source, tile layouts.
2. **A measurement the authors committed to their own repo** — the LDS bank/phase
   solver outputs and the benchmark JSON — quoted with the file it came from.
3. **A paper claim**, carried verbatim in `PAPER_CLAIMS` together with its
   attribution string, so no downstream reader has to work out whether a number is
   ours.

There is no code path in `kore/data/hipkittens.py` that invents a speedup, and
`tests/test_hipkittens_ingest.py` asserts that any row stating a multiplier also
states where the multiplier came from.

### The rows are not the `FULL_KERNEL` contract

`kore/policy/format.py`'s `SYSTEM_PROMPT` trains the model to emit a
self-contained ROCm/Triton kernel inside a `FULL_KERNEL:` block, which the
environment then compiles and benchmarks. HipKittens kernels are C++ that
`#include "kittens.cuh"`, a header that does not exist in the eval harness.

Training HipKittens source as a `FULL_KERNEL` response would teach the model to
answer optimization requests with code that cannot build — negative transfer that
would look like clean, well-provenanced data on every metric we track. So these
rows deliberately mirror the existing `kernel_qa` slice instead: a knowledge
persona, `[system, user, assistant]`, teaching the transferable *reasoning* rather
than the library's API surface. A test pins this.

## Row taxonomy

The goal is a model that can **apply** these techniques to a kernel it has not
seen, not one that can recite the library. Eleven row types, grouped by what they
teach:

| `_qa_type` | Teaches |
|---|---|
| `hk_lds_bank_model` | The measured bank count and conflict-phase partition for one DS instruction width |
| `hk_lds_bank_asymmetry` | That the effective bank count is a property of the *instruction*, not the LDS — reads and writes disagree |
| `hk_swizzle_derivation` | The XOR swizzle for one shared-tile layout, and what each term displaces |
| `hk_swizzle_contrast` | Why two layouts of the same dtype need different swizzle constants, and why the pattern must not be extrapolated |
| `hk_bank_conflict_exercise` | The *procedure*: offsets → banks → group by phase → count. Worked with solver-computed ground truth |
| `hk_schedule_selection` | Choosing 8-wave ping-pong vs 4-wave interleave, and why wave specialization loses on CDNA |
| `hk_pattern_apply` | The ping-pong loop skeleton with the role of every barrier and intrinsic, plus a port checklist |
| `hk_kernel_anatomy` | One real kernel: what it computes, its schedule, the evidence for that schedule, the techniques it depends on |
| `hk_intrinsic_role` | One scheduling/memory intrinsic: what it is, when to reach for it, how it fails when misused |
| `hk_measured_baseline` | Realistic expectations from the authors' own committed benchmark numbers |
| `hk_naive_vs_hk` | Structural before/after: the ordered set of changes separating an obvious GEMM from a state-of-the-art one |

The `hk_naive_vs_hk` "before" listing is authored for that row, is explicitly
labelled as not being HipKittens code, and carries no performance number, because
we did not measure it.

## Verification

"Bank-conflict free" would otherwise be an assertion we copied. Instead:

1. Swizzles are **parsed** out of `st_shape.cuh`, not transcribed, so the rows
   cannot drift from upstream silently.
2. Parsing **fails loud**. A branch containing a `^` that yields no parsed term
   raises. This is not hypothetical: the first parser silently returned zero XOR
   terms for `st_16x128`, whose modulus is written `(16*128)`, which would have
   taught an identity swizzle for fp8 — a wrong answer no row-count test could
   see.
3. Every parsed swizzle is checked to be a **bijection** over the tile's byte
   offsets. A non-bijective swizzle aliases two elements onto one LDS address and
   corrupts data rather than merely slowing it down.
4. Each swizzle is then run against the authors' **measured** bank count and phase
   partition. All five non-identity swizzles are confirmed to take their layout
   from a 2- or 4-way conflict to exactly conflict-free:

   | layout | dtype | plain conflict degree | swizzled |
   |---|---|---|---|
   | `st_16x32` | bf16 | 2 | 1 |
   | `st_32x16` | bf16 | 2 | 1 |
   | `st_32x32` | bf16 | 4 | 1 |
   | `st_16x16_swizzled` | bf16, 4 B/lane | 2 | 1 |
   | `st_16x128` | fp8 | 4 | 1 |

5. Schedule and technique labels are **evidence-based**: a kernel is only
   described as using a pattern when that pattern is present in its source, and a
   kernel with no detectable pattern is reported as unclassified rather than
   guessed. The resulting labels reproduce the paper's split without being told
   it — ping-pong for GEMM and attention forward, 4-wave interleave for the
   imbalanced backward, wave specialization only in the comparison micros — which
   is the check that the detectors are reading the source correctly.

## Building the slice

```bash
git clone https://github.com/HazyResearch/HipKittens.git ~/third_party/HipKittens
PYTHONPATH=. python scripts/build_hipkittens_sft.py
```

Writes `data/b05factory/sft/hipkittens.jsonl` and
`data/b05factory/sft/hipkittens_report.json`. Exit status is non-zero if any row
is contaminated, so it can gate a build.

Three gates run before anything is written, because each catches something the
others miss: contamination (held-out task ids, `record_family` attribution, and
signal-n-gram overlap, the same three checks as
`scripts/audit_decontamination.py`), cross-corpus near-duplicate containment
against the existing mixture, and the 17,408-token length limit that
`scripts/build_sft_v3_mixture.py` enforces.

## Sizing it into the mixture

The slice is **small and dense** — a few dozen rows, because that is how much
non-redundant knowledge the repository actually contains. Inflating it by
templating the same lesson over every near-identical kernel would reproduce the
exact failure this ingestion is meant to avoid, so a near-duplicate gate drops
rows that restate an already-kept row (calibrated at 0.6 containment against a
measured spread where true clones sit at 0.9–1.0 and genuinely distinct rows at
0.2–0.6).

Consequently this slice cannot move behaviour on row count alone against a
~61k-row base, and should be **upsampled** when mixed rather than added once. The
current measured numbers live in `data/b05factory/sft/hipkittens_report.json`;
read them from there rather than pinning them in prose, since they change whenever
the upstream checkout does.
