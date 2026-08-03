# What it would cost to reach FlyDSL and repository-level tasks

KORE emits two kernel languages: Triton (1,334 tasks) and HIP C++ (188 tasks).
AgentKernelArena additionally contains 111 FlyDSL tasks and 9 repository tasks,
and none of them is reachable today. This document is the measured cost of
changing that, and the recommendation that follows from it.

Everything below is a fact from this host, not an estimate, unless it is labelled
as an estimate. Where a number came from a command, the command is given.

## Summary

| Axis | Reachable today | Env/ABI cost | Real cost | Verdict |
| --- | --- | --- | --- | --- |
| FlyDSL | no | ~1 line + tests | authoring seeds in an undocumented DSL | **Do not build now.** No published bar exists to beat, and the only corpus we may legitimately learn from targets a different architecture. |
| Repository | no | a second verifier contract | minutes-per-episode datagen, no SNR/paired-timing protocol | **Do not build as a training family. Do build one as a credibility artifact**, outside the RL loop. |

Both verdicts are "no" for the training pool, but for opposite reasons: FlyDSL is
cheap to wire and buys nothing checkable; repository work buys the thing that
actually persuades and is expensive to wire.

## FlyDSL

### What it is, and the prior it corrects

The working assumption going in was that FlyDSL is "a lot of surface for a DSL
only AgentKernelArena uses". The first half is right and the second half is
false, which changes the reasoning even though it does not change the answer.

FlyDSL is a real, installed, public package:

* `flydsl` 0.2.2 is already in this venv, Apache-2.0, 269 MB, and downloadable
  from PyPI (`pip download flydsl==0.2.2` succeeds). It is 92 Python files over
  an embedded 268 MB MLIR runtime. Its own summary calls it a "ROCm Domain
  Specific Language for layout algebra".
* `pip show flydsl` reports **`Required-by: amd-aiter`**. It is a dependency of
  AITER, AMD's production kernel library — not a benchmark-only artifact.
* In the AITER checkout, 62 files reference FlyDSL, and `ops/flydsl` is a
  first-class kernel-authoring directory beside the Triton one.

So FlyDSL is a production path. The case against supporting it has to be made on
different grounds, and it survives on those grounds.

### Why it still does not pay

**1. There is no published bar to beat.** The whole reason AgentKernelArena
matters here is that it makes "we beat Opus" checkable. The published numbers we
hold are `torch2hip` 6.89x, `hip2hip` 6.69x and `triton2triton` 2.13x
(`kore/eval/agent_kernel_arena.py`). There is no published FlyDSL number, so
scoring FlyDSL tasks produces a figure with nothing to compare it against. It
would add benchmark rows, not evidence.

**2. The example corpus we may legitimately use targets the wrong chip.** We must
not mine AgentKernelArena, so the only substantial body of worked FlyDSL we can
learn from is AITER's. Counting architecture mentions across AITER's FlyDSL
kernels: **gfx1250 in 26 files (159 mentions) against gfx950 in 17 files (80
mentions)**. AMD is leading FlyDSL with the newer architecture; our product
target is gfx950. Several kernels are named for it outright
(`grouped_moe_gfx1250.py`, `gemm_fp8fp4_gfx1250.py`, an entire `fmha_gfx1250/`
package).

**3. It ships no documentation.** There are no `.md` files, no examples and no
`project_urls` in the distribution. Learning it means reading AITER's 62 files
and the package source. A torch2flydsl kernel in AgentKernelArena averages 165
lines, so authoring our own seeds is a per-operator cost in a language with no
reference manual.

**4. Fewer tasks are reachable than the headline count.** 111 is the raw total;
AgentKernelArena's own preflight filters by `required_arch`, and 10 of the 15
`flydsl2flydsl` tasks are pinned to gfx942. On gfx950 the reachable set is
**101**: `torch2flydsl` 45, `triton2flydsl` 51, `flydsl2flydsl` 5.

### What it would actually cost

The environment cost is genuinely near zero, and it is worth stating precisely so
the decision is not made on a false impression of difficulty. A FlyDSL kernel is
a **Python** file, so it lands on the existing Triton path:

* `kore/env/hip_toolchain.py` would gain one entry in `CANDIDATE_FILENAMES`
  (`flydsl` -> `kernel.py`) and one in `SOURCE_LANGUAGES` (`python`).
* `kore/tasks/_genops.py` needs nothing: `_load_candidate` already falls back to
  `kernel.py` for any non-HIP backend.
* The reward-hack scanner needs nothing: its Python mode already applies.

The cost is entirely in authoring and in verifying seeds, plus one unknown worth
naming: FlyDSL JIT-compiles through MLIR at call time, and whether that compile
fits inside an episode budget on gfx950 is recorded in "Measured on this host"
below.

**Recommendation: do not build it now.** Revisit if AMD publishes a FlyDSL agent
bar, or if AITER's gfx950 FlyDSL coverage overtakes its gfx1250 coverage. Both
are cheap to re-check: the second is the arch-mention count above.

## Repository-level tasks

### What they are

Nine tasks, five over AITER and four over rocPRIM. Each is a `config.yaml` plus a
`task_runner.py` of 273-307 lines that clones the upstream repository, builds a
private virtualenv, installs eight packages, builds the project, and then runs
**the repository's own** correctness and performance harnesses. All
nine leave `required_arch` unset, so all nine are architecture-reachable here,
and the network they need works from this host (`git ls-remote` against
`github.com/ROCm/aiter.git` succeeds).

### Why they do not fit the training loop

This is a different verifier contract, not a new language. `KoreEnv` stages a
task directory, writes exactly one candidate file, runs `driver.py`, and parses
an SNR and a set of paired timings. A repository task instead needs:

* a repository checkout and a build per episode, or a warm shared one with all
  the cache-invalidation that implies (the AITER tree here is 1.6 GB, of which
  1.2 GB is 17 JIT-built `.so` files);
* an edit applied to a *named file inside a tree*, rather than a whole-file
  candidate;
* the repository's own harness for correctness and timing — which does not emit
  the paired, L2-flushed, AB/BA-balanced protocol our publication gate is defined
  on. **Every timing-integrity guarantee in `kore/tasks/_genops.py` would be
  unavailable**, so a repository episode could not produce a publication-grade
  speedup even when it worked.

The throughput consequence is the decisive one. A HIP episode costs roughly 15-30
seconds. A repository episode costs a clone plus a build plus the upstream test
suite: minutes, and for AITER tens of minutes cold. That is one to two orders of
magnitude more GPU-time per unit of learning signal, against a datagen budget
this project already measures in episodes per hour.

### Why the prior is still right about credibility

The prior — that repository-level work is what actually ships, and that
Kernel-Smith's credibility came from merged SGLang and LMDeploy PRs rather than
benchmark rows — holds, and nothing above contradicts it. What the measurement
changes is *where* that work belongs. It is an **output** of a good model, not
**training data** for one. A merged PR is produced once, by running the model
against a real repository, and its value is entirely in the artifact; putting the
same activity inside the RL loop pays the per-episode cost hundreds of thousands
of times to buy a signal the timing protocol cannot even grade.

**Recommendation: keep repository work, move it out of the loop.** Once the HIP
model is trained, point it at one repository task end-to-end as a credibility
artifact. rocPRIM is the cheaper first target: its headers are already present
(`/opt/rocm/include/rocprim`), and it needs a CMake build rather than AITER's JIT
stack. This costs a scripted run and no environment change, because it never
touches `KoreEnv`.

## Measured on this host

Commands, so each claim can be re-derived. `$AKA` is the AgentKernelArena
checkout (`KORE_AKA_ROOT`, default `~/third_party/AgentKernelArena`).

```bash
# task inventory by type, and reachability by required_arch
for d in $AKA/tasks/*/; do echo "$(basename $d) $(find $d -name config.yaml | wc -l)"; done

# FlyDSL is installed, public, and a dependency of AITER
pip show flydsl                      # 0.2.2, Apache-2.0, Required-by: amd-aiter
pip download flydsl==0.2.2 --no-deps # succeeds: it is on PyPI

# which architecture AITER leads FlyDSL with
rg -c gfx1250 $AITER/aiter/ops/flydsl --glob '*.py'   # 26 files, 159 hits
rg -c gfx950  $AITER/aiter/ops/flydsl --glob '*.py'   # 17 files,  80 hits
```

FlyDSL runtime status on gfx950: NOT YET MEASURED.
