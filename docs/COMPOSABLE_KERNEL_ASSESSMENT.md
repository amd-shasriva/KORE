# Composable Kernel: emit target, baseline, or neither

Companion to [`AITER_CORPUS_AND_REPO_ASSESSMENT.md`](AITER_CORPUS_AND_REPO_ASSESSMENT.md),
which ruled on AITER. This document answers one question about AMD's Composable
Kernel (`github.com/ROCm/composable_kernel`): should KORE add CK tasks, and if so
is CK something the model should **emit**, or a **baseline to beat**?

**Verdict: neither, as a family.** Do not train the model to emit CK, and do not
add a CK baseline lane. One narrow exception is worth keeping open and is stated
at the end. The reasoning is measurement, not preference, and the measurements
are given so they can be re-derived.

## Does the AITER-assembly reasoning transfer? Partly, and not the part that matters

The prior ruling on AITER's hand-written assembly was: it cannot be ingested
because it is **not in the repository**: AITER ships 2,867 pre-assembled ELF
code objects and zero `.s`/`.S`/`.asm` files, so there is no source text to learn
from.

That argument does **not** transfer to CK. CK's source is public, readable,
permissively licensed C++, and there is plenty of it. Anyone reusing the
assembly verdict here would be reusing the wrong half.

The argument that *does* transfer is the other line in the same table: the CK
slice was marked *"Skip. Template-heavy, low signal per token, and not our output
language."* That was a one-line judgement. What follows is the measurement behind
it, which is what makes it a finding rather than a preference.

## Why CK is not an emit target

### 1. Writing a CK kernel is selecting template arguments, not writing device code

This is the decisive fact and it is visible in CK's own GEMM example
(`example/01_gemm/gemm_xdl_fp16.cpp` upstream). The entire "kernel" is one type
alias:

```cpp
using DeviceGemmInstance1 = ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle
  < ALayout, BLayout, CLayout, ADataType, BDataType, CDataType, AccDataType,
    CShuffleDataType, AElementOp, BElementOp, CElementOp, GemmDefault, 1, 256,
    256, 128, 32, 8, 2, 16, 16, 8, 4, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8,
    8, 1, S<8, 32, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 4, 2, 0, 1, 2,
    S<1, 16, 1, 16>, 4, ck::LoopScheduler::Interwave, ck::PipelineVersion::v1 >;
```

Roughly forty template arguments, and **not one statement of device code**. There
is no loop to write, no load to coalesce, no MFMA to reach for, no tail to mask.
All of that lives inside CK's headers. The author's whole act is choosing tile
sizes, thread-cluster arrangements, vector widths and a pipeline version.

That action space already exists in this repository, and in a stronger form.
`kore/transform` is a verified, epsilon-typed transformation calculus exposed to
the policy as `list_transforms` / `apply_transform`; it offers exactly these moves
(`set_num_warps`, `retile_block`, `split_k`, `fp32_accumulator`) with side
conditions checked and an error budget accounted, so an inadmissible choice is
rejected instead of silently producing an out-of-contract kernel. Training the
model to emit CK would teach it to make the same class of decision through an
undocumented forty-slot template signature, with no side-condition checking, and
without ever writing the code those tiles feed.

The transferable skill here, picking tiles that map cleanly onto 64-lane
wavefronts and the MFMA cores, is already trained, on 1,331 Triton and 188 HIP
C++ tasks, where the model must additionally write the code that consumes them.
CK would add the parameter-picking without the kernel.

### 2. A CK response cannot satisfy the self-containment contract

`kore/policy/format.py` defines the response contract as `FULL_KERNEL`: *"the
ENTIRE kernel source, ready to run - not a diff, not a snippet"*, and the
verifier stages exactly one candidate file.

CK's own example is not self-contained. It opens with `#include "common.hpp"` and
ends by including `run_gemm_example.inc`, CK's private example harness. A CK
submission would be a type alias plus includes the harness has to supply.

This is precisely the case the project already ruled on, correctly, for
HipKittens: those kernels are C++ that `#include "kittens.cuh"`, so training them
as `FULL_KERNEL` responses would teach the model to answer with code the harness
cannot compile: negative transfer that looks like clean data. That decision is
recorded in [`HIPKITTENS_INGEST.md`](HIPKITTENS_INGEST.md), and the reasoning
applies to CK unchanged. Being AMD's own library does not change what the
compiler can build.

### 3. CK is not present where episodes are generated

Measured on this host and on the cluster login node:

* The CK submodule vendored in the AITER checkout
  (`3rdparty/composable_kernel`) is **uninitialized**: 0 `.hpp`, 0 `.cpp`.
* The SPUR login node has **no ROCm at all**: no `/opt/rocm`, no `hipcc`. So the
  question "are CK headers installed" cannot be answered there, only on a GPU
  node.
* Only **5 files** in AITER's `csrc` reference `ck/` or `ck_tile/` at all, which
  is worth knowing because it undercuts the impression that CK is the substance
  of AITER's C++ rather than one dependency among several.

`scripts/probe_composable_kernel.sh` answers the remaining part on a GPU node: it
records whether CK headers exist and whether a minimal CK translation unit
actually compiles for gfx950. It is bundled into
`scripts/spur_verify_spec_1node.sbatch` rather than given its own job, because
the QoS caps six concurrent nodes and taking one to run `ls` would take a node
another agent needs.

**If that probe reports no CK headers, the verdict hardens from "poor value" to
"impossible": a CK response could not be compiled by the verifier that scores it,
so every rollout would score zero.** See "Status" below for what has and has not
been measured.

## Why CK is not a baseline lane either

The instinct that CK belongs as a baseline rather than an output, the correct
instinct for AITER, is mostly already satisfied, and for a specific reason.

KORE's vendor lane is 110 tasks baselined on AITER and hipBLASLt. AITER is a CK
*consumer*: where AITER's fastest path for an operator is CK-backed, a task
graded against AITER is already being graded against CK-derived code. Adding a
separate CK bar for those operators would re-measure the same thing through a
build system we would have to stand up.

For GEMM specifically, the production bar is hipBLASLt, which is the tuned
vendor path; a hand-instantiated CK GEMM would be a *worse* baseline than the one
already in use, and a baseline that is easy to beat inflates a speedup. The
project's own rule applies: never weaken a baseline to make a number look better.

## The one exception worth keeping open

CK and `ck_tile` cover fused attention and some MoE shapes where AITER's Triton
path is not obviously the strongest implementation. If, for a **specific
operator**, the genuine production-best implementation is CK rather than AITER
Triton or hipBLASLt, then CK is the right bar for that operator, and grading
against anything else would overstate the win.

That is a per-operator measurement, not a family: check what AITER actually
dispatches for the operator, and if it is CK-backed, the existing AITER baseline
already captures it. This is cheap to re-check and does not require any CK task
to exist.

## Status

| Claim | Measured? |
| --- | --- |
| CK authoring is ~40 template arguments and no device-code statements | yes, from CK's upstream `example/01_gemm` |
| CK examples are not self-contained (`common.hpp`, `run_gemm_example.inc`) | yes, same source |
| AITER's vendored CK submodule is uninitialized here | yes, on this host |
| AITER `csrc` references `ck/` or `ck_tile/` in 5 files | yes, on this host |
| SPUR login node has no ROCm, so CK cannot be probed there | yes |
| Whether CK headers exist and compile on a **gfx950 node** | **not yet: `scripts/probe_composable_kernel.sh` is queued behind a datagen campaign holding the QoS node cap** |

The verdict does not depend on the unmeasured row. That probe can only make the
answer more negative: it can confirm CK is uncompilable where episodes run, or
find it installed, in which case the reasoning above still stands on its own.

## Re-deriving the measurements

```bash
# CK submodule state in the AITER checkout
find "$HOME/third_party/aiter/3rdparty/composable_kernel" -name '*.hpp' | wc -l

# how much of AITER's C++ actually reaches for CK
rg -l --glob '*.cu' --glob '*.cpp' --glob '*.hpp' 'ck/|ck_tile/' \
    "$HOME/third_party/aiter/csrc" | wc -l

# CK availability and compilability on a GPU node (run via sbatch, not on login)
bash scripts/probe_composable_kernel.sh /tmp/ck_probe.json
```
