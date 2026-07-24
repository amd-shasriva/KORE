# `kore/tasks` — kernel task registry

Every RL "environment instance" is a **kernel-optimization task**: a Triton kernel to make fast, an fp32 **reference oracle** for correctness, a **production vendor baseline** to beat (AITER / hipBLASLt / framework), a set of evaluation **shapes**, and a driver contract the verifier speaks. Tasks are discovered from `<task_id>/task.yaml` directories.

The registry task count is derived from the checked-in task set; do not copy a
historical total into supervisors or completion gates:

| Group | Prefix | Count | Baseline |
| --- | --- | --- | --- |
| Hand-authored | — | 55 | per-task (`reference.py`) |
| Generated | `gen_*` | 201 | torch / framework |
| Vendor-baselined | `genv_*` | 26 | real AITER / hipBLASLt |

`registry.all_tasks()` is the source of truth; re-derive the live count with:

```bash
PYTHONPATH=. python -c "from kore.tasks import registry; print(len(registry.all_tasks()))"
```

A further **16 op-class generator engines** under `kore/tasks/breadth/` materialize **1,052** additional verified `genb_*` task variants on demand (opt-in; not part of the checked-in registry total until generated — see [Breadth op-class generators](#breadth-op-class-generators)).

The registry also defines the **authoritative train / held-out split** by operator family and architecture, so generalization can never be leaked.

---

## Files

| File | Purpose |
| --- | --- |
| `base.py` | Task ABI: `Shape`, `Task`, `Task.from_dir()` — parses `task.yaml` |
| `registry.py` | Discovery, `operator_family`, `is_heldout`, `split_tasks`, `all_tasks`, `get_task` |
| `augment.py` | Deterministic shape augmentation (scale factors + an odd non-aligned shape) |
| `audit.py` | Live data-scale audit from the registry |
| `_genops.py` | Operator spec registry + `make_reference`, `seed_source`, generic `driver_main` |
| `generate_ops.py` | Writes `gen_<op>_<dtype>/` tasks (framework/torch baseline) |
| `vendor_ops.py` | Vendor-baselined op templates vs. real AITER kernels |
| `generate_vendor_ops.py` | Writes `genv_<op>_<dtype>/` tasks |
| `generate_breadth.py` | Writes `genb_<op>_<dtype>/` tasks from the `breadth/` engines |
| `breadth/` | 16 op-class authoring engines (+ CPU tests) for the `genb_*` expansion |
| `aiter_ref.py`, `aiter_ref_attn.py` | Shared AITER / hipBLASLt / framework baseline wrappers |
| `<task_id>/` | Per-task dir: `task.yaml`, `reference.py`, `seed_triton.py`, `driver.py` |

---

## The task contract

A task directory contains:

| File | Role |
| --- | --- |
| `task.yaml` | metadata + shapes (`minimal` / `primary` / `validation[]`), `snr_threshold`, `comparison_baseline` |
| `reference.py` | `parse_shape`, `get_inputs`, `ref_fn` (fp32 oracle), `baseline_fn` (production bar) |
| `seed_triton.py` | an admitted candidate implementation the policy edits; it must define the task entrypoint, pass `scan_for_hacks`, and compute through Triton rather than delegate to the oracle/framework/vendor op |
| `driver.py` | prints `SNR:`, `allclose:`, `median_ms:` — hand-authored or a shim to `_genops.driver_main` |

```python
@dataclass(frozen=True)
class Shape:
    name: str
    dims: dict[str, int]          # e.g. {"M": 4096, "N": 4096, "K": 4096}

@dataclass
class Task:
    task_id: str; operation: str; dtype: str; backend: str; gpu_target: str
    seed_kernel_name: str; snr_threshold: float; comparison_baseline: str
    shapes: list[Shape]; raw: dict
    @classmethod
    def from_dir(cls, d: Path) -> "Task"
```

---

## Train / held-out split

```python
TRAIN_ARCH  = "gfx950"                          # primary target: CDNA4 (MI350X / MI355X)
TRAIN_ARCHS = {"gfx950", "gfx942"}              # arches accepted into train (override: KORE_TRAIN_ARCHS)
HELDOUT_FAMILIES = ("mla", "paged_attention")
HELDOUT_TASKS    = {"mla_decode_bf16", "paged_attn_decode_bf16"}
```

A task is held out if **any** of these hold (`registry.is_heldout`):

1. its `task_id` is in `HELDOUT_TASKS`, **or**
2. its `operator_family()` is in `HELDOUT_FAMILIES` (`mla` or `paged_attention`), **or**
3. it targets a **foreign arch** (a `gpu_target` outside `TRAIN_ARCHS`).

This is the single source of truth used by both datagen (never trains on held-out) and eval (measures zero-shot transfer to the held-out families).

```mermaid
flowchart TD
  T[Task] --> ID{task_id in HELDOUT_TASKS?}
  ID -->|yes| HO[held-out: eval only]
  ID -->|no| F{"family in (mla, paged_attention)?"}
  F -->|yes| HO
  F -->|no| A{"gpu_target in TRAIN_ARCHS?"}
  A -->|no| HO
  A -->|yes| TR[train]
```

**Core attention is trained, not held out.** Flash-attention prefill / decode / sliding-window / varlen / fp8 all train, so the product model is strong at attention. Only the two *structurally distinct* families are withheld to measure genuine cross-family transfer: **MLA** (DeepSeek latent attention) and **paged-KV decode** (a different KV-cache mechanism).

**Why family-level, not task-level.** Reserving whole families (not just the two seed task ids) keeps any generated or mined MLA/paged variant out of training by its family, closing the last leakage path. `operator_family` therefore classifies `mla`/`paged` **before** the generic `attn` catch, so those variants never fall through into the trained `attention` bucket.

**Why deterministic.** The held-out set is a pure function of family + arch, independent of any seed, so datagen can exclude it with no seed coordination. `split_tasks(seed)` returns `{"train", "heldout", "seed"}`; `seed` only reorders *within* a split (for sharding / CV folds) and never moves a task across the boundary.

**Why gfx942 stays in train.** gfx942/CDNA3 shares the hardware lineage with the gfx950/CDNA4 target and runs correctly on-node, so previous-gen-tagged tasks and any in-flight gfx942 datagen keep training instead of being retroactively held out when the primary arch advanced to gfx950. A truly foreign arch (gfx1100, NVIDIA) is still held out.

> **Two family taxonomies exist by design.** `registry.operator_family` is the coarse split authority (the `mla` / `paged_attention` / `attention` / … buckets above). `kore.eval.generalization.family_of` is a richer 8-family classifier (attention, moe, gemm, norm, positional, quant, reduction, activation) used for offline leave-one-family-out analysis. The two are distinct; do not conflate them.

---

## Authoring new tasks

```mermaid
flowchart LR
  GO[generate_ops.py] --> GEN["gen_*/ dirs"]
  GVO[generate_vendor_ops.py] --> GENV["genv_*/ dirs"]
  GB[generate_breadth.py] --> GENB["genb_*/ dirs"]
  HAND[55 hand-authored tasks] --> REG
  GEN --> REG[registry discovery]
  GENV --> REG
  GENB --> REG
  REG --> TRAIN[train_tasks]
  REG --> HOLD[heldout_tasks]
```

- `_genops.py` defines 67 operators across the `unary`, `binary`, `reduce`, `fusion` (multi-kernel headroom), and `gemm_fusion` (hipBLASLt + epilogue headroom) families.
- `generate_ops.py` emits `gen_<op>_<dtype>/` tasks with a torch/framework baseline, expanding each operator across `bf16`/`fp16`/`fp32` (67 × 3 = 201 tasks).
- `generate_vendor_ops.py` emits the 26 `genv_<op>_<dtype>/` tasks (14 vendor ops × their dtype sweeps) graded against real AITER kernels with LLM-realistic shape tables.

---

## Breadth op-class generators

`kore/tasks/breadth/` holds **16 op-class authoring engines** — attention, MoE, GEMM, norm, quant, reduction, convolution, scan/SSM, sequence, sort/sparse, sampling, and training-op families. Each engine exposes the shared ABI (`OPS`, `SHAPES`, `make_reference`, `seed_source`) and ships CPU-side tests under `breadth/tests/`.

`generate_breadth.py` auto-discovers every conformant engine and writes `genb_<op>_<dtype>/` dirs, each with a `task.yaml`, a naive Triton candidate, and thin `reference.py`/`driver.py` shims. Before writing, the generator parses every candidate, requires a top-level function matching `operation`, and runs the environment's unchanged `scan_for_hacks`; one invalid engine seed aborts generation instead of fanning out bad artifacts. Together the engines materialize **1,052** task variants:

```bash
python -m kore.tasks.generate_breadth --list   # dry-run: list the genb_* ids
python -m kore.tasks.generate_breadth          # write the dirs into this checkout
```

Generation is opt-in and idempotent. A declared `seed_kernel_name` is an admission claim: the file is candidate code, never a torch/reference/vendor alias used to impersonate a Triton seed. If a future family cannot supply a legitimate generic candidate, it must add an explicit bootstrap/no-seed contract before materialization rather than emit delegated source or relax the scanner. Static admission does not prove device compilation or numerical correctness; run `scripts/verify_breadth.py` on gfx950 for that GPU proof.

Registry discovery globs `*/task.yaml`, so freshly written `genb_*` dirs are picked up with no code edits, and since none are named `mla`/`paged` they all land in TRAIN. Only run generation on a node whose task suite you intend to widen — never on a node whose in-flight run must keep a frozen task set.

---

## Baselines

Baselines are **production vendor kernels**, not torch-eager. `aiter_ref.py` / `aiter_ref_attn.py` wrap AITER (`aiter_rms_norm`, `aiter_fused_add_rms_norm`, `flash_attn_func`, `fused_moe`, `paged_attention_rocm`, …), hipBLASLt for GEMM, and torch only where AITER has no standalone op — always labeled via a `KORE_BASELINE_IMPL:<impl>` stderr sentinel, so "correct-but-slow vs. production" is never mistaken for "beats torch".

> fp8 e4m3 is arch-selected by `aiter_ref.FP8_DTYPE`: OCP `e4m3fn` on gfx950/CDNA4 (MI350X/MI355X — the native format and this node's default), FNUZ `e4m3fnuz` on gfx942/CDNA3. Override with `KORE_FP8_ENCODING=ocp|fnuz`.

---

## Environment variables

| Variable | Effect |
| --- | --- |
| `KORE_SHAPE_AUGMENT` | expand shapes via `augment_shapes` |
| `KORE_COMPILE_BASELINE` | `torch.compile`-fused baseline for fusion / gemm_fusion families |
| `KORE_VERIFIED_CORRECTNESS` | enable the adversarial input battery in the driver |
| `KORE_CORRECTNESS_TRIALS` | min reseeded correctness trials (default 5) |
| `KORE_BENCH_COLD` | L2-flush between timed iters (default 1) |
| `GPU_TARGET` | arch for Triton/HIP compilation |

---

## Gotchas

- `minimal` shapes are **correctness-only** — they are launch-overhead-bound, so the roofline analysis excludes them from `η` correlation.
- Registry discovery is **lazy-import-safe**: AITER/torch are imported only inside wrappers, so listing tasks never needs a GPU.
- `mutates_input` ops (e.g. `fused_add_rmsnorm`) clone inputs each bench call for fair timing.

See also: [`env`](../env/README.md) (how tasks are executed), [`analysis`](../analysis/README.md) (roofline over `task.operation`), [`reward`](../reward/README.md).
