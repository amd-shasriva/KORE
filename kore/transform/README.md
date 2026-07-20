# `kore/transform` — the ε-typed transformation calculus (a bounded RL action space)

`kore/transform` turns Triton-kernel optimization into a **typed, budget-constrained rewrite space**: a library of **13** pure source→source rewrites, each tagged by its *relation* to the kernel it edits —

- **`exact` (`≡`)** — *bit-preserving* (scheduling / layout / occupancy / boundary masks / independent-load reorder). A rewrite that can perturb output bits — even to *improve* precision — is never `exact`.
- **`approx` (`≈_ε`)** — a numeric *contract* within tolerance `ε` (`num_warps` cross-warp reassociation / `fp32_accumulator` ε≈0 / re-tiling / K-split / downcast IO / reassociated reductions / fast reciprocal), which **spends** from a finite `ErrorBudget`.

Everything here is **pure, deterministic, stdlib-only, CPU-only** (regex/AST-lite over the source string). It is exposed to the agent as the `list_transforms` / `apply_transform` tools (`agentic_transform_tools: true`, see [`kore/agent`](../agent/README.md)) and used as AlphaKernel's move generator (see [`kore/search`](../search/README.md)).

> **What the typing means.** The `exact`/`approx` relation is a **design-time action-space label, not a machine-checked proof of semantic equivalence.** Correctness is enforced **downstream** by the environment's SNR oracle, which builds, tests, and benches *every* rewritten kernel: an over-approximate or outright wrong rewrite fails the SNR gate and is rejected/pruned. The accurate description of this package is a **typed, budget-constrained rewrite action space with downstream numerical verification**. Its value is the **bounding**: a policy that can only compose these in-contract rewrites structurally *cannot* emit a `memset` / cache / timing exploit — the anti-reward-hack spine of the action space.

---

## Files

| File | Purpose |
| --- | --- |
| `calculus.py` | `Transformation` (one typed rewrite: `apply` / `side_conditions` / `epsilon` / metadata), `Action`, and the two engine functions: `admissible_actions` (the legal move set = the RL action space) and `apply_sequence` (run a rewrite trajectory with side-condition + budget gating and a full audit trail) |
| `budget.py` | `ErrorBudget` ε-accounting (per-`(op, dtype)` default table), the relation **lattice** (`compose_relation` / `compose_eps`), pure and dependency-free |
| `library.py` | The **13** concrete rewrites (**6** `exact` + **7** `approx`) as regex/AST-lite Triton edits, keyed off the same knob-token conventions as `kore.policy.format` / `kore.value.features` |
| `discover.py` | **Self-extending library (opt-in, off by default)**: `discover_transforms` / `merge_transforms` / `extend_library` propose candidate new rewrites (knob sweeps, vectorization widths, elementwise fusion) as conservatively-`approx` `Transformation`s. The curated `LIBRARY` is never mutated. Proposals are SNR-gated downstream, not verified in this package |
| `__init__.py` | Public API (`Transformation`, `ErrorBudget`, `apply_sequence`, `admissible_actions`, `LIBRARY` / `EXACT` / `APPROX`, …) |
| `tests/` | `test_transform.py` (per-transform behavior + action-space monotonicity), `test_relation_typing.py` (`exact` == bit-preserving invariant + ε-lattice composition), `test_discover.py` (self-extending proposals are well-typed, no-op-safe, opt-in, budget-composing) |

---

## The transform library (13)

| Relation | Name | Knob | Note |
| --- | --- | --- | --- |
| `≡` exact | `set_num_stages` | `num_stages` | software pipelining / LDS double-buffering |
| `≡` exact | `set_waves_per_eu` | `waves_per_eu` | gfx950 occupancy hint |
| `≡` exact | `swizzle_group_m` | `GROUP_M` | L2 super-grouping / program swizzle |
| `≡` exact | `vectorize_loads` | `vectorize` | `tl.max_contiguous` / `tl.multiple_of` hints (compiler-only) |
| `≡` exact | `add_mask_boundary` | `mask` | reuse an in-scope guard on an unmasked load/store |
| `≡` exact | `reorder_loads` | `layout` | swap two **data-independent** adjacent `tl.load`s |
| `≈_ε` approx | `set_num_warps` | `num_warps` | wavefront parallelism / occupancy; can reassociate a cross-warp reduction (ε `0.005`; bit-exact only for reduction-free kernels) |
| `≈_ε` approx | `fp32_accumulator` | `accumulator` | force the `tl.zeros` accumulator to `tl.float32`; precision-**improving**, ε≈0 (`1e-6`) but **not** bit-identical |
| `≈_ε` approx | `retile_block` | `block` | re-tile `BLOCK_M/N/K` (ε 0.02 for M/N; **0.06** for K, which reassociates the reduction) |
| `≈_ε` approx | `split_k` | `split_k` | split the K reduction across a 2nd grid axis with atomic partials; **zero-inits the atomic-add output** (correctness guard) (ε 0.05→0.24 by ways) |
| `≈_ε` approx | `downcast_dtype` | `dtype` | downcast IO to fp16/bf16/fp8, keeping the fp32 accumulator (ε 0.03/0.05/0.15) |
| `≈_ε` approx | `reassociate_reduction` | `reduction` | fuse the accumulate into `tl.dot(...,acc)` (ε 0.01) |
| `≈_ε` approx | `fast_math_recip` | `fast_math` | `1.0/x → tl.math.rcp(x)` (ε 0.03) |

`set_num_warps` and `fp32_accumulator` are typed `approx` because each can move output bits — a warp-count change can reassociate a cross-warp reduction, and forcing the accumulator to fp32 improves precision toward the reference but is not bit-identical — so `exact` stays strictly bit-preserving. `split_k` rewrites the epilogue `tl.store` into `tl.atomic_add`, which is correct only against a zero-initialized output, so it rewrites the returned allocation (`torch.empty[_like] → torch.zeros[_like]`) as part of the move.

---

## The calculus

**Relation lattice.** `compose_relation` is a lattice join with `exact` as bottom (strongest) and `approx` as top (weakest): `exact ⊔ exact = exact`, and any `approx` step makes the whole trajectory `approx` (a chain can never recover bit-exactness). The carried ε is the **weakest (max)** step tolerance, since a chain of numeric contracts is only as strong as its loosest link.

**Two orthogonal meters** (both exposed, because they answer different questions):

1. **Cumulative spend** (`ErrorBudget.spent` / `.remaining()`) — a conservative *additive* meter of allowed drift; this is what **gates admissibility** (an approx move whose ε no longer fits `remaining()` is inadmissible).
2. **Composed contract** (`composed_relation()` / `weakest_eps()`) — the *type the result carries*.

**The action space** (`admissible_actions(src, budget)`) is the set of legal `(transform, params)` moves for the current source: params pass side-conditions, the move actually changes the source, and — for approx — its ε still fits the budget. It is **monotone in the budget**: spending ε can only *remove* approx actions, never add any.

`apply_sequence(src, steps, budget)` executes a trajectory, gating each step in order by **(1) side conditions → (2) budget → (3) applicability**, committing survivors (approx spends ε, exact is recorded free) and returning `(new_src, applied, rejected, budget_state)` — a full audit trail.

```python
from kore.transform import ErrorBudget, admissible_actions, apply_sequence
budget = ErrorBudget.for_op(task.operation, task.dtype)   # per-(op,dtype) default ε
actions = admissible_actions(kernel_src, budget)           # legal moves now
new_src, applied, rejected, state = apply_sequence(
    kernel_src, [actions[i].as_step()], budget)            # apply + account
```

The per-`(op, dtype)` default budget is `dtype_tolerance × op_scale` (fp32 `0.02` … bf16 `0.10` … fp8 `0.25`; elementwise scales up, deep GEMM/attention reductions scale down), so low-precision dtypes — whose SNR gate is already relaxed — get a larger ε to spend.

---

## Relation typing

`exact` means **bit-preserving**: a rewrite that can move a bit is typed `approx`. `approx` means **intended within ε**. Both are backstopped by the downstream SNR gate, which is the correctness authority; the relation is an action-space prior, not a proof. The typing is regression-pinned in `tests/test_relation_typing.py`, which asserts the `exact` == bit-preserving invariant and the ε-lattice composition rules.

---

## Self-extending library (`discover.py`, opt-in — off by default)

`discover.py` **proposes** candidate new rewrites so the action space can grow beyond the 13 curated transforms without changing the default. Importing it does nothing; the curated `LIBRARY` is only extended when a caller explicitly merges proposals in.

**Strategies** (each CPU-only, pure source→source):

- **knob sweeps** — parameter-sweep variants of the existing numeric knobs (`num_warps` / `num_stages` / `waves_per_eu` / `GROUP_M` / block sizes / `SPLIT_K` / IO dtype) at values outside the base grid, *reusing* the base transform's guarded `apply` / `side_conditions`;
- **vectorization widths** — pinned `tl.max_contiguous(tl.multiple_of(...))` contiguity annotations at candidate widths;
- **elementwise fusion** — fuse two adjacent elementwise assignments by inlining the temporary.

Every proposal is a **no-op (`None`) when its precondition doesn't match, and never raises**, is namespaced `disc:` so it can't shadow a curated transform, and is typed **conservatively `approx`** with a floored ε (`≥ 0.01`, and `≥` the base transform's ε for those params), because a proposal is unverified.

```python
from kore.transform import admissible_actions, apply_sequence
from kore.transform.discover import discover_transforms, merge_transforms, extend_library

ext = extend_library(source=kernel_src)                 # base + relevant proposals (NEW list)
actions = admissible_actions(kernel_src, budget, library=ext)
new_src, applied, rejected, state = apply_sequence(
    kernel_src, [actions[i].as_step()], budget, library=ext)
```

- `discover_transforms(base, *, source=None, sweep_knobs=True, enable_fusion=True, enable_vectorize_widths=True, vectorize_widths=(4,8,16), max_proposals=64) -> list[Transformation]` — synthesize proposals (pruned to those that fire on `source`, if given).
- `merge_transforms(base, discovered, *, override=False) -> list[Transformation]` — registry-merge (base-first, de-duped, base wins collisions unless `override`); mutates nothing.
- `extend_library(*, source=None, base=LIBRARY, **flags) -> list[Transformation]` — convenience discover+merge into a NEW list.

**Orchestrator wiring.** The merged registry flows through the existing `library=` seam, so no orchestrator change is required: `TransformProposePolicy(library=ext)` (AlphaKernel move generator) and `admissible_actions(..., library=ext)` / `apply_sequence(..., library=ext)` (agent tools) all accept it. Proposals are appended **after** the curated moves (a curated-first prior), so a small `k_expand` sees the base set unchanged; enlarge `k` (or exhaust/prune the curated set) to surface proposals. `discover.py` does not verify equivalence or the ε contract: every proposed rewrite is build/test/benched by the SNR oracle exactly like a curated one, so a wrong or out-of-contract proposal fails the gate and is pruned. It is a candidate generator that broadens the in-contract action space, not a correctness authority.

---

## Wiring and safety

- **Agent tool** (`agentic_transform_tools: true`): `list_transforms` shows the currently-legal, in-budget moves; `apply_transform` applies one with ε-accounting and returns the rewritten source (an inadmissible / budget-exceeding move is **rejected**, leaving the source unchanged). The model then build/test/benches the result through the verified env ([`kore/agent`](../agent/README.md)).
- **Search move generator** (`TransformProposePolicy`): the same calculus is AlphaKernel's action space ([`kore/search`](../search/README.md)).
- **Verification is downstream and mandatory.** Every rewritten kernel is graded by the env's SNR oracle; an over-approximate move fails the gate and is rejected/pruned. Any transform error yields *no move* rather than raising into the RL loop.

The design point is the **relation typing plus finite ε-budget as the admissibility gate**: the action space *shrinks as numerical tolerance is consumed*, and the env's SNR oracle is the correctness authority.

See also: [`kore/search`](../search/README.md) (AlphaKernel over this calculus), [`kore/agent`](../agent/README.md) (the transform tools), [`kore/openended`](../openended/README.md), [`kore/policy`](../policy/README.md), [`kore/value`](../value/README.md), [`kore/reward`](../reward/README.md), [`kore/env`](../env/README.md).
