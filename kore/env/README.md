# `kore/env` — the verified GPU environment

`KoreEnv` is where a candidate kernel meets real silicon. It compiles the kernel, checks correctness on every shape against the fp32 oracle, benchmarks it cold-cache against the production baseline with variance control, optionally collects rocprofv3 counters, and caches the result. It is hardened against verdict forgery, mode-sniffing, stateful-timing hacks, and filesystem escape, and it separates **infra** failures (timeout / OOM / HIP flake) from **kernel** failures.

---

## Files

| File | Purpose |
| --- | --- |
| `kore_env.py` | `KoreEnv` — `step` / `evaluate` / `_run`; correctness + bench + profile |
| `replay.py` | `ReplayCache` — JSONL-backed `(task_id, source) → Observation` |

---

## The evaluation path

```mermaid
flowchart TD
  S[candidate source] --> H{scan_for_hacks?}
  H -->|yes| OH[Observation flagged_hack]
  H -->|no| C{replay cache hit?}
  C -->|yes| RET[return cached Observation]
  C -->|no| RUN[stage driver+reference read-only, write kernel.py]
  RUN --> COR[correctness on ALL shapes vs fp32 oracle]
  COR --> DET{determinism re-check?}
  DET --> BENCH[cold-cache bench candidate + reference per shape]
  BENCH --> PROF{"profile_reward_weight > 0?"}
  PROF --> OBS[Observation]
  OBS --> CACHE{cacheable? not infra_error}
  CACHE --> RC[(replay_taskid.jsonl)]
```

Key API:

```python
class KoreEnv:
    def step(self, source, full_validation=True, multi_shape=True) -> Observation
    def evaluate(self, task, source, shapes=None, do_bench=True) -> Observation
```

The returned `Observation` (defined in [`kore/reward`](../reward/README.md)) carries `compiled`, `snr_by_shape`, `wall_by_shape`, `baseline_by_shape`, `cv_pct`, `flagged_hack`, `infra_error`, and optional `profile_efficiency`.

---

## Anti-hack hardening

| Attack | Defense |
| --- | --- |
| Verdict forgery (print fake `SNR:`) | parse the **last** regex match; re-check correctness *after* the timed loop |
| Mode sniffing (behave differently when benched) | randomized warmup/iters per bench run |
| Stateful timing | post-timing correctness poison → whole eval flagged as hack |
| One-easy-shape win | `wall_ms = max` over shapes, `snr_db = min` over shapes |
| Filesystem escape | staged in a temp workdir; reference/driver copied read-only (chmod 444) |
| Runaway process | `timeout` + process-group `killpg` in `_exec` (no `RLIMIT_AS` — ROCm needs a huge VA space) |

> **Concurrency and `RLIMIT_NPROC`.** `RLIMIT_NPROC` is **per-UID**: it counts every process and thread the user owns, not just the child, so a low per-subprocess soft cap throttles the *whole user*. Under concurrent datagen — dozens of workers spawning thousands of torch/OpenBLAS threads — a low cap makes OpenBLAS `blas_thread_init` fail and `import numpy` die inside the driver, which marks **every** candidate `compiled=False`. `_preexec` therefore raises the soft limit to the hard cap, and `_env` caps `OPENBLAS/OMP/MKL/NUMEXPR_NUM_THREADS=4` so the driver spawns a bounded thread pool instead of one BLAS thread per core (×96) per subprocess. Runaway containment is the `timeout` + `killpg` in `_exec`, not a per-child nproc cap.

**Infra vs. kernel classification** (`_classify`): timeouts, OOM, and HIP flakes are `infra_error=True`; they are **never cached** and **never scored as incorrect**, so a transient node problem cannot poison the replay cache or penalize a good kernel.

---

## Determinism gate

When `CONFIG.verifier_determinism_check` is on, the primary shape is re-run; if SNR drifts by more than `determinism_snr_tol_db` (10 dB) the kernel is judged non-deterministic (incorrect). An infra flake on the re-run is treated as *inconclusive*, preserving the original correct verdict.

---

## Replay cache

```python
def source_key(task_id, source) -> str      # SHA256(task_id + NUL + source)
class ReplayCache:
    def get(self, task_id, source) -> Optional[Observation]
    def put(self, task_id, source, obs) -> None
```

JSONL records are filtered to the current `Observation` field set on load, so schema evolution (e.g. removing a field) never causes a silent cache miss. Cacheability rule: `(compiled or error_text) and not infra_error`.

---

## Profiling

When `profile_reward_weight > 0`, `_collect_profile` runs rocprofv3 with `--bench-mode` on the primary shape and produces a `profile_efficiency ∈ [0,1]` (see [`kore/verifier`](../verifier/README.md) for counter sets and [`kore/reward`](../reward/README.md) for how it shapes reward). rocprof requires `--bench-mode`; without it candidate/reference profiles are degenerate.

`collect_counters(source, shape=primary)` is the public rocprofv3 PMC entry point: it stages an isolated workdir, profiles the candidate, and returns aggregated `{counter: value}` (the gfx950 derived metrics `MemUnitStalled` / `OccupancyPercent`, plus captured `vgpr_count` / `lds_bytes` / `num_warps`) or `None` when the profiler is unavailable — fully fail-safe.

These counters are diagnostic inputs. They do not become reward or potential
terms unless a fingerprint-pinned P0 artifact contains a PASS for the task's
operator family under the same physical model. The controlled reanalysis found
no passing family, so `physics_shaping_weight=0`, live physics counter collection
is off, and `phi_potential` returns `None` without explicit evidence. See
[`kore/reward`](../reward/README.md) and `docs/P0_RESULTS.md`.

Counter collection may still be enabled deliberately for observability. Raw
quad-cycle waits are never divided by instruction counts, and MOPS are not
treated as instructions.

---

## Config knobs (from `kore/config.py`)

| Knob | Effect |
| --- | --- |
| `verifier_determinism_check` | re-run primary shape; drift → incorrect |
| `min_variance_runs` / `max_variance_runs` / `cv_threshold_pct` | early-stop benching when CV is low enough |
| `warmup_iters` / `bench_iters` | base warmup/measure counts (randomized per run) |
| `profile_reward_weight` | trigger PMC collection + dense shaping |

See also: [`tasks`](../tasks/README.md), [`reward`](../reward/README.md), [`verifier`](../verifier/README.md).
