# `scripts/` - campaign orchestration & launchers

Everything needed to run KORE end-to-end: the campaign orchestrator, the portable conductor + tmux launchers, the FSDP launch helper, and smoke tests.

---

## Files

| Script | Purpose |
| --- | --- |
| `run_campaign.py` | The orchestrator: 9 default stages (plus 2 opt-in, `reverify` and `evolve`), manifest resume, retention gates, CLI |
| `run_conductor_14b.sh` | **Portable** full-14B launcher (repo-root-relative, loads `.env.local`, project venv) |
| `tmux_campaign.sh` | Run the conductor launcher in a durable detached tmux session |
| `run_full_14b.sh` | Legacy full-14B launcher with **hardcoded** dev-node paths (`/root/Kore-rl/kore`) |
| `run_e2e_14b.sh` | Bounded end-to-end validation run |
| `run_grpo_resilient.sh` | GRPO launcher for a **heavily-shared node**: dynamically picks the GPUs that are free right now and retries across transient VRAM spikes from other users' jobs |
| `run_v2_reuse_14b.sh` | v2 rerun that **reuses v1 kernels** instead of regenerating: `reverify` (re-measure existing kernels, no teacher) → `datagen` (coverage holes only) → `build` → `midtrain..eval` |
| `launch_distributed.sh` | `accelerate launch` wrapper for a single FSDP stage |
| `kore_supervise.py` | Keep a `run_campaign.py --full-ft` invocation alive across transient deaths (relaunch, resumes via the manifest, sparse ALERTs; `--force`/`--stages` are env-gated - see below) |
| `kore_monitor.py` | **Read-only** stage/health monitor: tails the live log, emits ALERT lines (no launching) |
| `kore_pause_after_datagen.py` | Halt the run cleanly at the **datagen→build boundary** so new code can land before the training stages |
| `kore_resume_supervise.py` | Wait for the pause sentinel, then auto-relaunch **build→eval** on the new code + supervise |
| `grpo_smoke.py`, `sft_smoke.py`, `smoke_env.py` | Tiny real runs to prove a subsystem works |
| `test_amd_gateway.py` | One-call check that the Claude gateway key works |
| `_repro_grpo_step.py` | Minimal GRPO-step reproduction harness |
| `eval_bakeoff_multi.py` | Matched-budget bake-off across the improvement ladder (seed → base → midtrain → SFT → DPO, optionally vs. Opus): `fast_p`, correctness rate, geomean speedup |
| `prove_dense_reward.py` | One-off proof that the dense hardware-counter reward (`--profile-reward`) is actually live on-silicon (not silently inert): exercises `env.collect_counters → roofline_dense_score` on a real compute-bound kernel |
| `prove_new_ops.py` | GPU-proves a batch of new task seeds compile + pass rigorous correctness + beat baseline; exits non-zero on any failure (a poison gate before registering new tasks) |
| `_validate_benchboth.py` | A/B check that a batched timing path (`--bench-both`) reproduces the legacy per-impl timing's speedup distribution within run-to-run variance |

---

## The campaign orchestrator

```mermaid
flowchart LR
  subgraph resume[manifest resume]
    M["campaign_manifest.json<br/>done_stages + ckpts"]
  end
  MT[midtrain] --> DG[datagen] --> AG[agentic] --> BU[build] --> SFT[sft] --> DPO[dpo] --> GRPO[grpo] --> SP[soup] --> EV[eval]
  M -.->|skip if done_stages ∧ artifact_ok| MT
  M -.-> DG
  M -.-> SFT
```

`DEFAULT_STAGES` (9 stages, run in this order when `--stages` is omitted) is `midtrain, datagen, agentic, build, sft, dpo, grpo, soup, eval` - midtrain runs FIRST because continued-pretraining trains on a general Triton/HIP corpus, not on the task-specific datagen output, so it has no dependency on `datagen`/`agentic`/`build`. A `--stages` list you pass explicitly is executed in the order you give it (e.g. the live run's `--stages midtrain,build,sft,dpo,grpo,soup,eval` skips `datagen`/`agentic` because that data was already generated in a prior invocation and is being reused/resumed).

**Resume logic.** After each stage the manifest records `done_stages` and the real checkpoint path (atomic write). On restart a stage is skipped only if it is in `done_stages` **and** `_artifact_ok(stage)` finds its on-disk artifact - so a stale "done" flag with a missing checkpoint correctly re-runs. `--force --stages <s>` re-runs regardless. Datagen additionally resumes at shard level (see [`kore/data`](../kore/data/README.md)).

**Retention gates** run after midtrain/sft/dpo/grpo; a FAIL hard-stops the campaign (see [`kore/eval`](../kore/eval/README.md)).

**Full-FT dispatch.** Under `--full-ft`, each of `midtrain`/`sft`/`dpo`/`grpo` is rendered into a resolved JSON and shelled out to `scripts/launch_distributed.sh <stage> <resolved.json>` (`accelerate launch`); see [`configs/README.md`](../configs/README.md) for exactly how the resolved config is built (`_launch_distributed`) and where it's written under `<data_root>/launch/`.

### Key CLI flags (defaults)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model` | `Qwen/Qwen3-14B` | base model |
| `--stages` | 9 defaults | comma-list subset of `reverify,datagen,evolve,agentic,build,midtrain,sft,dpo,grpo,soup,eval`; the default omits the opt-in `reverify` and `evolve` |
| `--dry-run` | off | import-check + print plan, no GPU/side effects |
| `--force` | off | re-run requested stages ignoring the manifest |
| `--full-ft` / `--lora` | `--lora` | full-parameter FSDP vs. LoRA bring-up |
| `--teacher` | `claude` | teacher backend |
| `--data-root` | `data` | shard + manifest root |
| `--datagen-workers` | 0 (=1/GPU) | parallel datagen concurrency |
| `--dpo-rounds` | 2 | iterative on-policy DPO rounds (>1 enables DAgger) |
| `--grpo-curriculum` | on | correctness→latency two-phase GRPO |
| `--adaptive-steps` | off | plateau early-stop for GRPO |
| `--use-hf` | off | real HF retention benches + general replay |
| `--sft-total` | 20000 | SFT mix cap |
| `--split-seed` | 0 | reorders within train/held-out (split itself is fixed) |

---

## Running the full campaign (recommended path)

```bash
bash scripts/tmux_campaign.sh              # start in a durable tmux session 'kore14b'
tmux attach -t kore14b                     # watch (Ctrl-b d to detach)
tail -f runs/full/logs/campaign_*.log      # follow the log
bash scripts/tmux_campaign.sh --status     # status without attaching
```

`run_conductor_14b.sh` is portable (resolves the repo root from its own path, uses `~/kore-venv`, sources `.env.local`, prepends the venv `bin` to `PATH` so `accelerate` resolves for FSDP) and overridable via env: `KORE_STAGES`, `KORE_DATAGEN_WORKERS` (default 32), `KORE_PY`, `KORE_TMUX`. Datagen/agentic are teacher-API-bound, so they oversubscribe the 8 GPUs (~4×) for throughput; training stages use full-parameter FSDP.

> Use `run_conductor_14b.sh` everywhere. `run_full_14b.sh` hardcodes dev-node paths and will not run on conductor.

---

## Supervision, monitoring & the datagen→build pause

Four Python helpers wrap a live, multi-day campaign. They all emit sparse `ALERT ` lines (plus periodic `HEARTBEAT` lines) so a `tail -F` with notify-on-output on `"ALERT "` pings the operator only on events that matter: **stage transitions**, real **errors** (`Traceback` / `ERROR` / OOM / `CUDA|HIP error`), **retention-gate** failures/hard-stops, and completion/death. Each reaps **only `shasriva`-owned** processes - root's shared workers are never touched.

| Script | Launches? | Scope | Role |
| --- | --- | --- | --- |
| `kore_monitor.py` | no (read-only) | any live campaign | Observe + alert only |
| `kore_supervise.py` | yes | `build..eval` by default (`KORE_SUP_STAGES` extends it, e.g. to `midtrain,datagen,...`) | Keep the run alive across deaths |
| `kore_pause_after_datagen.py` | no (stops procs) | at `datagen→build` | Halt cleanly so new code can land |
| `kore_resume_supervise.py` | yes | `build..eval` | Auto-resume on the new code + supervise |

**`kore_monitor.py`** - a **read-only** watcher. It polls the newest campaign log (from `/tmp/kore_foldin_logpath.txt`, else the newest `runs/full/logs/campaign_foldin_*.log`) and the campaign pid every `KORE_MONITOR_POLL_S` (default 180s), and additionally alerts on a **429 rate-limit *storm*** (a big jump, not the odd retry). It self-terminates on `campaign complete` or a confirmed death (pid gone two consecutive polls). It never launches or kills the campaign.

**`kore_supervise.py`** - keeps a `run_campaign.py --full-ft` invocation alive across transient deaths. Its default `--stages` is `build,sft,dpo,grpo,soup,eval` (env-overridable via `KORE_SUP_STAGES`, e.g. to prepend `midtrain` and/or `datagen`) - resume relies on the campaign manifest + `shard_done`, so `--force` is only appended when `KORE_SUP_FORCE=1` is set (default off, so a relaunch resumes rather than re-running completed stages). Each attempt: reap our stale workers → (re)launch `run_campaign.py --full-ft --use-hf --teacher claude --adaptive-steps [--force] …` → poll its log, alerting on the events above. On a non-completion exit it relaunches with bounded retries (`KORE_SUP_MAX_RETRIES`, default 12) + cooldown (`KORE_SUP_COOLDOWN_S`, default 90s); it stops on completion or exhausted retries. Writes the active log path to `/tmp/kore_foldin_logpath.txt`. `KORE_DATAGEN_WORKERS` defaults to 64 in this script (higher than `run_conductor_14b.sh`'s 32 default below) and is passed through as `--datagen-workers`; it also hard-sets `KORE_VERIFIED_CORRECTNESS`/`KORE_COMPILE_BASELINE`/`KORE_SHAPE_AUGMENT`/`KORE_BENCH_COLD=1` (via `setdefault`, so an explicit env still wins) so every training subprocess it launches inherits the rigorous verification gates.

### The paradigm-v2 fold-in (pause → resume)

The new code must land **before** the build/train stages (they shell out to fresh processes that re-import the changed code), but datagen was already running. `kore_pause_after_datagen.py` + `kore_resume_supervise.py` are the paired mechanism to swap in the new code at the `datagen→build` boundary without losing datagen work:

```mermaid
flowchart LR
  SUP[kore_supervise.py<br/>datagen running] --> PAUSE[kore_pause_after_datagen.py]
  PAUSE -->|"282 shards OR 'stage start: build'"| STOP["stop supervisor, then campaign<br/>write /tmp/kore_paused_after_datagen"]
  STOP --> RES[kore_resume_supervise.py]
  RES -->|sentinel seen| RELAUNCH["run_campaign.py --force<br/>--stages build,midtrain,sft,dpo,grpo,soup,eval"]
  RELAUNCH --> EVAL[build → … → eval on new code]
```

- **`kore_pause_after_datagen.py`** polls every 30s for the boundary - trigger = **all 282 group shards** present in `data/full14b/groups/` **OR** the log shows `stage start: build`. On trigger it stops the **supervisor first** (so it can't relaunch into build), then the campaign + spawn workers, and writes the sentinel `/tmp/kore_paused_after_datagen`. Datagen shards are on disk, so nothing is lost; build is re-run fresh from the new code on resume.
- **`kore_resume_supervise.py`** waits for that sentinel, reaps stale procs, then relaunches with `--stages build,midtrain,sft,dpo,grpo,soup,eval --force` (datagen is done and skipped; `--force` re-runs **build** on the new code, which a fresh process picks up via the editable install) and supervises `build→eval` across transient deaths (same bounded-retry + ALERT machinery as `kore_supervise.py`). Writes its log path to `/tmp/kore_resume_logpath.txt`.

---

## Ephemeral-node resume playbook

Files persist under your account, and the campaign is manifest + shard resumable. If a reservation ends mid-run: re-reserve the node, then re-run `bash scripts/tmux_campaign.sh` - it continues from where it stopped.

---

## Smoke tests

```bash
PYTHONPATH=. python scripts/smoke_env.py          # GPU/env sanity
PYTHONPATH=. python scripts/test_amd_gateway.py   # teacher gateway key
PYTHONPATH=. python scripts/grpo_smoke.py --task rmsnorm_aiter   # a few real GRPO steps
bash scripts/launch_distributed.sh sft configs/sft_14b_full.json --dry-run
```

See also: [`configs/`](../configs/README.md), [`docs/DISTRIBUTED.md`](../docs/DISTRIBUTED.md).
