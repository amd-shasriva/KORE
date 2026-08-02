# End-to-end serving gate — provisioning a real backend on gfx950

`kore/eval/e2e_sglang_vllm.py` is KORE's last gate: a kernel that wins in the isolated
microbenchmark verifier only counts if the win **survives inside a production inference server
with no accuracy regression** (KORE.pdf §4.7). This document is the operator's half — how to get a
real OpenAI-compatible server running on this hardware, how to point the gate at it, and what it
measured when we did.

Until this was written the gate had never been reached. Neither `vllm` nor `sglang` was installed
anywhere on the node, so `VLLM_AVAILABLE` and `SGLANG_AVAILABLE` were both `False` and
`e2e_throughput` / `e2e_accuracy` raised `E2ENotProvisioned`. The gate now runs, and
`runs/e2e_gate/` holds the first artifacts it has ever produced.

---

## The architecture: the engine does not live in the training environment

**This split is permanent on this node, for the same class of reason `docs/P0_RESULTS.md` records
for the AITER baseline.** It is a deliberate design property, not a workaround.

The gate module never imports vLLM or SGLang. It speaks HTTP to an OpenAI-compatible
`/v1/chat/completions` endpoint using `requests`, falling back to stdlib `urllib`. Consequently:

- the inference engine and its pinned `torch` live in **their own container or venv**;
- `/home/shasriva/kore-venv` — `torch 2.10.0+rocm7.0` + `triton 3.6.0` + `aiter`, the stack the
  14B SFT/DPO/GRPO stages run on — never has to satisfy an engine's dependency solver. vLLM pins
  torch to an exact build (`torch==2.11.0+gitd0c8b1f` for vLLM 0.26.0); installing it into the
  training venv would replace the working ROCm stack;
- `VLLM_AVAILABLE` / `SGLANG_AVAILABLE` are **diagnostics, not preconditions**. Both read `False`
  on the training box and the gate still runs for real. Do not "fix" them by installing an engine
  into the training venv.

```
  training venv (untouched)              container / separate venv
  ┌───────────────────────────┐          ┌──────────────────────────────┐
  │ kore.eval.e2e_sglang_vllm │  HTTP    │ SGLang or vLLM on ROCm       │
  │ torch 2.10.0+rocm7.0      │ ───────► │ its own torch, its own glibc │
  │ triton 3.6.0, aiter       │  :30000  │ serving the model on 1 GPU   │
  └───────────────────────────┘          └──────────────────────────────┘
```

---

## Why the prebuilt vLLM-ROCm wheels cannot be installed on this host

This was tried first, because a venv is lighter than a container. It is **blocked by glibc**, and
the block is in the dependency, not in vLLM:

| Property | This host | vLLM ROCm wheel index |
| --- | --- | --- |
| ROCm | 7.2.3 | `vllm-0.26.0+rocm723` — matches |
| Python | 3.12 via `uv --managed-python` | `cp312` — matches |
| glibc | **2.34** | `vllm-…-manylinux_2_34_x86_64.whl` — matches |
| torch (vLLM's exact pin) | — | `torch-2.11.0+gitd0c8b1f-cp312-cp312-**manylinux_2_35**_x86_64.whl` |

The vLLM wheel itself is built for `manylinux_2_34` and would install. Its pinned torch is
published **only** for `manylinux_2_35`, and every torch on
`https://wheels.vllm.ai/rocm/` carries that tag (checked for both the current `rocm723` line and
the older `0.18.0/rocm700` line). `uv` therefore fails to resolve:

```
× No solution found when resolving dependencies:
╰─▶ Because torch==2.11.0+gitd0c8b1f has no wheels with a matching platform tag
    (e.g., `manylinux_2_34_x86_64`) and all versions of vllm depend on
    torch==2.11.0+gitd0c8b1f, we can conclude that all versions of vllm cannot be used.
```

Note the upstream docs (`docs.vllm.ai` "GPU / ROCm") still describe only the `rocm700` and
`rocm721` variants at `glibc >= 2.35`; the index has since moved to a `rocm723` /
`manylinux_2_34` vLLM wheel, so read the index, not the table. The remaining venv routes and why
they were not taken inside the time budget:

- **Force the `manylinux_2_35` torch onto glibc 2.34** — installable with `--no-deps` plus a tag
  override, but the ABI mismatch it papers over is exactly the kind of failure that surfaces as a
  segfault mid-benchmark. Not worth it for a gate whose output is a numeric claim.
- **Build vLLM from source against a glibc-2.34-compatible ROCm torch** — supported
  (`PYTORCH_ROCM_ARCH="gfx950"; python3 setup.py develop`), but a multi-hour compile, and AMD's own
  index publishes no `gfx950` torch that both matches this glibc and satisfies vLLM's exact pin.
- **Upgrade the host glibc** — out of scope on a shared box.

**A container sidesteps all of this**, because it brings its own userspace: glibc, ROCm
user-mode libraries, torch and the engine. Only the kernel driver (`/dev/kfd`, `amdgpu`) is shared,
and that is version-tolerant. Use a container.

> `/home/shasriva/kore-vllm-venv` was created during this investigation (Python 3.12.13 via
> `uv venv --managed-python`) and left in place as evidence; it contains only `pip`. Nothing was
> ever installed into `/home/shasriva/kore-venv`.

---

## Serving a model (verified procedure)

### 1. Pick a free GPU and find its render node

Only expose the one card. `--device=/dev/dri` would hand the container every GPU on the node.

```bash
rocm-smi --showuse                    # pick an idle GPU; note its index
rocm-smi --showbus                    # index -> PCI address, e.g. GPU[4] = 0000:88:00.0
for d in /dev/dri/renderD*; do        # PCI address -> render node
  echo "$d -> $(basename "$(readlink -f /sys/class/drm/$(basename $d)/device)")"
done                                  # 0000:88:00.0 -> /dev/dri/renderD168
```

### 2. Launch the server

The image used here — and the only one these measurements were taken with — is an SGLang-on-ROCm
build for MI35x, `primussafe/sglang:v0.5.12-rocm720-mi35x-profilerfix`, already present on this
node. The published vLLM equivalents `rocm/vllm:latest` and `vllm/vllm-openai-rocm:latest` both
resolve in the registry and expose the same OpenAI API, so the gate would not need to change to
use them, but neither was pulled or exercised here; treat the serve flags below as SGLang-specific.

```bash
SNAP=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/<sha>

docker run -d --name kore-e2e-sglang \
  --device=/dev/kfd --device=/dev/dri/renderD168 \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined --shm-size 32g \
  -v "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-14B:/models/qwen3-14b:ro" \
  -p 127.0.0.1:30000:30000 \
  primussafe/sglang:v0.5.12-rocm720-mi35x-profilerfix \
  python -m sglang.launch_server \
    --model-path /models/qwen3-14b/snapshots/<sha> \
    --served-model-name Qwen3-14B \
    --host 0.0.0.0 --port 30000 \
    --attention-backend aiter --mem-fraction-static 0.898
```

- Mount the **whole** `models--Qwen--Qwen3-14B` directory, not just the snapshot: the snapshot's
  files are relative symlinks into a sibling `blobs/`, which will not resolve otherwise. `:ro`
  keeps a root-in-container process from writing into your cache.
- `-p 127.0.0.1:30000:30000` binds loopback only; do not publish an unauthenticated engine.
- `--attention-backend` is the flag that swaps the kernel under test.
- Wait for readiness with `curl -s -o /dev/null -w '%{http_code}' localhost:30000/health` returning
  `200`. Qwen3-14B takes roughly 90 s (weight load ~5 s, CUDA-graph capture ~25 s).
- Verify the card: `docker exec kore-e2e-sglang python -c "import torch; print(torch.cuda.device_count())"`
  must print `1`.

### 3. Run the gate

From **any** Python that can reach the port — including the training venv, read-only:

```bash
# measure one endpoint (no decision: the gate cannot accept without a baseline)
python -m kore.eval.e2e_sglang_vllm --engine sglang --model Qwen3-14B \
  --base-url http://127.0.0.1:30000 --requests 32

# the real protocol: two servers differing ONLY in the registered kernel
python -m kore.eval.e2e_sglang_vllm --engine sglang --model Qwen3-14B \
  --base-url http://127.0.0.1:30000 --candidate-url http://127.0.0.1:30001 \
  --served-kernel fused_add_rmsnorm_bf16 \
  --requests 32 --json runs/e2e_gate/<run>.json

# or gate against a previously measured stock number, one server at a time
python -m kore.eval.e2e_sglang_vllm --engine sglang --model Qwen3-14B \
  --base-url http://127.0.0.1:30001 --baseline-tokens-per-s 95.44 --baseline-accuracy 1.0
```

Programmatically, `e2e_gate_endpoints(...)` does the same in one call; `e2e_measure(...)` returns
the `(throughput, accuracy)` pair for a single endpoint.

### 4. Point the test suite at it

```bash
KORE_E2E_BASE_URL=http://127.0.0.1:30000 \
KORE_E2E_CANDIDATE_URL=http://127.0.0.1:30001 \
KORE_E2E_MODEL=Qwen3-14B KORE_E2E_ENGINE=sglang \
python -m pytest -m gpu tests/test_gpu_e2e_serving_gate.py -q
```

Unset or unreachable, every test skips with the reason. `tests/test_e2e_serving_gate.py` runs the
identical client path against a stdlib stub endpoint and needs no GPU, no engine and no network,
so CI keeps testing the logic.

---

## Measured on this node

**Model:** `Qwen/Qwen3-14B` (bf16, TP=1) — the base model of KORE's own 14B training stack. No
mid-train checkpoint existed yet, so this is the base weights, not a KORE-trained model.
**Hardware:** one AMD Instinct MI350X (gfx950/CDNA4), GPU 4, ROCm 7.2.3 host driver.
**Engine:** SGLang 0.5.12 in the ROCm 7.2.0 container above.
**Workload:** the module's built-in 4-prompt workload, 32 requests × 128 max new tokens for
throughput, 512 max new tokens for accuracy, temperature 0, issued sequentially.

| # | Run | throughput (tok/s) | accuracy | decision |
| --- | --- | --- | --- | --- |
| 1 | AITER attention measured twice against itself (null control) | 95.70 → 95.46 (−0.25%) | 1.00 → 1.00 (4/4) | **REJECT** |
| 2 | baseline Triton attention → candidate AITER attention | 99.35 → 95.44 (−3.93%) | 1.00 → 1.00 (4/4) | **REJECT** |
| 3 | baseline AITER attention → candidate Triton attention | 95.44 → 96.62 (+1.24%) | 1.00 → 1.00 (4/4) | **ACCEPT** |

Artifacts: `runs/e2e_gate/qwen3_14b_sglang_gfx950.json`,
`…_aiter_vs_triton.json`, `…_triton_vs_aiter.json`.

**The gate mechanism works.** Row 1 is the null control — the same server, no kernel change,
measured twice — and the gate rejects it, because `throughput_improved` requires a strict
improvement over the baseline. A no-op does not pass. Rows 2 and 3 show both decisions are
reachable from real measurements. Accuracy held at 4/4 in every run, so every decision here turned
on throughput alone.

**The kernel comparison is suggestive, not established, and the numbers say why.** Triton
attention came out ahead of AITER in both pairings (+4.1% in row 2's framing, +1.24% in row 3), so
the *direction* is consistent. But the two Triton measurements were **99.35 and 96.62 tok/s** —
the same backend, the same flags, the same card, differing only in a server restart. That is a
2.8% spread, which is the same order as the effect being measured, and an order of magnitude
larger than the 0.25% within-process null in row 1. **The honest read is that the two attention
backends are within restart-to-restart variance of each other on this workload.** Two paired
measurements cannot separate them.

That distinction matters more than the result. The cheap null control (row 1) understates the real
uncertainty, because it holds the server process fixed and so measures only sampling noise, not
the run-to-run variance a genuine kernel A/B is subject to — a candidate must be relaunched to be
tested at all. **Any real gate decision on a margin below ~5% needs repeated relaunches of both
sides and a paired statistic over them** (`kore/eval/paired_stats.py` provides exactly that:
paired bootstrap CI, Wilcoxon, sign test). Nothing in this run should be cited as evidence that
either attention backend is faster.

### Reproduce

```bash
python -m kore.eval.e2e_sglang_vllm --engine sglang --model Qwen3-14B \
  --base-url http://127.0.0.1:30000 --candidate-url http://127.0.0.1:30001 \
  --requests 32 --max-new-tokens 128 --accuracy-max-new-tokens 512 \
  --json runs/e2e_gate/rerun.json
```

---

## Caveats an operator must know

- **Measure the variance you are actually subject to.** Re-measuring one live server estimates
  sampling noise only (0.25% here). Relaunching the same server and measuring again gave a 2.8%
  spread. A kernel A/B necessarily crosses a relaunch, so the second number is the relevant one,
  and a single paired run cannot resolve a margin smaller than it.
- **`tokens/s` is a whitespace-word proxy by default.** `_count_tokens_default` counts
  `\S+` runs, not BPE tokens, so the absolute number is not comparable to a vLLM benchmark
  harness. It is *fair* — both sides of a comparison use it — and exact accounting is available by
  passing `count_tokens=` (e.g. `lambda s: len(tok.encode(s))`). Only compare numbers produced the
  same way.
- **Requests are issued sequentially.** `Workload.concurrency` is carried but `e2e_throughput`
  replays the workload in a plain loop, so what is measured is single-stream decode latency, not
  server saturation throughput. For a kernel A/B that is arguably the cleaner signal; for a
  capacity claim it is the wrong measurement.
- **The prompts repeat.** `_workload_prompts` cycles the four built-in prompts, so a server with a
  radix/prefix cache serves most requests warm. Identical on both sides, but it means prefill is
  nearly free and the number is decode-dominated. Pass `Workload(prompts=[...])` with distinct
  prompts, or launch with `--disable-radix-cache`, for a prefill-sensitive comparison.
- **Reasoning models need a real token budget.** Qwen3 thinks before answering and thinking is on
  by default. At `max_tokens=64` every reply is truncated mid-thought and scores 0/4 — which looks
  like an accuracy catastrophe and is actually a budget error. Hence the separate
  `--accuracy-max-new-tokens` (512). Relatedly, a server launched with `--reasoning-parser qwen3`
  returns `content: null` on a truncated reply and puts the text in `reasoning_content`; the client
  falls back to that field so those tokens are still counted.
- **The built-in accuracy workload is a smoke test, not an eval.** Four trivial questions detect a
  kernel that corrupts generation; they will not detect a subtle quality regression. Pass
  `tasks=` (or use `kore.eval.retention`) for a real held-out set before making a no-regression
  claim.
- **A `--mem-fraction-static` given to both servers can still yield different KV pools** (1.08M vs
  1.30M tokens here) because the attention backends reserve different workspaces. At batch size 1
  this does not affect decode rate, but it is a real asymmetry if you move to a batched workload.
- **Two servers do not fit on one MI350X at `--mem-fraction-static 0.898`.** Either lower it for
  both, or measure sequentially and use `--baseline-tokens-per-s`, which is what was done here.

## Teardown

```bash
docker rm -f kore-e2e-sglang kore-e2e-sglang-triton
```

Leaving a 14B server resident holds ~250 GB of HBM on that card.
