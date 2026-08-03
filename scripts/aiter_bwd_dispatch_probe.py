"""Probe which backward kernel AITER dispatches to on gfx950, and time it.

Why this exists
---------------
HipKittens (arXiv 2511.08083) reports that AITER's Llama GQA *backward* reaches
only ~30% of SoTA on MI355X. That is a claim about a paper's harness, not ours.
This script answers the mechanical question underneath it on our own hardware:
for a given attention config, *which* AITER backward kernel actually runs?

``aiter.ops.mha._flash_attn_backward`` picks between two implementations:

  * ``fmha_v3_bwd`` -- the hand-written asm backward (the fast path), and
  * ``mha_bwd``     -- the Composable Kernel backward (the fallback),

via ``can_impl_fmha_v3_bwd(...) | can_impl_fmha_v3_bwd_gfx950()``. Both gates
reject ``deterministic=True`` (the generic gate unconditionally; the gfx950 gate
unless ``seqlen_k <= 256``), and ``flash_attn_func`` *defaults* to
``deterministic=True``.

Phase A observes the dispatch with no kernel build at all: it replaces the two
dispatch targets with sentinels that raise on entry. The real gate logic in
``_flash_attn_backward`` still runs unmodified, so the observation is faithful,
but neither kernel is ever compiled or launched. This matters because neither
backward module is prebuilt in this environment -- a naive probe pays a long JIT
build before telling you anything.

Phase B restores the real ops and times the backward with the SAME cold-cache
CUDA-event median the KORE task harness uses (``kore.tasks._attn_common``:
L2 flush between iters, median of sorted per-iter event deltas).

Every row records the AITER commit so a number can never drift from the build
that produced it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback


def aiter_commit(aiter_file: str) -> dict:
    """Resolve the git commit of the *installed* aiter (not any other clone)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(aiter_file)))
    info = {"aiter_path": root, "commit": None, "committed_at": None, "dirty": None}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
        info["committed_at"] = subprocess.check_output(
            ["git", "-C", root, "log", "-1", "--format=%cI"], text=True).strip()
        info["dirty"] = bool(subprocess.check_output(
            ["git", "-C", root, "status", "--porcelain"], text=True).strip())
    except Exception:  # noqa: BLE001 - a non-git install still yields a usable row
        pass
    return info


class _Dispatched(Exception):
    """Raised by a sentinel to report which backward AITER chose."""

    def __init__(self, which: str):
        super().__init__(which)
        self.which = which


def _configs():
    """Sweep the axes the gfx950 gate keys on, anchored on real model shapes.

    ``D`` crosses the ``hdim_q > 64 and hdim_q <= 128`` window from both sides,
    ``hkv`` crosses MHA/GQA/MQA, and ``deterministic`` crosses the flag that
    ``flash_attn_func`` defaults to True.
    """
    cfgs = []
    # Qwen3-Coder-30B-A3B-Instruct attention geometry (32 Q heads / 4 KV, D=128)
    # at training-like sequence lengths, both causal senses.
    for causal in (True, False):
        for det in (True, False):
            cfgs.append(dict(name=f"qwen3coder_gqa_D128_causal{int(causal)}_det{int(det)}",
                             B=2, S=2048, H=32, HKV=4, D=128,
                             causal=causal, deterministic=det, window=(-1, -1, 0)))
    # head_dim sweep across the gate window (64 excluded: the test is strict >64).
    for D, H, HKV in ((64, 32, 4), (128, 32, 4), (192, 16, 4), (256, 16, 4)):
        cfgs.append(dict(name=f"hdim{D}_gqa_causal_det0",
                         B=2, S=2048, H=H, HKV=HKV, D=D,
                         causal=True, deterministic=False, window=(-1, -1, 0)))
    # MHA vs MQA at the favoured head_dim.
    for HKV, tag in ((32, "mha"), (1, "mqa")):
        cfgs.append(dict(name=f"{tag}_D128_causal_det0",
                         B=2, S=2048, H=32, HKV=HKV, D=128,
                         causal=True, deterministic=False, window=(-1, -1, 0)))
    # Sliding-window: the gate rejects swa outright.
    cfgs.append(dict(name="swa1024_gqa_D128_causal_det0",
                     B=2, S=2048, H=32, HKV=4, D=128,
                     causal=True, deterministic=False, window=(1024, 0, 0)))
    # Short sequence: the only case where gfx950 tolerates deterministic=True.
    cfgs.append(dict(name="short_S256_gqa_D128_causal_det1",
                     B=2, S=256, H=32, HKV=4, D=128,
                     causal=True, deterministic=True, window=(-1, -1, 0)))
    return cfgs


def _make_qkv(cfg, torch):
    dev = "cuda"
    def mk(h):
        return torch.randn(cfg["B"], cfg["S"], h, cfg["D"], dtype=torch.bfloat16,
                           device=dev, requires_grad=True)
    return mk(cfg["H"]), mk(cfg["HKV"]), mk(cfg["HKV"])


def _forward(cfg, torch, aiter):
    q, k, v = _make_qkv(cfg, torch)
    out = aiter.flash_attn_func(
        q, k, v, causal=cfg["causal"], window_size=tuple(cfg["window"]),
        deterministic=cfg["deterministic"],
    )
    return q, k, v, out


def phase_a(cfgs, torch, aiter, mha_mod):
    """Observe the dispatch choice without building or launching either kernel."""
    real_v3, real_ck = mha_mod.fmha_v3_bwd, mha_mod.mha_bwd

    def sentinel(which):
        def _fn(*_a, **_kw):
            raise _Dispatched(which)
        return _fn

    rows = []
    for cfg in cfgs:
        row = dict(cfg)
        row["window"] = list(cfg["window"])
        try:
            q, k, v, out = _forward(cfg, torch, aiter)
            dout = torch.randn_like(out)
            mha_mod.fmha_v3_bwd = sentinel("fmha_v3_bwd_asm")
            mha_mod.mha_bwd = sentinel("mha_bwd_ck")
            try:
                out.backward(dout)
                row["dispatch"] = "no_dispatch_observed"
            except _Dispatched as d:
                row["dispatch"] = d.which
            finally:
                mha_mod.fmha_v3_bwd, mha_mod.mha_bwd = real_v3, real_ck
        except Exception as e:  # noqa: BLE001 - record, never fabricate
            row["dispatch"] = "error"
            row["error"] = f"{type(e).__name__}: {e}"[:400]
            mha_mod.fmha_v3_bwd, mha_mod.mha_bwd = real_v3, real_ck
        del_keys = [x for x in ("q", "k", "v") if x in row]
        for x in del_keys:
            row.pop(x)
        print(f"[phase-a] {row['name']:44s} -> {row['dispatch']}", flush=True)
        rows.append(row)
        torch.cuda.empty_cache()
    return rows


def _time_backward(out, dout, leaves, torch, flush_l2, warmup, iters):
    """Cold-cache CUDA-event median, identical in method to the KORE task harness."""
    def step():
        for t in leaves:
            t.grad = None
        out.backward(dout, retain_graph=True)

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        flush_l2()
        st[i].record(); step(); en[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(st, en))
    return times


def _time_sdpa(cfg, torch, flush_l2, warmup, iters):
    """torch SDPA autograd backward on the same shapes, same timing method."""
    import torch.nn.functional as F

    q, k, v = _make_qkv(cfg, torch)
    # SDPA wants (B, H, S, D); enable_gqa lets it broadcast KV heads natively.
    qt, kt, vt = (t.transpose(1, 2) for t in (q, k, v))
    out = F.scaled_dot_product_attention(
        qt, kt, vt, is_causal=cfg["causal"], enable_gqa=(cfg["HKV"] != cfg["H"]))
    dout = torch.randn_like(out)
    return _time_backward(out, dout, (q, k, v), torch, flush_l2, warmup, iters)


def phase_b(cfgs, torch, aiter, mha_mod, flush_l2, warmup, iters, build_budget_s):
    """Time the real AITER backward, and torch SDPA backward, on the same shapes."""
    rows = []
    for cfg in cfgs:
        row = dict(cfg)
        row["window"] = list(cfg["window"])
        # --- AITER ---
        t0 = time.time()
        try:
            q, k, v, out = _forward(cfg, torch, aiter)
            dout = torch.randn_like(out)
            observed = {"which": None}
            real_v3, real_ck = mha_mod.fmha_v3_bwd, mha_mod.mha_bwd

            def wrap(fn, which):
                def _fn(*a, **kw):
                    observed["which"] = which
                    return fn(*a, **kw)
                return _fn

            mha_mod.fmha_v3_bwd = wrap(real_v3, "fmha_v3_bwd_asm")
            mha_mod.mha_bwd = wrap(real_ck, "mha_bwd_ck")
            try:
                times = _time_backward(out, dout, (q, k, v), torch, flush_l2,
                                       warmup, iters)
            finally:
                mha_mod.fmha_v3_bwd, mha_mod.mha_bwd = real_v3, real_ck
            row["aiter_dispatch"] = observed["which"]
            row["aiter_median_ms"] = times[len(times) // 2]
            row["aiter_min_ms"] = times[0]
            row["aiter_max_ms"] = times[-1]
            row["aiter_build_plus_bench_s"] = round(time.time() - t0, 1)
        except Exception as e:  # noqa: BLE001
            row["aiter_error"] = f"{type(e).__name__}: {e}"[:400]
            row["aiter_elapsed_s"] = round(time.time() - t0, 1)
        torch.cuda.empty_cache()

        # --- torch SDPA autograd backward, same shapes/method ---
        try:
            st = _time_sdpa(cfg, torch, flush_l2, warmup, iters)
            row["sdpa_median_ms"] = st[len(st) // 2]
            row["sdpa_min_ms"] = st[0]
        except Exception as e:  # noqa: BLE001
            row["sdpa_error"] = f"{type(e).__name__}: {e}"[:400]
        torch.cuda.empty_cache()

        if row.get("aiter_median_ms") and row.get("sdpa_median_ms"):
            row["sdpa_over_aiter"] = round(
                row["sdpa_median_ms"] / row["aiter_median_ms"], 4)
        print(f"[phase-b] {row['name']:44s} "
              f"dispatch={row.get('aiter_dispatch')} "
              f"aiter={row.get('aiter_median_ms')} "
              f"sdpa={row.get('sdpa_median_ms')} "
              f"err={row.get('aiter_error','')}", flush=True)
        rows.append(row)
        if build_budget_s and (time.time() - t0) > build_budget_s:
            print(f"[phase-b] build budget {build_budget_s}s exceeded; stopping "
                  f"after {len(rows)} configs", flush=True)
            break
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=21)
    ap.add_argument("--phase-b", action="store_true",
                    help="also time the real kernels (may JIT-build for a long time)")
    ap.add_argument("--build-budget-s", type=int, default=2400)
    args = ap.parse_args()

    import torch

    from kore.tasks._attn_common import _flush_l2

    if not torch.cuda.is_available():
        print("no GPU visible; refusing to emit numbers", file=sys.stderr)
        return 2
    arch = (torch.cuda.get_device_properties(0).gcnArchName or "").split(":")[0]

    import aiter
    import aiter.ops.mha as mha_mod

    meta = {
        "arch": arch,
        "device_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "aiter": aiter_commit(aiter.__file__),
        "timing": {
            "method": "cold-cache CUDA-event median (kore.tasks._attn_common)",
            "warmup": args.warmup, "iters": args.iters, "l2_flush": True,
        },
        "host": os.uname().nodename,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(json.dumps(meta, indent=2), flush=True)
    if arch != "gfx950":
        print(f"WARNING: arch is {arch}, not gfx950", file=sys.stderr, flush=True)

    cfgs = _configs()
    result = {"meta": meta, "phase_a_dispatch": [], "phase_b_timing": []}
    try:
        result["phase_a_dispatch"] = phase_a(cfgs, torch, aiter, mha_mod)
    except Exception:  # noqa: BLE001
        result["phase_a_error"] = traceback.format_exc()[-2000:]

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    if args.phase_b:
        try:
            result["phase_b_timing"] = phase_b(
                cfgs, torch, aiter, mha_mod, _flush_l2,
                args.warmup, args.iters, args.build_budget_s)
        except Exception:  # noqa: BLE001
            result["phase_b_error"] = traceback.format_exc()[-2000:]

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
