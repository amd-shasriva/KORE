#!/usr/bin/env python3
"""Standalone SFT retention gate - the gating test WITHOUT any (re)training.

The campaign's sft stage couples a full-FT (re)train with the retention gate, so
"just gate the finished SFT" ends up re-running SFT when the step schedule rescales
to a different GPU count. This script runs ONLY the gate: it scores the base model
vs the finished SFT checkpoint on the retention suite and applies the gate, then
stops (it never touches DPO). On PASS it marks 'sft' done in the campaign manifest
so a later `run_campaign` resume proceeds straight to DPO.

It reuses the fixed gate stack: the hardened HumanEval parse, the per-benchmark
score CACHE (so a mid-gate kill resumes), the capability-only gate keys (mtbench is
stub-judged -> advisory), and GPU-aware loading (pins device_map to --gpu-ids so it
stays on the free GPUs of a shared node). 2x 14B fits on 1-2 GPUs.

    python scripts/run_sft_gate.py --candidate runs/full/sft --gpu-ids 3,4 --mark-done
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile

from kore.ops.runtime import SecurityError, deprecated_entrypoint

# capability benchmarks with real (non-stub) scorers; mtbench uses the length/overlap
# stub judge here, so it is advisory (not a hard gate) - matches run_campaign.
_GATE_KEYS = ("mmlu", "humaneval", "ifeval", "bfcl", "livecodebench")


def _fp(p: str) -> str:
    pp = Path(str(p))
    try:
        key = pp / "config.json"
        mt = key.stat().st_mtime if key.exists() else pp.stat().st_mtime
    except Exception:  # noqa: BLE001
        mt = 0.0
    return hashlib.sha1(f"{pp}|{mt:.0f}".encode()).hexdigest()[:12]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-14B")
    ap.add_argument("--candidate", default="runs/full/sft")
    ap.add_argument("--gpu-ids", default="",
                    help="comma-separated HIP/torch GPU indices to pin to (NOT rocm-smi "
                         "physical ids - they differ on this node; use "
                         "run_sft_gate_dynamic.sh / gpu_pick_hip.py to map physical->HIP)")
    ap.add_argument("--epsilon", type=float, default=0.02)
    ap.add_argument("--manifest", default="data/full14b/campaign_manifest.json")
    ap.add_argument("--mark-done", action="store_true",
                    help="on PASS, add 'sft' to the manifest done_stages")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the deprecated gate plan without loading a model")
    a = ap.parse_args(argv)
    if not deprecated_entrypoint(
        "scripts/run_sft_gate.py",
        "run retention evaluation in a scheduler allocation and record its result "
        "through the supported campaign entrypoint",
        dry_run=a.dry_run,
    ):
        print(
            f"[gate] DRY-RUN base={a.base} candidate={a.candidate} "
            f"manifest={a.manifest} mark_done={a.mark_done}"
        )
        return 0

    # Pin GPUs via the ENVIRONMENT *before* importing anything that initialises
    # torch/HIP (the kore.eval.* imports below pull torch in transitively). Setting
    # HIP_VISIBLE_DEVICES after torch's first CUDA context is a silent no-op - which
    # is why an in-Python device_map pin can leak the model onto busy/co-tenant GPUs.
    # HIP only (not ROCR) to avoid a broken composed remap. If a launcher already
    # masked the devices, respect that existing mask.
    if a.gpu_ids.strip() and not os.environ.get("HIP_VISIBLE_DEVICES"):
        os.environ["HIP_VISIBLE_DEVICES"] = a.gpu_ids.strip()
    visible = os.environ.get("HIP_VISIBLE_DEVICES", "(all)")

    from kore.eval.gates import format_gate_report, retention_gate
    from kore.eval.retention import run_retention_suite
    from kore.policy.serve import load_generate

    cache_dir = Path(a.candidate).parent / "retention_cache"
    print(f"[gate] base={a.base} candidate={a.candidate} "
          f"HIP_VISIBLE_DEVICES={visible} cache={cache_dir}", flush=True)

    # The env mask above already restricts device_map="auto" to the idle GPUs, so
    # pass gpu_ids=None to avoid a redundant (and too-late) in-Python remask.
    base_gen = load_generate(a.base, gpu_ids=None)
    cand_gen = load_generate(a.candidate, gpu_ids=None)
    base = run_retention_suite(base_gen, cache_dir=cache_dir,
                               cache_tag=f"sft_base_{_fp(a.base)}")
    cand = run_retention_suite(cand_gen, cache_dir=cache_dir,
                               cache_tag=f"sft_cand_{_fp(a.candidate)}")

    b = {k: v for k, v in base["scores"].items() if k in _GATE_KEYS}
    c = {k: v for k, v in cand["scores"].items() if k in _GATE_KEYS}
    res = retention_gate(b, c, epsilon=a.epsilon)
    print(format_gate_report(res, title="KORE retention gate [sft] (standalone)"), flush=True)
    print(f"[gate] base scores: {base['scores']}", flush=True)
    print(f"[gate] cand scores: {cand['scores']}", flush=True)
    print(f"[gate] mtbench (advisory, stub-judged): base={base['scores'].get('mtbench')} "
          f"cand={cand['scores'].get('mtbench')}", flush=True)

    if not res.passed:
        print("[gate] RESULT: FAIL", flush=True)
        return 1
    print("[gate] RESULT: PASS", flush=True)
    if a.mark_done:
        mp = Path(a.manifest)
        info = mp.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise SecurityError(f"refusing non-regular campaign manifest: {mp}")
        if info.st_uid != os.getuid():
            raise SecurityError(f"campaign manifest owner mismatch: {mp}")
        m = json.loads(mp.read_text())
        ds = set(m.get("done_stages", []))
        ds.add("sft")
        m["done_stages"] = sorted(ds)
        m["sft_ckpt"] = a.candidate
        fd, raw_tmp = tempfile.mkstemp(prefix=f".{mp.name}.", dir=mp.parent)
        tmp = Path(raw_tmp)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w") as handle:
                handle.write(json.dumps(m, indent=2) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, mp)
        finally:
            tmp.unlink(missing_ok=True)
        print(f"[gate] marked sft done in manifest -> {m['done_stages']} "
              f"(a later run_campaign resume proceeds to DPO)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
