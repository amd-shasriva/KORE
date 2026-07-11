"""Task data-coverage audit (Pillar 2): make "100% coverage" measurable + fixable.

"Cover 100% of everything" has two senses; this module owns the first and reports
on the second:

  1. DATA coverage — every TRAIN task must have non-empty ``repair`` + ``groups`` +
     ``wins`` shards. A task with a missing/empty shard is a hole: the policy never
     sees repair transitions, preferences, or a win demo for that operator. The
     shipped data had ~5-28 tasks short of full coverage.
  2. SPACE coverage — the op x dtype frontier the task generator emits (see
     ``kore.tasks.generate_ops.FAMILY_DTYPES``). Reported here per (family, dtype)
     so gaps (e.g. no generated fp8/int8 elementwise) are visible.

:func:`coverage_report` returns a structured report; :func:`undercovered_tasks`
lists exactly which tasks need (re)generation, so datagen can TARGET the holes
instead of blindly re-running everything. Pure (registry + filesystem), no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

# The three per-task datagen products that together constitute full coverage.
REQUIRED_KINDS: tuple[str, ...] = ("repair", "groups", "wins")


def _shard_count(data_root: Path, kind: str, task_id: str) -> int:
    """Number of JSONL lines in ``<data_root>/<kind>/<task_id>.jsonl`` (0 if absent)."""
    p = data_root / kind / f"{task_id}.jsonl"
    if not p.is_file():
        return 0
    try:
        return sum(1 for ln in p.read_text().splitlines() if ln.strip())
    except OSError:
        return 0


def task_coverage(data_root, task_ids: Iterable[str],
                  kinds: tuple[str, ...] = REQUIRED_KINDS) -> dict[str, dict]:
    """Per-task shard counts + a ``full`` flag. ``{task_id: {kind: n, ..., full: bool}}``."""
    data_root = Path(data_root)
    out: dict[str, dict] = {}
    for tid in task_ids:
        counts = {k: _shard_count(data_root, k, tid) for k in kinds}
        counts["full"] = all(counts[k] > 0 for k in kinds)
        out[tid] = counts
    return out


def _train_task_ids() -> list[str]:
    """Train (non-held-out) task ids from the registry; [] if unavailable."""
    try:
        from kore.tasks.registry import train_tasks
        return sorted(t.task_id for t in train_tasks())
    except Exception:  # noqa: BLE001
        return []


def undercovered_tasks(data_root, task_ids: Optional[Iterable[str]] = None,
                       kinds: tuple[str, ...] = REQUIRED_KINDS) -> dict[str, list[str]]:
    """``{task_id: [missing_kinds]}`` for every train task missing a required kind."""
    ids = list(task_ids) if task_ids is not None else _train_task_ids()
    cov = task_coverage(data_root, ids, kinds)
    return {tid: [k for k in kinds if c[k] == 0]
            for tid, c in cov.items() if not c["full"]}


# The FRONTIER ROCm/AMD inference-kernel capabilities a world-class dataset should
# cover, grounded in AITER (the default ROCm kernel library for vLLM/SGLang) +
# Composable Kernel. Each entry: capability -> (substrings that identify a covering
# task.operation/task_id, note). "Covered" = >=1 registry task matches. This makes
# frontier holes MEASURABLE (rather than asserted); closing a hole requires authoring
# + GPU-verifying a seed (offline fabrication of unverified kernels is a shortcut).
FRONTIER_OPS: dict[str, tuple[tuple[str, ...], str]] = {
    "gemm_dense":        (("gemm",), "dense GEMM (hipBLASLt/CK)"),
    "gemm_fp8":          (("gemm_fp8", "a8w8"), "FP8 W8A8 GEMM (CK/ASM)"),
    "gemm_int8":         (("a8w8_int8", "gemm_a8w8"), "INT8 W8A8 GEMM"),
    "gemm_blockscale":   (("blockscale",), "block-scale FP8 GEMM (DeepSeek-V3)"),
    "gemm_batched":      (("batched_gemm",), "batched GEMM (bmm)"),
    "gemm_mxfp4":        (("mxfp4", "a4w4", "fp4"), "MXFP4 / A4W4 GEMM (MI350)"),
    "attention_mha":     (("flash_attn", "mha"), "flash / MHA prefill+decode"),
    "attention_mla":     (("mla",), "multi-head latent attention (DeepSeek)"),
    "attention_paged":   (("paged_attn", "paged"), "paged KV-cache attention"),
    "moe_router":        (("topk_softmax", "moe_router", "topk"), "MoE top-k routing"),
    "moe_fused":         (("fused_moe", "moe_silu", "moe"), "fused MoE expert compute"),
    "norm_rms":          (("rmsnorm",), "RMSNorm"),
    "norm_layer":        (("layernorm",), "LayerNorm"),
    "norm_fused_add":    (("fused_add_rmsnorm",), "fused residual-add RMSNorm"),
    "rope":              (("rope",), "rotary position embedding"),
    "quant_fp8":         (("quant_fp8", "quant"), "FP8 per-token/channel quant"),
    "activation_gated":  (("silu_mul", "gelu_mul", "reglu"), "gated MLP activation"),
    "softmax":           (("softmax",), "softmax"),
    "elementwise":       (("relu", "gelu", "add", "mul"), "elementwise/pointwise"),
    "reduction":         (("row_sum", "row_max", "row_rms"), "row reductions"),
    # Known frontier gaps that need authoring + GPU verification (no shortcut):
    "softmax_causal":    (("causal_softmax", "masked_softmax"), "causal/masked softmax"),
    "kv_cache_ops":      (("kv_cache", "reshape_cache", "append_kv"), "KV-cache write/gather"),
    "rope_kvcache":      (("rope_kvcache", "rope_cache"), "fused RoPE + KV-cache write"),
    "quant_int4":        (("int4", "w4a16", "awq", "gptq"), "INT4/MXFP4 weight quant"),
    "sampling":          (("sampling", "top_p", "top_k_sampl"), "top-p/top-k token sampling"),
    "collective":        (("allreduce", "all_reduce", "all_gather", "reduce_scatter"),
                          "multi-GPU collectives (needs multi-GPU harness)"),
}


def frontier_coverage() -> dict:
    """Which FRONTIER_OPS capabilities the registry currently covers vs the holes."""
    try:
        from kore.tasks.registry import all_tasks
        ops = [f"{getattr(t, 'operation', '') or ''} {t.task_id}".lower() for t in all_tasks()]
    except Exception:  # noqa: BLE001
        return {}
    covered, missing = {}, {}
    for cap, (subs, note) in FRONTIER_OPS.items():
        n = sum(1 for o in ops if any(s in o for s in subs))
        (covered if n else missing)[cap] = {"n_tasks": n, "note": note}
    return {
        "n_capabilities": len(FRONTIER_OPS),
        "n_covered": len(covered),
        "n_missing": len(missing),
        "frontier_pct": round(100.0 * len(covered) / len(FRONTIER_OPS), 1),
        "covered": covered,
        "missing": missing,
    }


def space_coverage() -> dict:
    """The generated op x dtype frontier (family -> dtypes) + per-dtype gaps."""
    try:
        from kore.tasks._genops import DTYPES
        from kore.tasks.generate_ops import FAMILY_DTYPES
    except Exception:  # noqa: BLE001
        return {}
    all_dtypes = set(DTYPES)
    per_family = {fam: {"emitted": list(dts),
                        "missing": sorted(all_dtypes - set(dts))}
                  for fam, dts in FAMILY_DTYPES.items()}
    return {"all_dtypes": sorted(all_dtypes), "per_family": per_family}


def coverage_report(data_root, task_ids: Optional[Iterable[str]] = None) -> dict:
    """Full data + space coverage report."""
    ids = list(task_ids) if task_ids is not None else _train_task_ids()
    cov = task_coverage(data_root, ids)
    n = len(ids)
    n_full = sum(1 for c in cov.values() if c["full"])
    per_kind = {k: sum(1 for c in cov.values() if c[k] > 0) for k in REQUIRED_KINDS}
    under = undercovered_tasks(data_root, ids)
    return {
        "n_train_tasks": n,
        "n_full_coverage": n_full,
        "coverage_pct": round(100.0 * n_full / n, 2) if n else 0.0,
        "per_kind_covered": per_kind,
        "per_kind_pct": {k: round(100.0 * v / n, 2) if n else 0.0
                         for k, v in per_kind.items()},
        "n_undercovered": len(under),
        "undercovered": under,
        "space": space_coverage(),
        "frontier": frontier_coverage(),
    }


def _main(argv: Optional[list[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="KORE task data-coverage audit")
    p.add_argument("data_root", help="campaign data root (e.g. data/full14b)")
    p.add_argument("--json", action="store_true", help="emit the full JSON report")
    p.add_argument("--undercovered", action="store_true", help="list only the holes")
    a = p.parse_args(argv)
    rep = coverage_report(a.data_root)
    if a.json:
        print(json.dumps(rep, indent=2))
    elif a.undercovered:
        for tid, missing in sorted(rep["undercovered"].items()):
            print(f"{tid}: missing {', '.join(missing)}")
    else:
        print(f"train tasks: {rep['n_train_tasks']}  full coverage: "
              f"{rep['n_full_coverage']} ({rep['coverage_pct']}%)  "
              f"undercovered: {rep['n_undercovered']}")
        for k, pct in rep["per_kind_pct"].items():
            print(f"  {k}: {rep['per_kind_covered'][k]}/{rep['n_train_tasks']} ({pct}%)")
        fr = rep.get("frontier") or {}
        if fr:
            print(f"\nfrontier op coverage: {fr['n_covered']}/{fr['n_capabilities']} "
                  f"({fr['frontier_pct']}%)")
            for cap, d in sorted(fr.get("missing", {}).items()):
                print(f"  MISSING {cap}: {d['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
