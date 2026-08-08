#!/usr/bin/env python
"""Rank every mineable task by how much a win on it is actually worth.

The pool-HIP sweep was optimising the wrong quantity. Measured over the 6,457
gated pool tasks and the 20k rows they produced:

  * 86% have a baseline under 100us and the median is 17us. A kernel that
    finishes in 17us is dominated by launch overhead, so there is no tiling, LDS
    staging, MFMA scheduling or pipelining to demonstrate -- the ceiling on
    "skill expressible" is near zero however many of them you mine.
  * primary_elements is ~1M for 84% of them, so the pool is not merely small, it
    is uniformly small: there is no size axis to select along.
  * Attention is 1.3% of the pool and quantization 0.2%, while MoE is a single
    task -- exactly the families carrying the arena's highest bars.
  * The wins reflect that: median 2.29x, max 3171x. A 3000x speedup is not a
    kernel achievement, it is a pathological torch baseline.

Meanwhile the registry holds 434 tasks in those families -- 170 attention
including twenty flash-attention variants, 115 MoE, 91 quant/fp8, 58 gemm -- in
real frontier dtypes (fp8_e4m3fn, int4_w4a16, mxfp4, bf16) and against vendor
baselines. Exactly ONE of them had been mined into v5.

So this ranks by expected value of a win rather than by availability:

    score = family_weight x dtype_weight x baseline_weight x size_weight

Registry tasks outrank pool tasks of the same family because their baseline is a
tuned vendor kernel (AITER, hipBLASLt) rather than an unfused torch module:
beating hipBLASLt by 1.2x is a real result, beating eager torch by 38x is a
statement about the baseline. Low-precision dtypes outrank fp32 because that is
where CDNA4 has hardware the model must learn to reach (MFMA fp8/fp6/fp4), and
because fp32 barely exists in production inference.

    python scripts/select_frontier_tasks.py --out runs/frontier_tasks.txt
    python scripts/select_frontier_tasks.py --report        # no files written
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
REGISTRY = REPO / "kore" / "tasks"
POOL = REPO / "data" / "pool_hip_ok" / "tasks"

#: Families ordered by what the arena actually spends its hardest tasks on and
#: where AMD publishes its highest Opus bars (torch2hip 6.89x, hip2hip 6.69x).
#: A family absent here is not mined at all -- the point of this pass is to stop
#: spending GPU-hours on elementwise activations.
FAMILY_WEIGHT = {
    "attention": 10.0,   # flash/paged/MLA: the hardest kernels in the suite
    "moe": 9.0,          # fused expert dispatch, 2-stage, top-k routing
    "quant": 8.0,        # fp8/mxfp4/int4 -- CDNA4 matrix-core territory
    "gemm": 6.0,         # tiling, LDS, MFMA; the canonical optimisation target
    "norm_fusion": 3.0,  # fused rmsnorm+quant etc: real but lower ceiling
}

FAMILY_PATTERNS = (
    ("attention", r"attn|attention|mha|mqa|gqa|flash|paged|mla|kv_?cache|rope"),
    # (^|_)moe(_|$) rather than \bmoe\b: underscore is a word character, so the
    # word-boundary form matches neither fused_moe_silu_bf16 nor
    # genb_moe_align_block_offsets_bf16 -- it silently dropped real MoE tasks
    # while the substring form would swallow "moment", "smoequant" and friends.
    ("moe", r"(?:^|_)moe(?:_|$)|expert|top_?k|router|gate_?net|grouped_gemm"),
    ("quant", r"quant|fp8|mxfp|fp4|fp6|int8|int4|w4a16|a8w8|awq|gptq|blockscale|smoothquant"),
    ("gemm", r"gemm|matmul|\bbmm\b|batched_gemm"),
    ("norm_fusion", r"fused_.*norm|norm.*fused|rmsnorm_quant|add_rmsnorm"),
)

#: Low precision is where the matrix cores are. fp32 kernels cannot reach the
#: fp8/fp6/fp4 MFMA paths at all, so a win there teaches a strictly smaller
#: lesson -- and production inference is not fp32.
DTYPE_WEIGHT = {
    "mxfp4": 3.0, "fp4": 3.0, "fp6": 2.8, "int4_w4a16": 2.8,
    "fp8_e4m3fn": 2.5, "fp8_e5m2": 2.5, "fp8": 2.5,
    "int8": 2.0, "bf16": 1.8, "fp16": 1.6, "tf32": 1.2, "fp32": 1.0,
}

#: A vendor baseline is the whole game. Beating a tuned AITER/hipBLASLt kernel by
#: 1.2x is a frontier result; beating an unfused eager-torch module by 38x says
#: only that the baseline was never a kernel.
BASELINE_WEIGHT = {"vendor": 3.0, "compile": 1.6, "torch": 1.0, "external_pool": 0.8}

#: Below this the kernel is launch-bound and there is nothing to optimise. The
#: pool's median is 17us, which is why almost none of it survives.
MIN_ELEMENTS = 1 << 20


def classify_family(text: str) -> str | None:
    t = text.lower()
    for label, pattern in FAMILY_PATTERNS:
        if re.search(pattern, t):
            return label
    return None


def _read_meta(task_dir: pathlib.Path) -> dict:
    """task.yaml is JSON for generated tasks and YAML for hand-authored ones.

    Only a handful of keys matter, so rather than take a YAML dependency for the
    hand-authored minority, fall back to line-wise extraction. A key we cannot
    read defaults to the conservative choice, never the flattering one.
    """
    y = task_dir / "task.yaml"
    if not y.is_file():
        return {}
    text = y.read_text(errors="ignore")
    if text.lstrip().startswith("{"):
        try:
            return json.loads(text)
        except Exception:  # noqa: BLE001 - fall through to regex
            pass
    out: dict = {}
    for key in ("dtype", "op_family", "taxonomy_family", "comparison_baseline",
                "baseline_tier", "backend"):
        m = re.search(rf'^\s*{key}\s*:\s*"?([\w./+-]+)"?\s*$', text, re.M)
        if m:
            out[key] = m.group(1)
    return out


def _baseline_kind(meta: dict) -> str:
    blob = " ".join(str(meta.get(k, "")) for k in
                    ("comparison_baseline", "baseline_tier", "baseline_kind")).lower()
    if "external_pool" in blob:
        return "external_pool"
    if any(v in blob for v in ("aiter", "hipblaslt", "rocblas", "vendor", "ck")):
        return "vendor"
    if "compile" in blob:
        return "compile"
    return "torch"


def _elements(meta: dict) -> int:
    prov = meta.get("provenance") or {}
    n = prov.get("primary_elements")
    if isinstance(n, int) and n > 0:
        return n
    shapes = (meta.get("shapes") or {}).get("primary")
    if isinstance(shapes, dict) and shapes:
        total = 1
        for v in shapes.values():
            if isinstance(v, int) and v > 0:
                total *= v
        return total
    return 0


def score_task(name: str, meta: dict, source: str) -> tuple[float, dict]:
    """Expected value of mining this task, and the reasoning behind it."""
    hay = f"{name} {meta.get('op_family','')} {meta.get('taxonomy_family','')} " \
          f"{(meta.get('provenance') or {}).get('module_name','')}"
    fam = classify_family(hay)
    if fam is None:
        return 0.0, {"reason": "family not in scope"}

    dtype = str(meta.get("dtype") or "fp32").lower()
    dtype_w = next((w for d, w in sorted(DTYPE_WEIGHT.items(),
                                         key=lambda kv: -len(kv[0]))
                    if d in dtype), 1.0)
    base = _baseline_kind(meta)
    elems = _elements(meta)

    # Registry tasks are hand-authored or vendor-generated against a tuned
    # baseline; pool tasks are KernelBook modules against eager torch. Same
    # family, very different lesson.
    src_w = 1.0 if source == "registry" else 0.55
    # Size gates rather than scales: past the launch-bound threshold, more
    # elements do not make a kernel more instructive.
    #
    # Unknown size is only evidence of smallness for pool tasks, where
    # primary_elements is always recorded. Registry tasks declare shapes in a
    # form this does not always parse, and they were sized deliberately by hand
    # -- penalising them for that would rank a 17us KernelBook module above
    # flash attention purely on missing metadata.
    if elems >= MIN_ELEMENTS:
        size_w = 1.0
    elif elems:
        size_w = 0.25
    else:
        size_w = 1.0 if source == "registry" else 0.6

    s = FAMILY_WEIGHT[fam] * dtype_w * BASELINE_WEIGHT[base] * src_w * size_w
    return s, {"family": fam, "dtype": dtype, "baseline": base,
               "elements": elems, "source": source}


def collect() -> list[tuple[float, str, dict]]:
    out = []
    for root, source in ((REGISTRY, "registry"), (POOL, "pool")):
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir() or d.name.startswith((".", "_")):
                continue
            meta = _read_meta(d)
            if not meta and source == "registry":
                continue
            s, why = score_task(d.name, meta, source)
            if s > 0:
                out.append((s, d.name, why))
    out.sort(key=lambda r: -r[0])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="write registry ids here")
    ap.add_argument("--out-pool", default=None, help="write pool ids here")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--report", action="store_true", help="print only")
    args = ap.parse_args()

    ranked = [r for r in collect() if r[0] >= args.min_score]
    if args.limit:
        ranked = ranked[: args.limit]
    if not ranked:
        print("no task scored above the threshold", file=sys.stderr)
        return 2

    fam = collections.Counter(r[2]["family"] for r in ranked)
    src = collections.Counter(r[2]["source"] for r in ranked)
    dty = collections.Counter(r[2]["dtype"] for r in ranked)
    base = collections.Counter(r[2]["baseline"] for r in ranked)
    print(f"=== {len(ranked)} tasks selected ===")
    for label, c in (("family", fam), ("source", src), ("baseline", base)):
        print(f"\n  by {label}:")
        for k, n in c.most_common():
            print(f"    {k:16} {n:5}")
    print("\n  by dtype (top 8):")
    for k, n in dty.most_common(8):
        print(f"    {k:16} {n:5}")
    print("\n  score histogram (pick --min-score from this, not from a guess):")
    edges = [0, 2, 3, 5, 8, 12, 18, 25, 40, 10 ** 9]
    for lo, hi in zip(edges, edges[1:]):
        n = sum(1 for s, _, _ in ranked if lo <= s < hi)
        if n:
            reg = sum(1 for s, _, w in ranked
                      if lo <= s < hi and w["source"] == "registry")
            print(f"    score {lo:>3}-{hi if hi < 10**9 else '+':>4}: {n:5}"
                  f"  ({reg} registry / {n - reg} pool)")
    print("\n  top 12 by score:")
    for s, n, w in ranked[:12]:
        print(f"    {s:7.1f}  {n[:52]:54} {w['family']:11} {w['dtype']:11} {w['baseline']}")

    if args.report:
        return 0
    for path, want in ((args.out, "registry"), (args.out_pool, "pool")):
        if not path:
            continue
        ids = [n for _, n, w in ranked if w["source"] == want]
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(ids) + "\n")
        print(f"\nwrote {len(ids)} {want} ids -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
