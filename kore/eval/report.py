"""Reporting for KORE eval results (PURE: formatting + JSON/MD I/O only).

Turns the dicts produced by :mod:`kore.eval.bakeoff` into human-readable
markdown (fast_p report, bake-off comparison table) and persists a run as both
JSON (machine) and markdown (human).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _fmt_num(x: Optional[float], nd: int = 3) -> str:
    if x is None:
        return "-"
    return f"{x:.{nd}f}"


def _curve_items(results: dict) -> list[tuple[float, float]]:
    """Normalize a results dict to a sorted [(p, fast_p)] list."""
    curve = results.get("fast_p_curve")
    if curve:
        return [(float(p), float(v)) for p, v in curve]
    fp = results.get("fast_p", {})
    return sorted((float(p), float(v)) for p, v in fp.items())


def _curve_items_torch(results: dict) -> list[tuple[float, float]]:
    """Sorted [(p, fast_p)] for the torch-eager curve (empty if absent)."""
    curve = results.get("fast_p_curve_vs_torch")
    if curve:
        return [(float(p), float(v)) for p, v in curve]
    fp = results.get("fast_p_vs_torch", {})
    return sorted((float(p), float(v)) for p, v in fp.items())


def format_fastp_report(results: dict) -> str:
    """Markdown report for a single policy's fast_p results."""
    lines: list[str] = []
    lines.append("# KORE fast_p report")
    lines.append("")
    mode = results.get("mode", "?")
    budget = results.get("budget", "?")
    n = results.get("n", 0)
    num_correct = results.get("num_correct", "?")
    lines.append(f"- **mode**: {mode}")
    lines.append(f"- **budget** (max benches/task): {budget}")
    lines.append(f"- **split size (n)**: {n}")
    lines.append(f"- **correct**: {num_correct}/{n}")
    gms = results.get("geometric_mean_speedup")
    if gms is not None:
        lines.append(f"- **geometric-mean speedup (correct-only)**: {_fmt_num(gms)}x")
    lines.append("")
    lines.append("| p | fast_p |")
    lines.append("| --- | --- |")
    for p, v in _curve_items(results):
        lines.append(f"| {p:g} | {_fmt_pct(v)} |")
    lines.append("")
    # Headline numbers per the plan (p in {1, 1.5}).
    fp = {p: v for p, v in _curve_items(results)}
    f1 = fp.get(1.0)
    f15 = fp.get(1.5)
    if f1 is not None or f15 is not None:
        headline = []
        if f1 is not None:
            headline.append(f"fast_1 = {_fmt_pct(f1)}")
        if f15 is not None:
            headline.append(f"fast_1.5 = {_fmt_pct(f15)}")
        lines.append("**Headline** (vs production baseline): " + ", ".join(headline))
        lines.append("")

    # Optional SECOND curve vs torch-eager (KernelBench-comparable).
    torch_items = _curve_items_torch(results)
    if torch_items:
        lines.append("## fast_p vs torch-eager (KernelBench-comparable)")
        lines.append("")
        gmt = results.get("geometric_mean_speedup_vs_torch")
        if gmt is not None:
            lines.append(f"- **geometric-mean speedup vs torch-eager (correct-only)**: {_fmt_num(gmt)}x")
        lines.append("| p | fast_p (vs torch) |")
        lines.append("| --- | --- |")
        for p, v in torch_items:
            lines.append(f"| {p:g} | {_fmt_pct(v)} |")
        lines.append("")

    return "\n".join(lines)


def format_bakeoff_table(results: dict) -> str:
    """Markdown table comparing multiple policies at a matched budget.

    Accepts the dict returned by ``matched_budget_bakeoff`` (has a ``policies``
    key) or a plain ``{name: policy_results}`` mapping.
    """
    policies = results.get("policies", results)
    budget = results.get("budget")
    ranking = results.get("ranking_by_fast1")

    # Collect the union of p thresholds across policies (stable order).
    ps: list[float] = []
    for res in policies.values():
        for p, _ in _curve_items(res):
            if p not in ps:
                ps.append(p)
    ps.sort()

    lines: list[str] = []
    lines.append("# KORE matched-budget bake-off")
    lines.append("")
    if budget is not None:
        lines.append(f"- **budget** (max benches/task): {budget}")
    if results.get("n") is not None:
        lines.append(f"- **split size (n)**: {results.get('n')}")
    lines.append("")

    header = ["policy"] + [f"fast_{p:g}" for p in ps] + ["geomean", "correct"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    names = ranking if ranking else list(policies.keys())
    for name in names:
        res = policies[name]
        fp = {p: v for p, v in _curve_items(res)}
        row = [str(name)]
        row += [_fmt_pct(fp.get(p, 0.0)) for p in ps]
        row.append(f"{_fmt_num(res.get('geometric_mean_speedup'))}x")
        row.append(f"{res.get('num_correct', '?')}/{res.get('n', '?')}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    if ranking:
        lines.append(f"**Ranking by fast_1**: {' > '.join(str(r) for r in ranking)}")
        lines.append("")
    return "\n".join(lines)


def format_multiseed_report(agg: dict) -> str:
    """Markdown for a multi-seed fast_p result: mean +/- 95% CI over seeds."""
    lines: list[str] = []
    lines.append("# KORE fast_p report (multi-seed)")
    lines.append("")
    seeds = agg.get("seeds", [])
    lines.append(f"- **seeds** ({agg.get('num_seeds', len(seeds))}): {seeds}")
    lines.append(f"- **mode**: {agg.get('mode', '?')}")
    lines.append(f"- **budget** (max benches/task): {agg.get('budget', '?')}")
    lines.append(f"- **split size (n)**: {agg.get('n', '?')}")
    nc = agg.get("num_correct_mean_ci")
    if nc is not None:
        lines.append(f"- **correct** (mean +/- CI95): {_fmt_num(nc['mean'])} +/- {_fmt_num(nc['ci95'])}")
    gm = agg.get("geomean_mean_ci")
    if gm is not None:
        lines.append(f"- **geomean speedup** (mean +/- CI95): {_fmt_num(gm['mean'])} +/- {_fmt_num(gm['ci95'])}x")
    lines.append("")
    lines.append("| p | fast_p (mean) | CI95 | [lo, hi] |")
    lines.append("| --- | --- | --- | --- |")
    for p, mc in sorted(agg.get("fast_p_mean_ci", {}).items()):
        lines.append(
            f"| {float(p):g} | {_fmt_pct(mc['mean'])} | +/-{_fmt_pct(mc['ci95'])} "
            f"| [{_fmt_pct(mc['lo'])}, {_fmt_pct(mc['hi'])}] |"
        )
    lines.append("")
    torch_ci = agg.get("fast_p_vs_torch_mean_ci")
    if torch_ci:
        lines.append("## fast_p vs torch-eager (mean +/- CI95)")
        lines.append("")
        lines.append("| p | fast_p (mean) | CI95 |")
        lines.append("| --- | --- | --- |")
        for p, mc in sorted(torch_ci.items()):
            lines.append(f"| {float(p):g} | {_fmt_pct(mc['mean'])} | +/-{_fmt_pct(mc['ci95'])} |")
        lines.append("")
    return "\n".join(lines)


def format_pass_at_k(res: dict) -> str:
    """Markdown for an unbiased pass@k / fast_p@k best-of-N result."""
    lines: list[str] = []
    lines.append("# KORE best-of-N (unbiased pass@k / fast_p@k)")
    lines.append("")
    lines.append(f"- **split size (n)**: {res.get('n', '?')}")
    lines.append("")
    lines.append("| k | pass@k |")
    lines.append("| --- | --- |")
    for k, v in sorted(res.get("pass_at_k", {}).items()):
        lines.append(f"| {k} | {_fmt_pct(v)} |")
    lines.append("")
    fpk = res.get("fast_p_at_k", {})
    if fpk:
        lines.append("| (k, p) | fast_p@k |")
        lines.append("| --- | --- |")
        for key, v in fpk.items():
            lines.append(f"| {key} | {_fmt_pct(v)} |")
        lines.append("")
    return "\n".join(lines)


def _is_bakeoff(results: dict) -> bool:
    return "policies" in results or "ranking_by_fast1" in results


def _is_multiseed(results: dict) -> bool:
    return "fast_p_mean_ci" in results


def _is_pass_at_k(results: dict) -> bool:
    return "pass_at_k" in results and "fast_p" not in results


def render_markdown(results: dict) -> str:
    """Pick the right formatter for a results dict."""
    if _is_multiseed(results):
        return format_multiseed_report(results)
    if _is_pass_at_k(results):
        return format_pass_at_k(results)
    if _is_bakeoff(results):
        return format_bakeoff_table(results)
    return format_fastp_report(results)


def _json_default(o):
    # Dataclasses / objects that aren't JSON-native fall back to their dict/str.
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)


def save_report(results: dict, path) -> dict:
    """Persist ``results`` as JSON + markdown.

    ``path`` may include an extension (stripped) or be a stem. Writes
    ``<stem>.json`` and ``<stem>.md``. Returns the two paths.
    """
    p = Path(path)
    stem = p.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    json_path = stem.with_suffix(".json")
    md_path = stem.with_suffix(".md")
    json_path.write_text(json.dumps(results, indent=2, default=_json_default))
    md_path.write_text(render_markdown(results))
    return {"json": str(json_path), "md": str(md_path)}
