"""Held-out language-model loss on kernel source: what did continued pretraining buy?

A mid-train stage is continued PRETRAINING. Its objective is next-token
prediction on a corpus, so the measurement that speaks directly to that
objective is next-token loss on held-out text from the same domain - not kernel
generation, which additionally requires instruction following, a parseable
response contract, a compiler and a correctness oracle to all cooperate.

This module measures exactly that, and it exists because it is the only
comparison on this scope with enough statistical power to resolve a small
effect. Single-shot kernel generation gives ONE bit per task - roughly 34 bits
over the whole held-out scope - so an improvement has to be enormous to clear an
exact McNemar test. Teacher-forced loss over the same 34 held-out kernels gives
tens of thousands of paired per-token measurements, and the per-document paired
delta is tight enough to separate "no effect" from "an effect too small to see
in generation".

What makes the number defensible:

  * The documents are the held-out tasks' own ``seed_triton.py`` sources, taken
    from the generalization scope, i.e. after the 11 tasks whose kernel source
    leaked into the midtrain corpus are removed. The contamination record names
    ``kore-task:<task_id>:seed_triton.py`` as the leaked reference, so the
    surviving 34 seed sources are precisely the held-out documents this stage
    never saw. :func:`assert_documents_uncontaminated` re-checks that against the
    registry rather than trusting this docstring.
  * Both arms are scored on the IDENTICAL token sequence.
    :func:`compare_documents` refuses to compare two arms whose tokenization of a
    document differs (a mid-train that resized or re-ordered the vocabulary would
    otherwise produce a per-token number that is not paired), so the delta is a
    true paired difference in bits per token.
  * The effect size is reported in BITS PER TOKEN with a paired bootstrap CI and
    an exact sign test over documents, plus the corpus-level totals. Bits per
    token rather than perplexity because differences add, CIs are symmetric, and
    a per-document mean is not distorted by document length.

Import-safe: torch/transformers are only reached inside
:func:`load_torch_nll_scorer`. Everything else is pure and CPU-testable.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

SCHEMA_SCORES = "kore.heldout-lm-scores.v1"
SCHEMA_REPORT = "kore.heldout-lm-report.v1"

LN2 = math.log(2.0)

#: A scorer: ``nll(text) -> {"sum_nll_nats", "n_tokens", "tokens_sha"}``.
NLLFn = Callable[[str], dict]


@dataclass
class Document:
    """One held-out document to score."""

    doc_id: str
    task_id: str
    kind: str
    text: str

    @property
    def chars(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("text")
        d["chars"] = self.chars
        d["text_sha"] = hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:12]
        return d


def heldout_documents(tasks: Sequence, kinds: Sequence[str] = ("seed",)) -> list[Document]:
    """Build held-out documents from ``tasks``.

    ``kinds`` selects which per-task artifact to score:
      * ``"seed"``      - ``seed_triton.py``, the Triton kernel source itself.
        This is the document the contamination record is written against, so it
        is the one whose held-out status is actually established.
      * ``"reference"``  - ``reference.py``, the fp32 oracle + baseline wrapper.
        Torch rather than Triton, so it measures a different thing; included as
        a contrast, not as the headline.
    """
    docs: list[Document] = []
    for task in tasks:
        for kind in kinds:
            text = _task_artifact(task, kind)
            if text and text.strip():
                docs.append(Document(doc_id=f"{task.task_id}:{kind}",
                                     task_id=task.task_id, kind=kind, text=text))
    return docs


def _task_artifact(task, kind: str) -> str:
    if kind == "seed":
        try:
            return task.seed_source
        except Exception:  # noqa: BLE001 - a task without a readable seed is skipped
            return ""
    if kind == "reference":
        p = getattr(task, "reference_path", None)
        if p is None:
            return ""
        p = Path(p)
        return p.read_text(encoding="utf-8") if p.is_file() else ""
    raise ValueError(f"unknown document kind {kind!r} (known: 'seed', 'reference')")


#: Bundled benches whose text doubles as a general-domain LM probe.
GENERAL_PROBE_BENCHES: tuple[str, ...] = ("mmlu", "humaneval", "mtbench")


def general_documents(benches: Sequence[str] = GENERAL_PROBE_BENCHES,
                      data_dir=None) -> list[Document]:
    """General-domain documents (English + generic Python) for a forgetting probe.

    Built from the bundled retention samples so it needs no network and no
    ``datasets``. Specialization is only a problem if it costs general ability,
    and bits/token on out-of-domain text detects that with orders of magnitude
    more resolution than a 14-item accuracy smoke test.

    **This probe is NOT decontaminated.** The mid-train recipe mixes 30% general
    replay (``general_replay_frac``), and canonical benchmark items may well sit
    inside it, which would flatter the trained arm here. Read it as a
    "did general LM quality collapse" instrument, never as a benchmark score.
    """
    from kore.eval.retention import load_bench

    docs: list[Document] = []
    for bench in benches:
        for i, item in enumerate(load_bench(bench, data_dir)):
            text = _general_text(bench, item)
            if text and len(text.strip()) > 40:
                docs.append(Document(doc_id=f"general:{bench}:{i}",
                                     task_id=f"general:{bench}", kind="general",
                                     text=text))
    return docs


def _general_text(bench: str, item: dict) -> str:
    if bench == "mmlu":
        choices = "\n".join(f"{c}" for c in item.get("choices", []))
        return f"{item.get('question', '')}\n{choices}"
    if bench in ("humaneval", "livecodebench"):
        return (item.get("prompt", "") or "") + (item.get("canonical_solution", "") or "")
    if bench == "mtbench":
        return (item.get("question", "") or "") + "\n" + (item.get("reference") or "")
    return item.get("prompt") or item.get("question") or ""


def assert_documents_uncontaminated(docs: Sequence[Document]) -> dict:
    """Fail unless every document's task is in the generalization scope.

    A held-out-loss claim over a task whose source reached the training corpus is
    not a held-out claim, and this is the check that makes the distinction
    load-bearing instead of a comment.
    """
    from kore.tasks.registry import (
        filter_generalization_scope,
        generalization_eval_ids,
        taxonomy_digest,
    )

    task_ids = sorted({d.task_id for d in docs})
    kept, dropped = filter_generalization_scope(task_ids)
    if dropped:
        raise AssertionError(
            "held-out LM loss requested over contaminated / unadjudicable tasks: "
            f"{dict(sorted(dropped.items()))}")
    # ``filter_generalization_scope`` only strikes CONTAMINATED ids; a TRAINING
    # task passes it untouched. Held-out loss over a trained task is not
    # held-out loss, so require positive membership in the scope rather than
    # merely the absence of a contamination record.
    scope = set(generalization_eval_ids())
    trained = sorted(t for t in kept if t not in scope)
    if trained:
        raise AssertionError(
            "held-out LM loss requested over tasks that are NOT in the "
            f"generalization scope (they were trained on): {trained}")
    return {"taxonomy_digest": taxonomy_digest(), "n_documents": len(docs),
            "n_tasks": len(kept), "task_ids": list(kept)}


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score_documents(docs: Sequence[Document], nll: NLLFn, *, arm: str,
                    log: Optional[Callable[[str], None]] = None) -> list[dict]:
    """Score every document with ``nll`` and return per-document rows."""
    rows: list[dict] = []
    for i, doc in enumerate(docs):
        out = nll(doc.text)
        n_tok = int(out["n_tokens"])
        total = float(out["sum_nll_nats"])
        rows.append({
            "arm": arm,
            **doc.to_dict(),
            "n_tokens": n_tok,
            "sum_nll_nats": total,
            "nats_per_token": (total / n_tok) if n_tok else None,
            "bits_per_token": (total / n_tok / LN2) if n_tok else None,
            "tokens_sha": out.get("tokens_sha"),
        })
        if log and (i + 1) % 5 == 0:
            log(f"[{arm}] scored {i + 1}/{len(docs)} documents")
    return rows


def write_scores(path, rows: Sequence[dict], **meta) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(
        {"schema": SCHEMA_SCORES, "n_rows": len(rows), **meta, "rows": list(rows)},
        indent=2, default=str))
    return p


def read_scores(path) -> tuple[dict, list[dict]]:
    obj = json.loads(Path(path).read_text())
    rows = obj.pop("rows", [])
    return obj, rows


def corpus_totals(rows: Sequence[dict]) -> dict:
    """Corpus-level aggregate: one bits/token over every token in every document."""
    tok = sum(int(r["n_tokens"]) for r in rows)
    nats = sum(float(r["sum_nll_nats"]) for r in rows)
    return {
        "n_documents": len(rows),
        "n_tokens": tok,
        "sum_nll_nats": nats,
        "bits_per_token": (nats / tok / LN2) if tok else None,
        "perplexity": (math.exp(nats / tok) if tok else None),
    }


# --------------------------------------------------------------------------- #
# The paired comparison (PURE)
# --------------------------------------------------------------------------- #
def compare_documents(cand_rows: Sequence[dict], ref_rows: Sequence[dict], *,
                      name_a: str = "candidate", name_b: str = "reference",
                      n_boot: int = 10000, seed: int = 0, alpha: float = 0.05,
                      require_matched_tokens: bool = True) -> dict:
    """Paired per-document comparison of bits/token (candidate minus reference).

    A NEGATIVE effect means the candidate assigns higher probability to held-out
    kernel source, i.e. continued pretraining helped on its own objective.
    """
    from kore.eval.paired_stats import paired_comparison

    a_by = {r["doc_id"]: r for r in cand_rows}
    b_by = {r["doc_id"]: r for r in ref_rows}
    doc_ids = [r["doc_id"] for r in cand_rows if r["doc_id"] in b_by]
    if not doc_ids:
        raise ValueError("no documents in common between the two arms")

    mismatched = [
        d for d in doc_ids
        if a_by[d]["n_tokens"] != b_by[d]["n_tokens"]
        or (a_by[d].get("tokens_sha") and b_by[d].get("tokens_sha")
            and a_by[d]["tokens_sha"] != b_by[d]["tokens_sha"])
    ]
    if mismatched and require_matched_tokens:
        raise AssertionError(
            "the two arms tokenized these documents differently, so a per-token "
            f"comparison would not be paired: {mismatched[:5]} "
            f"({len(mismatched)} of {len(doc_ids)}). Pass "
            "require_matched_tokens=False only if you intend to compare across "
            "different tokenizations, and say so in the write-up.")

    a_bits = [float(a_by[d]["bits_per_token"]) for d in doc_ids]
    b_bits = [float(b_by[d]["bits_per_token"]) for d in doc_ids]
    deltas = [x - y for x, y in zip(a_bits, b_bits)]
    cmp_ = paired_comparison(deltas=deltas, n_boot=n_boot, seed=seed, alpha=alpha,
                             headline="wilcoxon")

    a_tot = corpus_totals([a_by[d] for d in doc_ids])
    b_tot = corpus_totals([b_by[d] for d in doc_ids])
    improved = sum(1 for d in deltas if d < 0)
    return {
        "schema": SCHEMA_REPORT,
        "candidate": name_a,
        "reference": name_b,
        "n_documents": len(doc_ids),
        "matched_tokenization": not mismatched,
        "n_mismatched_documents": len(mismatched),
        "per_document": [
            {"doc_id": d, "task_id": a_by[d]["task_id"], "kind": a_by[d]["kind"],
             "n_tokens": a_by[d]["n_tokens"],
             "candidate_bits_per_token": a_by[d]["bits_per_token"],
             "reference_bits_per_token": b_by[d]["bits_per_token"],
             "delta_bits_per_token": a_by[d]["bits_per_token"] - b_by[d]["bits_per_token"]}
            for d in doc_ids
        ],
        "corpus": {"candidate": a_tot, "reference": b_tot,
                   "delta_bits_per_token": (
                       (a_tot["bits_per_token"] - b_tot["bits_per_token"])
                       if a_tot["bits_per_token"] is not None
                       and b_tot["bits_per_token"] is not None else None),
                   "perplexity_ratio": (
                       (a_tot["perplexity"] / b_tot["perplexity"])
                       if a_tot["perplexity"] and b_tot["perplexity"] else None)},
        "paired": cmp_.to_dict(),
        "n_documents_improved": improved,
        "verdict": _lm_verdict(cmp_, improved, len(doc_ids), name_a, name_b, alpha),
    }


def _lm_verdict(cmp_, improved: int, n: int, name_a: str, name_b: str,
                alpha: float) -> dict:
    eff = cmp_.effect_size
    lo, hi = cmp_.ci
    sig = cmp_.p_value < alpha and not (lo <= 0.0 <= hi)
    if eff < 0:
        direction = "candidate_better" if sig else "candidate_better_not_significant"
    elif eff > 0:
        direction = "reference_better" if sig else "reference_better_not_significant"
    else:
        direction = "tie"
    return {
        "direction": direction,
        "significant": sig,
        "p_value": cmp_.p_value,
        "statement": (
            f"{name_a} minus {name_b}: {eff:+.4f} bits/token "
            f"(95% CI [{lo:+.4f}, {hi:+.4f}], wilcoxon p={cmp_.p_value:.3g}); "
            f"{improved}/{n} documents improved. Negative is better for {name_a}."),
    }


def format_lm_report(result: dict) -> str:
    """Compact ASCII markdown for a :func:`compare_documents` result."""
    p = result["paired"]
    corp = result["corpus"]
    lines = [
        "# Held-out LM loss on held-out Triton kernel source",
        "",
        f"- **candidate**: `{result['candidate']}`",
        f"- **reference**: `{result['reference']}`",
        f"- **documents**: {result['n_documents']} "
        f"({corp['candidate']['n_tokens']:,} tokens, identical tokenization: "
        f"{result['matched_tokenization']})",
        "",
        "| arm | bits/token | perplexity |",
        "| --- | --- | --- |",
        f"| {result['candidate']} | {corp['candidate']['bits_per_token']:.4f} "
        f"| {corp['candidate']['perplexity']:.3f} |",
        f"| {result['reference']} | {corp['reference']['bits_per_token']:.4f} "
        f"| {corp['reference']['perplexity']:.3f} |",
        "",
        f"- **paired per-document effect**: {p['effect_size']:+.4f} bits/token, "
        f"95% CI [{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}]",
        f"- **p-values**: wilcoxon {p['p_wilcoxon']:.3g}, sign {p['p_sign']:.3g}, "
        f"bootstrap {p['p_bootstrap']:.3g}",
        f"- **documents improved**: {result['n_documents_improved']}/"
        f"{result['n_documents']}",
        f"- **corpus-level delta**: {corp['delta_bits_per_token']:+.4f} bits/token "
        f"(perplexity ratio {corp['perplexity_ratio']:.4f}x)",
        "",
        f"**Verdict: {result['verdict']['direction']}** - "
        f"{result['verdict']['statement']}",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The torch scorer (GPU side)
# --------------------------------------------------------------------------- #
def load_torch_nll_scorer(model_id: str, *, dtype: str = "bfloat16",
                          device_map: str = "auto", revision: Optional[str] = None,
                          max_length: int = 8192):
    """Load ``model_id`` and return ``(nll, close, info)``.

    ``nll(text)`` runs ONE teacher-forced forward pass over the raw document (no
    chat template - this measures the pretraining objective, not chat behaviour)
    and returns the summed next-token negative log-likelihood in nats, the number
    of predicted positions, and a digest of the token ids so the caller can prove
    the two arms scored the identical sequence.

    ``dtype`` is applied to both arms for the same reason as in
    :func:`kore.eval.checkpoint_ab.load_hf_batch_generate`: an FSDP
    ``FULL_STATE_DICT`` training output lands on disk as fp32 while a Hub base is
    bf16, and comparing a bf16 model against an fp32 one would confound
    precision with training. Loss is accumulated in fp32 regardless.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rev = {"revision": revision} if revision else {}
    tok = AutoTokenizer.from_pretrained(model_id, **rev)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=getattr(torch, dtype), device_map=device_map, **rev)
    model.eval()

    def nll(text: str) -> dict:
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=max_length, add_special_tokens=False)["input_ids"]
        if ids.shape[1] < 2:
            return {"sum_nll_nats": 0.0, "n_tokens": 0, "tokens_sha": None}
        ids = ids.to(model.device)
        with torch.no_grad():
            logits = model(input_ids=ids).logits
        # fp32 log-softmax: a bf16 reduction over a 151k-way vocabulary loses
        # enough precision to matter at the 1e-3 bits/token resolution this
        # comparison is trying to resolve.
        logprobs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
        target = ids[:, 1:]
        picked = logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        total = float(-picked.sum().item())
        flat = ids[0].tolist()
        sha = hashlib.sha256(
            ",".join(str(i) for i in flat).encode("utf-8")).hexdigest()[:12]
        return {"sum_nll_nats": total, "n_tokens": int(target.numel()),
                "tokens_sha": sha}

    def close() -> None:
        import gc
        nonlocal model, tok
        model = None
        tok = None
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - teardown is best effort
            pass

    info: dict[str, Any] = {
        "model_id": model_id, "dtype": dtype, "revision": revision,
        "max_length": max_length,
        "vocab_size": int(getattr(model.config, "vocab_size", 0)),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
    }
    return nll, close, info


def compare_by_kind(cand_rows: Sequence[dict], ref_rows: Sequence[dict],
                    *, name_a: str, name_b: str, **kw) -> dict:
    """Run :func:`compare_documents` separately per document ``kind``.

    The kinds answer different questions and must not be pooled: ``seed`` is
    held-out Triton kernel source (the target domain), ``reference`` is the
    torch oracle for the same task (adjacent but not Triton), and ``general`` is
    out-of-domain text whose job is to detect forgetting.
    """
    kinds = [k for k in ("seed", "reference", "general")
             if any(r["kind"] == k for r in cand_rows)]
    out: dict[str, Any] = {"schema": SCHEMA_REPORT, "candidate": name_a,
                           "reference": name_b, "by_kind": {}}
    for kind in kinds:
        a = [r for r in cand_rows if r["kind"] == kind]
        b = [r for r in ref_rows if r["kind"] == kind]
        if a and b:
            out["by_kind"][kind] = compare_documents(
                a, b, name_a=name_a, name_b=name_b, **kw)
    return out


def format_by_kind_report(result: dict) -> str:
    """Markdown for a :func:`compare_by_kind` result (one section per kind)."""
    titles = {
        "seed": "Held-out Triton kernel source (the target domain)",
        "reference": "Held-out torch reference/oracle source (adjacent domain)",
        "general": "General-domain text (forgetting probe, NOT decontaminated)",
    }
    parts = [f"# Held-out LM loss: `{result['candidate']}` vs `{result['reference']}`",
             ""]
    for kind, res in result.get("by_kind", {}).items():
        parts.append(f"## {titles.get(kind, kind)}")
        parts.append("")
        parts.append(format_lm_report(res).split("\n", 2)[2])
    return "\n".join(parts)


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m kore.eval.heldout_lm",
        description="Paired held-out LM-loss comparison of two arms "
                    "(bits/token on held-out kernel source).")
    p.add_argument("--candidate", required=True, help="candidate lm_scores JSON")
    p.add_argument("--reference", required=True, help="reference lm_scores JSON")
    p.add_argument("--out", required=True, help="output path stem")
    p.add_argument("--name-candidate", default=None)
    p.add_argument("--name-reference", default=None)
    p.add_argument("--allow-tokenizer-mismatch", action="store_true",
                   help="compare arms whose tokenization differs (say so in the "
                        "write-up: the per-token delta is then not paired)")
    a = p.parse_args(argv)

    cand_meta, cand_rows = read_scores(a.candidate)
    ref_meta, ref_rows = read_scores(a.reference)
    name_a = a.name_candidate or cand_meta.get("arm") or "candidate"
    name_b = a.name_reference or ref_meta.get("arm") or "reference"
    result = compare_by_kind(
        cand_rows, ref_rows, name_a=name_a, name_b=name_b,
        require_matched_tokens=not a.allow_tokenizer_mismatch)
    result["meta"] = {"candidate": cand_meta, "reference": ref_meta}
    stem = Path(a.out).with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2, default=str))
    md = format_by_kind_report(result)
    stem.with_suffix(".md").write_text(md)
    print(md)
    return 0


__all__ = [
    "SCHEMA_SCORES",
    "SCHEMA_REPORT",
    "LN2",
    "Document",
    "GENERAL_PROBE_BENCHES",
    "heldout_documents",
    "general_documents",
    "assert_documents_uncontaminated",
    "score_documents",
    "write_scores",
    "read_scores",
    "corpus_totals",
    "compare_documents",
    "compare_by_kind",
    "format_lm_report",
    "format_by_kind_report",
    "load_torch_nll_scorer",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
