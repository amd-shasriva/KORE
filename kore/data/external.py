"""External public-dataset ingestion for KORE — breadth WITHOUT diluting quality.

Public Triton corpora add breadth + syntax/op coverage, but they are generic
(NVIDIA / ``torch.compile``-generated, not MI300X-optimized) and unverified on
gfx942. So KORE does **not** raw-mix them. Every external kernel passes a strict
frontier gate before it can enter training:

  1. **License gate** — only sources whose license permits training/distillation
     for our use are admitted; others are flagged and skipped (see :data:`SOURCES`).
  2. **Decontamination** — dropped if it overlaps the held-out eval split by
     operator family or n-gram (reuses :mod:`kore.data.decontam`).
  3. **On-hardware verification** — the Triton kernel must COMPILE and pass the
     correctness gate on our gfx942 env (:class:`~kore.env.kore_env.KoreEnv`-style
     ``step``). Generic NVIDIA kernels that don't run on CDNA3 are dropped; the
     survivors are real, verified-on-target kernels.
  4. **Contract normalization** — survivors are rewritten into KORE's canonical
     SYSTEM_PROMPT + ANALYSIS/PROPOSED_CHANGE/FULL_KERNEL contract so they are
     indistinguishable from native KORE SFT rows.

This module is import-safe (no network / no ``datasets`` / no torch at import) —
loaders import lazily so the registry + conversion + decontam logic stay CPU- and
offline-testable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Optional


# --------------------------------------------------------------------------- #
# Source registry — license-aware (the frontier-quality gate starts here).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExternalSource:
    name: str
    hf_id: str                     # HuggingFace dataset id (or "" for local-only)
    kind: str                      # "pairs" | "traces" | "eval"
    license: str
    # train_ok: may we use it to TRAIN/distill a model under our terms?
    train_ok: bool
    note: str = ""


# Curated as of the 2026 landscape (see the research notes). ``train_ok=False``
# means "usable for EVAL / task-design inspiration only" — never mixed into SFT.
SOURCES: dict[str, ExternalSource] = {
    # KernelBench: eval benchmark (250 PyTorch modules, L1-L3); now supports the AMD
    # ``hip``/gfx942 backend. Use as ADDITIONAL held-out generalization eval, not train.
    "kernelbench": ExternalSource(
        "kernelbench", "ScalingIntelligence/KernelBench", "eval", "MIT", True,
        "eval benchmark; use as extra held-out generalization eval (gfx942 hip backend)"),
    # KernelBench-Triton / TritonBench-T taxonomy (176 tasks, 15 categories incl.
    # quantization/multi-precision) — task-design inspiration + eval.
    "kernelbenchx": ExternalSource(
        "kernelbenchx", "BonnieWang/KernelBenchX", "eval", "research", True,
        "176-task taxonomy + before/after corpus; eval + coverage inspiration"),
    # KernelBook: ~torch.compile PyTorch->Triton pairs. RAIL-D w/ 'Researcher
    # Reciprocity' use-restriction -> NOT admitted to training by default (flagged).
    "kernelbook": ExternalSource(
        "kernelbook", "GPUMODE/KernelBook", "pairs",
        "OpenRAIL-D+ReciprocitY", False,
        "RAIL-D reciprocity restriction: verify legal fit before training use"),
    # Multi-turn Triton reasoning traces (iterative refinement). Admitted only after
    # verify + contract normalization (traces are teacher NL, quality varies).
    "kb_multiturn_traces": ExternalSource(
        "kb_multiturn_traces", "", "traces", "research", True,
        "multi-turn refinement traces; verify + normalize before use"),
}


def license_ok(source: str) -> bool:
    """True iff ``source`` may be used to TRAIN under our terms (else eval-only)."""
    s = SOURCES.get(source)
    return bool(s and s.train_ok and s.kind in ("pairs", "traces"))


def license_report() -> list[dict]:
    """Human/audit-readable summary of every source's admission status."""
    return [{"name": s.name, "kind": s.kind, "license": s.license,
             "train_ok": s.train_ok, "admitted_for_training": license_ok(s.name),
             "note": s.note} for s in SOURCES.values()]


# --------------------------------------------------------------------------- #
# Normalized external record
# --------------------------------------------------------------------------- #
@dataclass
class ExternalKernel:
    source: str                    # registry key
    torch_src: str                 # the PyTorch module / op description (the "prompt")
    triton_src: str                # the candidate Triton kernel
    op_hint: str = ""              # inferred operator family / name (for decontam)
    meta: dict = field(default_factory=dict)

    def key(self) -> str:
        return hashlib.sha256(
            (self.source + "\x00" + self.triton_src).encode("utf-8", "ignore")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Loading (lazy; offline-safe)
# --------------------------------------------------------------------------- #
def load_pairs(source: str, *, limit: Optional[int] = None,
               local_rows: Optional[Iterable[dict]] = None) -> list[ExternalKernel]:
    """Load ``(torch, triton)`` pairs for ``source``.

    ``local_rows`` (an iterable of dicts) bypasses the network — used by tests and
    by air-gapped runs that pre-download the parquet. Otherwise the HuggingFace
    ``datasets`` library is imported lazily. Never raises on a missing dependency:
    returns ``[]`` and lets the caller degrade to native data only.
    """
    src = SOURCES.get(source)
    if src is None:
        raise ValueError(f"unknown external source {source!r}; known: {list(SOURCES)}")
    rows: Iterable[dict]
    if local_rows is not None:
        rows = local_rows
    else:
        try:
            from datasets import load_dataset  # lazy, optional
        except Exception:  # noqa: BLE001 - offline / not installed
            return []
        try:
            ds = load_dataset(src.hf_id, split="train")
            rows = (dict(r) for r in ds)
        except Exception:  # noqa: BLE001 - network / gated / missing
            return []
    out: list[ExternalKernel] = []
    for r in rows:
        torch_src = _first(r, ("python_code", "pytorch", "torch", "prompt", "code", "input"))
        triton_src = _first(r, ("triton_code", "triton", "kernel", "output", "completion"))
        if not torch_src or not triton_src:
            continue
        out.append(ExternalKernel(source=source, torch_src=torch_src, triton_src=triton_src,
                                  op_hint=_infer_op_hint(torch_src), meta={"raw_keys": list(r)[:8]}))
        if limit and len(out) >= limit:
            break
    return out


def _first(d: dict, keys: tuple[str, ...]) -> str:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _infer_op_hint(torch_src: str) -> str:
    """Best-effort operator-family hint from a torch snippet (for decontam)."""
    try:
        from kore.data.mutate import infer_family
    except Exception:  # noqa: BLE001
        infer_family = None  # type: ignore
    t = (torch_src or "").lower()
    for kw in ("attention", "softmax", "layernorm", "rmsnorm", "matmul", "gemm",
               "conv", "gelu", "silu", "relu", "rope", "moe", "quant", "embedding"):
        if kw in t:
            return kw
    return infer_family(t) if infer_family else "unknown"


# --------------------------------------------------------------------------- #
# Decontamination + verification + contract normalization
# --------------------------------------------------------------------------- #
def decontaminate(rows: list[ExternalKernel], *, heldout_op_families: set[str],
                  heldout_ngrams: Optional[set[str]] = None,
                  n: int = 8) -> tuple[list[ExternalKernel], int]:
    """Drop external rows that leak the held-out eval split (family or n-gram)."""
    kept: list[ExternalKernel] = []
    dropped = 0
    fams = {f.lower() for f in (heldout_op_families or set())}
    for r in rows:
        if r.op_hint and r.op_hint.lower() in fams:
            dropped += 1
            continue
        if heldout_ngrams and _ngram_overlap(r.triton_src, heldout_ngrams, n):
            dropped += 1
            continue
        kept.append(r)
    return kept, dropped


def _ngram_overlap(text: str, ref_ngrams: set[str], n: int) -> bool:
    toks = (text or "").split()
    for i in range(0, max(0, len(toks) - n + 1)):
        if " ".join(toks[i:i + n]) in ref_ngrams:
            return True
    return False


def verify_filter(rows: list[ExternalKernel], verify: Callable[[ExternalKernel], bool],
                  *, log: Callable[[str], None] = lambda m: None) -> tuple[list[ExternalKernel], dict]:
    """Keep only external kernels that COMPILE + pass correctness on our gfx942 env.

    ``verify(row) -> bool`` is injected (the caller wires it to a real
    ``KoreEnv``-style check) so this stays GPU-free + unit-testable. Returns
    ``(survivors, stats)``. This is the gate that turns generic public Triton into
    verified-on-target frontier data (or drops it)."""
    survivors: list[ExternalKernel] = []
    n_ok = n_bad = 0
    for r in rows:
        try:
            ok = bool(verify(r))
        except Exception:  # noqa: BLE001 - a broken external kernel never aborts the pass
            ok = False
        if ok:
            survivors.append(r)
            n_ok += 1
        else:
            n_bad += 1
    stats = {"in": len(rows), "verified": n_ok, "dropped": n_bad}
    log(f"external verify_filter: {stats}")
    return survivors, stats


def to_sft_rows(rows: list[ExternalKernel]) -> list[dict]:
    """Normalize verified external kernels into KORE's canonical SFT chat rows.

    The torch snippet becomes the task prompt; the (verified) Triton kernel becomes
    a canonical ANALYSIS/PROPOSED_CHANGE/FULL_KERNEL assistant turn — so external
    rows are contract-identical to native KORE SFT data. Provenance is recorded.
    """
    from kore.data.prompts import SYSTEM_PROMPT
    from kore.policy.format import format_assistant_turn

    def _prompt(r: "ExternalKernel") -> str:
        # External kernels have no Task object, so build a contract-shaped prompt
        # directly from the reference PyTorch (mirrors build_task_prompt's format).
        return (f"Optimize a {r.op_hint} kernel for AMD gfx942 (backend: triton). "
                f"Implement the following PyTorch reference as an optimized Triton kernel. "
                f"Return ANALYSIS, PROPOSED_CHANGE, and a complete FULL_KERNEL.\n\n"
                f"Reference:\n```python\n{r.torch_src}\n```")

    out: list[dict] = []
    for r in rows:
        analysis = (f"This is a verified Triton implementation for a `{r.op_hint}`-class "
                    f"operator (external source: {r.source}, re-verified on gfx942). It "
                    f"compiles and matches the reference within the SNR gate on CDNA3.")
        proposed = "Adopt this verified kernel as the implementation."
        assistant = format_assistant_turn(analysis, proposed, r.triton_src)
        out.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _prompt(r)},
                {"role": "assistant", "content": assistant},
            ],
            "_provenance": {"external_source": r.source, "op_hint": r.op_hint,
                            "key": r.key(), "verified_gfx942": True},
        })
    return out


def ingest(source: str, *, heldout_op_families: set[str],
           verify: Optional[Callable[[ExternalKernel], bool]] = None,
           limit: Optional[int] = None,
           local_rows: Optional[Iterable[dict]] = None,
           heldout_ngrams: Optional[set[str]] = None,
           log: Callable[[str], None] = lambda m: None) -> dict:
    """Full frontier gate: license -> load -> decontaminate -> verify -> normalize.

    Returns ``{"admitted": bool, "sft_rows": [...], "stats": {...}}``. When
    ``verify`` is None the verification gate is SKIPPED (rows are marked
    unverified and NOT normalized — a run must supply a real gfx942 verifier to
    admit external data into training)."""
    if not license_ok(source):
        log(f"external {source}: NOT admitted for training (license/kind gate)")
        return {"admitted": False, "sft_rows": [], "stats": {"reason": "license"}}
    raw = load_pairs(source, limit=limit, local_rows=local_rows)
    deco, n_deco = decontaminate(raw, heldout_op_families=heldout_op_families,
                                 heldout_ngrams=heldout_ngrams)
    if verify is None:
        log(f"external {source}: loaded={len(raw)} decontam_dropped={n_deco} "
            f"(no verifier supplied -> not admitted)")
        return {"admitted": False, "sft_rows": [],
                "stats": {"loaded": len(raw), "decontam_dropped": n_deco, "reason": "unverified"}}
    survivors, vstats = verify_filter(deco, verify, log=log)
    sft_rows = to_sft_rows(survivors)
    stats = {"loaded": len(raw), "decontam_dropped": n_deco, **vstats, "sft_rows": len(sft_rows)}
    log(f"external {source} ingested: {stats}")
    return {"admitted": True, "sft_rows": sft_rows, "stats": stats}


__all__ = [
    "ExternalSource", "SOURCES", "ExternalKernel",
    "license_ok", "license_report", "load_pairs",
    "decontaminate", "verify_filter", "to_sft_rows", "ingest",
]
