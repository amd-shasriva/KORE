"""Stage-0 mid-train corpus assembly (continued pretraining on ROCm/HIP/Triton).

KORE's Stage-0 continues pretraining the base model on a domain corpus so the
policy enters SFT already fluent in the ROCm/HIP/Triton/Composable-Kernel world
(the strong-distribution-shift regime the plan calls out). This module assembles
that corpus from REAL local sources on the box - no network, fully offline and
deterministic - and mixes in a small general-replay slice to guard against
catastrophic forgetting during the shift.

Sources (each reported separately in the returned counts):
  - ``kore_tasks``            : the KORE task seed kernels + references + drivers
                                (``kore/tasks/*/*.py``).
  - ``pytorch_triton_pairs``  : PyTorch reference <-> Triton seed kernel pairs
                                built from each task's ``reference.py`` +
                                ``seed_triton.py`` (real torch->Triton examples).
  - ``triton``                : Triton kernel Python files found under the local
                                repos (GEAK / KernelBench / KernelForge* / vllm).
  - ``rocm_hip``              : HIP/CUDA/Composable-Kernel source (``*.cu``,
                                ``*.cuh``, ``*.hip``, ``*.cpp``, ``*.hpp``,
                                ``*.h``, ``*.cc``) under the local repos.
  - ``docs``                  : ROCm / rocprof / tuning / perf-guide markdown
                                docs under the local repos (path-filtered so the
                                corpus stays a kernel/ROCm corpus, not all md).
  - ``general_replay``        : ~``config.general_replay_frac`` general shards
                                (code/math/chat/IF/tool-use) via
                                :func:`kore.data.general_replay.load_general_replay`
                                (offline bundled fallback).

Output: JSONL of ``{"text": <chunk>, "source": <source>}`` rows, chunked to
``config.max_seq_length`` (a char-budget approximation so it stays CPU/offline -
no tokenizer download) and deduplicated by normalized-text hash.

Everything is deterministic given ``seed`` (sorted file walks + seeded replay
sampling), so two builds from the same tree produce byte-identical output.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Callable, Iterable, Optional

from kore.data.general_replay import REPLAY_KINDS, load_general_replay
from kore.obs import get_logger

log = get_logger("data.midtrain_corpus")

# Char-per-token approximation for the chunker. Mid-train chunking is by
# characters (deterministic, offline, no tokenizer download); ~4 chars/token is
# the standard rule of thumb for code+English, so a ``max_seq_length`` token
# budget maps to ``max_seq_length * CHARS_PER_TOKEN`` characters.
CHARS_PER_TOKEN = 4

# HIP / CUDA / Composable-Kernel source extensions.
_HIP_EXTS = (".cu", ".cuh", ".hip", ".cpp", ".cc", ".hpp", ".h")

# Markers that identify a Triton kernel Python file (any one is enough).
_TRITON_MARKERS = ("import triton", "triton.jit", "triton.language", "tl.")


def _is_heldout_task_dir(dir_name: str) -> bool:
    """True if ``kore/tasks/<dir_name>`` is a held-out eval task (or family).

    Used to decontaminate the pretrain corpus: the held-out generalization set
    (task-level: paged-KV decode + MLA; plus any reserved family) must never enter
    training as source text. Core attention (prefill/decode/sliding/varlen/fp8) now
    TRAINS, so it is intentionally NOT excluded. Import is lazy + guarded so the
    corpus builder still runs if the registry is unavailable.
    """
    try:
        from kore.data.decontam import _family_of, heldout_families, heldout_task_ids
        if dir_name in heldout_task_ids():           # task-level holdout (paged / MLA)
            return True
        return _family_of(dir_name) in heldout_families()   # family-level holdout (if any)
    except Exception:  # noqa: BLE001 - registry missing -> do not exclude (safe)
        return False

# Path keywords that keep the ``docs`` slice a ROCm/kernel/perf corpus rather
# than pulling in every unrelated markdown file in the repos.
_DOC_KEYWORDS = (
    "rocprof", "tuning", "perf", "optimize", "occupancy", "triton", "hip",
    "rocm", "kernel", "amd", "gpu", "mi300", "mi200", "gfx", "matmul", "gemm",
    "attention", "quant", "fp8", "bf16",
)

# Directory names we never descend into (build/cache/vendor noise).
_SKIP_DIR_PARTS = frozenset({
    "__pycache__", ".git", "node_modules", ".venv", "venv", "build", "dist",
    ".mypy_cache", ".pytest_cache", ".egg-info",
})


def _is_skippable(path: Path) -> bool:
    parts = set(path.parts)
    if parts & _SKIP_DIR_PARTS:
        return True
    return any(p.endswith(".egg-info") for p in path.parts)


def _read_text(path: Path, max_chars: int) -> Optional[str]:
    """Best-effort UTF-8 read of a source file, truncated to ``max_chars``.

    Returns ``None`` for unreadable/binary/empty files so the caller can skip.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except (UnicodeDecodeError, OSError, ValueError):
        return None
    text = text.strip()
    if not text:
        return None
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


# --------------------------------------------------------------------------- #
# Source-tree discovery (offline / local only)
# --------------------------------------------------------------------------- #
def _kore_task_root() -> Optional[Path]:
    import kore
    root = Path(kore.__file__).resolve().parent / "tasks"
    return root if root.is_dir() else None


def discover_repo_roots() -> list[Path]:
    """Locate the local source repos (GEAK/KernelBench/KernelForge*/vllm/...).

    Checks ``KORE_REPOS_DIR`` then a set of candidate locations relative to the
    cwd and the installed ``kore`` package. Returns every existing candidate
    (de-duplicated, order-stable) so a build works regardless of where it runs.
    """
    import kore

    candidates: list[Path] = []
    env = os.environ.get("KORE_REPOS_DIR")
    if env:
        candidates.append(Path(env))
    pkg = Path(kore.__file__).resolve()   # .../<REPO>/kore/__init__.py
    candidates += [
        Path.cwd() / "repos",
        Path.cwd().parent / "repos",
        pkg.parents[1] / "repos",   # .../<REPO>/kore/__init__.py -> .../<REPO>/repos (CORRECT)
        pkg.parents[2] / "repos",   # extra fallback for a nested <repo>/kore/kore layout
    ]
    seen: set[Path] = set()
    out: list[Path] = []
    for c in candidates:
        try:
            rc = c.resolve()
            # is_dir() stats the path, which raises PermissionError (an OSError) for
            # e.g. /root on a non-root box - must be inside the guard, not after it.
            if rc in seen or not rc.is_dir():
                continue
        except OSError:
            continue
        seen.add(rc)
        out.append(rc)
    return out


# --------------------------------------------------------------------------- #
# File collection
# --------------------------------------------------------------------------- #
def _collect_files(
    roots: Iterable[Path],
    exts: tuple[str, ...],
    max_files: int,
    scan_budget: int,
    content_filter: Optional[Callable[[str], bool]] = None,
    max_chars_per_file: int = 200_000,
) -> list[tuple[Path, str]]:
    """Deterministically collect up to ``max_files`` ``(path, text)`` pairs.

    Walks ``roots`` for files with the given extensions in a globally sorted
    order (by relative path), reading at most ``scan_budget`` candidates and
    keeping those that pass ``content_filter`` (if any). Sorting makes the result
    order-stable and therefore the whole corpus deterministic.
    """
    # Gather (sort_key, path) so ordering is stable across roots.
    cands: list[tuple[str, Path]] = []
    for root in roots:
        for ext in exts:
            for p in root.rglob(f"*{ext}"):
                if _is_skippable(p) or not p.is_file():
                    continue
                try:
                    key = str(p.relative_to(root))
                except ValueError:
                    key = str(p)
                cands.append((f"{key}\x00{p}", p))
    cands.sort(key=lambda kp: kp[0])

    out: list[tuple[Path, str]] = []
    scanned = 0
    for _, p in cands:
        if len(out) >= max_files or scanned >= scan_budget:
            break
        scanned += 1
        text = _read_text(p, max_chars_per_file)
        if text is None:
            continue
        if content_filter is not None and not content_filter(text):
            continue
        out.append((p, text))
    return out


# --------------------------------------------------------------------------- #
# Chunking + dedup
# --------------------------------------------------------------------------- #
def chunk_text(text: str, budget_chars: int) -> list[str]:
    """Split ``text`` into <= ``budget_chars`` chunks on line boundaries.

    Accumulates whole lines up to the budget; a single line longer than the
    budget is hard-split so no emitted chunk ever exceeds ``budget_chars``.
    Deterministic and independent of any tokenizer.
    """
    if budget_chars <= 0:
        return [text] if text else []
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for line in text.splitlines(keepends=True):
        # Hard-split an over-long single line.
        while len(line) > budget_chars:
            if buf:
                chunks.append("".join(buf))
                buf, size = [], 0
            chunks.append(line[:budget_chars])
            line = line[budget_chars:]
        if size + len(line) > budget_chars and buf:
            chunks.append("".join(buf))
            buf, size = [], 0
        buf.append(line)
        size += len(line)
    if buf:
        chunks.append("".join(buf))
    return [c.strip() for c in chunks if c.strip()]


def _norm_hash(text: str) -> str:
    return hashlib.sha1(" ".join(text.split()).encode("utf-8")).hexdigest()


def _messages_to_text(messages: list[dict]) -> str:
    """Render chat ``messages`` into a plain-text completion document."""
    parts = []
    for m in messages:
        role = str(m.get("role", "")).strip() or "user"
        content = str(m.get("content", "")).strip()
        if content:
            parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def _load_kernelbook_pairs(n: int, max_chars: int) -> list:
    """Stream real (PyTorch module -> Triton) pairs from GPUMODE/KernelBook (HF).

    Returns ``[(pseudo_path, doc_text), ...]`` formatted like the local task pairs.
    Fully fail-safe: any error (missing datasets dep / offline / schema drift)
    returns [] so the corpus build never breaks. Used as corpus text only.
    """
    try:
        from datasets import load_dataset
    except Exception:
        return []
    out: list = []
    try:
        ds = load_dataset("GPUMODE/KernelBook", split="train", streaming=True)
        for i, ex in enumerate(ds):
            if len(out) >= n or i >= n * 8:
                break
            py = ex.get("python_code") or ex.get("pytorch_code")
            tri = ex.get("triton_code") or ex.get("original_triton_code")
            if not (isinstance(py, str) and isinstance(tri, str) and py.strip() and tri.strip()):
                continue
            doc = (f"# PyTorch module\n\n{py.strip()[:max_chars]}\n\n"
                   f"# Equivalent Triton kernel\n\n{tri.strip()[:max_chars]}\n")
            out.append((Path(f"kernelbook/pair_{i}.py"), doc))
    except Exception:
        return out  # partial results are fine; never raise
    return out


def _load_amd_kernels(n: int, max_chars: int) -> list:
    """Stream REAL AMD MI300 passing kernels from GPUMODE/kernelbot-data (HF).

    The ``amd_successful_submissions`` subset holds ~60k competition kernels that
    PASSED correctness on real MI300 hardware (fp8-gemm, MoE, MLA-decode, all2all,
    mxfp4, ...) - the highest-signal AMD-native (gfx942) kernel corpus available.
    Unlike KernelBook (NVIDIA/Inductor Triton), these are hand-optimized for AMD
    and carry the ``#!POPCORN`` problem header, so the model sees real gfx942
    idioms. ``code`` is stored as raw bytes; we decode + keep only passing rows.
    Fully fail-safe: any error returns partial/empty so the build never breaks.
    """
    try:
        from datasets import load_dataset
    except Exception:
        return []
    out: list = []
    try:
        ds = load_dataset("GPUMODE/kernelbot-data", "amd_successful_submissions",
                          split="train", streaming=True)
        for i, ex in enumerate(ds):
            if len(out) >= n or i >= n * 8:
                break
            if ex.get("run_passed") is False:   # subset is passing, but be strict
                continue
            code = ex.get("code")
            if isinstance(code, (bytes, bytearray)):
                code = code.decode("utf-8", errors="ignore")
            if not (isinstance(code, str) and code.strip()):
                continue
            out.append((Path(f"amd_kernels/sub_{i}.py"), code.strip()[:max_chars]))
    except Exception:
        return out  # partial results are fine; never raise
    return out


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _near_dedup_corpus(rows: list[dict], threshold: float = 0.7,
                       num_perm: int = 64, bands: int = 16) -> list[dict]:
    """Collapse near-duplicate corpus chunks via MinHash-LSH (The-Stack-v2 practice:
    MinHash + banded LSH, Jaccard 0.7). Exact-hash dedup upstream catches
    byte-identical chunks; this catches token-level near-dups (forked / reformatted /
    renamed kernels copied across repos) so continued-pretraining isn't dominated by
    redundant copies. O(n * bands) bucketing + confirm-with-Jaccard; keeps one
    representative per near-dup cluster (first-seen, so order is stable)."""
    from collections import defaultdict

    from kore.data.dedup import jaccard, minhash_signature

    n = len(rows)
    if n < 2:
        return rows
    rows_per_band = max(1, num_perm // bands)
    sigs = [minhash_signature(r.get("text", ""), num_perm=num_perm) for r in rows]
    buckets: dict = defaultdict(list)
    for i, sig in enumerate(sigs):
        for b in range(bands):
            band = tuple(sig[b * rows_per_band:(b + 1) * rows_per_band])
            buckets[(b, band)].append(i)

    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for idxs in buckets.values():
        if len(idxs) < 2:
            continue
        a = idxs[0]
        for b0 in idxs[1:]:
            if _find(a) == _find(b0):
                continue
            if jaccard(sigs[a], sigs[b0]) >= threshold:  # confirm (avoid band false-pos)
                parent[_find(b0)] = _find(a)

    seen_root: set = set()
    kept: list[dict] = []
    for i, r in enumerate(rows):
        root = _find(i)
        if root in seen_root:
            continue
        seen_root.add(root)
        kept.append(r)
    return kept


# Source weighting: oversample the highest-signal channels for continued
# pretraining. The real MI300 AMD competition kernels and the torch->Triton
# translation pairs are the most direct signal for the target task (writing fast
# device kernels), so they are seen more epochs than bulk repo code. Factors are
# deliberately conservative (<=2x) - heavy repetition risks CPT memorization /
# forgetting. Override via KORE_MIDTRAIN_WEIGHTS="src=factor,src=factor".
_DEFAULT_SOURCE_WEIGHTS: dict[str, float] = {
    "amd_kernels": 2.0,           # real gfx942 MI300 kernelbot submissions
    "pytorch_triton_pairs": 2.0,  # torch -> Triton (the core task, paired)
    "kore_tasks": 1.5,            # our own verified frontier tasks
    "kernelbook": 1.5,            # torch -> Triton translation corpus
}


# Held-out generalization concepts (MLA / paged-attention). Files whose PATH hits
# these are kept OUT of the CPT corpus so it never teaches the ops the eval measures
# as "unseen" (verbatim n-gram decontam misses differently-written impls in
# vllm/aiter/repos -- audit THEME B/C4). Anchored to path-segment boundaries so it
# only matches real MLA/paged files, not incidental substrings.
_HELDOUT_CONCEPT_RE = re.compile(
    # match the held-out concepts as path tokens INCLUDING concatenated spellings the
    # old regex missed: `paged_attention` (canonical vLLM/AITER: paged_attention_v1.cu),
    # `flashmla`, and `paged_decode`/`paged_prefill` (audit R2 midtrain I2).
    r"(mla|multi.?head.?latent|flashmla|paged?[_.\-]?(attn|attention|kv|cache|decode|prefill))",
    re.IGNORECASE)


def _is_heldout_concept(path) -> bool:
    try:
        return bool(_HELDOUT_CONCEPT_RE.search(str(path)))
    except Exception:  # noqa: BLE001
        return False


def _source_weights() -> dict[str, float]:
    """Return the source->oversample-factor map (env-overridable)."""
    w = dict(_DEFAULT_SOURCE_WEIGHTS)
    env = os.environ.get("KORE_MIDTRAIN_WEIGHTS", "").strip()
    if env:
        for part in env.split(","):
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            try:
                w[k.strip()] = max(1.0, float(v))
            except ValueError:
                pass
    return w


def build_midtrain_corpus(
    out_path,
    config,
    seed: int = 0,
    use_hf: bool = False,
    *,
    source_roots: Optional[list] = None,
    task_root=None,
    max_files_per_source: int = 5000,
    scan_budget: int = 40000,
    max_chars_per_file: int = 200_000,
) -> dict:
    """Assemble the Stage-0 continued-pretraining corpus and write it to disk.

    Args:
        out_path: destination JSONL path (parents created).
        config: a :class:`~kore.policy.configs.MidTrainConfig` (uses
            ``max_seq_length`` and ``general_replay_frac``).
        seed: determinism seed (drives general-replay sampling).
        use_hf: allow real HF general-replay sources (falls back to bundled
            offline samples on any failure). Kernel sources are always local.
        source_roots: override the repo roots to search (defaults to
            :func:`discover_repo_roots`). Tests pass a tmp source tree here.
        task_root: override the KORE task root (defaults to ``kore/tasks``).
        max_files_per_source: per-source file cap (bounds corpus size + cost).
        scan_budget: max candidate files inspected per collection pass.
        max_chars_per_file: truncate each source file to this many chars.

    Returns:
        A report dict: ``{"out_path", "total", "counts": {source: n},
        "n_dropped_dup", "general_frac", "max_seq_length", "budget_chars",
        "repo_roots"}``.
    """
    out_path = Path(out_path)
    # Env-tunable volume so a frontier midtrain can ingest thousands of MI300-native
    # kernels + KernelBook pairs + repo Triton/HIP/CK/docs (defaults already raised
    # 400 -> 5000 for a rich domain corpus; the highest-signal source is the ~60k
    # gfx942-native AMD kernels, previously capped at 400).
    max_files_per_source = int(os.environ.get("KORE_MIDTRAIN_MAX_FILES", max_files_per_source))
    scan_budget = int(os.environ.get("KORE_MIDTRAIN_SCAN_BUDGET", scan_budget))
    budget_chars = max(1, int(config.max_seq_length) * CHARS_PER_TOKEN)
    frac = float(getattr(config, "general_replay_frac", 0.15) or 0.0)

    repo_roots = [Path(r) for r in source_roots] if source_roots is not None \
        else discover_repo_roots()
    repo_roots = [r for r in repo_roots if r.is_dir()]
    troot = Path(task_root) if task_root is not None else _kore_task_root()

    # (source_label, list[(path, text)]) built from local files only.
    collected: list[tuple[str, list[tuple[Path, str]]]] = []

    # 1. KORE task Python (seed kernels, references, drivers) - EXCLUDING the
    # held-out generalization families (MLA / paged-KV decode) + arch-specific
    # tasks, so the eval set never leaks into pretraining (decontamination,
    # Pillar 5). Task-suite infrastructure files (_genops.py, base.py, ...) live
    # directly under the root and are kept; only per-task <task_id>/ dirs of a
    # held-out family are dropped.
    task_py: list[tuple[Path, str]] = []
    if troot is not None and troot.is_dir():
        task_py = _collect_files(
            [troot], (".py",), max_files=max_files_per_source,
            scan_budget=scan_budget, max_chars_per_file=max_chars_per_file,
            content_filter=lambda t: "__pycache__" not in t and len(t) > 20,
        )
        task_py = [(p, t) for (p, t) in task_py if not _is_heldout_task_dir(p.parent.name)]
    collected.append(("kore_tasks", task_py))

    # 2. PyTorch -> Triton pairs from each task's reference.py + seed_triton.py.
    pairs: list[tuple[Path, str]] = []
    if troot is not None and troot.is_dir():
        for task_dir in sorted(p for p in troot.iterdir() if p.is_dir()):
            if _is_heldout_task_dir(task_dir.name):  # decontamination (Pillar 5)
                continue
            ref = task_dir / "reference.py"
            seed_k = task_dir / "seed_triton.py"
            if not (ref.is_file() and seed_k.is_file()):
                continue
            ref_t = _read_text(ref, max_chars_per_file)
            seed_t = _read_text(seed_k, max_chars_per_file)
            if not (ref_t and seed_t):
                continue
            doc = (
                f"# PyTorch reference implementation ({task_dir.name})\n\n"
                f"{ref_t}\n\n"
                f"# Equivalent Triton kernel for {task_dir.name}\n\n"
                f"{seed_t}\n"
            )
            pairs.append((task_dir / "pair.py", doc))
    collected.append(("pytorch_triton_pairs", pairs))

    # 2b. REAL PyTorch->Triton pairs from KernelBook (HF, use_hf only). ~18k verified
    # (nn.Module -> Triton) pairs from torch.compile/Inductor - the best supervised
    # translate-and-fuse corpus. Used as CORPUS TEXT only (not executed), so the
    # NVIDIA/libdevice flavor of the Triton is fine for teaching the pattern.
    kb_pairs: list[tuple[Path, str]] = []
    if use_hf:
        kb_max = int(os.environ.get("KORE_MIDTRAIN_KERNELBOOK_MAX", str(max_files_per_source)))
        kb_pairs = _load_kernelbook_pairs(n=kb_max, max_chars=max_chars_per_file)
    collected.append(("kernelbook", kb_pairs))

    # 2c. REAL AMD MI300 passing kernels from GPUMODE/kernelbot-data (HF, use_hf
    # only). ~60k gfx942-native competition kernels (fp8-gemm/MoE/MLA/mxfp4/...) that
    # passed correctness on real MI300 - the highest-signal AMD-native corpus, which
    # KernelBook (NVIDIA/Inductor Triton) does not cover.
    amd_kernels: list[tuple[Path, str]] = []
    if use_hf:
        # The ~60k gfx942/MI300-native passing kernels are the HIGHEST-signal source,
        # so pull MANY more than other sources (up-weight); the MinHash near-dedup
        # below collapses the redundant competition submissions (many per problem)
        # into a diverse, deduped set of real AMD kernels. Env-tunable.
        amd_max = int(os.environ.get("KORE_MIDTRAIN_AMD_MAX",
                                     str(max(max_files_per_source, 25000))))
        amd_kernels = _load_amd_kernels(n=amd_max, max_chars=max_chars_per_file)
    collected.append(("amd_kernels", amd_kernels))

    # 3. Triton kernel Python files across the repos.
    triton_files: list[tuple[Path, str]] = []
    if repo_roots:
        triton_files = _collect_files(
            repo_roots, (".py",), max_files=max_files_per_source,
            scan_budget=scan_budget, max_chars_per_file=max_chars_per_file,
            content_filter=lambda t: any(m in t for m in _TRITON_MARKERS),
        )
    collected.append(("triton", triton_files))

    # 4. HIP / CUDA / Composable-Kernel source.
    hip_files: list[tuple[Path, str]] = []
    if repo_roots:
        hip_files = _collect_files(
            repo_roots, _HIP_EXTS, max_files=max_files_per_source,
            scan_budget=scan_budget, max_chars_per_file=max_chars_per_file,
        )
    collected.append(("rocm_hip", hip_files))

    # 5. ROCm / rocprof / tuning docs (path-filtered markdown).
    doc_files: list[tuple[Path, str]] = []
    if repo_roots:
        doc_files = _collect_files(
            repo_roots, (".md",), max_files=max_files_per_source,
            scan_budget=scan_budget * 3, max_chars_per_file=max_chars_per_file,
            content_filter=None,
        )
        # Match keywords against the file's own name + nearest parent dirs (NOT
        # the top-level repo dir, whose name - e.g. "KernelBench" - would else
        # sweep in every markdown file). Content-sniff the head as a fallback.
        def _doc_relevant(p: Path, t: str) -> bool:
            tail = "/".join(part.lower() for part in p.parts[-3:])
            if any(k in tail for k in _DOC_KEYWORDS):
                return True
            return any(k in t[:2000].lower() for k in _DOC_KEYWORDS)

        doc_files = [(p, t) for (p, t) in doc_files if _doc_relevant(p, t)][:max_files_per_source]
    collected.append(("docs", doc_files))

    # ------------------------------------------------------------------ #
    # SOTA quality gate (The-Stack-v2 / StarCoder2 style): drop minified,
    # autogenerated, vendored/3rd-party, boilerplate/high-repetition, and
    # license-only files BEFORE chunking so continued-pretraining sees clean,
    # high-signal domain code (and real dense kernels are preserved - the gate is
    # tuned not to reject long-line/low-comment kernels). Docs use the prose gate.
    # Gated by KORE_MIDTRAIN_QUALITY. Applied to every source incl. the mined AMD /
    # KernelBook sets so even those are screened for junk.
    # ------------------------------------------------------------------ #
    # Concept-level decontam: drop files whose PATH matches a held-out generalization
    # op (MLA / paged-attention) across ALL repo sources so the CPT corpus never
    # teaches the concepts the eval measures as unseen (audit THEME B/C4). Gated by
    # KORE_MIDTRAIN_DECONTAM_CONCEPTS.
    if os.environ.get("KORE_MIDTRAIN_DECONTAM_CONCEPTS", "1") != "0":
        _dc: list[tuple[str, list[tuple[Path, str]]]] = []
        _n_concept = 0
        for source, files in collected:
            keep = [(p, t) for (p, t) in files if not _is_heldout_concept(p)]
            _n_concept += len(files) - len(keep)
            _dc.append((source, keep))
        collected = _dc
        if _n_concept:
            log.info("midtrain concept decontam (held-out MLA/paged)", dropped=_n_concept)

    if os.environ.get("KORE_MIDTRAIN_QUALITY", "1") != "0":
        from kore.data.corpus_quality import quality_filter
        _filtered: list[tuple[str, list[tuple[Path, str]]]] = []
        _q_dropped = 0
        for source, files in collected:
            kept, st = quality_filter(files, is_doc=(source == "docs"))
            _filtered.append((source, kept))
            _q_dropped += st.get("dropped", 0)
            if st.get("dropped", 0):
                log.info("midtrain quality filter",
                         source=source, kept=st["kept"], dropped=st["dropped"],
                         reasons=st.get("drop_reasons", {}))
        collected = _filtered

    # ------------------------------------------------------------------ #
    # Chunk kernel-domain sources + dedup.
    # ------------------------------------------------------------------ #
    # Dedup is on normalized CONTENT only (no per-file provenance header is baked
    # into the text), so byte-identical files collapse to one set of chunks and
    # the ``source`` field remains the provenance channel.
    seen: set[str] = set()
    rows: list[dict] = []
    counts: dict[str, int] = {}
    n_dropped = 0

    for source, files in collected:
        n_src = 0
        for path, text in files:
            for chunk in chunk_text(text, budget_chars):
                h = _norm_hash(chunk)
                if h in seen:
                    n_dropped += 1
                    continue
                seen.add(h)
                rows.append({"text": chunk, "source": source})
                n_src += 1
        counts[source] = n_src

    # Near-duplicate collapse (SOTA: The-Stack-v2 MinHash-LSH, Jaccard 0.7). The
    # exact-hash pass above only catches byte-identical chunks; this removes the
    # token-level near-dups (forked/reformatted/renamed kernels copied across the
    # many repos we ingest) so continued-pretraining isn't dominated by redundant
    # copies. Applied to the kernel-domain chunks only (general replay is added
    # below and is already deduped upstream). Gated by KORE_MIDTRAIN_NEAR_DEDUP.
    n_near = 0
    if os.environ.get("KORE_MIDTRAIN_NEAR_DEDUP", "1") != "0" and len(rows) > 1:
        # Translation-pair sources (torch<->triton) are an intentional signal whose
        # value is the PAIRING even when the triton half also appears as raw code, so
        # they are exempt from near-dedup; only raw-code/doc redundancy is collapsed.
        _pair_srcs = {"pytorch_triton_pairs", "kernelbook"}
        _dedupable = [r for r in rows if r["source"] not in _pair_srcs]
        _kept = _near_dedup_corpus(_dedupable, threshold=0.7)
        _kept_ids = {id(r) for r in _kept}
        _new_rows = [r for r in rows if r["source"] in _pair_srcs or id(r) in _kept_ids]
        n_near = len(rows) - len(_new_rows)
        rows = _new_rows
        if n_near:
            kept_by_src: dict[str, int] = {}
            for r in rows:
                kept_by_src[r["source"]] = kept_by_src.get(r["source"], 0) + 1
            counts = {k: kept_by_src.get(k, 0) for k in counts}

    # Text-level decontamination backstop (Pillar 5): drop any chunk whose n-grams
    # overlap the HELD-OUT reference sources OR the RETENTION eval benchmarks. The
    # kore_tasks/pairs dir-filter above already excludes held-out KORE tasks; this also
    # catches a held-out kernel (e.g. an MLA / paged-KV-decode impl) copied into a mined
    # source (KernelBook / amd_kernels / repo Triton), AND -- new in R2 -- any general-
    # replay shard that carries an eval-benchmark question (MMLU/HumanEval/...), which
    # would otherwise train-on-test and inflate the retention gate. Safe no-op if the
    # reference sources can't be loaded.
    # Build the decontam n-gram set ONCE (held-out KORE kernels + the retention eval
    # benchmarks) and reuse it for BOTH the kernel rows here AND the general-replay
    # chunks appended later -- the replay slice (OpenCodeInstruct/OpenThoughts/tulu)
    # is the one most likely to carry a verbatim eval question, so it MUST be
    # decontaminated too (audit R2 midtrain I1: replay previously bypassed decontam).
    _decontam_ng: set = set()
    n_decontam = 0
    if os.environ.get("KORE_DECONTAM", "1") != "0":
        from kore.data.decontam import (build_heldout_ngrams, contaminated_by_text,
                                        eval_benchmark_texts)
        _eval_src = (eval_benchmark_texts()
                     if os.environ.get("KORE_DECONTAM_EVAL_BENCH", "1") != "0" else None)
        _decontam_ng = build_heldout_ngrams(8, extra_sources=_eval_src)
        if _decontam_ng:
            kept = [r for r in rows if not contaminated_by_text(r["text"], _decontam_ng, 8, 0.10)]
            n_decontam = len(rows) - len(kept)
            rows = kept
            if n_decontam:
                kept_by_src: dict[str, int] = {}
                for r in rows:
                    kept_by_src[r["source"]] = kept_by_src.get(r["source"], 0) + 1
                counts = {k: kept_by_src.get(k, 0) for k in counts}

    # ------------------------------------------------------------------ #
    # Source weighting: up-weight the highest-signal channels (real MI300 AMD
    # kernels + torch->Triton pairs) by oversampling their chunks so CPT sees
    # them for more effective epochs than bulk repo code. Fractional factors use
    # a seeded Bernoulli for the remainder (deterministic). Applied BEFORE replay
    # so the ~frac general slice is preserved against the weighted kernel total.
    # Gated by KORE_MIDTRAIN_WEIGHTING.
    # ------------------------------------------------------------------ #
    n_weighted = 0
    weights = _source_weights()
    if weights and os.environ.get("KORE_MIDTRAIN_WEIGHTING", "1") != "0":
        wrng = random.Random(seed + 7)
        wrows: list[dict] = []
        for r in rows:
            w = weights.get(r["source"], 1.0)
            reps = int(w)
            if wrng.random() < (w - reps):
                reps += 1
            for _ in range(max(1, reps)):
                wrows.append(r)
        n_weighted = len(wrows) - len(rows)
        rows = wrows
        if n_weighted:
            wc: dict[str, int] = {}
            for r in rows:
                wc[r["source"]] = wc.get(r["source"], 0) + 1
            counts = {k: wc.get(k, 0) for k in counts}
            log.info("midtrain source weighting", added=n_weighted,
                     weights={k: v for k, v in weights.items() if v != 1.0})

    n_kernel = len(rows)

    # ------------------------------------------------------------------ #
    # General replay: ~frac of the FINAL total (n_general = frac/(1-frac)*n_kernel).
    # ------------------------------------------------------------------ #
    n_general_target = 0
    if 0.0 < frac < 1.0 and n_kernel > 0:
        n_general_target = round(frac / (1.0 - frac) * n_kernel)
    n_general = 0
    if n_general_target > 0:
        # Distribute a SAMPLE budget across kinds; each sample yields >=1 chunk,
        # so we collect chunks in a stable order and stop exactly at the CHUNK
        # target (keeping the general slice at ~frac of the final total).
        kinds = list(REPLAY_KINDS)
        base = n_general_target // len(kinds)
        rem = n_general_target - base * len(kinds)
        per_kind = {k: base + (1 if i < rem else 0) for i, k in enumerate(kinds)}
        done = False
        for i, kind in enumerate(kinds):
            if done:
                break
            want = per_kind[kind]
            if want <= 0:
                continue
            replay = load_general_replay(kind, want, seed=seed + 1 + i, use_hf=use_hf)
            for r in replay:
                if done:
                    break
                text = _messages_to_text(r.get("messages", []))
                if not text:
                    continue
                for chunk in chunk_text(text, budget_chars):
                    if n_general >= n_general_target:
                        done = True
                        break
                    h = _norm_hash(chunk)
                    if h in seen:
                        continue
                    # Decontaminate the replay slice against held-out kernels + eval
                    # benchmarks (audit R2 midtrain I1) -- a mined general row carrying
                    # a HumanEval/MMLU item or a held-out MLA/paged kernel is train-on-
                    # test / leakage and must not enter CPT.
                    if _decontam_ng:
                        from kore.data.decontam import contaminated_by_text as _cbt
                        if _cbt(chunk, _decontam_ng, 8, 0.10):
                            continue
                    seen.add(h)
                    rows.append({"text": chunk, "source": "general_replay"})
                    n_general += 1
    counts["general_replay"] = n_general

    # ------------------------------------------------------------------ #
    # Write JSONL (deterministic order: kernel sources then general replay).
    # ------------------------------------------------------------------ #
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(rows)
    general_frac = (n_general / total) if total else 0.0
    report = {
        "out_path": str(out_path),
        "total": total,
        "counts": counts,
        "n_dropped_dup": n_dropped,
        "n_dropped_near_dup": n_near,
        "n_dropped_decontam": n_decontam,
        "n_weighted_added": n_weighted,
        "general_frac": round(general_frac, 4),
        "max_seq_length": int(config.max_seq_length),
        "budget_chars": budget_chars,
        "repo_roots": [str(r) for r in repo_roots],
    }
    log.info("midtrain corpus built", **{
        "out": str(out_path), "total": total, "general_frac": report["general_frac"],
        "dropped_dup": n_dropped, "dropped_near_dup": n_near, "dropped_decontam": n_decontam,
        **{f"n_{k}": v for k, v in counts.items()},
    })
    return report
