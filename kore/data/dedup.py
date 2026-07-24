"""Near-duplicate kernel detection + dedup (Pillar 5 hygiene).

Exact-hash dedup (``build_datasets.dedup_by_source_hash``) only collapses
byte-identical kernels, but the KORE corpus is dominated by NEAR-duplicates: the
same kernel with a renamed variable, a reflowed comment, a blank-line change, or
a tweaked constant appears dozens of times (in the shipped data ~148 kernels
recur >=50x). Training on those wastes capacity and over-weights a few templates.

Two complementary canonicalizers, both PURE (stdlib only - ast/hashlib/re):

  * :func:`structural_fingerprint` - parse the kernel as Python, drop docstrings,
    and hash the AST STRUCTURE with local identifiers alpha-renamed by first
    occurrence. Two kernels that differ only by variable naming / whitespace /
    comments collapse to the same fingerprint (type-1/type-2 clone detection).
    Falls back to a whitespace/comment-stripped text hash when the source does
    not parse (e.g. inline HIP/C strings).

  * :func:`minhash_signature` + :func:`jaccard` - k-shingle MinHash for FUZZY
    near-dup detection (estimated Jaccard) to catch structural near-dups that
    differ by a handful of tokens (a constant, an extra line).

:func:`dedup_near` collapses candidates keeping the BEST representative per
structural cluster (by a caller-supplied scorer, default: prefer the fastest /
most-preferred) and caps how many near-dups of any one structure survive.
"""

from __future__ import annotations

import ast
import hashlib
import re
from collections import Counter
from typing import Any, Callable, Iterable, Optional

# ``self`` is scope-significant; imported module *paths* remain semantic in
# ``ast.alias.name``, while their local aliases are normalized like other names.
_KEEP_NAMES = {"self"}


class _AlphaRename(ast.NodeTransformer):
    """Rename local Name/arg identifiers to canonical v0,v1,... by first use.

    Attribute names (``x.attr``), keyword arg names, and module roots in
    ``_KEEP_NAMES`` are preserved so ``tl.dot`` vs ``tl.load`` never collapse.
    """

    def __init__(self) -> None:
        self._map: dict[str, str] = {}

    def _canon(self, name: str) -> str:
        if name in _KEEP_NAMES:
            return name
        if name not in self._map:
            self._map[name] = f"v{len(self._map)}"
        return self._map[name]

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = self._canon(node.id)
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = self._canon(node.arg)
        node.annotation = None  # annotations are noise for structure
        return node

    def visit_alias(self, node: ast.alias) -> ast.AST:
        # Preserve the imported module path (``triton.language`` vs ``torch``)
        # but normalize a local ``as tl`` / ``as language`` spelling.
        if node.asname:
            node.asname = self._canon(node.asname)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        # Function names are provenance noise for clone/leakage detection: copied
        # kernels are routinely renamed when moved between repos.
        node.name = self._canon(node.name)
        node = self.generic_visit(node)  # type: ignore[assignment]
        node.returns = None
        _strip_docstring(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node.name = self._canon(node.name)
        node = self.generic_visit(node)  # type: ignore[assignment]
        node.returns = None
        _strip_docstring(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node.name = self._canon(node.name)
        return self.generic_visit(node)


def _strip_docstring(node: ast.AST) -> None:
    body = getattr(node, "body", None)
    if (isinstance(body, list) and body and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)):
        body.pop(0)


def _strip_comments_text(source: str) -> str:
    """Whitespace/comment-normalized text (fallback fingerprint for non-Python)."""
    lines = []
    for line in (source or "").splitlines():
        # drop full-line and trailing comments (best-effort; not string-aware)
        line = re.sub(r"#.*$", "", line)
        line = line.rstrip()
        if line.strip():
            lines.append(line.strip())
    return "\n".join(lines)


def structural_fingerprint(source: str) -> str:
    """Structure-only fingerprint (formatting/comment/rename invariant)."""
    src = (source or "").strip()
    if not src:
        return "empty"
    try:
        tree = ast.parse(src)
        for n in ast.walk(tree):
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef)):
                _strip_docstring(n)
        tree = _AlphaRename().visit(tree)
        ast.fix_missing_locations(tree)
        dump = ast.dump(tree, annotate_fields=False)
        return "ast:" + hashlib.sha1(dump.encode()).hexdigest()[:16]
    except SyntaxError:
        norm = _strip_comments_text(src)
        return "txt:" + hashlib.sha1(norm.encode()).hexdigest()[:16]


def content_hash(source: str) -> str:
    """Stable full-content SHA-256 used by provenance and exact dedup."""
    return "sha256:" + hashlib.sha256((source or "").encode("utf-8")).hexdigest()


def normalized_ast_fingerprint(source: str) -> Optional[str]:
    """Full SHA-256 of the alpha-normalized Python AST, or ``None`` if invalid.

    Unlike :func:`structural_fingerprint`, this never falls back to normalized
    text. That distinction matters to leakage reports: an ``ast`` reason proves
    both documents parsed and had the same normalized syntax tree.
    """
    src = (source or "").strip()
    if not src:
        return None
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _strip_docstring(node)
    tree = _AlphaRename().visit(tree)
    ast.fix_missing_locations(tree)
    dump = ast.dump(tree, annotate_fields=False, include_attributes=False)
    return "ast-sha256:" + hashlib.sha256(dump.encode("utf-8")).hexdigest()


def _attribute_name(node: ast.AST) -> str:
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    # Attribute names carry the operation semantics. The root binding is omitted
    # because import aliases are arbitrary; imported module paths are represented
    # separately by ``import:`` graph features below.
    return ".".join(reversed(parts)) if parts else type(node).__name__


def semantic_graph_features(source: str) -> tuple[str, ...]:
    """Return a normalized AST/data-flow graph feature multiset.

    Constants and local names are intentionally omitted, while call targets,
    operators, control-flow nodes, and parent->child edges are retained. This
    catches a copied kernel with renamed locals or tuned constants without
    equating unrelated kernels that merely share ``import triton`` boilerplate.
    Repeated features carry counts so a tiny skeleton cannot equal a real kernel.
    """
    try:
        tree = ast.parse((source or "").strip())
    except SyntaxError:
        return tuple()
    counts: Counter[str] = Counter()
    for parent in ast.walk(tree):
        ptype = type(parent).__name__
        if isinstance(parent, ast.Import):
            for alias in parent.names:
                counts[f"import:{alias.name}"] += 1
        elif isinstance(parent, ast.ImportFrom):
            counts[f"import:{parent.module or ''}"] += 1
        elif isinstance(parent, ast.Call):
            counts[f"call:{_attribute_name(parent.func)}"] += 1
        elif isinstance(parent, ast.BinOp):
            counts[f"binop:{type(parent.op).__name__}"] += 1
        elif isinstance(parent, ast.UnaryOp):
            counts[f"unary:{type(parent.op).__name__}"] += 1
        elif isinstance(parent, ast.BoolOp):
            counts[f"boolop:{type(parent.op).__name__}"] += 1
        elif isinstance(parent, ast.Compare):
            for op in parent.ops:
                counts[f"compare:{type(op).__name__}"] += 1
        elif isinstance(parent, (ast.For, ast.AsyncFor, ast.While, ast.If, ast.Try)):
            counts[f"control:{ptype}"] += 1
        for child in ast.iter_child_nodes(parent):
            ctype = type(child).__name__
            # Identifier/context leaf nodes add noise but no semantics.
            if ctype not in {"Load", "Store", "Del", "Name", "arg", "Constant"}:
                counts[f"edge:{ptype}>{ctype}"] += 1
    return tuple(f"{key}#{count}" for key, count in sorted(counts.items()))


def graph_fingerprint(source: str) -> Optional[str]:
    """Hash of :func:`semantic_graph_features`, or ``None`` for non-Python."""
    features = semantic_graph_features(source)
    if not features:
        return None
    payload = "\n".join(features).encode("utf-8")
    return "graph-sha256:" + hashlib.sha256(payload).hexdigest()


def graph_similarity(source_a: str, source_b: str) -> float:
    """Jaccard similarity between two normalized semantic-graph feature sets."""
    a, b = set(semantic_graph_features(source_a)), set(semantic_graph_features(source_b))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def token_shingles(source: str, k: int = 5) -> set[str]:
    """Token ``k``-shingles after comment/whitespace normalization."""
    toks = re.findall(r"[A-Za-z_][A-Za-z_0-9]*|[^\sA-Za-z_0-9]", _strip_comments_text(source))
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def minhash_signature(source: str, num_perm: int = 64, k: int = 5) -> tuple[int, ...]:
    """MinHash signature over k-token shingles (stdlib, deterministic)."""
    sh = token_shingles(source, k)
    if not sh:
        return tuple([0] * num_perm)
    sig = []
    for p in range(num_perm):
        seed = f"{p}:".encode()
        sig.append(min(int(hashlib.blake2b(seed + s.encode(), digest_size=8).hexdigest(), 16)
                       for s in sh))
    return tuple(sig)


def jaccard(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> float:
    if not sig_a or not sig_b or len(sig_a) != len(sig_b):
        return 0.0
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def directional_containment(candidate: str, reference: str, k: int = 8) -> dict:
    """Measure how much of ``reference`` is contained in ``candidate``.

    The denominator is the reference, never the candidate. A held-out kernel
    pasted into a very long training document therefore remains a 1.0 match
    instead of being diluted by unrelated candidate text.
    """
    cand = token_shingles(candidate, k)
    ref = token_shingles(reference, k)
    shared = cand & ref
    return {
        "containment": (len(shared) / len(ref)) if ref else 0.0,
        "candidate_coverage": (len(shared) / len(cand)) if cand else 0.0,
        "shared_shingles": len(shared),
        "reference_shingles": len(ref),
        "candidate_shingles": len(cand),
    }


def dedup_near(
    items: Iterable[dict],
    source_key: str = "source",
    scorer: Optional[Callable[[dict], float]] = None,
    per_fingerprint_cap: int = 1,
    fuzzy_threshold: float = 0.0,
    partition_key: Optional[str] = None,
    merge: Optional[Callable[[dict, list[dict]], dict]] = None,
) -> tuple[list[dict], dict]:
    """Collapse near-duplicate kernels, keeping the best representative(s).

    - Groups items by :func:`structural_fingerprint` (rename/format/comment
      invariant). Within a group keeps the top ``per_fingerprint_cap`` items by
      ``scorer`` (default: keep insertion order / all equal -> first).
    - When ``fuzzy_threshold > 0``, additionally merges groups whose MinHash
      Jaccard exceeds the threshold (catches token-level near-dups), keeping the
      globally best representative(s).

    Returns ``(kept, stats)``.
    """
    items = list(items)
    scorer = scorer or (lambda d: 0.0)
    # 1) exact-structural buckets. ``partition_key`` lets corpus assembly dedup
    # within a source channel while preserving intentional weighted channels.
    buckets: dict[tuple[Any, str], list[dict]] = {}
    for it in items:
        fp = structural_fingerprint(str(it.get(source_key, "")))
        partition = it.get(partition_key) if partition_key else None
        buckets.setdefault((partition, fp), []).append(it)

    # 2) optional fuzzy merge of buckets via representative MinHash
    if fuzzy_threshold > 0.0 and len(buckets) > 1:
        reps = {key: minhash_signature(str(grp[0].get(source_key, "")))
                for key, grp in buckets.items()}
        parent = {key: key for key in buckets}

        def _find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        keys = list(buckets)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                # Never bridge source partitions through fuzzy transitivity.
                if keys[i][0] != keys[j][0]:
                    continue
                if jaccard(reps[keys[i]], reps[keys[j]]) >= fuzzy_threshold:
                    parent[_find(keys[j])] = _find(keys[i])
        merged: dict[tuple[Any, str], list[dict]] = {}
        for key, grp in buckets.items():
            merged.setdefault(_find(key), []).extend(grp)
        buckets = merged

    kept: list[dict] = []
    for grp in buckets.values():
        grp_sorted = sorted(grp, key=scorer, reverse=True)
        winners = grp_sorted[:max(1, per_fingerprint_cap)]
        if merge is not None and winners:
            winners[0] = merge(winners[0], grp)
        kept.extend(winners)
    stats = {
        "n_in": len(items),
        "n_clusters": len(buckets),
        "n_kept": len(kept),
        "n_dropped": len(items) - len(kept),
    }
    return kept, stats


__all__ = [
    "content_hash",
    "structural_fingerprint",
    "normalized_ast_fingerprint",
    "semantic_graph_features",
    "graph_fingerprint",
    "graph_similarity",
    "token_shingles",
    "directional_containment",
    "minhash_signature",
    "jaccard",
    "dedup_near",
]
