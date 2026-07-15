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
from typing import Any, Callable, Iterable, Optional

# Identifiers we NEVER alpha-rename: module aliases + dunder-ish names that carry
# semantic meaning across kernels (renaming `tl` -> v0 is fine since it is
# consistent, but keeping the well-known roots avoids surprising collapses).
_KEEP_NAMES = {"tl", "triton", "torch", "hl", "self", "True", "False", "None"}


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

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        # keep the public entry-point NAME (semantic), rename only the body
        node = self.generic_visit(node)  # type: ignore[assignment]
        node.returns = None
        _strip_docstring(node)
        return node


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


def _shingles(source: str, k: int = 5) -> set[str]:
    toks = re.findall(r"[A-Za-z_][A-Za-z_0-9]*|[^\sA-Za-z_0-9]", _strip_comments_text(source))
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def minhash_signature(source: str, num_perm: int = 64, k: int = 5) -> tuple[int, ...]:
    """MinHash signature over k-token shingles (stdlib, deterministic)."""
    sh = _shingles(source, k)
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


def dedup_near(
    items: Iterable[dict],
    source_key: str = "source",
    scorer: Optional[Callable[[dict], float]] = None,
    per_fingerprint_cap: int = 1,
    fuzzy_threshold: float = 0.0,
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
    # 1) exact-structural buckets
    buckets: dict[str, list[dict]] = {}
    for it in items:
        fp = structural_fingerprint(str(it.get(source_key, "")))
        buckets.setdefault(fp, []).append(it)

    # 2) optional fuzzy merge of buckets via representative MinHash
    if fuzzy_threshold > 0.0 and len(buckets) > 1:
        reps = {fp: minhash_signature(str(grp[0].get(source_key, "")))
                for fp, grp in buckets.items()}
        parent: dict[str, str] = {fp: fp for fp in buckets}

        def _find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        fps = list(buckets)
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                if jaccard(reps[fps[i]], reps[fps[j]]) >= fuzzy_threshold:
                    parent[_find(fps[j])] = _find(fps[i])
        merged: dict[str, list[dict]] = {}
        for fp, grp in buckets.items():
            merged.setdefault(_find(fp), []).extend(grp)
        buckets = merged

    kept: list[dict] = []
    for grp in buckets.values():
        grp_sorted = sorted(grp, key=scorer, reverse=True)
        kept.extend(grp_sorted[:max(1, per_fingerprint_cap)])
    stats = {
        "n_in": len(items),
        "n_clusters": len(buckets),
        "n_kept": len(kept),
        "n_dropped": len(items) - len(kept),
    }
    return kept, stats


__all__ = [
    "structural_fingerprint",
    "minhash_signature",
    "jaccard",
    "dedup_near",
]
