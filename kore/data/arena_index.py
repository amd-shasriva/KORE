"""Screen training tasks against AgentKernelArena.

The repo already has a capable decontaminator, and it indexes KernelBench and
KORE's own held-out task sources. It has never indexed AgentKernelArena -- the
benchmark the project's headline claim is actually made on. The arena runner is a
scorer wired to nothing: it contains no reference to decontamination, holdout, or
contamination of any kind.

That gap is not hypothetical. Both the training pool and the arena's ``gpumode``
sub-suites descend from the same ``GPUMODE/KernelBook`` scrape, and 12 arena
tasks have a pool task whose PyTorch module is byte-identical once normalised --
``torch2hip/gpumode/10190_FusedLeakyReLU`` is ``kbk_fused_leaky_relu_070f69dc_fp32``
down to the ``self.scale * leaky_relu(...)`` expression. Because the twin of that
pool task carries a HIP kernel verified correct on real gfx950, training on it
hands the model a checked answer to a question the benchmark is about to ask.

Two comparisons, because they fail in different directions:

* **Exact normalised-AST identity** catches the byte-identical cases regardless of
  length. It cannot be defeated by renaming or reformatting and it has no
  threshold to tune.
* **Document-frequency-filtered shingle Jaccard** catches near-duplicates. The
  filter matters more than it sounds: every ``nn.Module`` shares a
  ``class X / __init__ / super().__init__()`` skeleton, which on a small module is
  most of the document, so unfiltered Jaccard rates unrelated three-line wrappers
  at 0.44. Dropping shingles that appear in more than ``DF_MAX_RATIO`` of the
  corpus removes the skeleton and leaves the computation.

Scored pairs separate cleanly -- real matches bottom out at 0.333 and the next
candidate sits far below -- so :data:`DEFAULT_THRESHOLD` is set in that gap rather
than at a round number.

Operator identity is deliberately *not* treated as contamination. The arena's
KernelBench suite is textually clean but asks for GELU, Softmax and LayerNorm,
and the pool holds 27 distinct GELU modules. You cannot train a kernel model
without training on GELU. What is screened here is a shared *source document*,
not a shared operation.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional

#: Shingles appearing in more than this fraction of pool documents are structural
#: boilerplate rather than content, and are dropped before scoring.
DF_MAX_RATIO = 0.005

#: Token n-gram width.
SHINGLE_N = 5

#: Below this a pair is operator similarity, not a shared document. Set inside the
#: empirical gap between the lowest true match and the highest coincidence.
DEFAULT_THRESHOLD = 0.30

#: Identifiers that carry meaning across codebases and must not be anonymised, or
#: every module collapses onto the same shape.
FRAMEWORK_NAMES = frozenset("""
torch nn functional F Tensor Module Parameter cuda device dtype float16 bfloat16
float32 int8 matmul mm bmm addmm einsum softmax log_softmax sigmoid tanh relu
leaky_relu gelu silu elu selu mish hardtanh clamp sqrt rsqrt exp log pow abs sum
mean var std max min argmax argmin cat stack reshape view permute transpose
squeeze unsqueeze expand repeat contiguous flatten chunk split masked_fill where
dropout layer_norm batch_norm group_norm conv1d conv2d conv3d linear embedding
scaled_dot_product_attention cross_entropy mse_loss nll_loss sigmoid_focal_loss
interpolate pad avg_pool2d max_pool2d adaptive_avg_pool2d normalize cumsum
forward __init__ super self return def class if else for in with as
""".split())

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUM = re.compile(r"\b\d+\.?\d*(e-?\d+)?\b", re.I)
_STR = re.compile(r"(\"\"\".*?\"\"\"|'''.*?'''|\"[^\"]*\"|'[^']*')", re.S)
_HARNESS = ("get_inputs", "get_init_inputs", "get_input_shapes")


class _Strip(ast.NodeTransformer):
    """Remove imports, docstrings, and the input-generation harness.

    The harness is stripped because AMD rewrote it for perf scale, so an
    identical module reads as different code if it is kept -- which would
    suppress exactly the matches this index exists to find.
    """

    def visit_Import(self, node):  # noqa: N802 - ast API
        return None

    def visit_ImportFrom(self, node):  # noqa: N802 - ast API
        return None

    def visit_FunctionDef(self, node):  # noqa: N802 - ast API
        if node.name in _HARNESS:
            return None
        node.body = _drop_docstring(node.body)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node):  # noqa: N802 - ast API
        node.body = _drop_docstring(node.body)
        self.generic_visit(node)
        return node

    def visit_Module(self, node):  # noqa: N802 - ast API
        node.body = _drop_docstring(node.body)
        self.generic_visit(node)
        return node


def _drop_docstring(body: list) -> list:
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)):
        return body[1:]
    return body


def normalize(source: str) -> str:
    """Canonical text for a PyTorch module, or '' when it will not parse."""
    if not source or not source.strip():
        return ""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    try:
        tree = _Strip().visit(tree)
        ast.fix_missing_locations(tree)
        text = ast.unparse(tree)
    except Exception:  # noqa: BLE001 - unparse chokes on exotic nodes
        return ""
    text = _STR.sub('"S"', text)
    text = _NUM.sub("N", text)

    # Anonymise author-chosen names, keep framework vocabulary. Without this a
    # module renamed from `Net` to `Model` looks unrelated; with it, the shape of
    # the computation is what is compared.
    mapping: dict[str, str] = {}

    def rename(m: re.Match) -> str:
        name = m.group(0)
        if name in FRAMEWORK_NAMES:
            return name
        if name not in mapping:
            mapping[name] = f"v{len(mapping)}"
        return mapping[name]

    return _IDENT.sub(rename, text)


def shingles(normalized: str, n: int = SHINGLE_N) -> set[str]:
    toks = normalized.split()
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def exact_hash(normalized: str) -> str:
    return hashlib.sha256(normalized.encode("utf-8", "ignore")).hexdigest()


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter) if inter else 0.0


class ArenaIndex:
    """A frozen pool-task -> arena-task contamination table.

    Matching is a dict lookup rather than a live comparison: the table is built
    once by ``scripts/v5_build_arena_index.py`` and committed as an artifact, so
    every build screens against exactly the same evidence and a rebuild is an
    explicit, reviewable act rather than a silent drift in what got excluded.
    """

    def __init__(self, table: dict[str, Any], threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self.meta = table.get("meta", {})
        raw = table.get("matches", {})
        self._matches: dict[str, tuple[str, float]] = {
            k: (str(v[0]), float(v[1])) for k, v in raw.items()
        }

    @classmethod
    def load(cls, path: Any, threshold: float = DEFAULT_THRESHOLD) -> "ArenaIndex":
        p = Path(path)
        return cls(json.loads(p.read_text()), threshold)

    @classmethod
    def empty(cls) -> "ArenaIndex":
        return cls({"meta": {"empty": True}, "matches": {}})

    def match(self, task_id: str) -> Optional[tuple[str, float]]:
        """The arena task this pool task duplicates, if any clears the threshold."""
        hit = self._matches.get(task_id)
        if hit is None:
            return None
        return hit if hit[1] >= self.threshold else None

    def __len__(self) -> int:
        return sum(1 for v in self._matches.values() if v[1] >= self.threshold)

    def __repr__(self) -> str:
        return (f"ArenaIndex(threshold={self.threshold}, blocking={len(self)}, "
                f"scored={len(self._matches)})")


def build_document_frequency(docs: Iterable[set[str]], max_ratio: float = DF_MAX_RATIO
                             ) -> set[str]:
    """Shingles common enough across the corpus to be structure, not content."""
    counts: dict[str, int] = {}
    total = 0
    for s in docs:
        total += 1
        for sh in s:
            counts[sh] = counts.get(sh, 0) + 1
    if not total:
        return set()
    cutoff = max(2, int(max_ratio * total))
    return {sh for sh, c in counts.items() if c > cutoff}


__all__ = [
    "ArenaIndex", "DEFAULT_THRESHOLD", "DF_MAX_RATIO", "SHINGLE_N",
    "build_document_frequency", "exact_hash", "jaccard", "normalize", "shingles",
]
