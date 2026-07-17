"""AlphaKernel: value-guided test-time search over kernel transformations.

AlphaKernel is the P1 (planning / test-time-search) component of KORE. It treats
the verified environment (:class:`kore.env.kore_env.KoreEnv`) as a **perfect
simulator**: every leaf is *exactly* labeled correct/incorrect by the oracle and,
when correct, *measured* by the timing harness. On top of that oracle it runs an
AlphaZero-style best-first search whose "moves" are kernel *transformations*
proposed by a policy and whose "value" is the pessimistic (LCB) measured speedup:

    root (seed kernel)
      |-- edit_1 --> kernel'      (compile -> correct? -> measure a few times)
      |-- edit_2 --> kernel''     (transposition of an earlier node -> reuse)
      ...

Design (mirrors the module docstrings of the interfaces it builds on):

1. **Node = a kernel state** with a semantic ``fingerprint`` (hash of the
   canonicalized source + IO signature), a ``status`` (correct / incorrect /
   compile_fail / pruned), streaming measurement stats (``speedup_mean``,
   ``speedup_lcb``, ``n_measures``, ``var``), a value-model ``prior``, a
   ``roofline_ub`` admissible ceiling, ``children``/``parents`` (a DAG, via the
   transposition table), a ``visit_count`` and a ``best_descendant_reward``.

2. **Selection** is best-first with a PUCT acquisition over the frontier using the
   PESSIMISTIC value (``best_descendant_reward``, backed up from measured
   ``speedup_lcb``) + the value-model prior exploration term + a novelty/diversity
   bonus. A node whose ``roofline_ub <= incumbent.speedup_lcb`` is provably
   dominated and is NEVER selected (admissible branch-and-bound).

3. **Expansion** asks the (pluggable) policy for up to ``k`` candidate edits,
   scores them with the value model to set priors, and adds the children -
   deduplicating by fingerprint against a global transposition table so an
   equivalent kernel reached by a different path is *linked* (DAG), inheriting the
   exact measured value with no re-measurement.

4. **Leaf eval** runs the exact oracle: compile/correctness via the env. An
   incorrect kernel is a LOW-value but *repairable* node (not dead); a correct
   kernel is then measured a few times through the perf oracle, keeping the LCB.

5. **Backup is MAX**, not mean: a node's ``best_descendant_reward`` is the best
   value anywhere in its subtree, so the search commits to the single best kernel
   it can reach (test-time search, not policy averaging).

6. **Measurement allocation** is Successive Halving (:mod:`kore.search.bandit`):
   many candidates get 1-2 measurements, survivors get more to tighten the LCB.

7. **Budget** is a hard global cap on verifier calls; the anytime **incumbent** is
   ``argmax`` over CORRECT nodes by ``speedup_lcb`` and its reported value is
   monotone non-decreasing.

The module is import-light and CPU-only: it touches the GPU only through the
injected ``env`` and reuses the existing reward/value/roofline interfaces without
modifying them, so it is fully exercisable with scripted fakes.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, runtime_checkable

from kore.config import CONFIG
from kore.reward.physics import compute_kernel_reward
from kore.search.bandit import Budget, CallbackArm, MeasureStats, successive_halving

# Node value tiers on a unified scalar so MAX-backup and selection respect the
# reward ladder (compile_fail < infra < incorrect < correct). Correct nodes sit a
# fixed base above every non-correct node and are then ordered by their pessimistic
# measured speedup, so "improve a correct kernel" always outranks "repair a broken
# one" while a broken kernel stays selectable (repairable, not dead).
_Q_CORRECT_BASE: float = 1.0
_Q_INCORRECT: float = 0.0
_Q_INFRA: float = -0.5
_Q_COMPILE_FAIL: float = -1.0


# --------------------------------------------------------------------------- #
# Semantic fingerprint (transposition key): canonicalized source + IO signature
# --------------------------------------------------------------------------- #
_TRIPLE_STR = re.compile(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'')
_LINE_COMMENT = re.compile(r"#.*")
_HSPACE = re.compile(r"[ \t]+")
_DEF = re.compile(r"def\s+([A-Za-z_]\w*)\s*\(([^)]*)\)")


def canonicalize_source(source: str) -> str:
    """Normalize a kernel source so cosmetically-different-but-equivalent kernels
    map to the same string: strip docstrings + ``#`` comments, collapse runs of
    horizontal whitespace, and drop blank lines. This is deliberately syntactic
    (no execution): two kernels that differ only by comments/formatting share a
    fingerprint and are deduped, while any change to a token, tile size, or
    statement produces a distinct node."""
    s = _TRIPLE_STR.sub(" ", source or "")
    out: list[str] = []
    for line in s.splitlines():
        line = _LINE_COMMENT.sub("", line)
        line = _HSPACE.sub(" ", line).strip()
        if line:
            out.append(line)
    return "\n".join(out)


def io_signature(source: str) -> str:
    """Extract a coarse IO signature: the ``name(arg,arg,...)`` of every ``def``
    (type/default annotations stripped). Folded into the fingerprint so two bodies
    that canonicalize alike but expose different entry-point signatures do not
    collide."""
    sigs: list[str] = []
    for m in _DEF.finditer(source or ""):
        name = m.group(1)
        args = [a.split(":")[0].split("=")[0].strip()
                for a in m.group(2).split(",") if a.strip()]
        sigs.append(f"{name}({','.join(args)})")
    return "|".join(sigs)


def fingerprint(source: str, io_sig: Optional[str] = None) -> str:
    """Stable semantic hash of a kernel = ``sha256(canonical_source + io_sig)``.

    ``io_sig`` may be supplied by the caller (e.g. the task's fixed driver IO
    contract); otherwise it is derived from the source's ``def`` signatures.
    """
    canon = canonicalize_source(source)
    sig = io_sig if io_sig is not None else io_signature(source)
    h = hashlib.sha256()
    h.update(canon.encode("utf-8", "ignore"))
    h.update(b"\x1f")
    h.update(sig.encode("utf-8", "ignore"))
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Pluggable interfaces (all fakeable in tests)
# --------------------------------------------------------------------------- #
@dataclass
class Edit:
    """One proposed kernel transformation = the full resulting child source plus
    human-readable metadata (the transform ``name`` and optional ``meta``). The
    policy is responsible for turning a diff/instruction into a complete kernel."""

    source: str
    name: str = "edit"
    meta: dict = field(default_factory=dict)


@dataclass
class ProposeContext:
    """Read-only view of the node being expanded, handed to the policy so it can
    condition its transformations on the current kernel and its measured state."""

    source: str
    depth: int
    correct: bool
    speedup_lcb: Optional[float]
    fingerprint: str
    task: object


@runtime_checkable
class ProposePolicy(Protocol):
    """A move generator: propose kernel transformations for a search node."""

    def propose(self, state: ProposeContext) -> list[Edit]: ...


@runtime_checkable
class ValueModel(Protocol):
    """A cheap surrogate that scores candidate sources (higher = more promising)
    BEFORE any GPU measurement, used to set PUCT priors. Optional: when omitted,
    AlphaKernel falls back to :func:`kore.value.rerank.score_candidates` (a fitted
    model if installed, else the source heuristic)."""

    def score(self, sources: list[str], task: object) -> list[float]: ...


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class AlphaKernelConfig:
    """Search hyper-parameters (all with production-reasonable defaults)."""

    c_puct: float = 1.5           # PUCT exploration weight on the value prior
    c_novelty: float = 0.25       # diversity/novelty bonus weight
    lcb_z: float = 1.0            # LCB pessimism (sigmas); see bandit.DEFAULT_LCB_Z
    k_expand: int = 4             # max candidate edits requested per expansion
    sh_eta: int = 2               # Successive-Halving elimination factor
    sh_min_measures: int = 2      # first-look measurements per candidate (>=2 so
    #                               variance is estimable before the first cut)
    sh_max_measures: int = 4      # measurement cap for a surviving arm
    max_iters: int = 1000         # hard cap on expansions (budget is the real cap)
    reward_mode: str = "speedup"  # compute_kernel_reward mode ("speedup"/"residual")
    multi_shape: bool = True      # env.step multi-shape (worst-shape discipline)
    step_cost: int = 1            # budget units per env.step


# --------------------------------------------------------------------------- #
# Node
# --------------------------------------------------------------------------- #
@dataclass(eq=False)  # identity semantics (mutable DAG node); hashable by id
class Node:
    source: str
    fingerprint: str
    depth: int = 0
    prior: float = 0.0                       # normalized value-model prior (PUCT P)
    value_prior: float = 0.0                 # raw value-model score (diagnostic)
    edit_name: str = "root"
    roofline_ub: float = float("inf")        # admissible speedup ceiling (B&B)

    # exact-oracle verdict
    evaluated: bool = False
    compiled: bool = False
    correct: bool = False
    infra: bool = False
    snr_db: Optional[float] = None
    status: str = "new"                       # new/correct/incorrect/compile_fail/infra/pruned
    pruned: bool = False

    # measurement (perf oracle)
    stats: MeasureStats = field(default_factory=MeasureStats)

    # search bookkeeping
    children: list["Node"] = field(default_factory=list)
    parents: list["Node"] = field(default_factory=list)
    expanded: bool = False
    visit_count: int = 0
    self_value: float = float("-inf")
    best_descendant_reward: float = float("-inf")

    # -- measurement conveniences (design's "measured {...}") ------------- #
    @property
    def n_measures(self) -> int:
        return self.stats.n

    @property
    def speedup_mean(self) -> float:
        return self.stats.mean

    @property
    def speedup_lcb(self) -> float:
        return self.stats.lcb

    @property
    def var(self) -> float:
        return self.stats.var


# --------------------------------------------------------------------------- #
# Value-model scoring (adapter over the pluggable ValueModel / rerank fallback)
# --------------------------------------------------------------------------- #
def _value_scores(value_model, sources: list[str], task) -> list[float]:
    """Score candidate sources with the value model (higher = better prior).

    Resolution order: an explicit ``.score(sources, task)`` (the fakeable
    AlphaKernel contract) -> a fitted rerank ``ValueModel`` via
    :func:`kore.value.rerank.score_candidates` -> a plain callable -> the rerank
    source-heuristic (model=None). Every branch is defensive so a value-model
    failure degrades to uniform priors rather than breaking the search."""
    if not sources:
        return []
    if value_model is not None and hasattr(value_model, "score"):
        try:
            return [float(x) for x in value_model.score(list(sources), task)]
        except Exception:  # noqa: BLE001 - never let the surrogate break search
            pass
    try:
        from kore.value.rerank import score_candidates
        model = value_model if (value_model is not None
                                and hasattr(value_model, "predict")) else None
        return [float(x) for x in score_candidates(list(sources), task=task, model=model)]
    except Exception:  # noqa: BLE001 - numpy/value deps unavailable
        pass
    if callable(value_model):
        try:
            return [float(value_model(s)) for s in sources]
        except Exception:  # noqa: BLE001
            pass
    return [0.0] * len(sources)


def _softmax(xs: list[float]) -> list[float]:
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


def _struct_key(source: str):
    """Coarse structural signature for the novelty term: a kernel's schedule knobs
    (tile sizes, warps, tl.dot/reduction presence). Structurally-different kernels
    fall in different buckets, so the novelty bonus steers the search toward
    unexplored regions of schedule space rather than re-tuning one family."""
    try:
        from kore.value.features import extract_schedule_features
        s = extract_schedule_features(source)
        return (s.get("block_m"), s.get("block_n"), s.get("block_k"),
                s.get("num_warps"), bool(s.get("has_tl_dot")),
                bool(s.get("has_reduction_loop")))
    except Exception:  # noqa: BLE001 - value features unavailable
        return (len(source or ""),)


# --------------------------------------------------------------------------- #
# Roofline ceiling helper (admissible perf bound via kore.analysis.rooflines)
# --------------------------------------------------------------------------- #
def roofline_speedup_ceiling(task, baseline_ms: float, shape=None,
                             arch: Optional[str] = None) -> Optional[float]:
    """Admissible speedup ceiling = ``baseline_ms / T_min`` for the task's shape.

    ``T_min`` is the roofline lower bound on runtime (:func:`kore.analysis.
    rooflines.roofline`), so no correct kernel can exceed ``baseline_ms / T_min``.
    Returns None when the operator is not roofline-modelable or the roofline deps
    are unavailable (the caller then leaves ``roofline_ub = +inf`` -> no pruning).
    A production ``roofline_ub_fn`` can wrap this once a baseline is measured."""
    try:
        from kore.analysis.rooflines import (
            detect_arch, resolve_peaks, roofline, shape_to_str,
        )
    except Exception:  # noqa: BLE001
        return None
    if not baseline_ms or baseline_ms <= 0:
        return None
    a = arch or detect_arch()
    peaks = resolve_peaks(a)
    sh = shape or (task.shapes[0] if getattr(task, "shapes", None) else None)
    dims = getattr(sh, "dims", None)
    if not dims:
        return None
    rf = roofline(getattr(task, "task_id", "?"), getattr(task, "operation", ""),
                  getattr(task, "dtype", "fp32"), shape_to_str(dims), dims, peaks, a)
    if rf is None or not (rf.t_min_ms > 0):
        return None
    return baseline_ms / rf.t_min_ms


# --------------------------------------------------------------------------- #
# The search
# --------------------------------------------------------------------------- #
class _Search:
    """Internal search state machine. One instance per :func:`search` call."""

    def __init__(self, task, env, policy, value_model, budget, cfg,
                 roofline_ub_fn):
        self.task = task
        self.env = env
        self.policy = policy
        self.value_model = value_model
        self.cfg = cfg
        self.budget: Budget = budget if isinstance(budget, Budget) else Budget(int(budget))
        self.roofline_ub_fn = roofline_ub_fn

        self.dtype = getattr(task, "dtype", "fp32")
        self.snr_threshold = getattr(task, "snr_threshold", None)
        self.correctness_weight = float(getattr(CONFIG, "correctness_weight", 0.3))

        self.tt: dict[str, Node] = {}          # transposition table (fingerprint -> Node)
        self.env_calls: int = 0
        self.iters: int = 0
        self.n_transpositions: int = 0
        self.struct_counts: dict = {}

        self.incumbent: Optional[Node] = None
        self.incumbent_lcb: float = float("-inf")
        self.incumbent_trace: list[Optional[float]] = []
        self._baseline_ms: Optional[float] = None

    # -- env access ------------------------------------------------------- #
    def _env_step(self, source: str, do_bench: bool):
        """Run one ``env.step`` (the expensive verifier call) and score it.

        The :class:`Budget` unit for this call is reserved by the caller BEFORE
        invoking it (the correctness gate reserves in :meth:`_leaf_eval`; a
        measurement is reserved by :func:`successive_halving`), so there is exactly
        one ``budget.spend`` per ``env.step`` and ``budget.used == env_calls``.
        """
        obs = self.env.step(source, full_validation=do_bench,
                            multi_shape=self.cfg.multi_shape)
        self.env_calls += 1
        b = getattr(obs, "baseline_ms", None)
        if b:
            self._baseline_ms = b
        rr = compute_kernel_reward(obs, source, self.task, mode=self.cfg.reward_mode,
                                   dtype=self.dtype, snr_threshold=self.snr_threshold)
        return rr, obs

    # -- leaf eval: exact compile/correctness oracle ---------------------- #
    def _leaf_eval(self, node: Node) -> bool:
        """Run the exact oracle (no timing) to label the node. Returns False iff the
        budget was exhausted before it could be evaluated."""
        if not self.budget.spend(self.cfg.step_cost):
            return False
        rr, obs = self._env_step(node.source, do_bench=False)
        node.evaluated = True
        node.compiled = bool(getattr(obs, "compiled", False))
        node.infra = bool(getattr(obs, "infra_error", False)) or rr.tier == "infra"
        node.correct = bool(rr.correct)
        node.snr_db = getattr(obs, "snr_db", None)
        if node.infra:
            node.status = "infra"            # transient; low value but not dead
        elif not node.compiled:
            node.status = "compile_fail"
        elif not node.correct:
            node.status = "incorrect"        # repairable low-value node
        else:
            node.status = "correct"
        return True

    def _measure_once(self, node: Node) -> Optional[float]:
        """Draw one perf-oracle speedup sample for a correct node.

        The budget unit is reserved by the Successive-Halving allocator before this
        runs, so it does not spend itself. Returns the sample, or None if the timed
        verdict carried no speedup. The caller (:class:`~kore.search.bandit.
        CallbackArm`) folds the value into the node's shared :class:`MeasureStats`."""
        rr, _ = self._env_step(node.source, do_bench=True)
        if rr.speedup is None:
            return None
        if rr.correct:
            node.correct = True
        return float(rr.speedup)

    def _allocate_measures(self, nodes: list[Node]) -> None:
        """Successive-Halving measurement allocation over correct nodes (item 6)."""
        arms = [
            CallbackArm(key=n, sampler=(lambda n=n: self._measure_once(n)),
                        stats=n.stats, ceiling=n.roofline_ub)
            for n in nodes if n.correct
        ]
        if not arms:
            return
        for a in arms:
            a.stats.z = self.cfg.lcb_z
        successive_halving(
            arms, self.budget, eta=self.cfg.sh_eta,
            min_measures=self.cfg.sh_min_measures, max_measures=self.cfg.sh_max_measures,
            rank_key="lcb", incumbent_lcb=self.incumbent_lcb,
        )

    # -- node value + backup (MAX) ---------------------------------------- #
    def _node_q(self, node: Node) -> float:
        if node.correct and node.stats.n > 0:
            return _Q_CORRECT_BASE + node.stats.lcb   # pessimistic measured value
        if node.correct:
            return _Q_CORRECT_BASE                     # correct, measurement pending
        if node.status == "compile_fail":
            return _Q_COMPILE_FAIL
        if node.status == "infra":
            return _Q_INFRA
        # incorrect (compiled): repairable; tiny SNR-progress shaping keeps it above
        # a compile failure and gives a gradient toward the correctness gate.
        return _Q_INCORRECT + self._snr_shaping(node)

    def _snr_shaping(self, node: Node) -> float:
        if node.snr_db is None or not self.snr_threshold:
            return 0.0
        frac = max(0.0, min(1.0, node.snr_db / float(self.snr_threshold)))
        return 0.04 * frac  # << _Q_CORRECT_BASE so it can never reach a correct node

    def _backup(self, node: Node) -> None:
        """Propagate MAX ``best_descendant_reward`` up the DAG from ``node``."""
        stack = [node]
        seen: set = set()
        while stack:
            cur = stack.pop()
            if id(cur) in seen:
                continue
            seen.add(id(cur))
            cur.self_value = self._node_q(cur)
            child_max = max((c.best_descendant_reward for c in cur.children),
                            default=float("-inf"))
            newv = max(cur.self_value, child_max)
            if newv != cur.best_descendant_reward:
                cur.best_descendant_reward = newv
            for p in cur.parents:
                if id(p) not in seen:
                    stack.append(p)

    # -- incumbent (anytime best; monotone via running max) --------------- #
    def _update_incumbent(self) -> None:
        best = None
        for n in self.tt.values():
            if n.correct and n.stats.n > 0:
                if best is None or n.stats.lcb > best.stats.lcb:
                    best = n
        if best is not None and best.stats.lcb > self.incumbent_lcb:
            self.incumbent_lcb = best.stats.lcb
            self.incumbent = best
        self.incumbent_trace.append(
            self.incumbent_lcb if self.incumbent is not None else None)

    def _admissible(self, node: Node) -> bool:
        """Admissible branch-and-bound: a node whose roofline ceiling cannot beat
        the current incumbent LCB is dominated and must never be selected."""
        if self.incumbent_lcb == float("-inf"):
            return True
        return node.roofline_ub > self.incumbent_lcb

    # -- selection (best-first PUCT over the frontier) -------------------- #
    def _novelty(self, node: Node) -> float:
        cnt = self.struct_counts.get(_struct_key(node.source), 0)
        return 1.0 / math.sqrt(1.0 + cnt)

    def _selection_score(self, node: Node) -> float:
        q = node.best_descendant_reward
        parent_visits = max((p.visit_count for p in node.parents),
                            default=node.visit_count)
        explore = (self.cfg.c_puct * node.prior
                   * math.sqrt(1 + parent_visits) / (1 + node.visit_count))
        return q + explore + self.cfg.c_novelty * self._novelty(node)

    def _select(self) -> Optional[Node]:
        """Pick the best frontier node to expand; prune dominated ones en route."""
        frontier: list[Node] = []
        for n in self.tt.values():
            if n.pruned or n.expanded or not n.evaluated:
                continue
            if not self._admissible(n):
                n.pruned = True
                n.status = "pruned"           # skip the whole dominated subtree
                continue
            frontier.append(n)
        if not frontier:
            return None
        return max(frontier, key=self._selection_score)

    # -- expansion -------------------------------------------------------- #
    def _register(self, node: Node) -> None:
        self.tt[node.fingerprint] = node
        key = _struct_key(node.source)
        self.struct_counts[key] = self.struct_counts.get(key, 0) + 1

    @staticmethod
    def _link(parent: Node, child: Node) -> None:
        if child not in parent.children:
            parent.children.append(child)
        if parent not in child.parents:
            child.parents.append(parent)

    def _make_child(self, parent: Optional[Node], edit: Edit) -> tuple[Node, bool]:
        """Create-or-link a child from an edit. Returns (node, is_new). A fingerprint
        hit links to the existing node (DAG) and inherits its exact value with NO
        re-measurement (transposition)."""
        fp = fingerprint(edit.source)
        existing = self.tt.get(fp)
        if existing is not None:
            if parent is not None:
                self._link(parent, existing)
            self.n_transpositions += 1
            return existing, False
        child = Node(source=edit.source, fingerprint=fp,
                     depth=(parent.depth + 1 if parent else 0),
                     edit_name=edit.name)
        rub = self.roofline_ub_fn(edit.source, self.task) if self.roofline_ub_fn else None
        child.roofline_ub = float(rub) if rub is not None else float("inf")
        if parent is not None:
            self._link(parent, child)
        self._register(child)
        return child, True

    def _expand(self, node: Node) -> None:
        node.expanded = True
        node.visit_count += 1
        for p in node.parents:
            p.visit_count += 1

        ctx = ProposeContext(source=node.source, depth=node.depth,
                             correct=node.correct, speedup_lcb=node.speedup_lcb,
                             fingerprint=node.fingerprint, task=self.task)
        try:
            edits = list(self.policy.propose(ctx) or [])
        except Exception:  # noqa: BLE001 - a policy hiccup must not kill the search
            edits = []
        edits = edits[: self.cfg.k_expand]
        if not edits:
            self._backup(node)
            return

        scores = _value_scores(self.value_model, [e.source for e in edits], self.task)
        priors = _softmax(scores)
        new_children: list[Node] = []
        for edit, score, prior in zip(edits, scores, priors):
            child, is_new = self._make_child(node, edit)
            if not is_new:
                continue                       # transposition: linked, never re-measured
            child.value_prior = score
            child.prior = prior
            if self.budget.remaining <= 0:
                break                          # out of budget: stop evaluating children
            if self._leaf_eval(child):
                child.self_value = self._node_q(child)
                child.best_descendant_reward = child.self_value
                new_children.append(child)

        # Perf-oracle measurement allocation over the newly-correct children.
        self._allocate_measures([c for c in new_children if c.correct])
        for c in new_children:
            c.self_value = self._node_q(c)
            c.best_descendant_reward = c.self_value
        self._backup(node)

    # -- driver ----------------------------------------------------------- #
    def run(self, root_source: str) -> Node:
        root, _ = self._make_child(None, Edit(source=root_source, name="root"))
        if self._leaf_eval(root):
            self._allocate_measures([root] if root.correct else [])
            root.self_value = self._node_q(root)
            root.best_descendant_reward = root.self_value
        self._update_incumbent()

        while self.iters < self.cfg.max_iters and self.budget.remaining >= self.cfg.step_cost:
            node = self._select()
            if node is None:
                break
            self._expand(node)
            self._update_incumbent()
            self.iters += 1
        return root

    def stats(self, root: Node) -> dict:
        nodes = list(self.tt.values())
        by_status: dict = {}
        for n in nodes:
            by_status[n.status] = by_status.get(n.status, 0) + 1
        n_edges = sum(len(n.children) for n in nodes)
        return {
            "n_nodes": len(nodes),
            "n_correct": sum(1 for n in nodes if n.correct),
            "n_incorrect": by_status.get("incorrect", 0),
            "n_compile_fail": by_status.get("compile_fail", 0),
            "n_pruned": by_status.get("pruned", 0),
            "n_expanded": sum(1 for n in nodes if n.expanded),
            "n_edges": n_edges,
            "n_transpositions": self.n_transpositions,
            "max_depth": max((n.depth for n in nodes), default=0),
            "iterations": self.iters,
            "env_calls": self.env_calls,
            "budget_total": self.budget.total,
            "budget_used": self.budget.used,
            "root_best_descendant_reward": root.best_descendant_reward,
            "incumbent_trace": list(self.incumbent_trace),
            "sol_speedup_ceiling": (
                roofline_speedup_ceiling(self.task, self._baseline_ms)
                if self._baseline_ms else None),
        }


def search(root_source: str, task, env, policy, value_model=None, budget=64, *,
           config: Optional[AlphaKernelConfig] = None,
           roofline_ub_fn: Optional[Callable[[str, object], Optional[float]]] = None,
           seed: int = 0) -> dict:
    """Run AlphaKernel value-guided search from ``root_source``.

    Parameters
    ----------
    root_source : the seed kernel to search from (e.g. ``task.seed_source``).
    task        : a KORE :class:`~kore.tasks.base.Task` (or any object exposing
                  ``dtype``/``operation``/``shapes``/``snr_threshold``).
    env         : a verified environment exposing
                  ``step(source, full_validation, multi_shape) -> Observation``
                  (a :class:`~kore.env.kore_env.KoreEnv` in production).
    policy      : a :class:`ProposePolicy` (``propose(ProposeContext) -> [Edit]``).
    value_model : optional :class:`ValueModel` for PUCT priors; falls back to
                  :func:`kore.value.rerank.score_candidates`.
    budget      : int (verifier-call cap) or a :class:`~kore.search.bandit.Budget`.
    roofline_ub_fn : optional ``(source, task) -> Optional[float]`` giving a per-node
                  admissible speedup ceiling for branch-and-bound pruning; when
                  omitted no roofline pruning is applied.

    Returns a dict with:
      ``best_source``        - the incumbent (best CORRECT node by ``speedup_lcb``),
      ``best_speedup_lcb``   - its pessimistic measured speedup (None if none found),
      ``best_node``          - the incumbent :class:`Node` (or None),
      ``root``               - the root :class:`Node` (DAG entry point),
      ``tree_stats``         - counters incl. the monotone ``incumbent_trace``.
    """
    cfg = config or AlphaKernelConfig()
    engine = _Search(task, env, policy, value_model, budget, cfg, roofline_ub_fn)
    root = engine.run(root_source)
    inc = engine.incumbent
    return {
        "best_source": inc.source if inc is not None else None,
        "best_speedup_lcb": inc.speedup_lcb if inc is not None else None,
        "best_node": inc,
        "root": root,
        "tree_stats": engine.stats(root),
    }
