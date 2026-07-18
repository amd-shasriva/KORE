"""Production ``ProposePolicy`` for AlphaKernel: the verified transformation calculus.

:func:`kore.search.alphakernel.search` needs a ``ProposePolicy`` (a move generator
that expands a search node into child kernels). Until now no production adapter
existed -- only a ``FakePolicy`` in tests. This module is that adapter: it turns
:mod:`kore.transform` (the verified epsilon-typed transformation library -- 13 real
Triton source rewrites, each exact/approx with an epsilon cost + side conditions)
into the move generator, so AlphaKernel searches over PROVABLY-in-contract
transformations rather than free-form edits.

Each node is expanded by enumerating the transforms admissible for its kernel under
a fresh per-node epsilon budget and applying each to produce a child. The env's SNR
gate remains the hard numerical guard (an over-approximate child fails verification
-> pruned), so the budget is a sound action-space prior, not the correctness
authority. Pure/CPU (source rewrites only; no torch/model).

:func:`search_from_kernel` wires the adapter + AlphaKernel into one call usable at
serial-rollout time (search-then-distill) OR test time (search over a trained
policy's kernels), which is AlphaKernel's canonical, affordable home.
"""

from __future__ import annotations

from typing import Callable, Optional

from kore.search.alphakernel import (
    AlphaKernelConfig,
    Edit,
    ProposeContext,
    make_roofline_ub_fn,
    search,
)


class TransformProposePolicy:
    """Expand a search node using the verified transformation calculus.

    ``k`` bounds the children returned per expansion (AlphaKernel also passes its own
    ``k_expand``; the min applies). ``library`` overrides the default transform set.
    """

    def __init__(self, *, k: int = 4, library=None):
        self.k = max(1, int(k))
        self.library = library

    def _budget(self, task):
        from kore.transform import ErrorBudget
        op = getattr(task, "operation", None) or getattr(task, "task_id", "") or ""
        dtype = getattr(task, "dtype", "fp32") or "fp32"
        return ErrorBudget.for_op(op, dtype)

    def propose(self, state: ProposeContext) -> list[Edit]:
        """Return up to ``k`` child kernels, one per admissible transform. Fail-safe:
        any transform/import error yields no edit rather than raising into search."""
        src = state.source
        if not src:
            return []
        try:
            from kore.transform import admissible_actions, apply_sequence
        except Exception:  # noqa: BLE001 - transform optional -> no expansion
            return []
        try:
            actions = admissible_actions(src, self._budget(state.task), self.library)
        except Exception:  # noqa: BLE001
            return []
        edits: list[Edit] = []
        for a in actions:
            if len(edits) >= self.k:
                break
            try:
                # Fresh per-child budget: branching from one parent, each child spends
                # the transform's epsilon against the full task tolerance (the env's
                # SNR gate is the hard guard on any path that drifts too far).
                budget = self._budget(state.task)
                new_src, applied, rejected, _ = apply_sequence(
                    src, [a.as_step()], budget, self.library)
            except Exception:  # noqa: BLE001 - a bad rewrite is just skipped
                continue
            if applied and not rejected and new_src and new_src != src:
                edits.append(Edit(source=new_src, name=a.name,
                                  meta={"eps": a.eps, "relation": a.relation}))
        return edits


def search_from_kernel(root_source: str, task, env, *, budget: int = 64,
                       value_model=None, value_fn: Optional[Callable] = None,
                       reward_mode: str = "speedup",
                       k_expand: int = 4, max_depth: Optional[int] = None,
                       incumbent_min_measures: int = 1,
                       value_leaf_weight: float = 0.0,
                       roofline_ub_fn=None, seed: int = 0) -> dict:
    """Run AlphaKernel from ``root_source`` over the verified-transform action space.

    A single call that constructs the :class:`TransformProposePolicy` and runs
    :func:`kore.search.alphakernel.search`. Returns the search result dict
    (``best_source`` / ``best_speedup_lcb`` / ``best_node`` / ``tree_stats``). The
    ``env`` supplies the perfect-verification benches (a KoreEnv in production).

    Depth / breadth (item 3 -- SAFE defaults reproduce the historical shallow search)
    ----------------------------------------------------------------------------------
    ``budget`` (verifier-call cap), ``k_expand`` (candidate edits per expansion) and
    ``max_depth`` (max node depth to expand; None => unbounded) let the orchestrator
    dial the search deeper. The defaults (64 / 4 / None) are exactly the prior search.

    Value model (item 4)
    --------------------
    ``value_model`` -- a trained :class:`kore.value.model.ValueModel` (used via
    ``.predict``) -- or ``value_fn(sources, task) -> [float]`` set the PUCT priors
    (default: the rerank heuristic). ``value_leaf_weight`` > 0 additionally uses the
    value model as a bounded PRIOR leaf value for correct-but-unmeasured nodes.

    Branch-and-bound (item 1)
    -------------------------
    ``roofline_ub_fn`` -- an admissible ``(source, task) -> Optional[float]`` speedup
    ceiling -- turns roofline pruning ON. Build the production bound with
    :func:`kore.search.alphakernel.make_roofline_ub_fn`. Default None => OFF (prior
    behavior). ``incumbent_min_measures`` raises the sample floor a node needs before
    it can seed the (monotone) B&B pruning bound.
    """
    policy = TransformProposePolicy(k=k_expand)
    cfg = AlphaKernelConfig(
        reward_mode=reward_mode, k_expand=k_expand, max_depth=max_depth,
        incumbent_min_measures=incumbent_min_measures,
        value_leaf_weight=value_leaf_weight,
    )
    return search(root_source, task, env, policy, value_model=value_model,
                  budget=budget, config=cfg, roofline_ub_fn=roofline_ub_fn,
                  value_fn=value_fn, seed=seed)


# Re-exported for convenience so callers can enable branch-and-bound with a single
# import alongside search_from_kernel (see the module docstring / README).
__all__ = ["TransformProposePolicy", "search_from_kernel", "make_roofline_ub_fn"]
