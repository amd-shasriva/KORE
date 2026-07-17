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

from typing import Optional

from kore.search.alphakernel import AlphaKernelConfig, Edit, ProposeContext, search


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
                       value_model=None, reward_mode: str = "speedup",
                       k_expand: int = 4, seed: int = 0,
                       roofline_ub_fn=None) -> dict:
    """Run AlphaKernel from ``root_source`` over the verified-transform action space.

    A single call that constructs the :class:`TransformProposePolicy` and runs
    :func:`kore.search.alphakernel.search`. Returns the search result dict
    (``best_source`` / ``best_speedup_lcb`` / ``best_node`` / ``tree_stats``). The
    ``env`` supplies the perfect-verification benches (a KoreEnv in production); the
    optional ``value_model`` sets PUCT priors (falls back to the fitted reranker).
    """
    policy = TransformProposePolicy(k=k_expand)
    cfg = AlphaKernelConfig(reward_mode=reward_mode, k_expand=k_expand)
    return search(root_source, task, env, policy, value_model=value_model,
                  budget=budget, config=cfg, roofline_ub_fn=roofline_ub_fn, seed=seed)
