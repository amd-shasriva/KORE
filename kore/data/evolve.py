"""Evolutionary kernel datagen (FunSearch / AlphaEvolve / AVO + D-MAB).

An offline, native evolutionary loop that grows verified fast kernels for a task
and emits them as training data. Three ideas from the review are combined:

  * **D-MAB adaptive operator selection** (:class:`DMABBandit`). Instead of
    picking a mutation/strategy operator uniformly, a UCB1 bandit *learns* which
    operator pays off (reward = verified improvement), and a Page-Hinkley change
    detector restarts the bandit when the reward regime shifts (a new elite makes
    a previously-useless operator suddenly valuable). This is the Dynamic MAB
    (D-MAB) of Fialho et al. for adaptive operator selection.

  * **Island / MAP-Elites archive** (:class:`MapElitesArchive`). Elites are kept
    in a behavior-descriptor grid (op-family x speedup-bin x correctness) so the
    search preserves *diverse* strong kernels rather than collapsing onto one.
    Multiple islands evolve in parallel with periodic ring migration
    (:func:`migrate`), and the top elites are fed back as few-shot exemplars to
    the generator - FunSearch/AlphaEvolve best-shot multi-parent prompting.

  * **Value-model prefilter**. Each generation proposes several candidates; the
    cheap value model (``kore.value.rerank.rank_candidates``) ranks them and only
    the top-k are actually benched (Ansor-style measurement efficiency).

``evolve_task`` runs the island loop against any ``generator`` exposing
``.generate(messages) -> str`` (a teacher or the policy) and any environment
exposing ``.step(source, ...) -> Observation``. It returns verified
``WinRecord``s and ``RankedGroupRecord``s ready for the dataset pipeline. Nothing
here needs a GPU beyond what ``env.step`` uses, so with a StubTeacher + fake env
it runs on CPU in tests.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional

from kore.config import CONFIG
from kore.data.gen_groups import build_preferences
from kore.data.gen_groups import rank_candidates as _rank_results
from kore.data.mutate import apply_operator, infer_family, list_operators
from kore.data.prompts import (
    SYSTEM_PROMPT,
    build_turn_prompt,
    extract_kernel,
    format_assistant_turn,
    normalize_assistant,
)
from kore.data.schemas import RankedGroupRecord, WinRecord
from kore.env.replay import kernel_hash
from kore.obs import get_logger
from kore.reward.reward import Observation, compute_reward
from kore.value.rerank import rank_candidates as _value_rank

log = get_logger("data.evolve")


# --------------------------------------------------------------------------- #
# D-MAB: UCB1 + Page-Hinkley change-detection bandit
# --------------------------------------------------------------------------- #
class DMABBandit:
    """Dynamic multi-armed bandit for adaptive operator selection.

    Arm selection is UCB1::

        ucb(op) = mean_reward(op) + c * sqrt(ln(total) / count(op))

    (each unplayed arm is tried once first). A two-sided Page-Hinkley test runs on
    the reward stream; when it fires (the reward distribution has drifted) the
    bandit statistics are *restarted* so it re-explores - this is what makes it
    "dynamic" and robust to non-stationary rewards (D-MAB).

    ``pulls`` / ``reward_sums`` accumulate across restarts (for reporting/tests);
    ``counts`` / ``values`` are the live UCB stats that get reset on a restart.
    """

    def __init__(
        self,
        operators,
        c: float = 1.4,
        ph_delta: float = 0.05,
        ph_lambda: float = 8.0,
        seed: int = 0,
    ):
        self.operators = list(operators)
        if not self.operators:
            raise ValueError("DMABBandit needs at least one operator")
        self.c = float(c)
        self.ph_delta = float(ph_delta)
        self.ph_lambda = float(ph_lambda)
        self.rng = random.Random(seed)
        # live UCB stats (reset on change detection)
        self.counts = {op: 0 for op in self.operators}
        self.values = {op: 0.0 for op in self.operators}
        self.total = 0
        # cumulative stats (never reset)
        self.pulls = {op: 0 for op in self.operators}
        self.reward_sums = {op: 0.0 for op in self.operators}
        self.n_resets = 0
        self.history: list[tuple[str, float]] = []
        self._ph_reset()

    # -- Page-Hinkley (two-sided) --
    def _ph_reset(self) -> None:
        self._ph_n = 0
        self._ph_mean = 0.0
        self._ph_cum_inc = 0.0
        self._ph_min = 0.0
        self._ph_cum_dec = 0.0
        self._ph_max = 0.0

    def _page_hinkley(self, reward: float) -> bool:
        self._ph_n += 1
        self._ph_mean += (reward - self._ph_mean) / self._ph_n
        # detector for an INCREASE in the mean
        self._ph_cum_inc += reward - self._ph_mean - self.ph_delta
        self._ph_min = min(self._ph_min, self._ph_cum_inc)
        ph_inc = self._ph_cum_inc - self._ph_min
        # detector for a DECREASE in the mean
        self._ph_cum_dec += reward - self._ph_mean + self.ph_delta
        self._ph_max = max(self._ph_max, self._ph_cum_dec)
        ph_dec = self._ph_max - self._ph_cum_dec
        return ph_inc > self.ph_lambda or ph_dec > self.ph_lambda

    def _restart(self) -> None:
        self.counts = {op: 0 for op in self.operators}
        self.values = {op: 0.0 for op in self.operators}
        self.total = 0
        self.n_resets += 1
        self._ph_reset()

    # -- MAB interface --
    def select(self) -> str:
        unplayed = [op for op in self.operators if self.counts[op] == 0]
        if unplayed:
            return self.rng.choice(unplayed)
        logt = math.log(max(self.total, 1))
        best_op = self.operators[0]
        best_score = float("-inf")
        # shuffle so ties are broken fairly (and deterministically per-seed)
        ops = list(self.operators)
        self.rng.shuffle(ops)
        for op in ops:
            ucb = self.values[op] + self.c * math.sqrt(logt / self.counts[op])
            if ucb > best_score:
                best_score = ucb
                best_op = op
        return best_op

    def update(self, op: str, reward: float) -> None:
        if op not in self.counts:
            raise KeyError(f"unknown operator: {op!r}")
        reward = float(reward)
        self.counts[op] += 1
        self.total += 1
        self.values[op] += (reward - self.values[op]) / self.counts[op]
        self.pulls[op] += 1
        self.reward_sums[op] += reward
        self.history.append((op, reward))
        if self._page_hinkley(reward):
            self._restart()

    def mean_reward(self, op: str) -> float:
        n = self.pulls[op]
        return self.reward_sums[op] / n if n else 0.0

    def best_operator(self) -> str:
        """Operator with the highest cumulative mean reward (>=1 pull)."""
        played = [op for op in self.operators if self.pulls[op] > 0]
        pool = played or self.operators
        return max(pool, key=self.mean_reward)


# --------------------------------------------------------------------------- #
# MAP-Elites / island archive
# --------------------------------------------------------------------------- #
@dataclass
class EliteRecord:
    source: str
    correct: bool
    speedup: Optional[float]
    snr_db: Optional[float]
    op_family: str
    descriptor: tuple
    meta: dict = field(default_factory=dict)

    @property
    def fitness(self) -> float:
        # correctness dominates; among correct kernels, higher speedup wins.
        base = 1e6 if self.correct else 0.0
        return base + (self.speedup or 0.0)


def _speedup_bin(speedup: Optional[float], bins) -> int:
    if speedup is None:
        return -1
    b = 0
    for i, edge in enumerate(bins):
        if speedup >= edge:
            b = i
    return b


def behavior_descriptor(op_family: str, speedup: Optional[float], correct: bool, bins) -> tuple:
    """(op_family, speedup_bin, correctness) - the MAP-Elites cell key."""
    return (op_family, _speedup_bin(speedup, bins) if correct else -1, bool(correct))


class MapElitesArchive:
    """A behavior-descriptor grid keeping the best kernel per cell.

    Cell key = (op_family, speedup-bin, correctness). ``add``/``insert`` keep a
    candidate only if its cell is empty or it beats the incumbent's fitness, so
    the archive holds a *diverse* set of elites across the speedup landscape."""

    def __init__(self, speedup_bins=(1.0, 1.2, 1.5, 2.0, 3.0), seed: int = 0):
        self.speedup_bins = tuple(speedup_bins)
        self.cells: dict[tuple, EliteRecord] = {}
        self.rng = random.Random(seed)

    def descriptor(self, op_family: str, speedup: Optional[float], correct: bool) -> tuple:
        return behavior_descriptor(op_family, speedup, correct, self.speedup_bins)

    def add(self, record: EliteRecord) -> bool:
        cur = self.cells.get(record.descriptor)
        if cur is None or record.fitness > cur.fitness:
            self.cells[record.descriptor] = record
            return True
        return False

    def insert(self, source: str, correct: bool, speedup: Optional[float],
               snr_db: Optional[float], op_family: str, meta: Optional[dict] = None) -> bool:
        rec = EliteRecord(
            source=source, correct=bool(correct), speedup=speedup, snr_db=snr_db,
            op_family=op_family,
            descriptor=self.descriptor(op_family, speedup, correct),
            meta=meta or {},
        )
        return self.add(rec)

    def elites(self) -> list[EliteRecord]:
        return list(self.cells.values())

    def best(self, n: int = 1) -> list[EliteRecord]:
        return sorted(self.cells.values(), key=lambda r: r.fitness, reverse=True)[:n]

    def coverage(self) -> int:
        return len(self.cells)

    def __len__(self) -> int:
        return len(self.cells)


def migrate(src: MapElitesArchive, dst: MapElitesArchive, n: int = 1) -> int:
    """Copy ``src``'s top-``n`` elites into ``dst`` (kept only where better).

    Returns the number that landed as new/improved elites in ``dst``."""
    moved = 0
    for rec in src.best(n):
        if dst.add(rec):
            moved += 1
    return moved


# --------------------------------------------------------------------------- #
# evolve_task
# --------------------------------------------------------------------------- #
@dataclass
class EvolveConfig:
    islands: int = 2
    candidates_per_gen: int = 4
    prefilter_k: int = 2               # bench only the top-k after value prefilter
    migration_interval: int = 3        # migrate every N generations
    migrants: int = 1
    max_shots: int = 2                 # elites fed as few-shot exemplars
    improve_factor: float = 0.98       # a kept step must beat parent wall by >=2%
    speedup_bins: tuple = (1.0, 1.2, 1.5, 2.0, 3.0)
    operator_kind: str = "optimize"    # bandit operator set (see mutate.list_operators)
    seed: int = 0
    bandit_c: float = 1.4
    ph_delta: float = 0.05
    ph_lambda: float = 8.0
    model: Any = None                  # value model for prefilter (None -> heuristic)


@dataclass
class EvolveResult:
    wins: list
    groups: list
    archive: MapElitesArchive
    islands: list
    bandit: DMABBandit
    stats: dict


def _wall_us(obs: Observation) -> Optional[float]:
    return obs.wall_ms * 1000.0 if getattr(obs, "wall_ms", None) is not None else None


def _bench(env, task, source: str, cfg):
    """Evaluate one candidate; never raise (a crashing verifier -> compile_fail)."""
    try:
        obs = env.step(source, full_validation=True, multi_shape=True)
    except Exception as e:  # keep the loop alive
        obs = Observation(compiled=False, dtype=getattr(task, "dtype", "fp32"),
                          error_text=str(e)[:200])
    rr = compute_reward(obs, source, dtype=getattr(task, "dtype", "fp32"), cfg=cfg)
    return obs, rr


def _result_dict(source: str, obs: Observation, rr) -> dict:
    return {
        "source": source,
        "compiled": bool(obs.compiled),
        "correct": bool(rr.correct),
        "speedup": rr.speedup,
        "snr_db": obs.snr_db,
        "wall_us": _wall_us(obs),
    }


def _fewshot_messages(task, elites, parent_src, feedback, mode, max_shots) -> list[dict]:
    """FunSearch/AlphaEvolve best-shot multi-parent prompt: prepend the top
    elites as few-shot exemplars, then the standard optimization turn prompt."""
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    op = getattr(task, "operation", None) or getattr(task, "task_id", "kernel")
    for e in elites[:max_shots]:
        if not e.correct or not e.source:
            continue
        sp = f"{e.speedup:.2f}x" if e.speedup else "correct"
        msgs.append({"role": "user",
                     "content": f"Here is a strong prior {op} kernel ({sp}). Learn from it."})
        msgs.append({"role": "assistant",
                     "content": f"FULL_KERNEL:\n```python\n{e.source}\n```"})
    msgs.append({"role": "user",
                 "content": build_turn_prompt(parent_source=parent_src,
                                              feedback=feedback, mode=mode)})
    return msgs


def _feedback(obs: Observation, rr) -> str:
    if not obs.compiled:
        return f"FAILED to compile: {(obs.error_text or '')[:300]}"
    if not rr.correct:
        return f"Correct? NO. snr_db={obs.snr_db}. Fix correctness first."
    wall = _wall_us(obs)
    return (f"Correct? YES. wall={wall:.1f}us speedup={rr.speedup:.3f}x. Make it faster."
            if wall is not None else "Correct? YES. Make it faster.")


def evolve_task(
    task,
    generator,
    env,
    generations: int = 6,
    cfg: Optional[EvolveConfig] = None,
    reward_cfg=CONFIG,
) -> EvolveResult:
    """Run the island evolutionary loop on ``task``.

    ``generator`` is anything with ``.generate(messages) -> str`` (a teacher or
    the policy); ``env`` is anything with ``.step(source, full_validation=,
    multi_shape=) -> Observation``. Returns verified ``WinRecord``s and
    ``RankedGroupRecord``s, plus the final archive/bandit for inspection.
    """
    cfg = cfg or EvolveConfig()
    rng = random.Random(cfg.seed)
    family = infer_family(getattr(task, "operation", None) or getattr(task, "task_id", ""))
    operators = list_operators(cfg.operator_kind)
    bandit = DMABBandit(operators, c=cfg.bandit_c, ph_delta=cfg.ph_delta,
                        ph_lambda=cfg.ph_lambda, seed=cfg.seed)
    islands = [MapElitesArchive(cfg.speedup_bins, seed=cfg.seed + i)
               for i in range(max(1, cfg.islands))]

    with log.stage("evolve_task", task=getattr(task, "task_id", "?"),
                   generations=generations, islands=len(islands)):
        seed_src = task.seed_source
        seed_obs, seed_rr = _bench(env, task, seed_src, reward_cfg)
        seed_wall = _wall_us(seed_obs)
        for isl in islands:
            isl.insert(seed_src, seed_rr.correct, seed_rr.speedup, seed_obs.snr_db, family)

        best_src = seed_src
        best_wall = seed_wall
        best_snr = seed_obs.snr_db
        best_correct = seed_rr.correct

        groups: list[RankedGroupRecord] = []
        trajectory: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        feedback = _feedback(seed_obs, seed_rr)
        mode = "exploit"

        n_benched = 0
        n_correct = 0

        for gen in range(generations):
            isl = islands[gen % len(islands)]
            elites = isl.best(cfg.max_shots)
            parent = elites[0] if elites else None
            parent_src = parent.source if parent else best_src
            parent_wall = None
            if parent is not None and parent.correct and parent.speedup and seed_wall:
                parent_wall = seed_wall / parent.speedup
            parent_wall = parent_wall or best_wall

            # ----- propose candidates -----
            candidates: list[tuple[str, str]] = []  # (source, provenance)
            msgs = _fewshot_messages(task, elites, parent_src, feedback, mode, cfg.max_shots)
            trajectory.append(msgs[-1])
            resp = generator.generate(msgs)
            # store the canonical contract (Pillar 0) - raw teacher text may be loosely
            # shaped; build_sft also canonicalizes at the boundary as a backstop.
            trajectory.append({"role": "assistant", "content": normalize_assistant(resp)})
            gsrc = extract_kernel(resp)
            if gsrc:
                candidates.append((gsrc, "generator"))
            # operator moves selected by the D-MAB bandit
            while len(candidates) < cfg.candidates_per_gen:
                op = bandit.select()
                new_src, _hint = apply_operator(op, parent_src, rng)
                candidates.append((new_src, op))

            # ----- value-model prefilter: bench only the top-k -----
            srcs = [c[0] for c in candidates]
            order = _value_rank(srcs, task=task, model=cfg.model)
            keep = order[: max(1, cfg.prefilter_k)]

            results: list[dict] = []
            gen_improved = False
            for idx in keep:
                src, prov = candidates[idx]
                obs, rr = _bench(env, task, src, reward_cfg)
                n_benched += 1
                n_correct += 1 if rr.correct else 0
                results.append(_result_dict(src, obs, rr))
                isl.insert(src, rr.correct, rr.speedup, obs.snr_db, family)

                cand_wall = _wall_us(obs)
                # reward the operator by the VERIFIED improvement it produced
                improvement = 0.0
                if rr.correct and cand_wall is not None and parent_wall:
                    improvement = max(0.0, min(1.0, (parent_wall - cand_wall) / parent_wall))
                if prov in bandit.operators:
                    bandit.update(prov, improvement)

                # track global best (correct + faster)
                if rr.correct and (
                    not best_correct
                    or best_wall is None
                    or (cand_wall is not None and cand_wall < best_wall * cfg.improve_factor)
                ):
                    best_src, best_wall, best_snr, best_correct = src, cand_wall, obs.snr_db, True
                    gen_improved = True
                feedback = _feedback(obs, rr)

            mode = "exploit" if gen_improved else "explore"

            # ----- ranked group from this generation's benched candidates -----
            if len(results) >= 2:
                rorder = _rank_results(results)
                rank_of = {idx: pos for pos, idx in enumerate(rorder)}
                cand_recs = [
                    {"source": r["source"], "wall_us": r["wall_us"],
                     "snr_db": r["snr_db"], "rank": rank_of[i]}
                    for i, r in enumerate(results)
                ]
                prefs = build_preferences(results)
                if prefs:
                    groups.append(RankedGroupRecord(
                        task_id=task.task_id,
                        parent_id=kernel_hash(parent_src),
                        candidates=cand_recs,
                        preferences=prefs,
                        gpu=getattr(task, "gpu_target", "gfx950"),
                        operation=getattr(task, "operation", None),
                        arch=getattr(task, "gpu_target", None),
                    ))

            # ----- periodic ring migration between islands -----
            if len(islands) > 1 and (gen + 1) % cfg.migration_interval == 0:
                for i in range(len(islands)):
                    migrate(islands[i], islands[(i + 1) % len(islands)], cfg.migrants)

            log.progress(gen + 1, generations, "evolve",
                         best_correct=best_correct, best_wall_us=best_wall,
                         benched=n_benched, elites=sum(len(x) for x in islands))

        # merge islands into one archive view (best per cell across islands)
        merged = MapElitesArchive(cfg.speedup_bins, seed=cfg.seed)
        for isl in islands:
            for rec in isl.elites():
                merged.add(rec)

        # ----- emit a WinRecord if we netted a speedup over the seed -----
        wins: list[WinRecord] = []
        speedup = None
        if seed_wall and best_wall and best_wall > 0:
            speedup = seed_wall / best_wall
        is_win = best_correct and best_src != seed_src and speedup is not None and speedup > 1.0
        if is_win:
            # CONVERGENT transcript (audit R2 datagen C2): a WinRecord must teach the
            # clean seed -> VERIFIED-winning-kernel demo in the canonical contract, NOT
            # the raw multi-generation exploration ``trajectory`` (which interleaves
            # dead-ends/failed candidates and may not even END on best_src when the win
            # came from an OPERATOR mutation rather than the generator's text). Mirror
            # gold_wins.mint_gold_win so evolve wins are SFT-shaped identically to gold.
            from kore.data.gold_wins import _analysis as _win_analysis
            _op = str(getattr(task, "operation", None) or task.task_id or "kernel")
            _dt = task.dtype if getattr(task, "dtype", "") in (
                "bf16", "fp16", "fp32", "fp8", "int8", "int4") else None
            _win_asst = format_assistant_turn(
                _win_analysis(_op, float(best_wall), float(best_snr or 0.0),
                              float(speedup), dtype=_dt),
                f"Adopt the fastest verified implementation for `{_op}` "
                f"({speedup:.2f}x over the seed).",
                best_src)
            win_trajectory = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_turn_prompt(parent_source=seed_src,
                                                              mode="exploit")},
                {"role": "assistant", "content": _win_asst},
            ]
            wins.append(WinRecord(
                task_id=task.task_id,
                trajectory=win_trajectory,
                initial_wall_us=seed_wall,
                final_wall_us=best_wall,
                speedup=speedup,
                final_source=best_src,
                snr_db=best_snr,
                gpu=getattr(task, "gpu_target", "gfx950"),
                operation=getattr(task, "operation", None),
                arch=getattr(task, "gpu_target", None),
            ))

        stats = {
            "generations": generations,
            "n_benched": n_benched,
            "n_correct": n_correct,
            "best_speedup": speedup,
            "best_correct": best_correct,
            "n_elites": len(merged),
            "bandit_resets": bandit.n_resets,
            "operator_pulls": dict(bandit.pulls),
            "n_wins": len(wins),
            "n_groups": len(groups),
        }
        log.metric("evolve_summary", task=getattr(task, "task_id", "?"), **{
            k: v for k, v in stats.items() if k != "operator_pulls"
        })
        return EvolveResult(wins=wins, groups=groups, archive=merged,
                            islands=islands, bandit=bandit, stats=stats)
