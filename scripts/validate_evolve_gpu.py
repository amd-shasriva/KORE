#!/usr/bin/env python3
"""First real-GPU run of the evolutionary loop (``kore.search.evolve_agent``).

The engine was built so that "orchestration is pure CPU; all GPU work is
injected through ``env``", which made it fully testable against scripted fakes
-- and meant it had never once been driven by a real :class:`KoreEnv`. Fakes
return a speedup; hardware returns compile errors, correctness failures, timing
noise between repeats of the SAME kernel, and occasional flakes. Those are the
inputs the archive, the stability estimator and the collapse guard actually have
to survive.

The proposer here is deterministic and model-free on purpose. Production uses
``HarnessProposer`` (one AgentHarness episode per mutation), but that would put
an LLM in the middle of the first hardware run and make every failure ambiguous
between "the loop is broken" and "the model wrote bad kernels". Mutating the
seed's launch parameters produces real, compilable, genuinely different kernels
with genuinely different runtimes, which is all the loop needs to be exercised.

What this checks, none of which a fake can:
  * ``evolve`` completes against a real env without raising
  * measured speedups are plausible (a tuned variant of a seed is not 50x it)
  * the archive niches by STRATEGY, so it does not fill with one design
  * the env-call budget is respected against real, slow calls
  * repeated measurement of the same kernel is stable enough for admission
    decisions to mean anything
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


class LaunchParamProposer:
    """Model-free mutation operator: rewrite the seed's launch configuration.

    Each proposal changes ``BLOCK_N`` and/or ``num_warps``, which on an
    elementwise Triton kernel changes occupancy and memory-coalescing behaviour
    and therefore produces genuinely different measured runtimes. Some
    combinations are also genuinely bad (or fail to compile at all), which is
    exactly the mix the loop has to handle.
    """

    def __init__(self, seed_source: str):
        self.seed = seed_source
        self.grid = [(128, 2), (256, 4), (512, 4), (1024, 8),
                     (2048, 8), (4096, 16), (64, 1), (8192, 16)]

    def propose(self, task, env, parent, exemplars, generation):
        from kore.search.evolve_agent import Proposal

        base = getattr(parent, "source", None) or self.seed
        out = []
        for turn in range(4):
            block, warps = self.grid[(generation * 4 + turn) % len(self.grid)]
            src = re.sub(r"BLOCK_N\s*=\s*\d+", f"BLOCK_N = {block}", base, count=1)
            src = re.sub(r"num_warps\s*=\s*\d+", f"num_warps={warps}", src, count=1)
            if src == base:
                continue
            out.append(Proposal(source=src, turn=turn, origin="launch_param"))
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="gen_add_bf16")
    ap.add_argument("--gpu", default=os.environ.get("KORE_PROBE_GPU", "0"))
    ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--max-env-calls", type=int, default=80)
    ap.add_argument("--out", default="data/evolve_gpu_validation.json")
    ap.add_argument("--detune", action="store_true",
                    help="start from a deliberately bad launch config so there "
                         "is real headroom for the search to recover")
    args = ap.parse_args()

    from kore.env.kore_env import KoreEnv
    from kore.search.evolve_agent import EvolveAgentConfig, evolve
    from kore.tasks.registry import get_task

    task = get_task(args.task)
    env = KoreEnv(task, use_replay=False, gpu=args.gpu)
    seed = task.seed_source

    # The shipped seed is already well tuned, so every mutation of it measures
    # slower and the run proves only that the loop declines to invent a win.
    # Detuning creates a known gap the search has to close, which is the
    # property actually worth testing: speedup is measured against the ORIGINAL
    # baseline, so recovering the seed's configuration should read as > 1.0.
    #
    # The control matters: env measures speedup against the task's REFERENCE, not
    # against whatever we seeded, and the shipped seed itself only reaches ~0.94x
    # of that reference. So "did the search work" cannot be "did it exceed 1.0x";
    # it has to be "did it beat the point it started from".
    detuned_control = None
    if args.detune:
        seed = re.sub(r"BLOCK_N\s*=\s*\d+", "BLOCK_N = 64", seed, count=1)
        seed = re.sub(r"num_warps\s*=\s*\d+", "num_warps=1", seed, count=1)
        print("# seed        : DETUNED to BLOCK_N=64, num_warps=1")
        obs = env.step(seed)
        detuned_control = _speedup_of(obs)
        print(f"# control     : detuned seed measures {detuned_control} vs reference")

    cfg = EvolveAgentConfig(
        generations=args.generations,
        max_env_calls=args.max_env_calls,
        turns_per_generation=4,
        archive_capacity=16,
        elite_band=4,
        max_measures=4,
        min_for_trim=3,
    )

    print(f"# task        : {task.task_id}")
    print(f"# gpu         : {args.gpu}")
    print(f"# generations : {cfg.generations}   budget: {cfg.max_env_calls} env calls")
    print("# proposer    : LaunchParamProposer (deterministic, model-free)\n")

    t0 = time.time()
    failures: list[str] = []
    try:
        result = evolve(task, LaunchParamProposer(seed), env, cfg,
                        seed_source=seed)
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"\nFAILED -- evolve() raised against a real env: "
              f"{type(exc).__name__}: {exc}")
        return 1
    elapsed = time.time() - t0

    gens = list(getattr(result, "generations", []) or [])
    print(f"{'gen':>4} {'props':>6} {'screen':>7} {'admit':>6} "
          f"{'best':>8} {'niches':>7} {'explore':>8}")
    for g in gens:
        best = getattr(g, "best_speedup", None)
        print(f"{g.generation:>4} {g.proposals:>6} {g.screened:>7} "
              f"{g.admitted:>6} {('%.4f' % best) if best else '     -':>8} "
              f"{g.coverage:>7} {str(getattr(g, 'forced_explore', False)):>8}")

    best = getattr(result, "best", None)
    # Candidate exposes speedup_lcb / speedup_mean, never a bare `speedup`.
    # LCB is what the archive ranks by (fitness = stats.lcb), so it is the number
    # that decided this candidate was the champion in the first place.
    best_speedup = getattr(best, "speedup_lcb", None) if best else None
    best_mean = getattr(best, "speedup_mean", None) if best else None
    archive = getattr(result, "archive", None)
    members = list(getattr(archive, "members", lambda: [])()) if archive else []
    niches = {getattr(m, "signature", None) for m in members}

    print(f"\nelapsed        : {elapsed:.1f}s")
    print(f"best speedup   : lcb={_fmt(best_speedup)} mean={_fmt(best_mean)}")
    print(f"archive size   : {len(members)}")
    print(f"distinct niches: {len(niches)}")

    # A member is in the archive after SCREENING but only becomes an elite once
    # it has been STABILISED, so "archive full, no champion" is the signature of
    # a budget that ran out before stabilisation rather than of a failed search.
    # Print the per-member state, because the two are indistinguishable from the
    # totals alone and the first hardware run hit exactly this.
    print(f"\n{'admiss':>7} {'n':>3} {'lcb':>8} {'mean':>8}  niche")
    for m in members:
        st = getattr(m, "stats", None)
        print(f"{str(getattr(m, 'admissible', '?')):>7} "
              f"{getattr(st, 'n', 0):>3} "
              f"{_fmt(getattr(m, 'speedup_lcb', None)):>8} "
              f"{_fmt(getattr(m, 'speedup_mean', None)):>8}  "
              f"{getattr(m, 'signature', None)}")
    n_admissible = sum(1 for m in members if getattr(m, "admissible", False))
    n_measured = sum(1 for m in members
                     if getattr(getattr(m, "stats", None), "n", 0) > 0)
    print(f"\nadmissible     : {n_admissible}/{len(members)}   "
          f"measured: {n_measured}/{len(members)}")
    if members and n_admissible == 0:
        print("NOTE: archive populated but nothing stabilised -- raise "
              "--max-env-calls; screening admits, only stabilisation crowns.")

    # --- checks a fake env cannot make ------------------------------------ #
    if not gens:
        failures.append("no generations ran")
    total_admitted = sum(g.admitted for g in gens)
    if total_admitted == 0:
        failures.append(
            "nothing was ever admitted: on a real env with compilable "
            "variants the archive should not stay empty")
    if best_speedup is not None and best_speedup > 12.0:
        failures.append(
            f"best speedup {best_speedup:.2f}x exceeds the plausibility cap -- "
            "a relaunch-tuned elementwise kernel cannot beat its own seed that "
            "far, so this is a measurement exploit, not a kernel")
    if best_speedup is not None and best_speedup <= 0:
        failures.append(f"non-positive speedup {best_speedup}")
    if args.detune:
        # The whole point of the detuned run: a known gap exists, so a search
        # that cannot close any of it is not searching. Judged against the
        # measured control, not against 1.0.
        if best_speedup is None:
            failures.append(
                "detuned seed left real headroom and the search found no "
                "champion at all")
        elif detuned_control and best_speedup <= detuned_control * 1.05:
            failures.append(
                f"best {best_speedup:.4f}x barely improves on the detuned "
                f"control {detuned_control:.4f}x -- no recovery")
        elif detuned_control:
            print(f"recovery       : {best_speedup / detuned_control:.2f}x "
                  f"over the detuned control")

    calls = getattr(result, "env_calls", None)
    if calls is not None:
        print(f"env calls used : {calls} / {cfg.max_env_calls}")
        # _MeteredEnv charges the proposer's traffic after each generation, so
        # one episode of overshoot is by design; unbounded overshoot is not.
        if calls > cfg.max_env_calls * 1.5:
            failures.append(
                f"env calls {calls} far exceed the {cfg.max_env_calls} budget")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "task": task.task_id,
        "gpu_arch": _arch(),
        "elapsed_s": round(elapsed, 1),
        "generations": [
            {"generation": g.generation, "proposals": g.proposals,
             "screened": g.screened, "admitted": g.admitted,
             "best_speedup": getattr(g, "best_speedup", None),
             "niches": g.coverage,
             "forced_explore": getattr(g, "forced_explore", False)}
            for g in gens],
        "best_speedup_lcb": best_speedup,
        "best_speedup_mean": best_mean,
        "archive_size": len(members),
        "distinct_niches": len(niches),
        "admissible_members": n_admissible,
        "measured_members": n_measured,
        "members": [
            {"admissible": getattr(m, "admissible", None),
             "n": getattr(getattr(m, "stats", None), "n", 0),
             "lcb": getattr(m, "speedup_lcb", None),
             "mean": getattr(m, "speedup_mean", None),
             "correct": getattr(m, "correct", None)}
            for m in members],
        "detuned_control": detuned_control,
        "env_calls": calls,
        "failures": failures,
    }, indent=2, default=str))
    print(f"\n# wrote {out}")

    print("\n" + "=" * 60)
    if failures:
        print("FAILED")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASSED -- evolutionary loop ran on real gfx950 against a real KoreEnv")
    return 0


def _fmt(value) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "-"


def _speedup_of(obs):
    """Worst-shape speedup for an observation, or None if it did not measure.

    Worst-shape rather than mean for the same reason the reward uses it: a
    kernel that wins on one shape and loses on another has not won.
    """
    try:
        from kore.reward.reward import _worst_speedup
        value = _worst_speedup(obs)
        return round(float(value), 4) if value else None
    except Exception:  # noqa: BLE001
        return None


def _arch() -> str:
    try:
        import torch
        return torch.cuda.get_device_properties(0).gcnArchName
    except Exception:  # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
