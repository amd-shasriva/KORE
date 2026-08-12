"""Turn a target composition into concrete per-slice row counts.

v4's mixture was not chosen, it was whatever the pipeline happened to produce:
38,460 kernel rows, all of them one question shape, against a benchmark that asks
five. It scored 55.1% where its own base model scored 55.9%. So v5's composition
is stated as an explicit target and solved for, and the plan is written into the
receipt next to what was actually achieved, because the two differ whenever a
slice runs out of distinct rows and that difference is the honest part.

The benchmark's own distribution is the reference point, not the goal. Matching it
exactly would be right if every task were equally winnable, and they are not:
``torch2flydsl`` is at 0-7% for every model including Opus 5 and no mixture we can
build moves it, while ``torch2hip`` is 7.5% for us and 57.5% for Opus and is
mostly a "does it compile" problem that verified HIP directly addresses. Budget
should follow *recoverable* tasks, not raw task counts.

Two mechanisms, deliberately distinguished:

* **Subsampling** a slice that has more distinct rows than its target. Free --
  it costs coverage of a slice that had plenty.
* **Upsampling** a slice that has fewer. Never free: it repeats rows, trading
  diversity for presence, and past some multiple it teaches the specific example
  rather than the skill. Capped, and reported separately so a reader can see
  exactly how much of a slice is repetition rather than content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

#: AgentKernelArena, 413 runnable tasks on gfx950, by the question it asks.
#: This is a COVERAGE CHECK, not a target. See GENERAL_SHAPES.
BENCHMARK_SHAPES: dict[str, float] = {
    "optimize": 207 / 413,       # triton2triton 165, hip2hip 32, flydsl2flydsl 5, +5
    "torch2kernel": 97 / 413,    # torch2hip 57, torch2flydsl 40 stubs
    "port": 51 / 413,            # triton2flydsl
    "instruction": 31 / 413,     # instruction2triton
    "repo": 27 / 413,            # image_kernel 18, repository 9
}

#: What v5 actually targets: balanced coverage of the things a kernel engineer is
#: asked to do, rather than the proportions of any one benchmark.
#:
#: Matching a benchmark's distribution is how you overfit to it. The clearest
#: measurement of that failure is MultiPL-T's OCaml model, which gained 13 points
#: on the benchmark whose format it trained on and lost 7.7 points on a
#: differently-formatted benchmark testing the same language -- and format
#: specialisation is known to happen early in fine-tuning, before the model has
#: learned the content. A mixture shaped to one evaluation buys that evaluation
#: and pays for it everywhere else, which is a fair description of what v4 did.
#:
#: So the cells below are the skills, weighted for balance and floors:
#:
#:   repair       given a broken kernel and its error, fix it. Split out from
#:                optimize because diagnosing a failure is a different skill from
#:                improving something that already works, and it is the one most
#:                directly tied to correctness -- which every benchmark scores.
#:   optimize     given a working kernel, make it faster. Kept substantial rather
#:                than suppressed: an earlier draft cut this because AgentKernelArena
#:                rarely awards its speedup term, which is true of that harness and
#:                false of the skill. KernelBench-Verified measures speedup, and so
#:                does anyone actually shipping a kernel.
#:   torch2kernel read a PyTorch module, write the kernel. The single most
#:                universal task shape across every kernel benchmark that exists.
#:   port         re-express a kernel in another dialect.
#:   instruction  write a kernel from a written specification, no source given.
#:   language     the dialect itself: idioms, layouts, API surface.
GENERAL_SHAPES: dict[str, float] = {
    "optimize": 0.25,
    "torch2kernel": 0.22,
    "repair": 0.18,
    "port": 0.13,
    "instruction": 0.13,
    "language": 0.09,
}

#: By the language the model must emit.
BENCHMARK_DIALECTS: dict[str, float] = {
    "Triton": 210 / 413,
    "FlyDSL": 102 / 413,
    "HIP": 100 / 413,
}

#: Tasks recoverable if we matched Opus 5, by category. This, not the task count,
#: is where budget earns its keep.
RECOVERABLE: dict[str, int] = {
    "triton2triton/vllm": 22, "torch2hip/kernelbench": 14,
    "instruction2triton": 11, "triton2flydsl/aiter": 8,
    "triton2triton/rocmbench": 6, "torch2hip/gpumode": 6,
    "triton2flydsl/sglang": 5, "torch2flydsl": 3, "hip2hip/gpumode": 2,
}

#: Repetition beyond this stops adding coverage and starts adding memorisation.
MAX_UPSAMPLE = 4.0


@dataclass
class SliceResult:
    name: str
    available: int
    target: int
    emitted: int
    repeats: float
    note: str = ""

    @property
    def upsampled(self) -> bool:
        return self.repeats > 1.0

    @property
    def unique(self) -> int:
        return min(self.available, self.emitted)


@dataclass
class Plan:
    body: int
    shares: dict[str, float]
    results: list[SliceResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(r.emitted for r in self.results)

    @property
    def unique_total(self) -> int:
        return sum(r.unique for r in self.results)

    @property
    def repetition_rate(self) -> float:
        t = self.total
        return 0.0 if not t else 1.0 - self.unique_total / t

    def table(self) -> str:
        w = max([len(r.name) for r in self.results] + [12])
        lines = [f"{'slice':<{w}} {'avail':>9} {'target':>9} {'emitted':>9} "
                 f"{'x':>6} {'share':>7}"]
        total = self.total or 1
        for r in sorted(self.results, key=lambda r: -r.emitted):
            lines.append(f"{r.name:<{w}} {r.available:>9,} {r.target:>9,} "
                         f"{r.emitted:>9,} {r.repeats:>6.2f} "
                         f"{100 * r.emitted / total:>6.1f}%")
        lines.append(f"{'TOTAL':<{w}} {'':>9} {'':>9} {self.total:>9,} "
                     f"{'':>6} {100.0:>6.1f}%")
        return "\n".join(lines)


def solve_use_all(available: dict[str, int], shares: dict[str, float],
                  max_upsample: float = MAX_UPSAMPLE) -> Plan:
    """Keep every distinct row, and reach the target shares by upsampling only.

    Downsampling a slice to hit a ratio throws away distinct verified examples,
    and distinct examples are the thing that actually buys generalisation --
    validated-trajectory count scales log-linearly with no saturation in the range
    anyone has measured, while repeats of an existing problem fall off fast. The
    earlier draft of this build discarded roughly 20,000 distinct optimization
    rows to make "optimize" land on 25%, which is a bad trade: it paid real
    coverage for a number.

    So the largest slice is kept whole and defines the scale, and every other
    slice is repeated toward its target share, capped. Balance improves, nothing
    verified is thrown away, and where a slice cannot reach its share the plan
    says so instead of quietly shrinking the corpus to hide it.
    """
    anchor = max(
        ((n, available.get(n, 0) / max(1e-9, s))
         for n, s in shares.items() if available.get(n, 0) > 0),
        key=lambda kv: kv[1], default=(None, 0.0))
    body = int(round(anchor[1]))
    plan = Plan(body=body, shares=dict(shares))
    for name, share in shares.items():
        have = int(available.get(name, 0))
        target = int(round(body * share))
        if have <= 0:
            plan.results.append(SliceResult(name, 0, target, 0, 0.0,
                                            "no rows available"))
            continue
        emitted = min(target, int(have * max_upsample)) if have < target else have
        note = ""
        if have >= target:
            # Never drop distinct rows: emit the whole slice and let it run over.
            emitted = have
            if have > target:
                note = f"kept whole (+{have - target:,} over share)"
        elif emitted < target:
            note = f"capped at x{max_upsample:g}"
        plan.results.append(SliceResult(name, have, target, emitted,
                                        emitted / have, note))
    return plan


def solve(available: dict[str, int], shares: dict[str, float], body: int,
          max_upsample: float = MAX_UPSAMPLE,
          floors: Optional[dict[str, int]] = None) -> Plan:
    """Allocate ``body`` rows across slices to hit ``shares`` as closely as possible.

    A slice short of its target is upsampled up to ``max_upsample`` and then left
    short rather than repeated further. The shortfall is redistributed across the
    slices that still have distinct rows to give, so the body size is honoured
    without any slice being quietly inflated past the cap.
    """
    floors = floors or {}
    total_share = sum(shares.values()) or 1.0
    plan = Plan(body=body, shares=dict(shares))

    targets = {k: int(round(body * v / total_share)) for k, v in shares.items()}
    for k, floor in floors.items():
        if k in targets:
            targets[k] = max(targets[k], floor)

    shortfall = 0
    provisional: dict[str, SliceResult] = {}
    for name, target in targets.items():
        have = int(available.get(name, 0))
        if have <= 0:
            provisional[name] = SliceResult(name, 0, target, 0, 0.0, "no rows available")
            shortfall += target
            continue
        if have >= target:
            provisional[name] = SliceResult(name, have, target, target, 1.0)
            continue
        capped = min(target, int(have * max_upsample))
        note = "" if capped >= target else f"capped at x{max_upsample:g}"
        provisional[name] = SliceResult(name, have, target, capped,
                                        capped / have, note)
        shortfall += target - capped

    # Give the shortfall to slices that still hold unused distinct rows, in
    # proportion to what they have spare. Upsampling elsewhere to cover another
    # slice's gap would be repetition with no benchmark justification at all.
    spare = {n: r.available - r.emitted for n, r in provisional.items()
             if r.available > r.emitted}
    pool = sum(spare.values())
    if shortfall > 0 and pool > 0:
        for name, room in spare.items():
            add = min(room, int(round(shortfall * room / pool)))
            if add > 0:
                r = provisional[name]
                r.emitted += add
                r.repeats = r.emitted / r.available if r.available else 0.0
                r.note = (r.note + "; " if r.note else "") + f"+{add:,} backfill"

    plan.results = list(provisional.values())
    return plan


def describe_gap(achieved: dict[str, float], reference: dict[str, float]) -> str:
    """A readable comparison of what we built against what the benchmark asks."""
    keys = sorted(set(achieved) | set(reference), key=lambda k: -reference.get(k, 0))
    w = max([len(k) for k in keys] + [10])
    out = [f"{'':<{w}} {'built':>8} {'benchmark':>10} {'delta':>8}"]
    for k in keys:
        a, r = 100 * achieved.get(k, 0.0), 100 * reference.get(k, 0.0)
        out.append(f"{k:<{w}} {a:>7.1f}% {r:>9.1f}% {a - r:>+7.1f}")
    return "\n".join(out)


__all__ = ["BENCHMARK_DIALECTS", "BENCHMARK_SHAPES", "GENERAL_SHAPES",
           "MAX_UPSAMPLE", "Plan", "RECOVERABLE", "SliceResult", "describe_gap",
           "solve", "solve_use_all"]
