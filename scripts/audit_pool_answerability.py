#!/usr/bin/env python
"""Is a pool task's oracle a function of the tensors a kernel is handed?

The driver passes a candidate exactly what ``get_inputs()`` returns. A mined
``nn.Module`` that builds its own ``nn.Conv2d`` draws that weight from the torch
RNG, and the weight is not among those tensors -- so the oracle's answer depends
on a variable the kernel cannot read. This script measures that directly rather
than arguing it: rebuild each reference under two different weight seeds, feed
both the SAME inputs, and see whether the answer moves.

An output that moves is decisive. The only channel from a kernel to those weights
is re-executing the torch module, and
:func:`kore.reward.reward.scan_for_hacks` rejects that as delegation, so such a
task cannot be solved by any admissible kernel at any tolerance. It is
unanswerable, not hard.

Prints a per-family breakdown and exits non-zero only if it could not measure
anything, so it can gate a launch.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import sys


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pool-root", default="")
    p.add_argument("--sample", type=int, default=120,
                   help="specs to probe (0 = all; the full pool costs ~4 forwards each)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json-out", default="")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

    import torch  # noqa: F401 - imported for its side effect on thread limits

    from kore.data.task_mining import hidden_state_reason
    from kore.reward.reward import scan_for_hacks
    from kore.tasks import external

    root = external.pool_root(args.pool_root or None)
    index = root / external.POOL_INDEX_NAME
    prefilter = index.with_suffix(".jsonl.prefilter")
    # Measure the population the campaign actually drew from: the pre-gate index
    # when it exists, because that is the corpus whose failure rate is in question.
    source = prefilter if prefilter.is_file() else index
    if not source.is_file():
        print(f"no pool index at {source}", file=sys.stderr)
        return 2

    raws = [json.loads(line) for line in source.open(encoding="utf-8") if line.strip()]
    print(f"pool index      : {source.name} ({len(raws)} specs)")

    rng = random.Random(args.seed)
    chosen = list(raws)
    if args.sample and args.sample < len(chosen):
        chosen = rng.sample(chosen, args.sample)

    by_family = collections.defaultdict(lambda: collections.Counter())
    totals = collections.Counter()
    examples = []
    for raw in chosen:
        spec = external.ExternalTaskSpec.from_dict(raw)
        try:
            namespace = external.exec_module_source(spec.module_source)
            reason = hidden_state_reason(
                namespace, spec.entry_class, spec.init_args, spec.init_kwargs,
                spec.input_specs, spec.dtype,
            )
        except BaseException as exc:  # noqa: BLE001 - a mined module may do anything
            totals["unprobeable"] += 1
            by_family[spec.family]["unprobeable"] += 1
            if len(examples) < 5:
                examples.append({"task_id": spec.task_id,
                                 "verdict": "unprobeable",
                                 "detail": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        verdict = "hidden_state" if reason else "answerable"
        totals[verdict] += 1
        by_family[spec.family][verdict] += 1
        if reason and len(examples) < 5:
            examples.append({"task_id": spec.task_id, "verdict": verdict,
                             "detail": reason[:200]})

    probed = totals["hidden_state"] + totals["answerable"]
    if not probed:
        print("could not probe a single spec", file=sys.stderr)
        return 3

    print(f"probed          : {probed} (unprobeable {totals['unprobeable']})")
    print(f"oracle depends on hidden module state: {totals['hidden_state']}/{probed} "
          f"({100.0 * totals['hidden_state'] / probed:.1f}%)")
    print()
    print(f"{'family':16s} {'probed':>7s} {'hidden':>7s} {'hidden%':>8s}")
    for family, counts in sorted(by_family.items(), key=lambda kv: -sum(kv[1].values())):
        n = counts["hidden_state"] + counts["answerable"]
        if not n:
            continue
        print(f"{family:16s} {n:>7d} {counts['hidden_state']:>7d} "
              f"{100.0 * counts['hidden_state'] / n:>7.1f}%")

    # The second half of the argument: the seed the harness puts in the prompt is
    # itself refused by the integrity gate, so a pool episode starts from source
    # the oracle will never accept.
    seeds_scanned = 0
    seeds_refused = collections.Counter()
    tasks_dir = external.pool_tasks_dir(root)
    for raw in chosen:
        seed_path = tasks_dir / raw["task_id"] / external.POOL_SEED_KERNEL
        if not seed_path.is_file():
            continue
        seeds_scanned += 1
        verdict = scan_for_hacks(seed_path.read_text(encoding="utf-8"))
        if verdict:
            seeds_refused[verdict] += 1
    if seeds_scanned:
        refused = sum(seeds_refused.values())
        print()
        print(f"pool seeds refused by scan_for_hacks: {refused}/{seeds_scanned} "
              f"({100.0 * refused / seeds_scanned:.1f}%)")
        for reason, count in seeds_refused.most_common(5):
            print(f"    {count:>6d}  {reason}")

    result = {
        "index": str(source),
        "indexed": len(raws),
        "probed": probed,
        "unprobeable": totals["unprobeable"],
        "hidden_state": totals["hidden_state"],
        "answerable": totals["answerable"],
        "hidden_state_fraction": round(totals["hidden_state"] / probed, 4),
        "by_family": {k: dict(v) for k, v in sorted(by_family.items())},
        "seeds_scanned": seeds_scanned,
        "seeds_refused": sum(seeds_refused.values()),
        "seed_refusal_reasons": dict(seeds_refused),
        "examples": examples,
    }
    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
