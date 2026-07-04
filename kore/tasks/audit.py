"""Data-scale audit: quantify operator/shape/dtype coverage of the task suite.

Reviewers of a kernel-RL paper want the data scale stated precisely: how many
operators, across how many families, with how many shapes and dtypes, and how the
train/held-out split partitions them. This produces that report from the live
registry (so it can never drift from the actual tasks) and accounts for shape
augmentation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from kore.config import CONFIG
from kore.tasks.augment import augment_shapes
from kore.tasks.registry import all_tasks, is_heldout, operator_family


@dataclass
class DataScaleReport:
    n_operators: int
    n_train: int
    n_heldout: int
    families: dict[str, int]
    dtypes: dict[str, int]
    total_base_shapes: int
    total_effective_shapes: int
    shapes_per_op_min: int
    shapes_per_op_max: int
    baseline_tiers: dict[str, int] = field(default_factory=dict)
    heldout_families: list[str] = field(default_factory=list)
    vendor_baselines: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "KORE data-scale audit",
            f"  operators:        {self.n_operators} "
            f"(train {self.n_train}, held-out {self.n_heldout})",
            f"  families:         {len(self.families)}  {dict(self.families)}",
            f"  dtypes:           {dict(self.dtypes)}",
            # honest headroom breakdown: vendor (real AITER/hipBLASLt baseline) +
            # fusion (real multi-kernel headroom) are the "real speedup" ops;
            # elementwise are correctness-training (low speedup headroom).
            f"  baseline tiers:   {dict(self.baseline_tiers)}",
            # which REAL vendor/framework kernels the vendor-tier tasks are graded
            # against (the honest "beat the production library" bar).
            f"  vendor baselines: {dict(self.vendor_baselines)}",
            f"  base shapes:      {self.total_base_shapes}",
            f"  effective shapes: {self.total_effective_shapes} "
            f"(per-op {self.shapes_per_op_min}-{self.shapes_per_op_max})",
            f"  held-out family:  {self.heldout_families}",
        ]
        return "\n".join(lines)


def audit(shape_augment: bool | None = None, augment_max: int = 6) -> DataScaleReport:
    if shape_augment is None:
        shape_augment = getattr(CONFIG, "shape_augment", False)
    tasks = all_tasks()
    fams: Counter = Counter()
    dtypes: Counter = Counter()
    base_total = 0
    eff_total = 0
    per_op: list[int] = []
    heldout_fams: set[str] = set()
    tiers: Counter = Counter()
    vendor_baselines: Counter = Counter()

    for t in tasks:
        fams[operator_family(t)] += 1
        dtypes[t.dtype] += 1
        # honest headroom tier: generated tasks carry baseline_tier; hand-authored
        # tasks (with real AITER/hipBLASLt baselines) count as "vendor".
        tier = t.raw.get("baseline_tier", "vendor")
        tiers[tier] += 1
        if tier == "vendor":
            vendor_baselines[t.comparison_baseline or "unknown"] += 1
        base = t.shapes or []
        base_total += len(base)
        eff = augment_shapes(base, max_shapes=augment_max) if shape_augment else base
        n_eff = len(eff) or len(base)
        eff_total += n_eff
        per_op.append(n_eff)
        if is_heldout(t):
            heldout_fams.add(operator_family(t))

    return DataScaleReport(
        baseline_tiers=dict(tiers),
        vendor_baselines=dict(sorted(vendor_baselines.items())),
        n_operators=len(tasks),
        n_train=sum(1 for t in tasks if not is_heldout(t)),
        n_heldout=sum(1 for t in tasks if is_heldout(t)),
        families=dict(fams),
        dtypes=dict(dtypes),
        total_base_shapes=base_total,
        total_effective_shapes=eff_total,
        shapes_per_op_min=min(per_op) if per_op else 0,
        shapes_per_op_max=max(per_op) if per_op else 0,
        heldout_families=sorted(heldout_fams),
    )


if __name__ == "__main__":  # pragma: no cover
    print(audit().summary())
    print("--- with shape augmentation ---")
    print(audit(shape_augment=True).summary())
