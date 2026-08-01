"""KORE PMC verifier: rocprofv3 counter sets + toolchain output parsers.

Defines the named rocprofv3 performance-counter sets ``KoreEnv`` collects
(``pmc.COUNTER_SETS`` / ``GROUNDING_PASSES``) plus the pure, CPU-testable helpers
that turn raw counters into the bottleneck-grounding metrics KORE reasons about
(L2 hit-rate, HBM bytes, occupancy), and the ``parsers`` that decode rocprofv3 CSV
(LONG and WIDE layouts) into typed objects. Counter collection itself lives in
:mod:`kore.env`; the physics interpretation (stall / occupancy -> residual) lives
in :mod:`kore.reward` and :mod:`kore.analysis`.

Register pressure and occupancy are grounded on the rocprofv3 counters
(``vgpr_count`` / ``lds_bytes`` / ``num_warps``, captured in :mod:`kore.env`) and
interpreted against the verified per-arch limits in :mod:`kore.verifier.pmc` by
:func:`kore.analysis.roofline.est_occupancy` - NOT by scraping hipcc/clang text.
"""
