"""KORE PMC verifier: rocprofv3 counter sets + toolchain output parsers.

Defines the named rocprofv3 performance-counter sets ``KoreEnv`` collects
(``pmc.COUNTER_SETS`` / ``GROUNDING_PASSES``) plus the pure, CPU-testable helpers
that turn raw counters into the bottleneck-grounding metrics KORE reasons about
(L2 hit-rate, HBM bytes, occupancy), and the ``parsers`` that decode rocprofv3 CSV
(LONG and WIDE layouts) and hipcc/clang register output into typed objects.
Counter collection itself lives in :mod:`kore.env`; the physics interpretation
(stall / occupancy -> residual) lives in :mod:`kore.reward` and :mod:`kore.analysis`.
"""
