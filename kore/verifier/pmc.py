"""PMC counter sets for rocprofv3 hardware-counter collection.

The actual collection lives in ``kore.env.kore_env.KoreEnv._collect_profile``
(candidate-vs-reference, fail-safe, wired into the dense profiler reward). This
module just defines the named counter sets it draws from.
"""

from __future__ import annotations


# Standard counter sets for common analysis patterns
COUNTER_SETS = {
    "standard": [
        "SQ_INSTS_VALU_MFMA_BF16",
        "SQ_INSTS_VMEM",
        "SQ_WAIT_INST_LDS",
        "SQ_WAIT_INST_ANY",
    ],
    "full": [
        "SQ_INSTS_VALU",
        "SQ_INSTS_VALU_MFMA_BF16",
        "SQ_INSTS_VALU_MFMA_F16",
        "SQ_INSTS_VMEM",
        "SQ_INSTS_SALU",
        "SQ_WAIT_INST_LDS",
        "SQ_WAIT_INST_VMEM",
        "SQ_WAIT_INST_ANY",
    ],
    "memory": [
        "SQ_INSTS_VMEM",
        "SQ_WAIT_INST_VMEM",
        "TCP_TCC_READ_REQ_sum",
        "TCP_TCC_WRITE_REQ_sum",
    ],
    "compute": [
        "SQ_INSTS_VALU",
        "SQ_INSTS_VALU_MFMA_BF16",
        "SQ_INSTS_VALU_MFMA_F16",
        "SQ_INSTS_VALU_MFMA_F32",
        "SQ_INSTS_SALU",
    ],
}
