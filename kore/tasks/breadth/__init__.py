"""Breadth task-authoring engines (op families beyond the vendor-baselined core).

Each submodule mirrors the ``kore.tasks.vendor_ops`` ABI (OPS / OP_DTYPES / SHAPES /
make_reference / seed_source) so the same generic ``_genops`` driver + generator
machinery consume it unchanged. Tasks are torch-baselined: a torch fp32 reference
oracle (correctness) plus a torch eager baseline (the perf bar a fused Triton kernel
must beat).

  * ``conv``       - vision/CNN ops: conv2d (standard/depthwise/dilated), pooling,
                     spatial resize.
  * ``sort_sparse``- the sort/select sampling tail + structured/unstructured sparse
                     GEMM (2:4, block-sparse, SpMM, SDDMM).
  * ``train_ops``  - training-critical ops: loss heads (cross-entropy family) and
                     fused optimizer steps (AdamW / Lion / Muon / grad clip).
  * ``seq``        - sequence-model + conv1d ops: cumulative/associative scans,
                     the Mamba-1 selective SSM, Mamba-2 SSD, linear attention, and
                     the depthwise causal 1D conv.

Pure/CPU-importable: torch is imported lazily inside the GPU paths so registry
discovery never needs a GPU.
"""
