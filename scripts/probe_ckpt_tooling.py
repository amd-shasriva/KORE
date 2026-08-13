"""Probe which FSDP checkpoint-consolidation helpers this environment actually has.

Run on the cluster before switching the SFT run to sharded checkpoints: the
mid-training format is only safe to change if we can still produce a plain HF
checkpoint at the end for the next stage to load with from_pretrained.
"""

import accelerate
import torch
import transformers

print("accelerate", accelerate.__version__)
print("torch", torch.__version__)
print("transformers", transformers.__version__)

try:
    from accelerate.utils import merge_fsdp_weights  # noqa: F401

    print("merge_fsdp_weights: AVAILABLE")
except Exception as exc:  # pragma: no cover - probe
    print("merge_fsdp_weights: MISSING", exc)

try:
    import torch.distributed.checkpoint as dcp  # noqa: F401
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save  # noqa: F401

    print("dcp + dcp_to_torch_save: AVAILABLE")
except Exception as exc:  # pragma: no cover - probe
    print("dcp: MISSING", exc)

try:
    from transformers.trainer import Trainer  # noqa: F401
    import inspect
    from transformers.training_args import TrainingArguments

    sig = inspect.signature(TrainingArguments.__init__)
    print("save_only_model supported:", "save_only_model" in sig.parameters)
except Exception as exc:  # pragma: no cover - probe
    print("trainer probe failed:", exc)
