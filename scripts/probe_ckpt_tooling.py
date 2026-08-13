"""Probe the FSDP checkpoint APIs the SFT run depends on.

Two things are checked, both of which only bite at the very end of a multi-day
run if they are wrong:

  1. ``FullyShardedDataParallelPlugin.set_state_dict_type("FULL_STATE_DICT")``
     accepts a plain string. train_sft calls this so the FINAL save is a plain HF
     checkpoint even though periodic checkpoints are sharded.
  2. ``merge_fsdp_weights`` exists as the offline fallback.
"""

import inspect

import accelerate
import torch
import transformers

print("accelerate", accelerate.__version__)
print("torch", torch.__version__)
print("transformers", transformers.__version__)

from accelerate import FullyShardedDataParallelPlugin

sig = inspect.signature(FullyShardedDataParallelPlugin.set_state_dict_type)
print("set_state_dict_type signature:", sig)

# Does it accept the string form we pass, without a live FSDP model?
plugin = FullyShardedDataParallelPlugin(fsdp_version=1)
try:
    plugin.set_state_dict_type("FULL_STATE_DICT")
    print("set_state_dict_type('FULL_STATE_DICT'): OK ->", plugin.state_dict_type)
except Exception as exc:
    print("set_state_dict_type('FULL_STATE_DICT'): FAILED", type(exc).__name__, exc)

try:
    from accelerate.utils import merge_fsdp_weights

    print("merge_fsdp_weights:", inspect.signature(merge_fsdp_weights))
except Exception as exc:
    print("merge_fsdp_weights: MISSING", exc)
