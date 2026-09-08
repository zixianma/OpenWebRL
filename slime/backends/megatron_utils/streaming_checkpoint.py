"""Synchronous torch_dist save with bounded GPU-to-host copy-ahead.

Megatron's usual async-capable writer stages all tensors even when executing
synchronously. This implementation retains its tensor mapping and planner but
uses PyTorch's streaming filesystem writer, avoiding a second full CPU copy of
model and optimizer state during save.
"""
from pathlib import Path

import torch.distributed.checkpoint as dcp
from megatron.core.dist_checkpointing.strategies.torch import (
    MCoreSavePlanner,
    TorchDistSaveShardedStrategy,
    _replace_state_dict_keys_with_sharded_keys,
    mcore_to_pyt_state_dict,
)


class StreamingTorchDistSaveStrategy(TorchDistSaveShardedStrategy):
    def __init__(self):
        super().__init__('torch_dist', 1, thread_count=1)

    def save(self, sharded_state_dict, checkpoint_dir: Path):
        sharded_state_dict, _, _ = _replace_state_dict_keys_with_sharded_keys(
            sharded_state_dict, self.keep_only_main_replica,
        )
        pyt_state_dict = mcore_to_pyt_state_dict(sharded_state_dict, False)
        writer = dcp.FileSystemWriter(
            checkpoint_dir, thread_count=1, per_thread_copy_ahead=16 * 1024**2,
        )
        dcp.save(
            pyt_state_dict,
            storage_writer=writer,
            planner=MCoreSavePlanner(
                dedup_replicated_tensors=not self.keep_only_main_replica,
                flatten_state_dict=False,
            ),
        )
