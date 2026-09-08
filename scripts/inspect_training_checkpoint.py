"""Read-only metadata/file-extent checks for a trusted local training checkpoint.

Optionally read small tensor samples on CPU. This is not a full GPU resume.
"""
import argparse
import json
import math
from pathlib import Path

import torch
from torch.distributed.checkpoint import DefaultLoadPlanner, FileSystemReader, load


def inspect_checkpoint(root: Path, expected_updates: int | None = None, sample_payloads: bool = False) -> dict:
    iteration = int((root / 'latest_checkpointed_iteration.txt').read_text().strip())
    checkpoint = root / f'iter_{iteration:07d}'
    metadata = FileSystemReader(checkpoint).read_metadata()
    common = torch.load(checkpoint / 'common.pt', map_location='cpu', weights_only=False)
    if common['iteration'] != iteration:
        raise ValueError('Checkpoint marker and saved iteration disagree')
    batch_size = common['args'].global_batch_size
    scheduler_steps = common['opt_param_scheduler']['num_steps']
    if batch_size <= 0 or scheduler_steps % batch_size:
        raise ValueError('Scheduler counter is not a whole number of configured global batches')
    updates = scheduler_steps // batch_size
    if expected_updates is not None and updates != expected_updates:
        raise ValueError(f'Expected {expected_updates} updates, found {updates}')
    files = {}
    for entry in metadata.storage_data.values():
        path = checkpoint / entry.relative_path
        if path.resolve().parent != checkpoint.resolve():
            raise ValueError(f'Unexpected shard path: {entry.relative_path}')
        size = files.setdefault(entry.relative_path, path.stat().st_size)
        if entry.offset < 0 or entry.length <= 0 or entry.offset + entry.length > size:
            raise ValueError(f'Truncated or invalid shard extent: {entry.relative_path}')
    if not files or not metadata.state_dict_metadata:
        raise ValueError('No saved tensor/shard metadata')
    cursor = root / 'rollout' / f'global_dataset_state_dict_{iteration}.pt'
    if not cursor.is_file():
        raise ValueError('Missing matching rollout dataset cursor')
    payload_report = {'checked': False}
    if sample_payloads:
        selected, covered = {}, set()
        for index, entry in metadata.storage_data.items():
            item = metadata.state_dict_metadata[index.fqn]
            if entry.relative_path in covered or not hasattr(item, 'properties'):
                continue
            size = math.prod(item.size) * torch.empty((), dtype=item.properties.dtype).element_size()
            if size > 1024**2:
                continue
            selected[index.fqn] = torch.empty(item.size, dtype=item.properties.dtype, device='cpu')
            covered.add(entry.relative_path)
        if not selected:
            raise ValueError('No small tensor payloads available for sampling')
        load(selected, storage_reader=FileSystemReader(checkpoint),
             planner=DefaultLoadPlanner(flatten_state_dict=False), no_dist=True)
        if not all(torch.isfinite(value).all().item() for value in selected.values()):
            raise ValueError('Nonfinite sampled checkpoint tensor')
        payload_report = {'checked': True, 'finite': True, 'tensor_keys': list(selected),
                          'covered_shard_files': sorted(covered),
                          'uncovered_shard_files': sorted(set(files)-covered),
                          'tensor_bytes': sum(v.numel()*v.element_size() for v in selected.values())}
    return {
        'checkpoint': str(checkpoint.resolve()),
        'iteration': iteration,
        'completed_optimizer_updates_from_scheduler': updates,
        'scheduler_num_steps': scheduler_steps,
        'global_batch_size': batch_size,
        'state_dict_entries': len(metadata.state_dict_metadata),
        'stored_extents': len(metadata.storage_data),
        'shard_files': files,
        'shard_bytes': sum(files.values()),
        'dataset_cursor': str(cursor.resolve()),
        'validation': 'metadata, counters, cursor presence, and shard byte extents passed',
        'sampled_cpu_payloads': payload_report,
        'full_tensor_reload_verified': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--expected-updates', type=int)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--sample-payloads', action='store_true')
    args = parser.parse_args()
    report = json.dumps(inspect_checkpoint(args.run_directory, args.expected_updates, args.sample_payloads), indent=2) + '\n'
    if args.report:
        args.report.write_text(report)
    print(report, end='')


if __name__ == '__main__':
    main()
