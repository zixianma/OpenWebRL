#!/usr/bin/env python3
"""Reuse verified frozen-reference scores for the matched action-DPO run."""
import argparse
import json
from pathlib import Path

from train_arm_joint_sft import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-config', type=Path, required=True)
    parser.add_argument('--target-config', type=Path, required=True)
    args = parser.parse_args()
    source_config = json.loads(args.source_config.read_text())
    target_config = json.loads(args.target_config.read_text())
    if target_config['objective'] != 'dpo_action':
        raise ValueError('Target must be action-masked DPO')
    for field in ['actor', 'model_metadata_hashes', 'data', 'data_hashes', 'order_sha256', 'world_size']:
        if source_config[field] != target_config[field]:
            raise ValueError(f'Frozen-reference lineage changed: {field}')
    source_hash = digest(args.source_config)
    target_hash = digest(args.target_config)
    source_root = Path(source_config['output'])
    target_root = Path(target_config['output'])
    target_root.mkdir(parents=True, exist_ok=True)
    provenance = []
    for rank in range(target_config['world_size']):
        source = source_root / f'reference-rank-{rank}.jsonl'
        destination = target_root / source.name
        if destination.exists():
            raise ValueError(f'Refuse to overwrite {destination}')
        count = 0
        with source.open() as incoming, destination.open('x') as outgoing:
            for line in incoming:
                item = json.loads(line)
                if item['config_sha256'] != source_hash:
                    raise ValueError('Source reference-cache provenance mismatch')
                for branch in ['chosen', 'rejected']:
                    if not isinstance(item['scores'][branch].get('action'), (float, int)):
                        raise ValueError('Source cache lacks action scores')
                item['config_sha256'] = target_hash
                outgoing.write(json.dumps(item, separators=(',', ':')) + '\n')
                count += 1
        provenance.append({'rank': rank, 'source': str(source), 'source_sha256': digest(source),
                           'destination': str(destination), 'destination_sha256': digest(destination),
                           'cached_examples': count})
    write_json(target_root / 'cache-migration.json', {
        'reason': 'The base actor, data, order, and token masks are identical; reuse precomputed full/action reference scores.',
        'source_config': str(args.source_config), 'source_config_sha256': source_hash,
        'target_config': str(args.target_config), 'target_config_sha256': target_hash,
        'caches': provenance,
    })
    print(json.dumps({'target': str(target_root), 'caches': provenance}, indent=2))


if __name__ == '__main__':
    main()
