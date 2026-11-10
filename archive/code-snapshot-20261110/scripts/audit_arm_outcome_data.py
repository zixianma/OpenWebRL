#!/usr/bin/env python3
"""Bounded archive availability probe for an outcome-supervised reward pilot."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import resource
import time


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    started = time.monotonic()
    root = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
    manifest = root/'arm-turn-bonus-preparation/arm-refresh-20260924/baseline40/states.jsonl'
    bins = collections.defaultdict(set)
    with manifest.open() as stream:
        for line in stream:
            row = json.loads(line)
            bins[row['iteration']].add(row['source_group'])
    counts = collections.Counter()
    classifications = collections.Counter()
    rewards = collections.Counter()
    for iteration, paths in sorted(bins.items()):
        for filename in sorted(paths)[:5]:
            path = Path(filename)
            assert json.loads((path.parent/'manifest.json').read_text())['complete']
            with gzip.open(path, 'rt') as stream:
                group = json.load(stream)
            counts['groups'] += 1
            classifications[str(group.get('classification'))] += 1
            for tr in group['trajectories']:
                counts['trajectories'] += 1
                rewards[repr(tr.get('reward'))] += 1
                turns = tr['turns']
                images = [im for s in turns for im in ((s.get('multimodal_inputs') or {}).get('images') or [])]
                present = bool(images) and all(isinstance(im, dict) and 'archive_image' in im
                    and (path.parent.parent/im['archive_image']).is_file() for im in images)
                texts = [bool(s.get('prompt')) and bool(s.get('response')) for s in turns]
                masks = [bool(s.get('tokens')) and bool(s.get('loss_mask')) for s in turns]
                valid = tr.get('valid') is True and tr.get('reward') in (0, 1)
                counts['trajectories_with_existing_archived_images'] += present
                counts['valid_binary_trajectories'] += valid
                counts['valid_with_images_text_and_mask_fields'] += valid and present and all(texts) and all(masks)
                counts['turns'] += len(turns)
                counts['turns_with_prompt_response'] += sum(texts)
                counts['turns_with_token_loss_mask'] += sum(masks)
                counts['turns_with_binary_outcome'] += sum(s.get('reward') in (0, 1) for s in turns)
    result = dict(status='Bounded read-only availability probe; not a training manifest',
        selection='First five source groups per iteration in existing refresh manifest',
        source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        source_iterations=sorted(bins), counts=counts, classifications=classifications,
        trajectory_rewards=rewards,
        limitations=['Convenience sample; not a random estimate of the full corpus.',
                    'Checks file presence and fields, not every image hash, token boundary, mask value or terminal-judge provenance.',
                    'Null outcomes are excluded; never relabel unavailable attempts as failures.'],
        resources=dict(seconds=time.monotonic()-started,
            maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, gpus=0, api_calls=0))
    destination=Path(__file__).resolve().parents[1]/'openwebrl/docs/arm_results/rl_integration/outcome-reward-data-probe.json'
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
