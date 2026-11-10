#!/usr/bin/env python3
"""CPU-only, private transition indices for matched outcome-critic comparisons.

Labels are observed terminal returns, never action-correctness or counterfactual
advantages. Use causal_inputs() to resolve model inputs; never serialize entire
archive metadata into a model prompt. Output indices contain private provenance.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import time

RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
DEFAULT_SOURCE = RUNTIME / 'arm-turn-bonus-preparation/outcome-reward-20260927'
DEFAULT_OUTPUT = DEFAULT_SOURCE / 'critic-comparison-20261004'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def image_refs(turn, archive_root):
    refs = []
    for ref in (turn.get('multimodal_inputs') or {}).get('images', []):
        path = (archive_root / ref['archive_image']).resolve()
        if not path.is_relative_to(archive_root.resolve()) or not path.is_file():
            raise ValueError('Missing or out-of-root image')
        refs.append(dict(path=str(path), sha256=ref['sha256']))
    if not refs:
        raise ValueError('Missing screenshot')
    return refs


def causal_inputs(turns, index, archive_root, horizon, variant):
    """Return whitelisted inputs only, independently of trajectory outcome.

    prompt is the original pre-action actor conversation. The post variant adds
    the next causal prompt/images; it must not access the next actor response.
    Tool failures remain executed transitions. Missing execution receipts are
    excluded from the matched pair set, even if a parse-recovery turn follows.
    """
    if variant not in ('state_value', 'pre_action', 'post_action'):
        raise ValueError(variant)
    turn = turns[index]
    data = dict(goal=turn['metadata']['intent'],
                before_prompt=turn['prompt'],
                before_images=image_refs(turn, archive_root),
                elapsed_turns=index, remaining_turns=max(0, horizon-index))
    if variant != 'state_value':
        data['candidate_response'] = turn['response']
    if variant == 'post_action':
        if index + 1 >= len(turns) or not turn['metadata'].get('step_tool_responses'):
            raise ValueError('No observed executed nonterminal transition')
        nxt = turns[index+1]
        data.update(after_prompt=nxt['prompt'],
                    after_images=image_refs(nxt, archive_root),
                    after_remaining_turns=max(0, horizon-index-1),
                    tool_responses=[{k: r.get(k, '') for k in ('tool_name', 'tool_response')}
                                    for r in turn['metadata']['step_tool_responses']])
    return data


def make_indices(row, trajectory, horizon):
    turns = trajectory['turns']
    if not trajectory['valid'] or trajectory['reward'] not in (0, 1):
        raise ValueError('Invalid outcome')
    if len(turns) != row['turn_count'] or trajectory['reward'] != row['terminal_success']:
        raise ValueError('Source/manifest mismatch')
    if [t['metadata'].get('turn_index') for t in turns] != list(range(len(turns))):
        raise ValueError('Noncontiguous turns')
    if len({t['metadata']['intent'] for t in turns}) != 1 or not turns[0]['metadata']['intent']:
        raise ValueError('Inconsistent/empty goal')
    archive_root = Path(row['source_group']).parent.parent
    rows = []
    for i, turn in enumerate(turns):
        # Fail closed if the source lineage no longer has causal history.
        if i+1 < len(turns) and turn['response'].strip() not in turns[i+1]['prompt']:
            raise ValueError('Executed response absent from next causal history')
        terminal = i == len(turns)-1
        receipt = bool(turn['metadata'].get('step_tool_responses'))
        before = causal_inputs(turns, i, archive_root, horizon, 'state_value')
        pair = not terminal and receipt
        after = causal_inputs(turns, i, archive_root, horizon, 'post_action') if pair else None
        rows.append(dict(
            id=f"{row['id']}:{i}", trajectory_id=row['id'],
            task_group=row['task_group'], split=row['split'], iteration=row['iteration'],
            source_group=row['source_group'], source_sha256=row['source_sha256'],
            trajectory_offset=row['trajectory_offset'], turn_index=i, horizon=horizon,
            terminal=terminal, execution_receipt_present=receipt,
            matched_pre_post_eligible=pair,
            exclusion=None if pair else ('terminal_after_observation_unavailable' if terminal else 'missing_execution_receipt'),
            # Neither weights nor outcomes are model input fields.
            trajectory_weight=1/len(turns), terminal_return=row['terminal_success'],
            immediate_terminal_reward=row['terminal_success'] if terminal else 0,
            advantage_label=None, progress_label=None,
            before_prompt_sha256=sha(before['before_prompt'].encode()),
            before_images=before['before_images'],
            after_prompt_sha256=sha(after['after_prompt'].encode()) if after else None,
            after_images=after['after_images'] if after else [],
        ))
    matched = sum(r['matched_pre_post_eligible'] for r in rows)
    for r in rows:
        r['matched_trajectory_weight'] = 1/matched if r['matched_pre_post_eligible'] else 0
    return rows


def prepare(source, output):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source/'manifest.json').read_text())
    grouped = defaultdict(list)
    split_tasks = {}
    for split in ('train', 'dev', 'future'):
        raw = (source/f'{split}.jsonl').read_bytes()
        if sha(raw) != manifest['data_sha256'][split]:
            raise ValueError('Source manifest hash mismatch')
        rows = [json.loads(line) for line in raw.splitlines()]
        split_tasks[split] = {r['task_group'] for r in rows}
        for row in rows:
            if row['split'] != split:
                raise ValueError('Split mismatch')
            grouped[row['source_group']].append(row)
    for a, aset in split_tasks.items():
        for b, bset in split_tasks.items():
            if a != b and aset & bset:
                raise ValueError('Task leakage across splits')
    parts = defaultdict(list); unique_images = {}; launches = {}; source_count = 0
    for filename, rows in sorted(grouped.items()):
        path = Path(filename); raw = path.read_bytes()
        if any(sha(raw) != r['source_sha256'] for r in rows):
            raise ValueError('Archive hash mismatch')
        group = json.loads(gzip.decompress(raw)); source_count += 1
        folder_manifest = json.loads((path.parent/'manifest.json').read_text())
        launch_path = Path(folder_manifest['run_directory'])/'launch_manifest.json'
        if str(launch_path) not in launches:
            raw_launch = launch_path.read_bytes()
            if sha(raw_launch) != manifest['launch_provenance_sha256'][str(launch_path)]:
                raise ValueError('Launch provenance changed')
            launches[str(launch_path)] = json.loads(raw_launch)
        horizon = launches[str(launch_path)]['max_browser_steps']
        for row in rows:
            new = make_indices(row, group['trajectories'][row['trajectory_offset']], horizon)
            parts[row['split']].extend(new)
            for r in new:
                for im in r['before_images'] + r['after_images']:
                    unique_images[im['path']] = im['sha256']
    # Verify the bytes, not just existence/declarations. No tensors are loaded.
    image_bytes = 0
    for name, expected in sorted(unique_images.items()):
        raw = Path(name).read_bytes(); image_bytes += len(raw)
        if sha(raw) != expected:
            raise ValueError('Screenshot hash mismatch')
    stats = {}; hashes = {}
    for split, rows in parts.items():
        rows.sort(key=lambda r: r['id'])
        raw = ''.join(json.dumps(r, sort_keys=True)+'\n' for r in rows).encode()
        (output/f'{split}.jsonl').write_bytes(raw); hashes[split] = sha(raw)
        pairs = [r for r in rows if r['matched_pre_post_eligible']]
        stats[split] = dict(tasks=len(split_tasks[split]),
            trajectories=len({r['trajectory_id'] for r in rows}), prefixes=len(rows),
            matched_transitions=len(pairs),
            trajectories_with_matched_transitions=len({r['trajectory_id'] for r in pairs}),
            successful_return_pairs=sum(r['terminal_return'] for r in pairs),
            exclusions=dict(Counter(r['exclusion'] for r in rows if r['exclusion'])))
    report = dict(status='CPU data audit complete; neural critic fits not launched',
        source_manifest_sha256=sha((source/'manifest.json').read_bytes()),
        index_sha256=hashes, statistics=stats,
        source_archives_verified=source_count, unique_images_verified=len(unique_images),
        image_bytes_verified=image_bytes, horizon_values=sorted({v['max_browser_steps'] for v in launches.values()}),
        task_disjoint=True, benchmark_training_rows=0,
        target='Observed undiscounted terminal return under archived behavior; not action correctness',
        causal_fields=['goal', 'before_prompt', 'before_images', 'elapsed_turns', 'remaining_turns',
                       'candidate_response (Q/post only)', 'after_prompt (post only)',
                       'after_images (post only)', 'after_remaining_turns (post only)', 'tool_responses (post only)'],
        forbidden_inputs=['trajectory reward/success', 'eval', 'terminate_reason', 'total_steps',
                          'num_turns_in_trajectory', 'is_last_turn', 'next actor response', 'remaining actual trajectory length'],
        limitations=['Post-state missing for terminal turns; matched comparison excludes them.',
            'Observed behavior is a policy mixture (train iterations31-40, future81-90), not a fixed-policy Q oracle.',
            'Task and policy shift both change in future; they cannot be separated in this split.',
            'No counterfactual branch outcomes or direct progress/advantage labels.',
            'Input renderer must budget context/images identically and keep these index labels out of prompts.'],
        resources=dict(gpus=0, api_calls=0, wall_seconds=round(time.monotonic()-started, 2)))
    (output/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))
