#!/usr/bin/env python3
"""Independently check the paired replay ledger, barriers, choices and caps."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main(root, output):
    plan = read(root / 'plan.json')
    picks = {row['state_id']: row for row in read(root / 'model/frozen-picks.json')}
    with (root / 'data/panels.jsonl').open() as stream:
        panels = {row['state_id']: {'task_group': row['task_group'],
                                    'names': [a['name'] for a in row['actions']]}
                  for line in stream if line.strip() for row in [json.loads(line)]}
    assert len(panels) == len(picks) == 8 and len(plan['schedule']) == 640
    assert sha(root / 'model/value-head.npz') == read(root / 'model/head-manifest.json')['value_head_sha256']
    complete, rejected = [], []
    results = defaultdict(dict)
    with sqlite3.connect(root / 'state.sqlite3') as db:
        db.row_factory = sqlite3.Row
        for info in plan['states']:
            panel_id = info['id']
            panel_dir = root / 'states' / panel_id
            if (panel_dir / 'complete.json').exists():
                complete.append(info)
                release = read(panel_dir / 'release.json')
                assert len(release['ready_receipts']) == 40
                assert all(sha(proof['path']) == proof['sha256'] for proof in release['ready_receipts'])
                assert release['anchor_sha256'] == info['anchor_sha256']
            else:
                assert (panel_dir / 'reject.json').exists() and not (panel_dir / 'release.json').exists()
                rejected.append(info)
            cells = []
            for item in (item for item in plan['schedule'] if item['state'] == panel_id):
                attempts = [dict(row) for row in db.execute(
                    'SELECT * FROM attempts WHERE item_id=? ORDER BY started', (item['item_id'],))]
                released = [json.loads(a['result_json']) for a in attempts
                            if a['result_json'] and json.loads(a['result_json']).get('released')]
                if info in complete:
                    assert len(released) == 1 and attempts[-1]['status'] == 'finished'
                    value = released[0]
                    assert value['candidate'] == item['candidate'] and value['repeat'] == item['repeat']
                    cells.append(value)
                else:
                    assert not released
            assert len(cells) == (40 if info in complete else 0)
            results[info['source_state']][info['condition']] = cells
    assert len(complete) == 14 and len(rejected) == 2
    assert {x['source_state'] for x in rejected} == {'bd5e52daf254215acbd6885a'}
    counts = {}
    for condition in ('original', 'shared'):
        values = [value for arms in results.values() for value in arms[condition]]
        counts[condition] = {'panels': sum(info['condition'] == condition for info in complete),
                             'recorded': len(values), 'valid': sum(x['valid'] for x in values),
                             'invalid': sum(not x['valid'] for x in values),
                             'valid_success': sum(x['valid'] and x['reward'] > 0 for x in values)}
    paired_candidates = []
    for sid, arms in results.items():
        if any(len(arms.get(condition, [])) != 40 for condition in ('original', 'shared')):
            continue
        for candidate in range(5):
            original = [x for x in arms['original'] if x['candidate'] == candidate and x['valid']]
            shared = [x for x in arms['shared'] if x['candidate'] == candidate and x['valid']]
            if original and shared:
                a = sum(x['reward'] > 0 for x in original) / len(original)
                b = sum(x['reward'] > 0 for x in shared) / len(shared)
                paired_candidates.append((a, b, len(original), len(shared)))
    assert len(paired_candidates) == 15
    scoreable = []
    for sid, arms in results.items():
        names = set(panels[sid]['names'])
        if any(len(arms.get(condition, [])) != 40 for condition in ('original', 'shared')):
            continue
        rates = {}
        for condition in ('original', 'shared'):
            by_name = defaultdict(lambda: [0, 0])
            for value in arms[condition]:
                name = panels[sid]['names'][value['candidate']]
                if value['valid']:
                    by_name[name][0] += int(value['reward'] > 0)
                    by_name[name][1] += 1
            if set(by_name) != names or any(by_name[name][1] == 0 for name in names):
                break
            rates[condition] = {name: wins / valid for name, (wins, valid) in by_name.items()}
        if len(rates) == 2:
            pick = picks[sid]
            scoreable.append({'state_id': sid, 'task_group': panels[sid]['task_group'],
                              'changed': pick['actor_name'] != pick['value_name'],
                              'original_delta_pp': 100 * (rates['original'][pick['value_name']]
                                                          - rates['original'][pick['actor_name']]),
                              'shared_delta_pp': 100 * (rates['shared'][pick['value_name']]
                                                        - rates['shared'][pick['actor_name']])})
    assert len(scoreable) == 3
    published = read(output.parent / 'shared-prefix-replay-20261019.json')
    assert published['coverage']['recorded'] == 560 and published['coverage']['valid'] == 88
    assert abs(sum(x['original_delta_pp'] for x in scoreable) / 3
               - published['original']['value_minus_actor_pp']) < 1e-9
    assert abs(sum(x['shared_delta_pp'] for x in scoreable) / 3
               - published['shared']['value_minus_actor_pp']) < 1e-9
    first_release = min((root / 'states' / x['id'] / 'release.json').stat().st_mtime for x in complete)
    assert (root / 'model/frozen-picks.json').stat().st_mtime < first_release
    raw = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j',
                                   '369234,370406,370589,370595',
                                   '--format=JobIDRaw,State,ElapsedRaw'], text=True)
    scheduler = {job: {'state': status, 'seconds': int(seconds)}
                 for job, status, seconds in (line.split('|') for line in raw.splitlines())}
    assert set(scheduler) == {'369234', '370406', '370589', '370595'}
    elapsed = sum(x['seconds'] for x in scheduler.values())
    assert elapsed <= plan['resources']['total_seconds']
    with sqlite3.connect(root / 'state.sqlite3') as db:
        counters = dict(db.execute('SELECT key,value FROM counters'))
        outstanding = db.execute("SELECT COUNT(*) FROM reservations WHERE status='reserved'").fetchone()[0]
    assert outstanding == 0 and counters['active_browsers'] == 0
    assert counters['browser_sessions'] <= plan['limits']['browser_sessions']
    assert counters['local_sft_generations'] <= plan['limits']['local_sft_generations']
    assert counters['om2w_judge_http_attempts'] <= plan['limits']['canonical_judge_http_attempts']
    assert counters['canonical_judge_usd'] <= plan['limits']['canonical_judge_usd']
    audit = {'status': 'independent_ledger_audit', 'complete_panels': len(complete),
             'rejected_panels': len(rejected), 'by_condition': counts,
             'candidate_pairs_with_valid_both': len(paired_candidates),
             'candidate_pairs_possible_in_replayed_states': 35,
             'candidate_pairs_with_one_valid_draw_in_either_arm': sum(
                 min(a_valid, b_valid) == 1 for _, _, a_valid, b_valid in paired_candidates),
             'candidate_pairs_opposite_all_success_failure': sum(
                 (a == 0 and b == 1) or (a == 1 and b == 0)
                 for a, b, _, _ in paired_candidates),
             'candidate_pair_mean_shared_minus_original_pp': 100 * sum(
                 b - a for a, b, _, _ in paired_candidates) / len(paired_candidates),
             'scoreable_states': len(scoreable),
             'scoreable_task_groups': len({x['task_group'] for x in scoreable}),
             'changed_scoreable_states': sum(x['changed'] for x in scoreable),
             'original_value_minus_actor_pp': sum(x['original_delta_pp'] for x in scoreable) / 3,
             'shared_value_minus_actor_pp': sum(x['shared_delta_pp'] for x in scoreable) / 3,
             'picks_frozen_before_release': True, 'scheduler_attempts': scheduler,
             'total_h200_seconds': elapsed, 'observed_counters': counters,
             'unsettled_reservations': outstanding,
             'caveat': 'Three states in three tasks retain at least one valid draw for every tool name in both arms; no population-level inference.'}
    output.write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output)
