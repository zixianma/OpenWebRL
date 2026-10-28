#!/usr/bin/env python3
"""Per-cell paid-API cost estimate for the OM2W Table 5-style stages from saved ledgers (read-only).

Per-turn actor/selector cost per (actor, selector) is pooled from the Reddit Table 5
campaign's settled budget events (same viewer prompts, routes and prices); one turn is
one actor `terminal` request. The o4-mini terminal-judge cost per episode comes from
the OM2W WebPRM pilot journals. Live OM2W prompts were smaller per actor call than
Reddit's in those ledgers, so per-turn Reddit costs are used unscaled (conservative).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json
from pathlib import Path

N = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation')
REDDIT = N/'reddit-table5-190-v3/campaign'
PILOT = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009')
ACTORS = ('gpt-4o-mini-2024-07-18', 'gpt-4o-2024-05-13')
SELECTORS = ('no_search', 'gpt4omini', 'shepherd8b', 'arbiter7b', 'piotr_selectionarm')
JUDGE, JUDGE_PRICES = 'o4-mini-2025-04-16', (1.10e-6, 4.40e-6)
STAGE_TASKS = {'pilot10': 10, 'next90': 90, 'rest200': 200}


def reddit_rates(root=REDDIT):
    events = [json.loads(line) for line in (root/'shared-budget.json.events.jsonl').read_text().splitlines()]
    reserves = {e['id']: e for e in events if e['event'] == 'reserve'}
    purpose = {}
    for path in root.glob('attempt-*/broker.jsonl'):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if 'request_id' in row and 'purpose' in row:
                purpose[row['request_id']] = row['purpose']
    cells = defaultdict(lambda: dict(usd=0., turns=0, calls=0, prompt=0, completion=0))
    for event in events:
        if event['event'] != 'settle':
            continue
        reserve = reserves[event['id']]; cell = reserve.get('cell_id', '')
        if '/' not in cell or purpose.get(reserve['request_id']) == 'native_fuzzy':
            continue  # preflight controls and the WebArena fuzzy judge (replaced by the OM2W judge) excluded
        row = cells[cell]
        row['usd'] += float(event['actual']); row['calls'] += 1
        row['prompt'] += event['usage']['prompt_tokens']; row['completion'] += event['usage']['completion_tokens']
        row['turns'] += purpose.get(reserve['request_id']) == 'terminal'
    pooled = defaultdict(lambda: dict(usd=0., turns=0, cells=0, max_cell_usd=0.))
    for cell, row in cells.items():
        actor, selector = cell.split('/', 1)[1].split('__')
        target = pooled[(actor, selector)]
        target['usd'] += row['usd']; target['turns'] += row['turns']; target['cells'] += 1
        target['max_cell_usd'] = max(target['max_cell_usd'], row['usd'])
    return {key: dict(value, usd_per_turn=value['usd']/value['turns'] if value['turns'] else None)
            for key, value in pooled.items()}


def judge_episode_usd(root=PILOT):
    prompt = completion = calls = 0
    for path in glob.glob(str(root/'**/requests.jsonl'), recursive=True):
        pending = {}
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            if row.get('event') == 'request' and not row.get('local'):
                pending[json.dumps(row.get('reservation'))] = row['payload']['model']
            elif row.get('event') == 'response' and pending.get(json.dumps(row.get('reservation'))) == JUDGE:
                usage = row['response']['usage']; calls += 1
                prompt += usage['prompt_tokens']; completion += usage['completion_tokens']
    if not calls:
        raise ValueError('No saved OM2W judge usage')
    return dict(calls=calls, mean_prompt_tokens=prompt/calls, mean_completion_tokens=completion/calls,
                usd_per_episode=(prompt*JUDGE_PRICES[0]+completion*JUDGE_PRICES[1])/calls)


def estimate(turns=(15, 30)):
    rates, judge = reddit_rates(), judge_episode_usd()
    per_turn = {}
    for actor in ACTORS:
        for selector in SELECTORS:
            value = rates.get((actor, selector))
            if value is None or value['usd_per_turn'] is None:
                raise ValueError(f'No ledger rate for {actor}/{selector}')
            per_turn[(actor, selector)] = value['usd_per_turn']
    table = {}
    for t in turns:
        by_actor = {}
        for actor in ACTORS:
            cell = {s: per_turn[(actor, s)]*t+judge['usd_per_episode'] for s in SELECTORS}
            by_actor[actor] = dict(per_cell_usd=cell, per_task_five_selectors_usd=sum(cell.values()))
        table[f'turns_{t}'] = dict(by_actor=by_actor, stages={stage: dict(
            gpt4omini_actor_usd=n*by_actor[ACTORS[0]]['per_task_five_selectors_usd'],
            gpt4o_actor_usd=n*by_actor[ACTORS[1]]['per_task_five_selectors_usd'],
            total_usd=n*(by_actor[ACTORS[0]]['per_task_five_selectors_usd']+by_actor[ACTORS[1]]['per_task_five_selectors_usd']))
            for stage, n in {**STAGE_TASKS, 'subset100': 100, 'full300': 300}.items()})
    return dict(schema_version=1, method=__doc__.strip().splitlines()[0],
        reddit_ledger=str(REDDIT/'shared-budget.json.events.jsonl'), pilot_root=str(PILOT),
        ledger_rates={f'{a}__{s}': rates[(a, s)] for a, s in rates}, judge=judge, scenarios=table,
        caveat='Turn count per OM2W episode is unknown for these actors; 30 is the protocol cap (upper bound).')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    value = estimate()
    text = json.dumps(value, indent=1, sort_keys=True)
    if args.output:
        args.output.write_text(text+'\n')
    print(text)


if __name__ == '__main__':
    main()
