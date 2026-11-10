#!/usr/bin/env python3
"""Render the saved Jev pilot as a standalone dashboard; makes no API calls."""
import argparse
import hashlib
import heapq
import json
from pathlib import Path
from urllib.parse import urlsplit

import audit_arm_task_quality_jev as jev_api

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-turn-bonus-preparation/task-pool-expansion-20260922')
OUTPUT = REPO/'openwebrl/docs/arm_results/rl_integration/jev-quality-review.html'
DIMENSIONS = {
    'self_contained': ('Specified target', 'Is the target specified, or does the actor have an explicit choice?'),
    'single_episode': ('One episode', 'Does completion require observing future changes over multiple days?'),
    'completion_criterion': ('Completion', 'Is there a recognizable answer or stopping condition?'),
    'contradiction': ('Compatible requirements', 'Are explicit requirements mutually incompatible?'),
    'missing_prerequisite': ('Available inputs', 'Does the task explicitly require unavailable private inputs?'),
    'rubric_alignment': ('Rubric alignment', 'Does the reference rubric add hidden demands?'),
    'redundant_rubric': ('Rubric redundancy', 'Does the rubric repeat or subsume the same requirement? Advisory only.'),
}


def build(root, version='v1', current_pool=None):
    source = 'jev-pilot-10.jsonl' if version == 'v1' else 'screening-v2/candidate-review-75.jsonl'
    rows = [json.loads(line) for line in (root/source).read_text().splitlines()]
    jev = {}
    for path in (root/f'jev-quality-{version}/responses').glob('*.json'):
        record = json.loads(path.read_text())
        if record.get('status') == 'ok': jev[record['task_id']] = record
    gpt = {}
    for path in (root/f'gpt41-quality-pilot-{version}/responses').glob('*.json'):
        record = json.loads(path.read_text())
        if record.get('valid'): gpt[(record['task_id'],record['dimension'])] = record
    manual = json.loads((root/'pilot-manual-review-before-models.json').read_text())['notes']
    post_review={}
    if version=='v2':
        path=root/'screening-v2/assistant-quality-review.json'
        if path.exists():post_review={r['task_id']:r for r in json.loads(path.read_text())['decisions']}
    retained_ids = set()
    example_heaps = {}
    pool_count = 0
    if current_pool:
        # Bounded-memory stratified examples, not a task acceptance filter.
        with current_pool.open() as stream:
            for line in stream:
                row = json.loads(line)
                ident = str(row['task_id'])
                retained_ids.add(ident)
                pool_count += 1
                if ident in jev:
                    continue
                band = 'hard' if row['difficulty'] >= 7 else 'medium' if row['difficulty'] >= 4 else 'easy'
                key = (row['benchmark_name'], band)
                rank = int(hashlib.sha256(('manual-review-20260929:' + ident).encode()).hexdigest(), 16)
                heap = example_heaps.setdefault(key, [])
                item = (-rank, ident, row)
                if len(heap) < 2:
                    heapq.heappush(heap, item)
                elif item[:2] > heap[0][:2]:
                    heapq.heapreplace(heap, item)
    tasks = []
    for row in rows:
        ident = str(row['task_id']); record = jev[ident]
        if version == 'v2' and record['request'] != jev_api.payload(row):
            raise ValueError('Cached Jev request does not match current task/criteria: ' + ident)
        answers = []
        for dimension, (label, description) in DIMENSIONS.items():
            answer = record['response']['answers'][dimension]
            other = gpt.get((ident,dimension))
            probabilities = answer['probabilities']
            if abs(sum(probabilities.values())-1) > .001:
                raise ValueError('Invalid saved probability distribution')
            answers.append({'dimension':dimension, 'label':label, 'description':description,
                'choice':answer['choice'], 'confidence':answer['confidence'],
                'probabilities':probabilities, 'gpt_choice':other['label']['choice'] if other else None,
                'gpt_explanation':other['label']['explanation'] if other else 'Not included in the ten-task comparator.',
                'gpt_quote':other['label']['evidence_quote'] if other else None,
                'gpt_quote_verified':other.get('quote_verified') if other else None,
                'question':record['request']['questions'][dimension]})
        # Public WebGym task text and derived outputs only; no credentials or raw API envelopes.
        tasks.append({'id':ident, 'site':row['site_host'], 'instruction':row['task_name'],
                      'url': row['website'], 'source': row['benchmark_name'], 'difficulty': row['difficulty'],
                      'retained_current': ident in retained_ids if current_pool else None,
                      'cohort': 'cached_paired' if any(a['gpt_choice'] is not None for a in answers) else 'cached_jev_only',
                      'rubric':row['evaluator_reference'], 'manual_note':manual.get(ident,{}).get('note','No pre-model inspection note for this task.'),
                      'review':post_review.get(ident),
                      'answers':answers})
    fresh = []
    for key, heap in sorted(example_heaps.items()):
        for _, ident, row in sorted(heap, reverse=True):
            fresh.append({'id': ident, 'site': (urlsplit(row['website']).hostname or ''),
                'instruction': row['task_name'], 'url': row['website'],
                'source': row['benchmark_name'], 'difficulty': row['difficulty'],
                'retained_current': True, 'cohort': 'new_pool_unlabeled',
                'rubric': row['evaluator_reference'], 'answers': [],
                'manual_note': None, 'review': None})
    for task in [*tasks, *fresh]:
        parsed = urlsplit(task['url'])
        task['structural_checks'] = {'nonempty_instruction': bool(task['instruction'].strip()),
            'http_url_with_host': parsed.scheme in ('http', 'https') and bool(parsed.hostname),
            'reference_present': bool(task['rubric'])}
    paired = [a for task in tasks for a in task['answers'] if a['gpt_choice'] is not None]
    return {'tasks':tasks, 'new_pool_examples': fresh,
            'dimensions':[{'key':k,'label':v[0], 'description':v[1]} for k,v in DIMENSIONS.items()],
            'model':'jev-1.13.0', 'comparator':'gpt-4.1-2025-04-14',
            'paired_summary': {'tasks': sum(t['cohort'] == 'cached_paired' for t in tasks),
                'judgments': len(paired), 'agreements': sum(a['choice'] == a['gpt_choice'] for a in paired),
                'unverified_gpt_quotes': sum(a['gpt_quote_verified'] is False for a in paired)},
            'current_pool_count': pool_count,
            'review_fingerprint': hashlib.sha256(json.dumps([tasks, fresh], sort_keys=True).encode()).hexdigest(),
            'version':'webgym-quality-'+version, 'date':'2026-09-22 UTC' if version=='v1' else '2026-09-29 UTC'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--version',choices=['v1','v2'],default='v1')
    parser.add_argument('--manual-review', action='store_true')
    parser.add_argument('--current-pool', type=Path)
    args = parser.parse_args()
    data = build(args.input,args.version,args.current_pool)
    template = (REPO/'scripts/templates'/('task_quality_manual_review.html' if args.manual_review else 'jev_quality_review.html')).read_text()
    embedded = json.dumps(data,ensure_ascii=False).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    rendered = template.replace('__PILOT_DATA__',embedded)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(rendered)
    print(json.dumps({'output':str(args.output),'tasks':len(data['tasks']),
                      'judgments':sum(len(r['answers']) for r in data['tasks']),
                      'new_pool_examples':len(data['new_pool_examples']), 'paired':data['paired_summary']}))


if __name__ == '__main__': main()
