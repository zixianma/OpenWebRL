#!/usr/bin/env python3
"""Render the saved Jev pilot as a standalone dashboard; makes no API calls."""
import argparse
import json
from pathlib import Path

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


def build(root):
    rows = [json.loads(line) for line in (root/'jev-pilot-10.jsonl').read_text().splitlines()]
    jev = {}
    for path in (root/'jev-quality-v1/responses').glob('*.json'):
        record = json.loads(path.read_text())
        if record.get('status') == 'ok': jev[record['task_id']] = record
    gpt = {}
    for path in (root/'gpt41-quality-pilot-v1/responses').glob('*.json'):
        record = json.loads(path.read_text())
        if record.get('valid'): gpt[(record['task_id'],record['dimension'])] = record
    manual = json.loads((root/'pilot-manual-review-before-models.json').read_text())['notes']
    tasks = []
    for row in rows:
        ident = str(row['task_id']); record = jev[ident]
        answers = []
        for dimension, (label, description) in DIMENSIONS.items():
            answer = record['response']['answers'][dimension]
            other = gpt[(ident,dimension)]
            probabilities = answer['probabilities']
            if abs(sum(probabilities.values())-1) > .001:
                raise ValueError('Invalid saved probability distribution')
            answers.append({'dimension':dimension, 'label':label, 'description':description,
                'choice':answer['choice'], 'confidence':answer['confidence'],
                'probabilities':probabilities, 'gpt_choice':other['label']['choice'],
                'gpt_explanation':other['label']['explanation'],
                'question':record['request']['questions'][dimension]})
        # Public WebGym task text and derived outputs only; no credentials or raw API envelopes.
        tasks.append({'id':ident, 'site':row['site_host'], 'instruction':row['task_name'],
                      'rubric':row['evaluator_reference'], 'manual_note':manual[ident]['note'],
                      'answers':answers})
    return {'tasks':tasks, 'dimensions':[{'key':k,'label':v[0]} for k,v in DIMENSIONS.items()],
            'model':'jev-1.13.0', 'comparator':'gpt-4.1-2025-04-14',
            'version':'webgym-quality-v1', 'date':'2026-09-22 UTC'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args = parser.parse_args()
    data = build(args.input)
    template = (REPO/'scripts/templates/jev_quality_review.html').read_text()
    embedded = json.dumps(data,ensure_ascii=False).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    rendered = template.replace('__PILOT_DATA__',embedded)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(rendered)
    print(json.dumps({'output':str(args.output),'tasks':len(data['tasks']),
                      'judgments':sum(len(r['answers']) for r in data['tasks'])}))


if __name__ == '__main__': main()
