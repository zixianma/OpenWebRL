#!/usr/bin/env python3
"""Offline audit of saved ARM decisions; never calls models or browsers.

Task-level findings remain under the private runtime root. Only aggregate.json
is eligible for publication. Current running cohorts are explicit snapshots.
"""
import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics
import time

RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
HISTORICAL = RUNTIME/'arm-reproduction/runs/dedicated-282209-20260908T005547Z/full'
OCT4 = RUNTIME/'arm-turn-bonus-preparation/controlled-inference-20261004'
OCT6 = RUNTIME/'evaluations/selectionarm-rltasks-historical-20261006'


def read(path):
    return json.loads(Path(path).read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def candidate_seed(seed, task, turn, candidate):
    return int.from_bytes(hashlib.sha256(f'{seed}:{task}:{turn}:{candidate}'.encode()).digest()[:4], 'big') % (2**31-1)


def split(text):
    if '</think>' in text:
        thought, action = text.split('</think>', 1)
        thought = thought.rsplit('<think>', 1)[-1].strip()
    else:
        thought, action = '', text
    return {'thought': thought, 'action': action.strip()}


def canonical_action(text):
    text = text.replace('<|im_end|>', '').strip()
    if text.startswith('<think>') and '</think>' not in text:
        text = text[len('<think>'):].strip()
    calls = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', text, flags=re.S)
    if not calls:
        return text
    try:
        return json.dumps([json.loads(c) for c in calls], sort_keys=True, ensure_ascii=False)
    except json.JSONDecodeError:
        return text


def image_sha(value):
    if isinstance(value, dict):
        value = value.get('url', '')
    if not isinstance(value, str):
        return None
    if value.startswith('data:'):
        value = value.split(',', 1)[1]
    try:
        return sha(base64.b64decode(value, validate=True))
    except (ValueError, TypeError):
        return None


def check_trace(rows, artifact, seed, count):
    """Separate observed mismatches from unavailable logging evidence."""
    errors = Counter()
    messages = artifact.get('metadata', {}).get('messages', [])
    assistants = [m['content'] for m in messages if m.get('role') == 'assistant']
    comparisons = 0
    for row in rows:
        candidates = row['candidates']; turn = row['turn']; index = row['selected_index']
        if len(candidates) != count:
            errors['candidate_count'] += 1
        if row.get('candidate_seeds') != [candidate_seed(seed, row['task_id'], turn, i) for i in range(count)]:
            errors['candidate_seeds'] += 1
        if not 0 <= index < len(candidates):
            errors['selected_index_range'] += 1
            continue
        if count > 1 and not row.get('fallback'):
            parsed = re.findall(r'\{\s*"selection"\s*:\s*(\d+)\s*\}', row.get('selector_raw') or '')
            if not parsed or int(parsed[-1])-1 != index:
                errors['selector_index_mapping'] += 1
        if turn < len(assistants):
            comparisons += 1
            chosen = candidates[index]; executed = split(assistants[turn])
            if canonical_action(chosen['action']) != canonical_action(executed['action']):
                errors['chosen_action_vs_assistant_history'] += 1
            if chosen['thought'].strip() != executed['thought'].strip():
                errors['chosen_thought_vs_assistant_history'] += 1
    images = artifact.get('metadata', {}).get('full_image_list', [])
    last_turn = artifact.get('metadata', {}).get('turn_index')
    last_row = next((r for r in rows if r['turn'] == last_turn), None)
    judge_image_equals_last_pre_action = None
    if images and last_row:
        judge_image_equals_last_pre_action = image_sha(images[-1]) == last_row['screenshot_sha256']
    return dict(errors=dict(errors), executed_history_comparisons=comparisons,
                assistant_turns=len(assistants), trace_turns=len(rows),
                judge_image_equals_last_pre_action=judge_image_equals_last_pre_action)


def latest_trace_segment(rows):
    """Old append-only logs can contain an interrupted prefix before restart."""
    starts=[i for i,r in enumerate(rows) if r['turn']==0]
    return rows[starts[-1]:] if starts else rows


def cohorts():
    result = {}
    for mode in ('baseline', 'selection'):
        result['sep_'+mode] = [dict(artifact_path=str(p), seed=42,
            trace_paths=[str(HISTORICAL/mode/'selections'/(p.stem+'.jsonl'))], count=1 if mode=='baseline' else 5)
            for p in sorted((HISTORICAL/mode/'results').glob('*.json'))]
    for label, root, modes, subpath in [('oct4', OCT4, ('actor0','arm'), ''),
                                       ('oct6', OCT6, ('actor0','piotr','arm'), 'collection')]:
        for mode in modes:
            result[label+'_'+mode] = []
        for shard in (0,1):
            base = root/f'shard-{shard}'/subpath
            for p in sorted((base/'records').glob('*.json')):
                d = read(p)
                if d['mode'] not in modes:
                    continue
                result[label+'_'+d['mode']].append(dict(record=d, artifact_path=d['artifact'],
                    seed=d.get('seed'), count=1 if d['mode']=='actor0' else 5,
                    trace_paths=[x['path'] for x in d['trace_files']], artifact_sha=d['artifact_sha256']))
        if label == 'oct4':
            schedule = read(root/'schedule.json')
            seeds = {(t['task_id'], a['mode']):a['seed'] for t in schedule for a in t['attempts']}
        else:
            seeds = {(x['record']['task_id'], m):42 for m in modes for x in result[label+'_'+m]}
        for mode in modes:
            for entry in result[label+'_'+mode]:
                entry['seed'] = seeds[entry['record']['task_id'], mode]
    return result


def audit_cohort(entries):
    metrics = Counter(); statuses = Counter(); terms = Counter(); choices = Counter(); errors = Counter()
    unique_actions = Counter(); finish = Counter(); policies=Counter(); first = {}; private = []; lengths = []; steps = []
    for e in entries:
        path=Path(e['artifact_path']); data=path.read_bytes()
        if e.get('artifact_sha') and sha(data)!=e['artifact_sha']:
            raise ValueError('Committed artifact hash changed')
        artifact=json.loads(data); record=e.get('record',artifact); task=record['task_id']
        if task in first:
            raise ValueError('Duplicate task in cohort')
        rows=[json.loads(line) for p in e['trace_paths'] if Path(p).exists() for line in Path(p).read_text().splitlines() if line]
        committed_rows=latest_trace_segment(rows)
        metrics['preserved_interrupted_prefix_decisions']+=len(rows)-len(committed_rows)
        rows=committed_rows
        messages=artifact.get('metadata',{}).get('messages',[])
        if messages:
            policy=next((m['content'] for m in messages if m['role']=='system'),None)
            if isinstance(policy,str):
                policies[sha(policy.encode())]+=1
                metrics['recorded_system_policies']+=1
                metrics['blank_system_policies']+=not policy.strip()
        row0=next((r for r in rows if r['turn']==0),None)
        first[task]=row0
        proof=check_trace(rows, artifact, e['seed'], e['count']); errors.update(proof['errors'])
        for key in ['executed_history_comparisons','assistant_turns','trace_turns']:
            metrics[key]+=proof[key]
        if proof['judge_image_equals_last_pre_action'] is not None:
            metrics['judge_image_comparisons']+=1
            metrics['judge_image_equals_last_pre_action']+=proof['judge_image_equals_last_pre_action']
        valid=bool(record.get('valid')); success=valid and record.get('reward')==1
        metrics['episodes']+=1;metrics['valid']+=valid;metrics['successes']+=success
        status=artifact.get('status','missing');statuses[status]+=1
        term=artifact.get('terminate_reason',artifact.get('error_type','missing'))
        if term.startswith('generation_error:'):term='generation_error'
        terms[term]+=1
        judge=artifact.get('metadata',{}).get('reward',{});text=judge.get('judge_text','')
        verdict=bool(re.search(r'Status\s*:',text,re.I))
        metrics['judge_verdicts']+=verdict;metrics['successes_with_judge_verdict']+=bool(success and verdict)
        if verdict:
            label=re.search(r'Status\s*:\s*[\"“”\x27]?\s*(success|failure)\b',text,re.I)
            if not label or (label.group(1).lower()=='success')!=(record.get('reward')==1):errors['saved_verdict_vs_reward']+=1
        for row in rows:
            choices[str(row['selected_index'])]+=1
            actions=[canonical_action(c['action']) for c in row['candidates']]
            unique_actions[str(len(set(actions)))]+=1
            metrics['fallbacks']+=bool(row.get('fallback'))
            metrics['selected_action_equals_candidate0']+=actions[row['selected_index']]==actions[0]
            finish.update(row['finish_types'])
            metrics['selected_length_finish']+=row['finish_types'][row['selected_index']]=='length'
            for c in row['candidates']:lengths.append(len(c['thought'])+len(c['action']))
        steps.append(len(rows))
        if proof['errors'] or proof['assistant_turns'] != proof['trace_turns']:
            private.append(dict(task_id=task,**proof,status=status))
    summary=dict(metrics, status_counts=dict(statuses),termination_counts=dict(terms),selected_index_counts=dict(choices),
                 unique_action_counts=dict(unique_actions),candidate_finish_counts=dict(finish),integrity_mismatch_counts=dict(errors),system_policy_sha256_counts=dict(policies),
                 mean_proposal_characters=statistics.mean(lengths) if lengths else 0,mean_decisions=statistics.mean(steps) if steps else 0)
    return summary,first,private


def compare_first(left,right):
    c=Counter()
    for task in left.keys() & right.keys():
        a,b=left[task],right[task]
        if not a or not b:continue
        c['both_first_turn_present']+=1
        prompt=a['prompt_sha256']==b['prompt_sha256'];image=a['screenshot_sha256']==b['screenshot_sha256'];seed=a['candidate_seeds'][0]==b['candidate_seeds'][0]
        candidate=a['candidates'][0]==b['candidates'][0]
        c['same_prompt_text']+=prompt;c['same_image_bytes']+=image;c['same_candidate0_seed']+=seed;c['same_candidate0_output']+=candidate
        if prompt and image and seed:
            c['identical_first_request_inputs']+=1;c['identical_inputs_different_candidate0']+=not candidate
    return dict(c)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    start=time.time();groups=cohorts();summaries={};first={};private={}
    for name,entries in groups.items():
        summaries[name],first[name],private[name]=audit_cohort(entries)
        print(name,json.dumps({k:summaries[name][k] for k in ('episodes','successes','trace_turns','integrity_mismatch_counts')}),flush=True)
    pairs=[('sep_baseline','sep_selection'),('oct4_actor0','oct4_arm'),('oct6_actor0','oct6_piotr'),
           ('sep_baseline','oct6_actor0'),('sep_selection','oct6_piotr'),('oct6_piotr','oct6_arm')]
    result=dict(epoch=time.time(),snapshot_started=start,offline_only=True,cohorts=summaries,
                first_decision_comparisons={a+'_vs_'+b:compare_first(first[a],first[b]) for a,b in pairs},
                limitations=['October6 is an unfinished duration-selected snapshot.',
                             'Matching inputs with different outputs can reflect inference numerical/sampling scheduling; it does not isolate a bug.',
                             'Assistant-history consistency verifies the chosen response reached execution parsing, not that the website applied the action.',
                             'Historical full dependency environment was not immutably archived. September9 proxy sources are labeled separately.'])
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'aggregate.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.output/'private-discrepancies.json').write_text(json.dumps(private,indent=2)+'\n')
    print('aggregate',args.output/'aggregate.json')


if __name__=='__main__':main()
