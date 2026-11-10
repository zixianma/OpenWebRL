#!/usr/bin/env python3
"""Bounded CPU preparation of complete baseline trajectories for outcome RM.

No GPU/API calls, deserialization of tensors, or benchmark training examples.
Private manifests reference immutable archives; they do not duplicate payloads.
"""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import re
import resource
import time

RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
REPO = Path(__file__).resolve().parents[1]
OUTPUT = RUNTIME/'arm-turn-bonus-preparation/outcome-reward-20260927'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def norm(text):
    return re.sub(r'[^a-z0-9]+',' ',text.lower()).strip()


def stable(text):
    return sha(('outcome-reward-v1-seed42:'+text).encode())


def check_trajectory(trajectory, path, blocked_ids, blocked_intents):
    if trajectory.get('valid') is not True or trajectory.get('reward') not in (0,1):
        return 'invalid_or_nonbinary', None
    turns = trajectory['turns']
    if not turns or [s['metadata'].get('turn_index') for s in turns] != list(range(len(turns))):
        return 'incomplete_history', None
    if not turns[-1]['metadata'].get('is_last_turn'):
        return 'missing_terminal_marker', None
    aliases = set(); intents = set(); contexts = []; responses = []; images = set()
    for turn in turns:
        m = turn['metadata']
        aliases |= {str(m[k]) for k in ('task_id','old_task_id') if m.get(k) is not None}
        intents.add(norm(m.get('intent','')))
        n, ids, mask = turn.get('response_length'), turn.get('tokens'), turn.get('loss_mask')
        if (turn.get('remove_sample') or turn.get('status') == 'aborted' or not turn.get('prompt')
                or not turn.get('response') or not isinstance(n,int) or n <= 0
                or not isinstance(ids,list) or not n < len(ids) <= 32768
                or not isinstance(mask,list) or len(mask) != n or not any(mask)
                or any(v not in (0,1) for v in mask)):
            return 'unusable_executed_turn', None
        refs = (turn.get('multimodal_inputs') or {}).get('images') or []
        if not refs:
            return 'missing_images', None
        for im in refs:
            if not isinstance(im,dict) or 'archive_image' not in im or 'sha256' not in im:
                return 'invalid_image_reference', None
            image = (path.parent.parent/im['archive_image']).resolve()
            if not image.is_relative_to(path.parent.parent.resolve()) or not image.is_file():
                return 'missing_image_file', None
            images.add((str(image), im['sha256']))
        contexts.append(len(ids)); responses.append(sum(mask))
    if len(intents) != 1 or '' in intents or not aliases:
        return 'inconsistent_task_identity', None
    if aliases & blocked_ids or intents & blocked_intents:
        return 'benchmark_overlap', None
    return None, dict(task_aliases=sorted(aliases),normalized_intent=next(iter(intents)),
        terminal_success=int(trajectory['reward']),turn_count=len(turns),context_tokens=sum(contexts),
        maximum_context_tokens=max(contexts),supervised_response_tokens=sum(responses),
        image_count=len(images),image_references=[dict(path=p,sha256=h) for p,h in sorted(images)],
        accepted_for_rl=trajectory['accepted_for_rl'],termination=turns[-1]['metadata'].get('terminate_reason'))


def task_union(rows):
    parents = {}
    def find(k):
        parents.setdefault(k,k)
        while parents[k] != k:
            parents[k] = parents[parents[k]]; k = parents[k]
        return k
    for row in rows:
        aliases = ['intent:'+row['normalized_intent'], *['id:'+k for k in row['task_aliases']]]
        for alias in aliases[1:]:
            a,b = find(aliases[0]),find(alias)
            parents[max(a,b)] = min(a,b)
    for row in rows:
        row['task_group'] = sha(find('intent:'+row['normalized_intent']).encode())
    return rows


def prepare():
    resource.setrlimit(resource.RLIMIT_CPU,(180,185))
    resource.setrlimit(resource.RLIMIT_AS,(1536*1024**2,1536*1024**2))
    started = time.monotonic()
    if (OUTPUT/'manifest.json').exists():
        return json.loads((OUTPUT/'aggregate.json').read_text())
    benchmark = [json.loads(s) for s in (REPO/'openwebrl/data/eval/online-mind2web.jsonl').read_text().splitlines()]
    blocked_ids = {str(v) for r in benchmark for v in (r.get('task_id'),r['metadata'].get('task_id')) if v is not None}
    blocked_intents = {norm(r['metadata']['intent']) for r in benchmark}
    rows = []; exclusions = collections.Counter(); inventory = collections.Counter(); provenance = {}
    for folder in sorted((RUNTIME/'runs').glob('openwebrl-4b-reference-*/completed_rollout_archive/iteration_*')):
        iteration = int(folder.name.split('_')[1])
        if not (31 <= iteration <= 40 or 81 <= iteration <= 90):
            continue
        manifest = json.loads((folder/'manifest.json').read_text())
        if not manifest['complete']:
            raise ValueError('Archive not complete')
        launch_path = Path(manifest['run_directory'])/'launch_manifest.json'
        launch = json.loads(launch_path.read_text())
        if launch.get('judge_model') != 'gpt-4.1':
            raise ValueError('Outcome judge lineage changed')
        provenance[str(launch_path)] = sha(launch_path.read_bytes())
        for record in manifest['records']:
            path = folder/record['file']; raw = path.read_bytes()
            if len(raw) != record['bytes']:
                raise ValueError('Archive bytes changed')
            group = json.loads(gzip.decompress(raw)); inventory['groups'] += 1
            group_hash = sha(raw)
            for i,trajectory in enumerate(group['trajectories']):
                inventory['trajectories'] += 1
                reason,row = check_trajectory(trajectory,path,blocked_ids,blocked_intents)
                if reason:
                    exclusions[reason] += 1; continue
                row.update(id=stable(str(path)+':'+str(i)),source_group=str(path),source_sha256=group_hash,
                    trajectory_offset=i,iteration=iteration,source_classification=group['classification'])
                rows.append(row)
    task_union(rows)
    parts = collections.defaultdict(list); task_counts = collections.Counter()
    limits = dict(train=2000,dev=250,future=500)
    for row in sorted(rows,key=lambda x:stable(x['id'])):
        n = int(stable(row['task_group'])[:8],16)%100
        part = 'train' if n < 70 else 'dev' if n < 85 else 'future'
        if (row['iteration'] <= 40) != (part != 'future'):
            continue
        if len(parts[part]) >= limits[part] or task_counts[(part,row['task_group'])] >= 10:
            continue
        row['split'] = part; parts[part].append(row); task_counts[(part,row['task_group'])] += 1
    if any(len(parts[p]) < n for p,n in limits.items()):
        raise ValueError('Insufficient complete task-disjoint trajectories: '+str({p:len(v) for p,v in parts.items()}))
    tasks = {p:{r['task_group'] for r in rs} for p,rs in parts.items()}
    if tasks['train']&tasks['dev'] or tasks['train']&tasks['future'] or tasks['dev']&tasks['future']:
        raise ValueError('Task split leaked')
    stats = {}
    for part,rs in parts.items():
        mixed = 0
        for task in tasks[part]:
            mixed += len({r['terminal_success'] for r in rs if r['task_group']==task}) == 2
        stats[part] = dict(trajectories=len(rs),tasks=len(tasks[part]),tasks_with_both_outcomes=mixed,
            successes=sum(r['terminal_success'] for r in rs),turns=sum(r['turn_count'] for r in rs),
            supervised_response_tokens=sum(r['supervised_response_tokens'] for r in rs),
            context_tokens=sum(r['context_tokens'] for r in rs),
            maximum_context_tokens=max(r['maximum_context_tokens'] for r in rs),
            classifications=dict(collections.Counter(r['source_classification'] for r in rs)),
            terminal_reasons=dict(collections.Counter(str(r['termination']) for r in rs)))
    aggregate = dict(status='Complete-trajectory manifest prepared; model/processor parity and image hashes pending',
        policy='outcome-only baseline',training_iterations=[31,40],future_iterations=[81,90],
        judge_model='gpt-4.1',split='Task aliases and normalized intent union; hash70/15/15; cap10 trajectories/task; no label balancing',
        benchmark_training_rows=0,inventory=inventory,exclusions=exclusions,statistics=stats,
        resources=dict(seconds=time.monotonic()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,gpus=0,api_calls=0),
        limitations=['Image presence and declared hashes checked; bytes must be verified by training loader.',
                    'Saved token/mask shapes checked; processor and token boundary parity still require a model-environment test.',
                    'This is a data artifact, not a fitted reward model or a causal action-quality result.'])
    OUTPUT.mkdir(parents=True,exist_ok=True)
    hashes = {}
    for part,rs in parts.items():
        payload = ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs)
        (OUTPUT/f'{part}.jsonl').write_text(payload); hashes[part] = sha(payload.encode())
    (OUTPUT/'manifest.json').write_text(json.dumps(dict(data_sha256=hashes,launch_provenance_sha256=provenance,**aggregate),indent=2)+'\n')
    (OUTPUT/'aggregate.json').write_text(json.dumps(aggregate,indent=2)+'\n')
    (REPO/'openwebrl/docs/arm_results/rl_integration/outcome-reward-data-manifest.json').write_text(json.dumps(aggregate,indent=2)+'\n')
    return aggregate


if __name__ == '__main__':
    print(json.dumps(prepare(),indent=2))
