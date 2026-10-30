"""Resumable five-attempt actor screening; no ARM selection or policy updates."""
import asyncio
from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

from openwebrl.arm_rescue_yield import trajectory, write_json

POLICY = 'OpenWebRL-4B-SFT:iteration0'
JUDGE = 'gpt-4.1/action_history'


def key(task, attempt):
    return hashlib.sha256(f'{task}/screen/actor/{attempt}'.encode()).hexdigest()


def verify_record(record, task, attempt):
    if (record['task_id'] != task or record['attempt'] != attempt or
            record['policy_id'] != POLICY or record['judge_id'] != JUDGE or
            record['phase'] != 'screen' or record['mode'] != 'actor' or
            record['selector_calls'] != 0):
        raise ValueError('Screening identity/protocol mismatch')
    archive = Path(record['artifact'])/(hashlib.sha256(task.encode()).hexdigest()+'.pt')
    if not archive.is_file() or archive.stat().st_size == 0:
        raise ValueError('Missing durable trajectory archive')
    sidecar = json.loads(archive.with_suffix('.json').read_text())
    if (sidecar['task_id'] != task or sidecar['judge_model'] != 'gpt-4.1' or
            sidecar['judge_prompt_variant'] != 'action_history' or
            sidecar['metrics']['valid_trajectories'] != int(record['valid'])):
        raise ValueError('Saved verdict/validity mismatch')
    # Native OpenWebRL also emits -1 for actor formatting failures. Preserve
    # that raw reward; binary task success is (reward == 1), as in its evals.
    # Environment/judge invalidity remains in record['valid'], not this sign.
    if record['valid'] and record['reward'] not in (-1., 0., 1.):
        raise ValueError('Valid trajectory needs a supported native outcome')
    if record['valid'] and sidecar['reward_metadata'].get('combined') != record['reward']:
        raise ValueError('Saved terminal reward differs from screening record')


def summarize(records, ids, require_complete=False, verify_artifacts=True):
    allowed = {(task, attempt) for task in ids for attempt in range(5)}
    keyed = {}
    for r in records:
        k = (r['task_id'], r['attempt'])
        if k not in allowed or k in keyed:
            raise ValueError('Unexpected/duplicate task attempt')
        if verify_artifacts:
            verify_record(r, *k)
        keyed[k] = r
    complete = set(keyed) == allowed
    if require_complete and not complete:
        raise ValueError('Not every task has five primary attempt records')
    dispositions = {}
    for task in ids:
        group = [keyed.get((task, a)) for a in range(5)]
        if any(r is None for r in group):
            status = 'pending'
        elif not all(r['valid'] for r in group):
            status = 'unresolved_invalid'
        else:
            wins = sum(r['reward'] == 1. for r in group)
            status = 'all_failure' if wins == 0 else 'all_success' if wins == 5 else 'mixed'
        dispositions[task] = status
    valid = sum(r['valid'] for r in records)
    wins = sum(r['valid'] and r['reward'] == 1. for r in records)
    return dict(complete=complete, tasks=len(ids), primary_attempts=len(records),
        expected_attempts=len(allowed), valid_attempts=valid, successes=wins,
        invalid_attempts=len(records)-valid,
        native_format_failure_attempts=sum(r['valid'] and r['reward'] == -1. for r in records),
        overall=wins/len(records) if records else None,
        valid_only=wins/valid if valid else None,
        task_counts=dict(Counter(dispositions.values())), dispositions=dispositions,
        optimizer_updates=0, policy_id=POLICY, judge_id=JUDGE)


async def generate(args, sample, sampling_params, evaluation=False):
    if not evaluation:
        raise ValueError('This collection never trains')
    config = json.loads(Path(os.environ['OPENWEBRL_TASK_SCREEN_CONFIG']).read_text())
    task, attempt = str(sample.metadata['task_id']), sample.index % 5
    if task not in config['task_order']:
        raise ValueError('Unregistered task')
    record_path = Path(config['output'])/'records'/(key(task, attempt)+'.json')
    if record_path.exists():
        verify_record(json.loads(record_path.read_text()), task, attempt)
        return []
    bucket = config['task_order'].index(task)//400
    if (Path(config['output'])/f'judge-budget-{bucket}'/'halt.json').exists():
        raise RuntimeError('Judge budget halted; preserve partial screening')
    local = dict(config, output=config['attempt_output'],
                 selector_endpoint='http://127.0.0.1:1')
    # Baseline mode only generates one action; it never contacts this endpoint.
    await trajectory(args, sample, sampling_params, 'actor', attempt, 'screen', local)
    saved = Path(local['output'])/'screen_records'/(key(task, attempt)+'.json')
    record = json.loads(saved.read_text())
    verify_record(record, task, attempt)
    write_json(record_path, record)
    # Data is already durable; avoid accumulating 10K trajectories in Ray memory.
    return []


async def collect(args, rollout_id, config):
    import openwebrl.reward_browser as reward
    from openwebrl.arm_terminal_budget import CappedJudge
    from slime.rollout.sglang_rollout import eval_rollout_single_dataset
    from slime.rollout.base_types import RolloutFnEvalOutput

    if (args.judge_api_model != 'gpt-4.1' or args.judge_prompt_variant != 'action_history'
            or args.num_rollout != 0 or len(args.eval_datasets) != 1):
        raise ValueError('Frozen actor/native outcome protocol changed')
    dataset = args.eval_datasets[0]
    if (dataset.n_samples_per_eval_prompt, dataset.temperature, dataset.top_p,
            dataset.top_k, dataset.max_response_len) != (5, .8, 1., -1, 1024):
        raise ValueError('Screen decoding changed')
    rows = [json.loads(line) for line in Path(config['tasks']).read_text().splitlines()]
    ids = [str(r['metadata']['task_id']) for r in rows]
    if ids != config['task_order'] or len(ids) != len(set(ids)) or len(ids) != 2000:
        raise ValueError('Frozen task cohort changed')
    root = Path(config['output'])
    indexed = {(r['task_id'], r['attempt']):r for r in
        (json.loads(p.read_text()) for p in (root/'records').glob('*.json'))}
    summarize(list(indexed.values()), ids)  # Validate prior artifacts once at resume.
    old = reward._get_openai_client
    try:
        # Five disjoint 400-task blocks, each with a persistent $30 cap. The
        # unchanged judge wrapper bounds the whole screen to $150 /30K calls.
        for bucket in range(5):
            judge = CappedJudge(root/f'judge-budget-{bucket}', max_calls=6000, max_usd=30.)
            reward._get_openai_client = lambda **unused: judge
            try:
                start, end = bucket*400, (bucket+1)*400
                # The first two selected tasks are the startup check and count
                # toward the primary screen; no extra or cherry-picked tasks.
                boundaries = [0, 2, *range(22, 400, 20), 400] if bucket == 0 else list(range(0, 401, 20))
                for a, b in zip(boundaries, boundaries[1:]):
                    selected = rows[start+a:start+b]
                    pending = [r for r in selected if not all(
                        (root/'records'/(key(str(r['metadata']['task_id']), n)+'.json')).exists()
                        for n in range(5))]
                    if pending:
                        path = root/f'work-{bucket}-{a}.jsonl'
                        path.write_text(''.join(json.dumps(r)+'\n' for r in pending))
                        await eval_rollout_single_dataset(args, rollout_id,
                            replace(dataset, name=f'actor-screen-{bucket}-{a}', path=str(path)))
                    recent = []
                    for row in selected:
                        task = str(row['metadata']['task_id'])
                        for n in range(5):
                            r = json.loads((root/'records'/(key(task,n)+'.json')).read_text())
                            verify_record(r,task,n)
                            indexed[task,n] = r
                            recent.append(r)
                    records = list(indexed.values())
                    summary = summarize(records, ids, verify_artifacts=False)
                    write_json(root/'progress.json', {k:v for k,v in summary.items() if k != 'dispositions'})
                    if bucket == 0 and a == 0:
                        smoke = [r for r in records if r['task_id'] in ids[:2]]
                        if len(smoke) != 10 or sum(r['valid'] for r in smoke) < 5:
                            raise RuntimeError('Startup has fewer than five valid attempts; inspect before scaling')
                        write_json(root/'startup-check.json', dict(passed=True, primary_attempts=10))
                    if len(recent) >= 50 and sum(not r['valid'] for r in recent)/len(recent) > .5:
                        raise RuntimeError('More than 50% invalid in the completed batch; diagnose before scaling')
            finally:
                await judge.close()
        records = [json.loads(p.read_text()) for p in (root/'records').glob('*.json')]
        summary = summarize(records, ids, require_complete=True)
        write_json(root/'summary.json', summary)
        write_json(root/'all-failure-task-ids.json', [t for t,s in summary['dispositions'].items() if s == 'all_failure'])
        write_json(root/'collection-complete.json', {k:v for k,v in summary.items() if k != 'dispositions'})
        return RolloutFnEvalOutput(data={}, metrics={k:v for k,v in summary.items() if type(v) in (int,float)})
    finally:
        reward._get_openai_client = old


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    if not evaluation or args.num_rollout != 0:
        raise ValueError('Screening must execute zero optimizer updates')
    from slime.utils.async_utils import run
    config = json.loads(Path(os.environ['OPENWEBRL_TASK_SCREEN_CONFIG']).read_text())
    return run(collect(args, rollout_id, config))
