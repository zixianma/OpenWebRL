"""Resumable actor+SelectionARM collection on a frozen five-failure cohort."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

from openwebrl.arm_rescue_yield import stable_seed, trajectory, write_json
from openwebrl.arm_task_screen import POLICY, verify_record as verify_screen_record

ARM_REVISION = '81b452d800d9f859687074f82680dd5257e02d89'
SEED = 20261002


def key(task, attempt):
    return hashlib.sha256(f'{task}/guided/arm/{attempt}'.encode()).hexdigest()


def eligible_ids(summary, order):
    if not summary['complete'] or set(summary['dispositions']) != set(order):
        raise ValueError('Require the complete independently verified actor screen')
    return [task for task in order if summary['dispositions'][task] == 'all_failure']


def verify_record(record, task, attempt):
    if (record['phase'] != 'guided' or record['mode'] != 'arm' or
            record['selector_revision'] != ARM_REVISION or record['candidate_count'] != 5 or
            record['candidate_representation'] != 'full' or
            record['seed'] != stable_seed(SEED, POLICY, task, 'guided', 'arm', attempt)):
        raise ValueError('Guided collection identity changed')
    # Reuse the same archive, verdict, native-reward and actor/judge checks.
    verify_screen_record(dict(record, phase='screen', mode='actor', selector_calls=0), task, attempt)
    trace = Path(record['artifact'])/'selections'/(hashlib.sha256(task.encode()).hexdigest()[:20]+'.jsonl')
    rows = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
    if record['valid'] and (not rows or len(rows) != record['selector_calls']):
        raise ValueError('Valid ARM rollout is missing selection traces')
    turns = set()
    for row in rows:
        if (row['task_id'] != task or row['mode'] != 'selection' or
                len(row['candidates']) != 5 or len(row['candidate_seeds']) != 5 or
                not 0 <= row['selected_index'] < 5 or row['turn'] in turns):
            raise ValueError('Wrong or duplicate candidate trace')
        if any(set(c) != {'thought', 'action'} for c in row['candidates']):
            raise ValueError('Missing full reasoning/action alternatives')
        turns.add(row['turn'])
    if len(rows) > record['selector_calls']:
        raise ValueError('More selection traces than calls')
    return len(rows)


def summarize(records, ids, attempts=1, require_complete=False, verify_artifacts=True):
    expected = {(task, a) for task in ids for a in range(attempts)}
    indexed = {}
    for record in records:
        identity = record['task_id'], record['attempt']
        if identity not in expected or identity in indexed:
            raise ValueError('Unexpected/duplicate guided attempt')
        if verify_artifacts:
            verify_record(record, *identity)
        indexed[identity] = record
    complete = set(indexed) == expected
    if require_complete and not complete:
        raise ValueError('Guided cohort is incomplete')
    valid = sum(r['valid'] for r in records)
    wins = sum(r['valid'] and r['reward'] == 1 for r in records)
    rescued = [task for task in ids if any(indexed.get((task, a), {}).get('valid') and
               indexed[task, a]['reward'] == 1 for a in range(attempts))]
    return dict(complete=complete, tasks=len(ids), attempts_per_task=attempts,
                primary_attempts=len(records), expected_attempts=len(expected),
                valid_attempts=valid, invalid_attempts=len(records)-valid, successes=wins,
                overall=wins/len(records) if records else None,
                valid_only=wins/valid if valid else None, rescued_tasks=rescued,
                selector_calls=sum(r['selector_calls'] for r in records),
                selector_fallback_turns=sum(r['selector_fallback_turns'] for r in records),
                actor_requests=sum(r['actor_requests'] for r in records),
                actor_output_tokens=sum(r['actor_output_tokens'] for r in records),
                optimizer_updates=0, policy_id=POLICY, selector_revision=ARM_REVISION)


async def generate(args, sample, sampling_params, evaluation=False):
    if not evaluation:
        raise ValueError('Collection only')
    config = json.loads(Path(os.environ['OPENWEBRL_TASK_RESCUE_CONFIG']).read_text())
    task, attempt = str(sample.metadata['task_id']), sample.index % config['attempts_per_task']
    if task not in config['task_order']:
        raise ValueError('Task is outside the frozen eligible shard')
    path = Path(config['output'])/'records'/(key(task, attempt)+'.json')
    if path.exists():
        verify_record(json.loads(path.read_text()), task, attempt)
        return []
    if (Path(config['output'])/'judge-budget/halt.json').exists():
        raise RuntimeError('Judge budget halted')
    local = dict(config, output=config['attempt_output'])
    await trajectory(args, sample, sampling_params, 'arm', attempt, 'guided', local)
    record = json.loads((Path(local['output'])/'guided_records'/(key(task, attempt)+'.json')).read_text())
    record.update(selector_revision=ARM_REVISION, candidate_count=5, candidate_representation='full')
    verify_record(record, task, attempt)
    write_json(path, record)
    return []  # Durable files own tensors; do not accumulate all trajectories in Ray.


async def collect(args, rollout_id, config):
    import openwebrl.reward_browser as reward
    from openwebrl.arm_terminal_budget import CappedJudge
    from slime.rollout.sglang_rollout import eval_rollout_single_dataset
    from slime.rollout.base_types import RolloutFnEvalOutput
    if (args.num_rollout != 0 or args.judge_api_model != 'gpt-4.1' or
            args.judge_prompt_variant != 'action_history' or len(args.eval_datasets) != 1):
        raise ValueError('Actor/judge/collection protocol changed')
    dataset = args.eval_datasets[0]
    n = config['attempts_per_task']
    if (dataset.n_samples_per_eval_prompt, dataset.temperature, dataset.top_p,
            dataset.top_k, dataset.max_response_len) != (n, .8, 1., -1, 1024):
        raise ValueError('Decoding changed')
    rows = [json.loads(line) for line in Path(config['tasks']).read_text().splitlines()]
    ids = [str(row['metadata']['task_id']) for row in rows]
    manifest = json.loads(Path(config['partition_manifest']).read_text())
    if ids != config['task_order'] or ids != manifest['shards'][config['shard']]:
        raise ValueError('Frozen eligible cohort changed')
    root = Path(config['output'])
    indexed = {(r['task_id'], r['attempt']):r for r in
               (json.loads(p.read_text()) for p in (root/'records').glob('*.json'))}
    summarize(list(indexed.values()), ids, n)
    old = reward._get_openai_client
    judge = CappedJudge(root/'judge-budget', max_calls=config['judge_max_calls'], max_usd=config['judge_max_usd'])
    reward._get_openai_client = lambda **unused: judge
    try:
        boundaries = [0, 2, *range(22, len(rows), 20), len(rows)]
        for a, b in zip(boundaries, boundaries[1:]):
            selected = rows[a:b]
            pending = [row for row in selected if any((str(row['metadata']['task_id']), k) not in indexed for k in range(n))]
            if pending:
                path = root/f'work-{a}.jsonl'
                path.write_text(''.join(json.dumps(row)+'\n' for row in pending))
                await eval_rollout_single_dataset(args, rollout_id, replace(dataset, name=f'arm-rescue-{config["shard"]}-{a}', path=str(path)))
            recent = []
            for row in selected:
                task = str(row['metadata']['task_id'])
                for k in range(n):
                    record = json.loads((root/'records'/(key(task,k)+'.json')).read_text())
                    verify_record(record, task, k)
                    indexed[task,k] = record
                    recent.append(record)
            report = summarize(list(indexed.values()), ids, n, verify_artifacts=False)
            write_json(root/'progress.json', {k:v for k,v in report.items() if k != 'rescued_tasks'})
            if a == 0:
                if not any(r['valid'] and r['selector_calls'] > 0 and r['selector_fallback_turns'] == 0 for r in recent):
                    raise RuntimeError('No valid, genuinely ARM-guided startup rollout; inspect before scaling')
                write_json(root/'startup-check.json', dict(passed=True, primary_attempts=len(recent)))
            if len(recent) >= 20 and sum(not r['valid'] for r in recent) > len(recent)/2:
                raise RuntimeError('More than50% invalid in completed batch')
        report = summarize(list(indexed.values()), ids, n, require_complete=True)
        write_json(root/'summary.json', report)
        write_json(root/'collection-complete.json', {k:v for k,v in report.items() if k != 'rescued_tasks'})
        return RolloutFnEvalOutput(data={}, metrics={k:v for k,v in report.items() if type(v) in (int,float)})
    finally:
        reward._get_openai_client = old
        await judge.close()


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    if not evaluation or args.num_rollout != 0:
        raise ValueError('Zero optimizer updates required')
    from slime.utils.async_utils import run
    config = json.loads(Path(os.environ['OPENWEBRL_TASK_RESCUE_CONFIG']).read_text())
    return run(collect(args, rollout_id, config))
