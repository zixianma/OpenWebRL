"""Independent, resumable shards of the fixed 2K/five-attempt actor screen."""
from dataclasses import replace
import json
import os
from pathlib import Path

from openwebrl.arm_task_screen import generate, key, summarize, verify_record
from openwebrl.arm_rescue_yield import write_json

SHARDS = 8
TASKS_PER_SHARD = 250
JUDGE_USD_PER_SHARD = 18.75
JUDGE_CALLS_PER_SHARD = 3750


def partition(ids):
    if len(ids) != 2000 or len(set(ids)) != 2000:
        raise ValueError('Expected the exact unique 2K cohort')
    # Interleave the frozen hash order. Seed remains a function of task/attempt,
    # independent of the shard index, job, node, batch and scheduling order.
    return [ids[i::SHARDS] for i in range(SHARDS)]


def validate_partition(shards, ids):
    if shards != partition(ids):
        raise ValueError('Changed, missing or overlapping task shards')


def pending_rows(rows, record_root):
    pending = []
    for row in rows:
        task = str(row['metadata']['task_id'])
        missing = False
        for attempt in range(5):
            path = Path(record_root)/(key(task, attempt)+'.json')
            if path.exists():
                verify_record(json.loads(path.read_text()), task, attempt)
            else:
                missing = True
        if missing:
            pending.append(row)
    return pending


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
    manifest = json.loads(Path(config['partition_manifest']).read_text())
    validate_partition(manifest['shards'], manifest['task_order'])
    if ids != config['task_order'] or ids != manifest['shards'][config['shard']]:
        raise ValueError('Frozen shard changed')
    root = Path(config['output'])
    indexed = {(r['task_id'], r['attempt']): r for r in
        (json.loads(p.read_text()) for p in (root/'records').glob('*.json'))}
    summarize(list(indexed.values()), ids)
    old = reward._get_openai_client
    # One persistent account per disjoint shard: total <=$150/30K requests.
    # The existing generate helper uses bucket0 for all250 tasks.
    judge = CappedJudge(root/'judge-budget-0', max_calls=JUDGE_CALLS_PER_SHARD,
                        max_usd=JUDGE_USD_PER_SHARD)
    reward._get_openai_client = lambda **unused: judge
    try:
        boundaries = [0, 2, *range(22, TASKS_PER_SHARD, 20), TASKS_PER_SHARD]
        for a, b in zip(boundaries, boundaries[1:]):
            selected = rows[a:b]
            pending = pending_rows(selected, root/'records')
            if pending:
                path = root/f'work-{a}.jsonl'
                path.write_text(''.join(json.dumps(r)+'\n' for r in pending))
                await eval_rollout_single_dataset(args, rollout_id,
                    replace(dataset, name=f'actor-screen-shard{config["shard"]}-{a}', path=str(path)))
            recent = []
            for row in selected:
                task = str(row['metadata']['task_id'])
                for attempt in range(5):
                    r = json.loads((root/'records'/(key(task, attempt)+'.json')).read_text())
                    verify_record(r, task, attempt)
                    indexed[task, attempt] = r
                    recent.append(r)
            summary = summarize(list(indexed.values()), ids, verify_artifacts=False)
            write_json(root/'progress.json', {k:v for k,v in summary.items() if k != 'dispositions'})
            if a == 0:
                if len(recent) != 10 or sum(r['valid'] for r in recent) < 5:
                    raise RuntimeError('Startup has fewer than five valid attempts; inspect before scaling')
                write_json(root/'startup-check.json', dict(passed=True, primary_attempts=10))
            if len(recent) >= 50 and sum(not r['valid'] for r in recent)/len(recent) > .5:
                raise RuntimeError('More than50% invalid in the completed batch; diagnose before scaling')
        summary = summarize(list(indexed.values()), ids, require_complete=True)
        write_json(root/'summary.json', summary)
        write_json(root/'all-failure-task-ids.json',
                   [t for t,s in summary['dispositions'].items() if s == 'all_failure'])
        write_json(root/'collection-complete.json',
                   {k:v for k,v in summary.items() if k != 'dispositions'})
        return RolloutFnEvalOutput(data={}, metrics={k:v for k,v in summary.items() if type(v) in (int,float)})
    finally:
        reward._get_openai_client = old
        await judge.close()


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    if not evaluation or args.num_rollout != 0:
        raise ValueError('Screening must execute zero optimizer updates')
    from slime.utils.async_utils import run
    config = json.loads(Path(os.environ['OPENWEBRL_TASK_SCREEN_CONFIG']).read_text())
    return run(collect(args, rollout_id, config))
