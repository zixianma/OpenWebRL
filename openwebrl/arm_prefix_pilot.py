"""Training-only ARM-prefix feasibility experiment; no policy updates."""
import asyncio
from copy import copy, deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import time

from openwebrl.arm_inference import ActionSelector
from openwebrl.arm_rescue_yield import stable_seed, write_json
from openwebrl.arm_task_success import AuditSelector

PREFIXES = (0, 2, 4)
REPEATS = 2
JUDGE = 'gpt-4.1/action_history'


class PrefixSelector:
    """Select for exactly k turns, then continue in the same browser with K=1."""
    def __init__(self, prefix, endpoint, output, seed):
        if prefix not in PREFIXES:
            raise ValueError('Unregistered handoff length')
        self.prefix = prefix
        self.output = Path(output)
        self.guide = AuditSelector('selection', endpoint, self.output/'selections', seed=seed)
        self.actor = ActionSelector('baseline', endpoint, self.output/'actor', seed=seed)
        self.turns = []
        self.actor_requests = self.actor_output_tokens = self.selector_calls = 0
        self.suffix_requests = self.suffix_output_tokens = 0
        self.handoff = None

    async def __call__(self, *, infer, **kwargs):
        turn = kwargs['turn']
        if turn != len(self.turns):
            raise ValueError('Handoff requires contiguous zero-based turns')
        guided = turn < self.prefix
        if not guided and self.handoff is None:
            self.handoff = dict(turn=turn, url=kwargs['observation'].get('active_tab_url', ''),
                prompt_sha256=hashlib.sha256(kwargs['input_text'].encode()).hexdigest(),
                screenshot_sha256=hashlib.sha256(kwargs['observation']['screenshot']).hexdigest(),
                history_turns=len(kwargs['history']))
            write_json(self.output/'handoff.json', self.handoff)

        async def metered(*args, **kw):
            self.actor_requests += 1
            if not guided:
                self.suffix_requests += 1
            result = await infer(*args, **kw)
            self.actor_output_tokens += len(result[1])
            if not guided:
                self.suffix_output_tokens += len(result[1])
            return result

        self.selector_calls += int(guided)
        result, metadata = await (self.guide if guided else self.actor)(infer=metered, **kwargs)
        if metadata.get('fallback'):
            raise ValueError('Silent selector fallback is forbidden')
        entry = dict(turn=turn, guided=guided, selected_index=metadata['selected_index'])
        self.turns.append(entry)
        write_json(self.output/'handoff-trace.json', dict(prefix=self.prefix, turns=self.turns,
            handoff=self.handoff, actor_requests=self.actor_requests,
            actor_output_tokens=self.actor_output_tokens, selector_calls=self.selector_calls))
        return result, dict(metadata, prefix_turns=self.prefix, guided=guided,
            actor_suffix=not guided, eligible_for_future_suffix_loss=not guided)


def condition_order(seed, task, phase):
    conditions = [(prefix, repeat) for prefix in PREFIXES for repeat in range(REPEATS)]
    random.Random(stable_seed(seed, task, phase, 'order')).shuffle(conditions)
    return conditions


async def trajectory(args, sample, params, prefix, repeat, phase, config):
    from openwebrl.generate_browser import generate_turn_sample
    from openwebrl.reward_browser import reward_func
    from openwebrl.eval_monitor import save_task
    from slime.utils.rollout_transport import file_back_completed_group
    from slime.utils.trajectory_metrics import trajectory_metrics
    from slime.utils.types import Sample

    task = str(sample.metadata['task_id'])
    key = hashlib.sha256(f'{task}/{phase}/{prefix}/{repeat}'.encode()).hexdigest()
    artifact = Path(config['output'])/'trajectories'/key
    # Common random-number schedule across k; contexts diverge after intervention.
    seed = stable_seed(config['seed'], config['policy_id'], task, phase, repeat)
    selector = PrefixSelector(prefix, config['selector_endpoint'], artifact, seed)
    local, initial = copy(args), deepcopy(sample)
    initial.index = (200000 if phase == 'smoke' else 100000) + sample.index*10 + PREFIXES.index(prefix)*2 + repeat
    initial.session_id = None
    local.max_steps = config['max_steps']
    local.inference_step_timeout_secs = 180
    local.judge_timeout_secs = 120
    local.browser_action_selector = selector
    turns, error = [], None
    started = time.monotonic()
    try:
        turns = await generate_turn_sample(local, initial, dict(params, sampling_seed=seed))
        if turns and not any(t.remove_sample or t.status == Sample.Status.ABORTED for t in turns):
            rewards = await reward_func(local, turns)
            for turn, reward in zip(turns, rewards, strict=True):
                turn.reward = reward
    except Exception as exc:
        error = type(exc).__name__
        for turn in turns:
            turn.remove_sample = True
    finally:
        await asyncio.to_thread(file_back_completed_group, turns)
        await asyncio.to_thread(save_task, local, artifact, task, turns, error)
    valid = error is None and trajectory_metrics(local, [turns])['valid_trajectories'] == 1
    terminal = max(turns, key=lambda t: t.metadata.get('turn_index', 0)) if turns else None
    reward = terminal.get_reward_value(local) if valid else None
    record = dict(task_id=task, prefix=prefix, repeat=repeat, phase=phase, seed=seed,
        policy_id=config['policy_id'], judge_id=JUDGE, valid=valid, reward=reward,
        error_type=error, steps=len(turns), artifact=str(artifact), handoff=selector.handoff,
        actor_suffix_turns=sum(not t['guided'] for t in selector.turns),
        guided_turns=sum(t['guided'] for t in selector.turns),
        actor_requests=selector.actor_requests, actor_output_tokens=selector.actor_output_tokens,
        suffix_requests=selector.suffix_requests, suffix_output_tokens=selector.suffix_output_tokens,
        selector_calls=selector.selector_calls, elapsed_seconds=time.monotonic()-started)
    write_json(Path(config['output'])/f'{phase}_records'/f'{key}.json', record)
    print('[PrefixPilot] '+json.dumps(record), flush=True)
    return turns


def summarize(records, ids, policy_id):
    expected = {(task, prefix, repeat) for task in ids for prefix in PREFIXES for repeat in range(REPEATS)}
    keyed = {}
    for r in records:
        key = (r['task_id'], r['prefix'], r['repeat'])
        if key not in expected or key in keyed or r['policy_id'] != policy_id or r['judge_id'] != JUDGE:
            raise ValueError('Unexpected, duplicated, or changed-condition attempt')
        keyed[key] = r
    if set(keyed) != expected:
        raise ValueError('Every task requires every registered condition and repeat')
    success = lambda r: bool(r['valid'] and r['reward'] == 1.)
    by_prefix, paired = {}, {}
    for prefix in PREFIXES:
        rows = [keyed[task, prefix, repeat] for task in ids for repeat in range(REPEATS)]
        wins = sum(success(r) for r in rows)
        valid = sum(r['valid'] for r in rows)
        suffix_wins = sum(success(r) and r['actor_suffix_turns'] > 0 for r in rows)
        reached = sum(r['actor_suffix_turns'] > 0 for r in rows)
        by_prefix[str(prefix)] = dict(attempts=len(rows), successes=wins, valid=valid,
            invalid=len(rows)-valid, overall=wins/len(rows), valid_only=wins/valid if valid else None,
            successes_with_actor_suffix=suffix_wins, suffix_success_per_attempt=suffix_wins/len(rows),
            prefix_only_successes=wins-suffix_wins, reached_actor_suffix=reached,
            **{name:sum(r[name] for r in rows) for name in (
                'actor_requests','actor_output_tokens','selector_calls','elapsed_seconds','actor_suffix_turns')})
        if prefix:
            pairs = [(keyed[t, 0, rep], keyed[t, prefix, rep]) for t in ids for rep in range(REPEATS)]
            joint = [(a,b) for a,b in pairs if a['valid'] and b['valid']]
            task_deltas = [sum(int(success(keyed[t, prefix, rep]))-int(success(keyed[t, 0, rep]))
                for rep in range(REPEATS))/REPEATS for t in ids]
            rng = random.Random(20260926)
            boots = sorted(sum(rng.choice(task_deltas) for _ in ids)/len(ids) for _ in range(2000))
            paired[str(prefix)] = dict(overall_delta=sum(task_deltas)/len(ids),
                task_cluster_bootstrap_95ci=[boots[49],boots[1949]],
                improved=sum(not success(a) and success(b) for a,b in pairs),
                regressed=sum(success(a) and not success(b) for a,b in pairs),
                jointly_valid_attempts=len(joint),
                jointly_valid_delta=sum(int(success(b))-int(success(a)) for a,b in joint)/len(joint) if joint else None)
    return dict(complete=True,tasks=len(ids),trajectories=len(records),optimizer_updates=0,
        by_prefix=by_prefix,paired=paired,
        actor_two_attempt_any_success=sum(any(success(keyed[t,0,rep]) for rep in range(REPEATS)) for t in ids)/len(ids),
        limitation='Training-only feasibility, not learning or held-out performance. Guidance changes '
            'state distribution. Two actor retries are not exactly matched compute. Common seeds do not '
            'make dynamic websites deterministic. Handoff-conditional comparisons are selected populations; '
            'report unconditional suffix-success yield. Two prefix comparisons are exploratory.')


def verify_archives(records):
    for r in records:
        archive = Path(r['artifact'])/(hashlib.sha256(r['task_id'].encode()).hexdigest()+'.pt')
        if not archive.is_file() or not archive.stat().st_size or not archive.with_suffix('.json').is_file():
            raise ValueError('Missing saved trajectory or judge verdict')


async def generate(args, sample, sampling_params, evaluation=False):
    if not evaluation:
        raise ValueError('The prefix pilot never trains')
    config = json.loads(Path(os.environ['OPENWEBRL_ARM_QUALITY_CONFIG']).read_text())
    phase = sample.metadata.get('prefix_phase')
    task = str(sample.metadata['task_id'])
    if phase not in ('smoke','compare') or task not in config['task_order']:
        raise ValueError('Unregistered phase/task')
    turns = []
    for prefix, repeat in condition_order(config['seed'], task, phase):
        for halt in (Path(config['budget'])/'halt.json', Path(config['output'])/'selector-failure.json'):
            if halt.exists():
                raise RuntimeError('Stop after selector/judge failure')
        turns.extend(await trajectory(args, sample, sampling_params, prefix, repeat, phase, config))
    return turns


async def collect(args, rollout_id, config):
    import openwebrl.reward_browser as reward
    from openwebrl.arm_terminal_budget import CappedJudge
    from slime.rollout.sglang_rollout import eval_rollout_single_dataset
    from slime.rollout.base_types import RolloutFnEvalOutput
    if args.judge_api_model != 'gpt-4.1' or args.judge_prompt_variant != 'action_history':
        raise ValueError('Preserve the native terminal judge')
    rows = [json.loads(line) for line in Path(config['tasks']).read_text().splitlines()]
    ids = [str(row['metadata']['task_id']) for row in rows]
    if len(ids) != 48 or len(set(ids)) != 48 or ids != config['task_order'] or len(args.eval_datasets) != 1:
        raise ValueError('The fixed training-task panel changed')
    dataset = args.eval_datasets[0]
    if dataset.n_samples_per_eval_prompt != 1:
        raise ValueError('Repeats and conditions are owned by the pilot')
    output = Path(config['output'])
    old_client = reward._get_openai_client
    judge = CappedJudge(Path(config['budget']),max_calls=config['judge_max_calls'],max_usd=config['judge_max_usd'])
    reward._get_openai_client = lambda **unused:judge
    try:
        smoke_file = output/'smoke-tasks.jsonl'
        smoke_file.write_text(''.join(json.dumps(row)+'\n' for row in rows[:2]))
        smoke = replace(dataset,name='arm-prefix-smoke',path=str(smoke_file),
            metadata_overrides=dict(dataset.metadata_overrides,prefix_phase='smoke'))
        await eval_rollout_single_dataset(args,rollout_id,smoke)
        smoke_rows = [json.loads(p.read_text()) for p in (output/'smoke_records').glob('*.json')]
        sm = summarize(smoke_rows,ids[:2],config['policy_id'])
        if any(r['error_type'] for r in smoke_rows):
            raise RuntimeError('Smoke generation/judge failure')
        for prefix, m in sm['by_prefix'].items():
            if not m['valid'] or not m['reached_actor_suffix'] or (int(prefix)>0 and not m['selector_calls']):
                raise RuntimeError('Smoke lacks valid actor suffix or guided-prefix execution')
        verify_archives(smoke_rows)
        write_json(output/'smoke-gate.json',dict(passed=True,excluded_from_primary=True,summary=sm))
        primary = replace(dataset,name='arm-prefix-primary',
            metadata_overrides=dict(dataset.metadata_overrides,prefix_phase='compare'))
        await eval_rollout_single_dataset(args,rollout_id,primary)
        records = [json.loads(p.read_text()) for p in (output/'compare_records').glob('*.json')]
        summary = summarize(records,ids,config['policy_id'])
        verify_archives(records)
        write_json(output/'prefix-summary.json',summary)
        write_json(output/'collection-complete.json',dict(complete=True,tasks=48,trajectories=288,
            smoke_trajectories=12,optimizer_updates=0))
        metrics = {f'prefix/{prefix}/{k}':v for prefix,values in summary['by_prefix'].items()
            for k,v in values.items() if type(v) in (int,float)}
        return RolloutFnEvalOutput(data={},metrics=metrics)
    finally:
        reward._get_openai_client = old_client
        await judge.close()


def generate_rollout(args,rollout_id,data_source,evaluation=False):
    if not evaluation or args.num_rollout != 0:
        raise ValueError('No optimizer updates in prefix pilot')
    from slime.utils.async_utils import run
    config = json.loads(Path(os.environ['OPENWEBRL_ARM_QUALITY_CONFIG']).read_text())
    return run(collect(args,rollout_id,config))
