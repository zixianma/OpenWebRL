"""Fresh ARM-only fixed-100 task rollouts; terminal success, never local quality labels."""
import asyncio
import base64
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import statistics

from openwebrl.arm_inference import ActionSelector, candidate_seed, parse_selection, request_selection_result, split_response
from openwebrl.arm_quality_audit import permutation
from openwebrl.arm_rescue_yield import stable_seed, trajectory, write_json


class AuditSelector(ActionSelector):
    """Randomize ARM display order; persist candidates and reject silent fallbacks."""
    async def __call__(self, *, infer, url, input_text, sampling_params, images,
                       observation, history, task, task_id, turn, timeout):
        if self.mode == 'baseline':
            return await super().__call__(infer=infer,url=url,input_text=input_text,
                sampling_params=sampling_params,images=images,observation=observation,
                history=history,task=task,task_id=task_id,turn=turn,timeout=timeout)
        outputs=await asyncio.gather(*(infer(url,input_text,dict(sampling_params,
            sampling_seed=candidate_seed(self.seed,task_id,turn,i)),images,timeout_secs=timeout)
            for i in range(5)))
        candidates=[split_response(o[0]) for o in outputs]
        order=permutation(f'{task_id}/{turn}/{self.seed}','task-success-selector')
        record=dict(mode='selection',task_id=task_id,turn=turn,candidates=candidates,
            responses=[o[0] for o in outputs],response_tokens=[len(o[1]) for o in outputs],
            candidate_seeds=[candidate_seed(self.seed,task_id,turn,i) for i in range(5)],
            finish_types=[o[3] for o in outputs],permutation=order,selected_index=None,
            prompt_sha256=hashlib.sha256(input_text.encode()).hexdigest(),
            screenshot_sha256=hashlib.sha256(observation['screenshot']).hexdigest())
        path=self.output/f'{hashlib.sha256(str(task_id).encode()).hexdigest()[:20]}-{turn:02d}.json'
        write_json(path,record)
        try:
            reply=await request_selection_result(self.endpoint,dict(mode='selection',task=task,
                url=observation.get('active_tab_url',''),history=[split_response(h) for h in history],
                candidates=[candidates[i] for i in order],
                screenshot=base64.b64encode(observation['screenshot']).decode()),self.timeout,self.connect_timeout)
            selected=order[parse_selection(reply.get('raw',''),5)]
            record.update(selected_index=selected,selector_raw=reply.get('raw'),fallback=None)
        except Exception as exc:
            record['error_type']=type(exc).__name__;write_json(path,record)
            config_path=os.getenv('OPENWEBRL_ARM_QUALITY_CONFIG')
            if config_path:
                config=json.loads(Path(config_path).read_text())
                write_json(Path(config['output'])/'selector-failure.json',dict(error_type=type(exc).__name__,trace=str(path)))
            raise
        write_json(path,record)
        return outputs[selected],dict(selected_index=selected,mode='selection',fallback=None,trace_path=str(path))


def summarize(records, ids, policy_id):
    groups={}
    for r in records:
        if r['policy_id']!=policy_id or r['judge_id']!='gpt-4.1/action_history':
            raise ValueError('Checkpoint or terminal judge changed')
        if r['task_id'] not in ids or r['mode']!='arm' or r['task_id'] in groups:
            raise ValueError('Unknown, duplicate, or unauthorized actor-only attempt')
        groups[r['task_id']]=r
    if set(groups)!=set(ids):raise ValueError('ARM must finish every declared task')
    rows=[groups[t] for t in ids]
    success=lambda r:bool(r['valid'] and r['reward']==1.)
    wins=sum(map(success,rows));valid=sum(r['valid'] for r in rows)
    metrics=dict(tasks=len(ids),successes=wins,valid=valid,invalid=len(ids)-valid,
        overall=wins/len(ids),valid_only=wins/valid if valid else None,
        **{k:sum(r[k] for r in rows) for k in ['actor_output_tokens','actor_requests','selector_calls','selector_fallback_turns','elapsed_seconds']})
    return dict(complete=True,tasks=len(ids),by_mode=dict(arm=metrics),
        task_outcomes=[dict(task_id=t,arm=dict(valid=groups[t]['valid'],success=success(groups[t]),artifact=groups[t]['artifact'])) for t in ids],
        limitation='ARM-only fresh trajectories; historical controls differ in date and decoding. SFT historical judge differs. No selector-only causal or equal-compute claim.')


async def generate(args,sample,sampling_params,evaluation=False):
    if not evaluation:raise ValueError('Task audit is evaluation only')
    config=json.loads(Path(os.environ['OPENWEBRL_ARM_QUALITY_CONFIG']).read_text())
    task=str(sample.metadata['task_id']);phase=sample.metadata.get('success_audit_phase')
    if phase not in ('smoke','compare'):raise ValueError('Unknown audit phase')
    if task not in config['task_order']:raise ValueError('Task is outside the fixed100 cohort')
    turns=await trajectory(args,sample,sampling_params,'arm',1,phase,config)
    if (Path(config['budget'])/'halt.json').exists():raise RuntimeError('Terminal judge budget halted')
    if (Path(config['output'])/'selector-failure.json').exists():raise RuntimeError('Selector failed; no fallback or further scaling')
    return turns


async def collect(args,rollout_id,config):
    import openwebrl.arm_inference as inference
    import openwebrl.reward_browser as reward
    from openwebrl.arm_terminal_budget import CappedJudge
    from slime.rollout.sglang_rollout import eval_rollout_single_dataset
    from slime.rollout.base_types import RolloutFnEvalOutput
    if args.judge_api_model!='gpt-4.1' or args.judge_prompt_variant!='action_history':
        raise ValueError('Use native GPT-4.1/action-history terminal success')
    output=Path(config['output']);rows=[json.loads(x) for x in Path(config['tasks']).read_text().splitlines()]
    ids=[str(r['metadata']['task_id']) for r in rows]
    count=config.get('task_count',100)
    if count not in (100,300) or ids!=config['task_order'] or len(ids)!=count or len(set(ids))!=count or len(args.eval_datasets)!=1:
        raise ValueError('Prespecified task cohort changed')
    dataset=args.eval_datasets[0]
    if dataset.n_samples_per_eval_prompt!=1:raise ValueError('One ARM rollout per task')
    old_selector=inference.ActionSelector;old_client=reward._get_openai_client
    judge=CappedJudge(Path(config['budget']),max_calls=config.get('judge_max_calls'),max_usd=config.get('judge_max_usd'))
    inference.ActionSelector=AuditSelector;reward._get_openai_client=lambda **unused:judge
    try:
        smoke_file=output/'smoke-tasks.jsonl';smoke_file.write_text(''.join(json.dumps(r)+'\n' for r in rows[:3]))
        smoke=replace(dataset,name='arm-success-smoke',path=str(smoke_file),
            metadata_overrides=dict(dataset.metadata_overrides,success_audit_phase='smoke'))
        await eval_rollout_single_dataset(args,rollout_id,smoke)
        smoke_rows=[json.loads(p.read_text()) for p in (output/'smoke_records').glob('*.json')]
        smoke_summary=summarize(smoke_rows,ids[:3],config['policy_id'])
        valid=smoke_summary['by_mode']['arm']['valid']
        if not valid or not smoke_summary['by_mode']['arm']['selector_calls']:
            raise RuntimeError('No valid smoke trajectory with real ARM selection')
        if any(r['error_type'] for r in smoke_rows):raise RuntimeError('Smoke generation/judge exception')
        write_json(output/'smoke-gate.json',dict(passed=True,excluded_from_primary=True,summary=smoke_summary))
        primary=replace(dataset,name='arm-success-primary',
            metadata_overrides=dict(dataset.metadata_overrides,success_audit_phase='compare'))
        await eval_rollout_single_dataset(args,rollout_id,primary)
        records=[json.loads(p.read_text()) for p in (output/'compare_records').glob('*.json')]
        summary=summarize(records,ids,config['policy_id'])
        if config.get('comparison_limitation'):
            summary['limitation']=config['comparison_limitation']
        for r in smoke_rows+records:
            archive=Path(r['artifact'])/(hashlib.sha256(r['task_id'].encode()).hexdigest()+'.pt')
            if not archive.is_file() or not archive.stat().st_size or not archive.with_suffix('.json').is_file():
                raise ValueError('A rollout or verdict is missing')
        write_json(output/'success-summary.json',summary)
        write_json(output/'collection-complete.json',dict(complete=True,tasks=count,trajectories=count,
            smoke_trajectories=3,optimizer_updates=0))
        logged={'task_success/'+mode+'/'+key:value for mode,values in summary['by_mode'].items()
                for key,value in values.items() if type(value) in (int,float)}
        return RolloutFnEvalOutput(data={},metrics=logged)
    finally:
        inference.ActionSelector=old_selector;reward._get_openai_client=old_client
        await judge.close()


def generate_rollout(args,rollout_id,data_source,evaluation=False):
    if not evaluation or args.num_rollout!=0:raise ValueError('No optimizer updates in task audit')
    from slime.utils.async_utils import run
    config=json.loads(Path(os.environ['OPENWEBRL_ARM_QUALITY_CONFIG']).read_text())
    return run(collect(args,rollout_id,config))
