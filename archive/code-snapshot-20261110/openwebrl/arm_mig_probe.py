"""Bounded MIG plumbing test; never contributes to benchmark success estimates."""
import asyncio
import base64
import hashlib
import json
from pathlib import Path
import time

from openwebrl.arm_task_success import AuditSelector
from openwebrl.arm_rescue_yield import trajectory, write_json


class SerialAuditSelector(AuditSelector):
    """Keep the same five seeded candidates but only one actor request in flight."""
    async def __call__(self, *, infer, **kwargs):
        gate=asyncio.Lock()
        async def serial(*args,**options):
            async with gate:
                return await infer(*args,**options)
        return await super().__call__(infer=serial,**kwargs)


def archived_history_responses(history):
    """Archived audit history is already split; the live selector expects raw text."""
    result=[]
    for entry in history:
        if isinstance(entry,str):
            result.append(entry)
        elif (isinstance(entry,dict) and set(entry)=={'thought','action'}
              and all(isinstance(entry[k],str) for k in ('thought','action'))):
            result.append('<think>'+entry['thought']+'</think>'+entry['action'])
        else:
            raise ValueError('Unsupported archived history representation')
    return result


def validate_devices(actor,selector):
    for device in (actor,selector):
        if device['count']!=1 or not 15*1024**3<=device['memory_bytes']<=21*1024**3:
            raise ValueError('Each service must see exactly one 18-GB MIG slice')
        if not device.get('uuid'):raise ValueError('Missing CUDA device identity')
    if actor['uuid']==selector['uuid']:raise ValueError('Actor and selector share a slice')


def checked_records(root,expected):
    rows=[json.loads(p.read_text()) for p in Path(root).glob('*_records/*.json')]
    if {(r['task_id'],r['mode']) for r in rows}!=set(expected) or len(rows)!=len(expected):
        raise ValueError('Missing/duplicate pilot trajectories')
    for r in rows:
        archive=Path(r['artifact'])/(hashlib.sha256(r['task_id'].encode()).hexdigest()+'.pt')
        if not archive.is_file() or not archive.stat().st_size or not archive.with_suffix('.json').is_file():
            raise ValueError('Missing pilot rollout or verdict')
        if r['error_type'] or not r['valid']:raise ValueError('Pilot trajectory did not get a valid terminal judgment')
        if r['mode']=='arm' and (r['selector_fallback_turns'] or r['selector_calls']==0):
            raise ValueError('ARM selection was not executed correctly')
    return rows


def probe_args(config):
    from openwebrl.run_evaluate import EvalArgs
    args=EvalArgs(sglang_router_ip='127.0.0.1',sglang_router_port=config['actor_port'],
        hf_checkpoint=config['actor'],sglang_server_concurrency=1,rollout_temperature=.8,
        rollout_top_p=1.,rollout_top_k=-1,rollout_max_response_len=4096,
        rollout_max_context_len=32768,max_steps=30,context_num_screenshots=1,
        judge_api_model='gpt-4.1',judge_api_mode='served',judge_prompt_variant='action_history',
        turn_history_reasoning_mode='full',browser_response_format_mode='browser_env',
        browser_include_tool_response=1,inference_step_timeout_secs=120,judge_timeout_secs=120)
    # Native reward_func returns scalar rewards; its metadata contains the judge dict.
    args.reward_key=None;args.debug_trace_prob=0.
    return args


def recovered_records(config):
    receipt=config.get('recovery')
    if not receipt:return {}
    if receipt['actor']!=config['actor']:raise ValueError('Recovered actor identity mismatch')
    for file,digest in receipt['files_sha256'].items():
        if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=digest:
            raise ValueError('Recovered record/verdict changed')
    rows=checked_records(receipt['root'],[(receipt['task_id'],'arm')])
    row=rows[0]
    if (row['task_id']!=str(config['tasks'][0]['metadata']['task_id'])
        or row['policy_id']!='sft-mig-architecture-probe'
        or row['judge_id']!='gpt-4.1/action_history'):
        raise ValueError('Recovered trajectory identity mismatch')
    return {(row['task_id'],row['mode']):row}


async def run(config):
    import httpx
    import wandb
    from openwebrl.generate_browser import _run_inference_step
    from openwebrl.arm_terminal_budget import CappedJudge
    import openwebrl.arm_inference as inference
    import openwebrl.reward_browser as reward
    import openwebrl.generate_browser as generation
    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.http_utils import init_http_client
    from slime.utils.types import Sample

    root=Path(config['output'])
    generation._BROWSER_HOST_BLACKLIST_PATH=str(root/'navigation-failures.txt')
    args=probe_args(config)
    recovered=recovered_records(config)
    init_http_client(args)
    state=GenerateState(args);sampling=dict(state.sampling_params)
    tracking=wandb.init(entity='zixianma',project='openwebrl-evals',id=config['run_id'],
        name=config['run_id'],config=dict(diagnostic_only=True,actor=config['actor'],
        context=32768,candidates=5,actor_candidate_concurrency=1,live_tasks=2),
        dir=str(root),settings=wandb.Settings(x_disable_stats=True))
    judge=CappedJudge(config.get('judge_budget_root',root/'terminal-budget'));judge.MAX_USD=1.;judge.MAX_CALLS=6
    old_selector,old_client=inference.ActionSelector,reward._get_openai_client
    inference.ActionSelector=SerialAuditSelector;reward._get_openai_client=lambda **kw:judge
    try:
        endpoint=f"http://127.0.0.1:{config['actor_port']}"
        async with httpx.AsyncClient(timeout=240,trust_env=False) as client:
            info=(await client.get(endpoint+'/get_model_info')).json()
            if Path(info['model_path']).resolve()!=Path(config['actor']).resolve():
                raise ValueError('Actor service loaded a different checkpoint')
            # Exercise a full 32k KV footprint without spending API calls.
            token=state.tokenizer.encode(' web',add_special_tokens=False)[0]
            # SGLang reserves one context token: input + output must be < 32768.
            response=await client.post(endpoint+'/generate',json=dict(input_ids=[token]*32759,
                sampling_params=dict(temperature=0,max_new_tokens=8,ignore_eos=True)))
            response.raise_for_status();capacity=response.json()
            write_json(root/'context-capacity.json',dict(passed=True,input_tokens=32759,
                generation=capacity.get('meta_info',{}),synthetic=True,excluded_from_results=True))
        for row in config['states']:
            selector=SerialAuditSelector('selection',config['selector_endpoint'],root/'offline-selections',seed=20260921)
            started=time.monotonic()
            await selector(infer=_run_inference_step,url=endpoint+'/generate',input_text=row['actor_prompt'],
                sampling_params=sampling,images=['data:image/png;base64,'+base64.b64encode(Path(i['path']).read_bytes()).decode() for i in row['actor_images']],
                observation=dict(screenshot=Path(row['screenshot']).read_bytes(),active_tab_url=row['url']),
                history=archived_history_responses(row['history']),task=row['task'],task_id=row['id'],turn=row['turn'],timeout=240)
            tracking.log({'probe/offline_selection_seconds':time.monotonic()-started})
        expected=[]
        for index,mode in enumerate(['arm','actor0']):
            row=config['tasks'][index];task=str(row['metadata']['task_id']);expected.append((task,mode))
            if (task,mode) in recovered:
                write_json(root/'probe_records'/f'recovered-{mode}.json',recovered[(task,mode)])
                continue
            sample=Sample(index=index,prompt=row['prompt'],metadata=dict(row['metadata']))
            task_sampling=dict(sampling)
            if mode=='actor0':task_sampling.update(temperature=0.,top_k=1)
            await trajectory(args,sample,task_sampling,mode,1 if mode=='arm' else 0,'probe',dict(
                output=str(root),policy_id='sft-mig-architecture-probe',seed=20260921,
                selector_endpoint=config['selector_endpoint'],max_steps=30,inference_timeout=240))
        rows=checked_records(root,expected)
        result=dict(passed=True,diagnostic_only=True,checkpoint='SFT architecture proxy, not an RL checkpoint',
            offline_states=len(config['states']),browser_trajectories=len(rows),
            actor_tokens=sum(r['actor_output_tokens'] for r in rows),
            selector_calls=sum(r['selector_calls'] for r in rows),
            recovered_browser_trajectories=len(recovered),
            limitation='B/C and outcome-only RL checkpoint export/parity are separate required gates.')
        write_json(root/'probe-result.json',result);tracking.log({'probe/passed':1,**{f'probe/{k}':v for k,v in result.items() if type(v) in (int,float)}})
    finally:
        inference.ActionSelector, reward._get_openai_client=old_selector,old_client
        await judge.close();tracking.finish()
