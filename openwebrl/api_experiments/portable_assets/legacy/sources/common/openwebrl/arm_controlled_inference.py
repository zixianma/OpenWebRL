"""Paired full-episode inference experiment; immutable schedule, durable attempts.

One frozen SelectionARM episode and five ordinary episodes per benchmark task.
No training, outcome-based retries, task filtering, or stopping on success.
"""
import argparse
import asyncio
from collections import defaultdict
from contextvars import ContextVar
from copy import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from openwebrl.arm_rescue_yield import stable_seed, write_json

MODES = ('arm', 'actor0', 'actor1', 'actor2', 'actor3', 'actor4')
SEED = 20261004
ATTEMPT = ContextVar('controlled_inference_attempt', default=None)
PROTOCOL = dict(actor='OpenWebRL-4B-SFT:iteration0', selector_revision='81b452d800d9f859687074f82680dd5257e02d89',
    browser='local_process', temperature=.7, top_p=.9, top_k=-1, max_new_tokens=1024,
    max_steps=30, context_length=32768, context_num_screenshots=1, history='full',
    judge='o4-mini', judge_protocol='online_mind2web/AgentTrek', judge_response_tokens=4096,
    candidates=5, candidate_representation='full', optimizer_updates=0)


def attempt_key(task, mode):
    return hashlib.sha256(f'{SEED}/{task}/{mode}'.encode()).hexdigest()


def schedule(ids, shards=2):
    if len(ids) != len(set(ids)) or shards < 1:
        raise ValueError('Unique task identities and positive shard count required')
    result = []
    order = sorted(ids, key=lambda t: stable_seed(SEED,t,'task-order'))
    for i, task in enumerate(order):
        modes = list(MODES)
        random.Random(stable_seed(SEED,task,'mode-order')).shuffle(modes)
        result.append(dict(task_id=task, shard=i % shards, attempts=[dict(mode=m,
            key=attempt_key(task,m), seed=stable_seed(SEED,task,m,'generation')) for m in modes]))
    return result


def verify_record(record, item, attempt, root, protocol_sha):
    if (record['task_id'] != item['task_id'] or record['mode'] != attempt['mode'] or
        record['seed'] != attempt['seed'] or record['key'] != attempt['key'] or
        record['protocol_sha256'] != protocol_sha):
        raise ValueError('Attempt identity/protocol changed')
    artifact = Path(record['artifact'])
    if not artifact.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Artifact outside private experiment root')
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != record['artifact_sha256']:
        raise ValueError('Result artifact hash mismatch')
    if record['valid'] and record['reward'] not in (0,1):
        raise ValueError('Valid result must have a binary outcome')
    if record['valid']:
        if not record['trace_files']:
            raise ValueError('Valid result missing decision trace')
        for entry in record['trace_files']:
            path=Path(entry['path'])
            if not path.resolve().is_relative_to(Path(root).resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
                raise ValueError('Decision trace changed')
    return record


def summarize(records, task_ids, require_complete=False):
    expected={(t,m) for t in task_ids for m in MODES}; indexed={}
    for r in records:
        key=(r['task_id'],r['mode'])
        if key not in expected or key in indexed:
            raise ValueError('Unexpected/duplicate attempt')
        indexed[key]=r
    complete=len(indexed)==len(expected)
    if require_complete and not complete: raise ValueError('Incomplete experiment')
    out=dict(complete=complete, scheduled_tasks=len(task_ids), scheduled_attempts=len(expected),
             completed_attempts=len(records), completed_task_blocks=0)
    if not complete: return out
    out['completed_task_blocks']=len(task_ids)
    yes=lambda r: r['valid'] and r['reward']==1
    out['by_mode']={}
    for m in MODES:
        rows=[indexed[t,m] for t in task_ids];valid=sum(r['valid'] for r in rows);wins=sum(yes(r) for r in rows)
        out['by_mode'][m]=dict(successes=wins,valid=valid,overall=wins/len(rows),valid_only=wins/valid if valid else None,
            **{k:sum(r[k] for r in rows) for k in ('actor_requests','actor_output_tokens','actor_input_tokens',
                 'actor_client_request_seconds','browser_steps','browser_step_seconds','selector_calls',
                 'selector_input_tokens','selector_output_tokens','selector_service_seconds','elapsed_seconds')})
    out['pass_at_k']=[]
    for k in range(1,6):
        vals=[]; diffs=[]
        for t in task_ids:
            c=sum(yes(indexed[t,f'actor{i}']) for i in range(5))
            p=1-math.comb(5-c,k)/math.comb(5,k) if 5-c>=k else 1.
            vals.append(p);diffs.append(float(yes(indexed[t,'arm']))-p)
        # Paired task bootstrap; preserve every attempt within each task.
        rng=random.Random(SEED+k);boots=sorted(sum(diffs[rng.randrange(len(diffs))] for _ in diffs)/len(diffs) for _ in range(2000))
        out['pass_at_k'].append(dict(k=k,pass_at_k=sum(vals)/len(vals),arm_minus_pass_at_k=sum(diffs)/len(diffs),
            paired_task_bootstrap_95=[boots[49],boots[1949]],
            expected_actor_output_tokens=sum(indexed[t,f'actor{i}']['actor_output_tokens'] for t in task_ids for i in range(5))*k/5))
    arm_only=actor_only=both_valid=common_arm=common_actor=0
    for t in task_ids:
        a=yes(indexed[t,'arm']);b=any(yes(indexed[t,f'actor{i}']) for i in range(5))
        arm_only+=a and not b;actor_only+=b and not a
        if all(indexed[t,m]['valid'] for m in MODES):
            both_valid+=1;common_arm+=a;common_actor+=b
    out['paired_arm_actor5']=dict(arm_only=arm_only,actor5_only=actor_only,
        all_six_valid_tasks=both_valid,common_valid_arm_successes=common_arm,common_valid_actor5_successes=common_actor)
    out['cost_limitations']='Client request seconds overlap; allocation GPU-hours/device samples are whole-job costs, not per-policy kernel time. Missing server timing stays missing.'
    return out


class MeteredSelector:
    def __init__(self, selector, directory):
        self.selector=selector;self.directory=Path(directory)
        self.requests=self.output_tokens=self.input_tokens=0;self.client_seconds=0.

    async def __call__(self, *, infer, **kwargs):
        from openwebrl import generate_browser as generation
        async def metered(url,text,sampling_params,image_data,timeout_secs=None):
            ident=self.requests;self.requests+=1;started=time.monotonic()
            receipt=dict(request=ident,started_epoch=time.time(),sampling_seed=sampling_params['sampling_seed'])
            try:
                payload=dict(text=text,sampling_params=sampling_params,return_logprob=True)
                if image_data:payload['image_data']=image_data
                output=await asyncio.wait_for(generation.post(url,payload),timeout=timeout_secs)
                meta=output['meta_info'];pairs=meta['output_token_logprobs']
                tokens=[p[1] for p in pairs];logps=[p[0] for p in pairs]
                # Count backend tokens before the generator repairs delimiters.
                self.output_tokens+=meta.get('completion_tokens',len(tokens))
                if 'prompt_tokens' not in meta:raise ValueError('Actor input-token telemetry missing')
                self.input_tokens+=meta['prompt_tokens']
                receipt.update(status='received',meta={k:v for k,v in meta.items() if k not in ('output_token_logprobs','input_token_logprobs','output_top_logprobs','input_top_logprobs')})
                text=generation._ensure_im_end_w_new_line(output['text'],tokens,logps)
                return text,tokens,logps,meta.get('finish_reason',{}).get('type','stop')
            except BaseException as exc:
                receipt.update(status='failed',error_type=type(exc).__name__);raise
            finally:
                duration=time.monotonic()-started;self.client_seconds+=duration;receipt['client_seconds']=duration
                write_json(self.directory/f'actor-request-{ident:05d}.json',receipt)
        return await self.selector(infer=metered,**kwargs)


def make_judge(root):
    from openwebrl.arm_terminal_budget import CappedJudge
    from openai import AsyncOpenAI
    class Judge(CappedJudge):
        INPUT_USD_PER_MILLION=1.1
        OUTPUT_USD_PER_MILLION=4.4
        @staticmethod
        def upper_cost(messages):
            text_bytes=images=0
            for m in messages:
                content=m['content']
                if isinstance(content,str):text_bytes+=len(content.encode())
                else:
                    for part in content:
                        if part.get('type')=='image_url':images+=1
                        else:text_bytes+=len(json.dumps(part).encode())
            return ((text_bytes+images*20000+4096)*1.1+4096*4.4)/1e6

        def settle(self,ident,upper,usage):
            super().settle(ident,upper,usage)
            write_json(self.root/f'{ident:05d}-usage.json',dict(attempt=ATTEMPT.get(),model='o4-mini',usage=usage,
                usd=(usage['prompt_tokens']*1.1+usage['completion_tokens']*4.4)/1e6))

        async def create(self,**kwargs):
            if kwargs.get('model')!='o4-mini':raise ValueError('Wrong terminal judge model')
            upper=self.upper_cost(kwargs['messages']);ident=self.reserve(upper)
            reply=await self.client.chat.completions.create(**kwargs,max_completion_tokens=4096,store=False)
            self.settle(ident,upper,reply.usage.model_dump())
            write_json(self.root/f'{ident:05d}-response.json',dict(attempt=ATTEMPT.get(),response_model=reply.model,
                text=reply.choices[0].message.content,finish_reason=reply.choices[0].finish_reason))
            return reply
    client=AsyncOpenAI(api_key=os.getenv('JUDGE_API_KEY') or os.environ['OPENAI_API_KEY'],
                       base_url='https://api.openai.com/v1',max_retries=0,timeout=180)
    return Judge(root,client=client,max_calls=3600,max_usd=25.)


async def run(config):
    from openwebrl import run_evaluate as evaluation, generate_browser as generation
    from openwebrl.eval import reward_online_mind2web as reward
    from openwebrl.arm_inference import ActionSelector
    from slime.utils.http_utils import init_http_client
    import httpx
    root=Path(config['output']);root.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(Path(config['manifest']).read_text())
    if manifest['protocol']!=PROTOCOL:raise ValueError('Protocol changed')
    if hashlib.sha256(Path(config['tasks']).read_bytes()).hexdigest()!=manifest['task_file_sha256']:
        raise ValueError('Task file changed')
    schedule_all=json.loads(Path(config['schedule']).read_text())
    if hashlib.sha256(Path(config['schedule']).read_bytes()).hexdigest()!=manifest['schedule_sha256']:
        raise ValueError('Schedule changed')
    cohort=[r for r in schedule_all if r['shard']==config['shard']]
    tasks={t['task_id']:t for t in evaluation.load_tasks_from_jsonl(config['tasks'])}
    os.environ.update(SLIME_BROWSER_ENV_MODE='local_process',SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES='8',
        SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR=str(root/'browser_logs'),
        SLIME_BROWSER_LOCAL_PROCESS_PORT_START=str(config['browser_port_start']),
        SLIME_BROWSER_LOCAL_PROCESS_PORT_END=str(config['browser_port_start']+199),WANDB_PROJECT='openwebrl-evals')
    generation._BROWSER_HOST_BLACKLIST_PATH=str(root/'navigation-failures.txt')
    evaluation.reward_func=reward.reward_func
    judge=make_judge(root/'judge-budget');reward._get_openai_client=lambda **unused:judge
    async with httpx.AsyncClient(timeout=30,trust_env=False) as client:
        response=await client.get(f"http://127.0.0.1:{config['actor_port']}/get_model_info");response.raise_for_status()
        if Path(response.json()['model_path']).resolve()!=Path(config['actor']).resolve():raise ValueError('Wrong actor')
        response=await client.get(config['selector_endpoint']+'/health');response.raise_for_status();health=response.json()
        if health['mode']!='selection' or health.get('adapter') or health['model']!=config['selector_model']:
            raise ValueError('Wrong selector')
        write_json(root/'services.json',dict(selector=health,actor=config['actor']))
    args=evaluation.EvalArgs(sglang_router_ip='127.0.0.1',sglang_router_port=config['actor_port'],
        hf_checkpoint=config['actor'],max_steps=30,context_num_screenshots=1,judge_api_model='o4-mini',
        judge_api_mode='served',judge_timeout_secs=180,browser_response_format_mode='browser_env',
        turn_history_reasoning_mode='full',browser_include_tool_response=1,inference_step_timeout_secs=180,
        task_timeout_secs=1800,path_to_save_generated_samples=str(root/'samples'))
    args.rollout_temperature=.7;args.rollout_max_response_len=1024
    init_http_client(args)
    sem=asyncio.Semaphore(8)
    import wandb
    tracking=wandb.init(project='openwebrl-evals',group='ARM-controlled-inference-20261004',
        id=f'arm-controlled-20261004-shard{config["shard"]}',resume='allow',dir=str(root),config=PROTOCOL)
    protocol_sha=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()
    async def one(item):
        async with sem:
            for attempt in item['attempts']:
                record_path=root/'records'/(attempt['key']+'.json')
                if record_path.exists():
                    verify_record(json.loads(record_path.read_text()),item,attempt,root,protocol_sha);continue
                if (root/'judge-budget/halt.json').exists():raise RuntimeError('Judge budget exhausted')
                # A process interrupted before commitment leaves its directory;
                # preserve it and record a new physical attempt for the same slot.
                directory=root/'attempts'/attempt['key']/str(time.time_ns());directory.mkdir(parents=True)
                selector=MeteredSelector(ActionSelector('selection' if attempt['mode']=='arm' else 'baseline',
                    config['selector_endpoint'],directory/'selections',seed=attempt['seed'],record_telemetry=True),directory)
                local=copy(args);local.browser_action_selector=selector
                local.path_to_save_generated_samples=str(directory/'samples')
                browser=[];local.browser_step_timing_callback=lambda seconds:browser.append(seconds)
                started=time.monotonic();token=ATTEMPT.set(attempt['key']);error=None
                try:
                    result=await evaluation.evaluate_single_task(local,tasks[item['task_id']],
                        dict(temperature=.7,top_p=.9,top_k=-1,max_new_tokens=1024,sampling_seed=attempt['seed']),True)
                    valid=result.get('reward') in (0,1) and 'ABORTED' not in result.get('status','')
                    if result.get('metadata',{}).get('judge_timeout'):valid=False
                except Exception as exc:
                    error=type(exc).__name__;valid=False
                    result=dict(task_id=item['task_id'],reward=None,valid=False,error_type=error,error=str(exc))
                finally:ATTEMPT.reset(token)
                artifact=directory/'result.json';write_json(artifact,result)
                traces=[];selector_values=[];fallbacks=0
                for path in sorted((directory/'selections').glob('*.jsonl')):
                    traces.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                    for line in path.read_text().splitlines():
                        trace=json.loads(line);fallbacks+=bool(trace.get('fallback'))
                        if attempt['mode']=='arm':selector_values.append(trace.get('selector_telemetry',{}))
                if any(any(v.get(k) is None for k in ('input_tokens','output_tokens','seconds')) for v in selector_values):
                    raise ValueError('Selector telemetry missing; retain artifacts and stop')
                record=dict(key=attempt['key'],task_id=item['task_id'],mode=attempt['mode'],seed=attempt['seed'],
                    protocol_sha256=protocol_sha,valid=valid,reward=result.get('reward') if valid else None,error_type=error,
                    artifact=str(artifact),artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),trace_files=traces,
                    actor_requests=selector.requests,actor_output_tokens=selector.output_tokens,actor_input_tokens=selector.input_tokens,
                    actor_client_request_seconds=selector.client_seconds,browser_steps=len(browser),browser_step_seconds=sum(browser),
                    selector_calls=len(selector_values),selector_fallbacks=fallbacks,
                    selector_input_tokens=sum(v['input_tokens'] for v in selector_values),
                    selector_output_tokens=sum(v['output_tokens'] for v in selector_values),
                    selector_service_seconds=sum(v['seconds'] for v in selector_values),elapsed_seconds=time.monotonic()-started)
                verify_record(record,item,attempt,root,protocol_sha);write_json(record_path,record)
                tracking.log({'collection/completed_attempts':len(list((root/'records').glob('*.json'))),
                              'collection/last_attempt_seconds':record['elapsed_seconds']})
                print('CONTROLLED_ATTEMPT '+json.dumps({k:v for k,v in record.items() if k not in ('trace_files','artifact')}),flush=True)
    try:
        # Two prespecified tasks, included in the final cohort, exercise both
        # modes before broader collection. No filtering on success.
        await asyncio.gather(*(one(t) for t in cohort[:2]))
        pilot=[json.loads((root/'records'/(a['key']+'.json')).read_text()) for t in cohort[:2] for a in t['attempts']]
        if not any(r['valid'] and r['mode']=='arm' and r['selector_calls']>0 and r['browser_steps']>0 for r in pilot):
            raise RuntimeError('No functioning ARM/browser pilot; stop for diagnosis')
        if not any(r['valid'] and r['mode'].startswith('actor') and r['browser_steps']>0 for r in pilot):
            raise RuntimeError('No functioning ordinary-actor pilot; stop for diagnosis')
        write_json(root/'startup-gate.json',dict(passed=True,attempts=len(pilot),excluded=False))
        await asyncio.gather(*(one(t) for t in cohort[2:]))
        records=[json.loads(p.read_text()) for p in (root/'records').glob('*.json')]
        result=summarize(records,[t['task_id'] for t in cohort],require_complete=True)
        write_json(root/'summary.json',result)
        tracking.summary.update(result)
    finally:
        tracking.finish()
        await judge.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    # Preserve the established event-loop-policy initialization order.
    from openwebrl import run_evaluate
    asyncio.run(run(json.loads(args.config.read_text())))
