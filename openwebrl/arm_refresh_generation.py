"""Generate bounded fresh baseline40 candidate sets at archived training states."""
import asyncio, base64, json, os, time
from collections import Counter
from pathlib import Path
from openwebrl.arm_quality_audit import atomic_json, digest, permutation
from openwebrl.arm_inference import candidate_seed, split_response
from openwebrl.arm_refresh_format import SelectionFormatter

def validate_inputs(config):
    raw=Path(config['states']).read_bytes()
    if digest(raw)!=config['states_sha256']:raise ValueError('State manifest changed')
    rows=[json.loads(line) for line in raw.splitlines()]
    if len(rows)>3000 or len(rows)!=config['states_count'] or len({r['id'] for r in rows})!=len(rows):
        raise ValueError('Candidate-set budget/identity mismatch')
    tasks={part:{r['task_group'] for r in rows if r['split']==part} for part in ('train','dev')}
    if tasks['train']&tasks['dev']:raise ValueError('Task split overlap')
    for row in rows:
        if digest(row['actor_prompt'].encode())!=row['prompt_sha256']:raise ValueError('Prompt changed')
        for image in row['actor_images']:
            if digest(Path(image['path']).read_bytes())!=image['sha256']:raise ValueError('Image changed')
    return rows

async def collect(args,config):
    from openwebrl.generate_browser import _run_inference_step
    from slime.rollout.base_types import RolloutFnEvalOutput
    from slime.rollout.sglang_rollout import GenerateState
    from scripts.audit_arm_preference_pairs import canonical,parse_action,schemas
    rows=validate_inputs(config);out=Path(config['output']);records=out/'records';records.mkdir(exist_ok=True)
    gate=asyncio.Semaphore(8);formatter=SelectionFormatter();counts=Counter();started=time.monotonic()
    url=f'http://{args.sglang_router_ip}:{args.sglang_router_port}/generate'
    sampling=dict(temperature=.8,top_p=1.,top_k=-1,max_new_tokens=4096,skip_special_tokens=False)
    native=GenerateState(args).sampling_params
    for key in ('stop','stop_token_ids','no_stop_trim','spaces_between_special_tokens'):
        if key in native:sampling[key]=native[key]
    def save_progress():
        atomic_json(out/'progress.json',dict(counts=counts,targets=config['targets'],elapsed_seconds=time.monotonic()-started))
    async def one(row):
        async with gate:
            if counts['eligible_'+row['split']]>=config['targets'][row['split']]:return
            dest=records/(row['id']+'.json')
            if dest.exists():raise ValueError('Unexpected existing candidate record; explicit resume required')
            images=['data:image/png;base64,'+base64.b64encode(Path(i['path']).read_bytes()).decode() for i in row['actor_images']]
            seeds=[candidate_seed(20260925,row['id'],row['turn'],i) for i in range(5)]
            outputs=await asyncio.gather(*[_run_inference_step(url,row['actor_prompt'],dict(sampling,sampling_seed=s),images,timeout_secs=600) for s in seeds])
            record=dict(id=row['id'],task_id=row['task_id'],split=row['split'],actor='baseline40',
                checkpoint=config['checkpoint'],candidates=[split_response(o[0]) for o in outputs],
                responses=[o[0] for o in outputs],candidate_seeds=seeds,response_tokens=[len(o[1]) for o in outputs],
                finish_types=[o[3] for o in outputs],prompt_sha256=row['prompt_sha256'],
                screenshot_sha256=row['screenshot_sha256'],eligible=False)
            try:
                if any(f!='stop' for f in record['finish_types']):raise ValueError('Candidate not stopped')
                classes=[canonical(parse_action(c['action'],schemas(row['actor_prompt']))[0]) for c in record['candidates']]
                if len(set(classes))<2:raise ValueError('Only one distinct action')
                record.update(formatter.prepare(dict(row,**{k:record[k] for k in ('candidates',)}),permutation(row['id'],'refresh',20260925)))
                record.update(classes=classes,eligible=True)
                counts['eligible_'+row['split']]+=1
            except (ValueError,KeyError,TypeError) as exc:
                record['exclusion']=str(exc);counts['excluded:'+str(exc)]+=1
            atomic_json(dest,record);counts['states']+=1;counts['candidates']+=5
            counts['output_tokens']+=sum(record['response_tokens']);save_progress()
            if counts['states']%25==0 or counts['states']<=2:
                print(f'[ARMRefresh] {dict(counts)} elapsed={time.monotonic()-started:.1f}',flush=True)
    # First two test the exact restored actor and the archive/image path.
    for row in rows[:2]:await one(row)
    await asyncio.gather(*(one(row) for row in rows[2:]))
    complete=all(counts['eligible_'+p]>=n for p,n in config['targets'].items())
    result=dict(complete=complete,counts=counts,elapsed_seconds=time.monotonic()-started,optimizer_updates=0,browsers=0)
    atomic_json(out/'collection-complete.json',result)
    if not complete:raise ValueError('Eligible targets not reached within approved candidate-set budget')
    return RolloutFnEvalOutput(data={},metrics={'arm_refresh/states':counts['states'],'arm_refresh/output_tokens':counts['output_tokens']})

def generate_rollout(args,rollout_id,data_source,evaluation=False):
    if not evaluation or args.num_rollout!=0:raise ValueError('Offline candidate generation must be evaluation-only')
    from slime.utils.async_utils import run
    return run(collect(args,json.loads(Path(os.environ['OPENWEBRL_ARM_REFRESH_CONFIG']).read_text())))
