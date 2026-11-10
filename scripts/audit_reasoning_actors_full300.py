#!/usr/bin/env python3
"""Independently verify final600 collection, all attempts, budgets and W&B.

Read-only toward experiments and providers. The sole optional output is the
private final-audit.json used to gate aggregate export; never marks completion.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from openwebrl.luna_qwen_metrics import api_cost,digest,write_json
from openwebrl.reasoning_actor_policy import MODELS,metric_prices
from report_reasoning_actors_full300 import audited_records
from run_reasoning_actors_full300 import RESOURCES,ACTOR_LIMITS,JUDGE_LIMITS,validate_approval

TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}


def read(p):return json.loads(Path(p).read_text())


def receipt_accounting(paths,model,role,pricing,ledger,limits):
    ids=[];counts=Counter();known=reserved=0.;input_tokens=output_tokens=0
    for path in paths:
        row=read(path);ids.append(int(path.stem.split('-')[-1]));request=row['request']
        if request.get('model')!=model or row.get('role')!=role:
            raise ValueError('Unexpected receipt model or role')
        if role=='actor' and (request.get('reasoning')!={'effort':'high'} or request.get('max_output_tokens')!=4096
                or request.get('service_tier')!='default' or request.get('tool_choice')!='required'
                or request.get('parallel_tool_calls') is not False or request.get('store') is not False
                or any(k in request for k in ('temperature','top_p'))):
            raise ValueError('Actor reasoning or output cap changed')
        if role=='judge' and (request.get('seed')!=42 or request.get('max_completion_tokens')!=4096):
            raise ValueError('Judge seed or output cap changed')
        if row.get('response'):
            actual=row['response']['model']
            if actual!=model and not (role=='actor' and actual.startswith(model+'-')):
                raise ValueError('Returned model identity changed')
        if row.get('usage'):
            if row.get('response', {}).get('usage') != row['usage']:
                raise ValueError('Receipt usage differs from preserved provider response')
            known+=api_cost(row['usage'],role,pricing)['upper_usd'];counts['received']+=1
            input_tokens+=row['usage'].get('input_tokens',row['usage'].get('prompt_tokens',0))
            output_tokens+=row['usage'].get('output_tokens',row['usage'].get('completion_tokens',0))
        else:
            reserved+=row['reserved_usd'];counts['unknown_usage']+=1
    if (len(ids)!=len(set(ids)) or set(ids)!=set(range(1,ledger.get('calls',0)+1))
            or not math.isclose(known+reserved,ledger.get('charged_or_reserved_usd',0),rel_tol=1e-9,abs_tol=1e-7)
            or len(ids)>limits['max_calls'] or known+reserved>limits['max_usd']+1e-9):
        raise ValueError('All-attempt receipts fail exact ledger or approved-cap reconciliation')
    return dict(calls=len(ids),received=counts['received'],unknown_usage=counts['unknown_usage'],
        known_upper_usd=known,unresolved_reserved_usd=reserved,charged_or_reserved_usd=known+reserved,
        input_tokens=input_tokens,output_tokens=output_tokens,max_calls=limits['max_calls'],max_usd=limits['max_usd'])


def audit(root):
    root=Path(root).resolve();proposal,approval,execution=[read(root/p) for p in ['plan.json','approval.json','execution.json']]
    validate_approval(approval,proposal)
    if approval['execution_sha256']!=digest(execution):raise ValueError('Execution registration changed')
    source=Path(execution['source'])
    subprocess.check_output([str(root.parent/'venv/bin/python'),str(source/'scripts/run_reasoning_actors_full300.py'),
        '--root',str(root),'--validate'],text=True)
    jobs=[a['job_id'] for a in approval['attempts']]
    if len(jobs)!=len(set(jobs)):raise ValueError('Duplicate scheduler attempt')
    raw=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(jobs),
        '--format=JobIDRaw,State,ElapsedRaw,AllocTRES'],text=True)
    scheduler={}
    for line in raw.splitlines():
        v=line.split('|')
        if v[0] not in jobs:continue
        state=v[1].split()[0].rstrip('+');tres=dict(x.split('=',1) for x in v[3].split(',') if '=' in x)
        if (v[0] in scheduler or state not in TERMINAL or tres.get('cpu')!='16'
                or tres.get('mem') not in ['32G','32768M'] or int(tres.get('gres/gpu','0'))!=0):
            raise ValueError('Require terminal exact-profile scheduler accounting for every attempt')
        scheduler[v[0]]=dict(job_id=v[0],state=state,elapsed_seconds=int(v[2]))
    if set(scheduler)!=set(jobs):raise ValueError('Missing final scheduler rows')
    report=dict(epoch=time.time(),proposal_sha256=digest(proposal),execution_sha256=digest(execution),approval_sha256=digest(approval),
        collection_verified=False,scheduler_accounting_verified=False,api_accounting_verified=False,
        wandb_verified=False,episodes_by_mode={},errors=[],public_accounting={'scheduler':{},'api':{}},remote_runs=[])
    inventory=[];judge_paths=[];worker_runs=[]
    for mode,ref in execution['arms'].items():
        fp=read(ref['plan']);rows,items=audited_records(fp,mode,MODELS[mode]);inventory.extend(items)
        report['episodes_by_mode'][mode]=len(rows)
        attempts=[a for a in approval['attempts'] if a['pool']==mode];elapsed=sum(scheduler[a['job_id']]['elapsed_seconds'] for a in attempts)
        if elapsed>RESOURCES['total_seconds']:raise ValueError('Separate six-hour scheduler budget exceeded')
        current=read(root/'pools'/mode/'current.json');job=attempts[-1]['job_id']
        if (current.get('job_id')!=job or current.get('complete') is not True
                or scheduler[job]['state']!='COMPLETED'):
            raise ValueError('Current attempt lacks successful terminal controller endpoint')
        journal=read(root/'pools'/mode/'attempts'/job/'workers.json')
        if len(journal)!=4 or any(x.get('returncode')!=0 for x in journal):
            raise ValueError('All four owned workers must have exited successfully')
        for entry in journal:
            directory=Path(fp['output'])/'workers'/entry['worker_id']
            if read(directory/'status.json').get('complete') is not True:raise ValueError('Worker sentinel missing')
            worker_runs.append(dict(id=fp['wandb_prefix']+'-w'+entry['worker_id'],model=MODELS[mode]))
        output=Path(fp['output']);actor_paths=list(output.glob('attempts/*/*/requests/actor-*.json'))
        judge_paths.extend(output.glob('attempts/*/*/requests/judge-*.json'))
        report['public_accounting']['api'][mode]=receipt_accounting(actor_paths,MODELS[mode],'actor',fp['pricing'],
            read(root/'budget'/mode/'actor-budget/ledger.json'),ACTOR_LIMITS[mode])
        report['public_accounting']['scheduler'][mode]=dict(attempts=[scheduler[a['job_id']] for a in attempts],
            total_seconds=elapsed,approved_seconds=21600,CPU_core_seconds=elapsed*16,memory_GiB_seconds=elapsed*32,GPU_seconds=0)
    report['public_accounting']['api']['judge']=receipt_accounting(judge_paths,'o4-mini-2025-04-16','judge',
        metric_prices('gpt-6-luna'),read(root/'budget/judge-budget/ledger.json'),JUDGE_LIMITS)
    from dotenv import load_dotenv
    import wandb
    load_dotenv('/gpfs/projects/krishna/zixianma/OpenWebRL/.env')
    api=wandb.Api(timeout=30);entity=os.getenv('WANDB_ENTITY') or api.default_entity
    for expected in worker_runs:
        run=api.run(entity+'/openwebrl-evals/'+expected['id'])
        if (run.state!='finished' or run.project!='openwebrl-evals'
                or run.config.get('actor')!=expected['model'] or run.config.get('reasoning_effort')!='high'):
            raise ValueError('Final W&B state or model/effort identity mismatch')
        report['remote_runs'].append(dict(id=run.id,project=run.project,state=run.state))
    report.update(collection_verified=True,scheduler_accounting_verified=True,api_accounting_verified=True,
        wandb_verified=True,primary_record_inventory_sha256=digest(sorted(inventory)),finished_epoch=time.time())
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path);p.add_argument('--write',action='store_true');a=p.parse_args()
    try:result=audit(a.root)
    except Exception as exc:
        print(json.dumps(dict(verified=False,error_type=type(exc).__name__,error=str(exc))));sys.exit(1)
    if a.write:write_json(a.root/'supervisor/final-audit.json',result)
    print(json.dumps({k:result[k] for k in ['collection_verified','episodes_by_mode','scheduler_accounting_verified','api_accounting_verified','wandb_verified']}))
