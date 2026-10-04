#!/usr/bin/env python3
"""Two-GPU full-300 Sol selection evaluation, with two retained pilot tasks first."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

from evaluate_baseline_checkpoint import REPO,RUNTIME
from resume_baseline import allocation,clean_environment,source_command,write_json
from run_arm_turn_bonus_calibration import selector_preflight
from run_arm_turn_bonus_cycles import INITIAL,tail

CONTROL=RUNTIME/'arm-reproduction/runs/dedicated-282209-20260908T005547Z/full'
TASKS=REPO/'openwebrl/data/eval/online-mind2web.jsonl'
PYTHON=RUNTIME/'venv/bin/python'
EVAL_PYTHON=RUNTIME/'arm-reproduction/venv/bin/python'



def read_result_summaries(directory):
    """Read one trajectory at a time; comparisons need no screenshot payloads."""
    rows = {}
    for path in Path(directory).glob('*.json'):
        row = json.loads(path.read_text())
        key = row['task_id']
        if key in rows:
            raise ValueError('Duplicate task result')
        rows[key] = {field: row.get(field) for field in ('task_id', 'valid', 'reward')}
    return rows


def plan(job,minutes=120):
    if not 1<=minutes<=120: raise ValueError('Inference allocation cap is 120 minutes')
    tasks=[json.loads(x) for x in TASKS.read_text().splitlines()]
    ids=[r.get('metadata',{}).get('task_id') or r.get('id') or f'task_{i}' for i,r in enumerate(tasks)]
    if len(ids)!=300 or len(set(ids))!=300: raise ValueError('Expected exact released 300-task set')
    control=json.loads((CONTROL/'selection/manifest.json').read_text())
    if ids!=control['task_ids'] or hashlib.sha256(TASKS.read_bytes()).hexdigest()!=control['task_file_sha256']:
        raise ValueError('Historical task set mismatch')
    root=RUNTIME/f'evaluations/sol-selection300-{job}'
    port_base=11000+2*(int(job)%4000) if str(job).isdigit() else 27100
    selector_port=31000+int(job)%20000 if str(job).isdigit() else 27511
    root_results=read_result_summaries(CONTROL/'baseline/results')
    # Deterministic historical-success pilot order; all 300 still evaluated once.
    reliable=[i for i,t in enumerate(ids) if root_results[t].get('valid') and root_results[t].get('reward')==1]
    pilots=reliable[:2]
    if len(pilots)!=2: raise ValueError('Missing historical pilot candidates')
    rest=[i for i in range(300) if i not in pilots]
    segments=[dict(name=f'pilot-{i}',gpu=i,indices=[pilots[i]],parallel=1) for i in range(2)]
    segments += [dict(name=f'shard-{i}',gpu=i,indices=rest[i::2],parallel=16) for i in range(2)]
    for segment in segments:
        i=segment['gpu']; first=28000+i*500
        segment['command']=[str(EVAL_PYTHON),'-m','openwebrl.arm_eval','--mode','selection','--actor',str(INITIAL),
            '--actor-port',str(port_base+i),'--selector-endpoint',f'http://127.0.0.1:{selector_port}',
            '--task-file',str(TASKS),'--task-indices',','.join(map(str,segment['indices'])),
            '--browser-port-start',str(first),'--browser-port-end',str(first+399),
            '--parallel',str(segment['parallel']),'--output',str(root/segment['name']),
            '--seed','42','--temperature','0.7','--top-p','0.9','--max-new-tokens','1024',
            '--max-steps','30','--task-timeout','1800','--judge-model','o4-mini','--env-file',str(REPO/'.env')]
    return dict(job_id=str(job),output=str(root),task_ids=ids,segments=segments,selector='gpt-5.6-sol',
        selector_reasoning_effort='medium',actor_port_base=port_base,selector_port=selector_port,
        requested_resources=dict(gpus=2,hours=minutes/60,gpu_hours=2*minutes/60,cpus=16,memory_gib=240),
        protocol={k:control[k] for k in ['actor','task_file_sha256','seed','sampling','max_steps','judge','judge_protocol']})


def report(p,require_complete=True):
    from openwebrl.arm_eval import summarize
    from summarize_arm_reproduction import paired_report
    root=Path(p['output']); current={}; manifests=[]
    for segment in p['segments']:
        output=root/segment['name']; manifest=output/'manifest.json'
        if manifest.exists():
            m=json.loads(manifest.read_text()); manifests.append(m)
            for k,v in p['protocol'].items():
                if m[k]!=v: raise ValueError(f'Historical protocol mismatch: {k}')
        for key,row in read_result_summaries(output/'results').items():
            if key in current: raise ValueError('Duplicate result across pilot/shard segments')
            current[key]=row
    if p.get('diagnostic_only'):
        expected={p['task_ids'][i] for s in p['segments'] for i in s['indices']}
        if require_complete and set(current)!=expected: raise ValueError('Diagnostic pilot tasks incomplete')
        result=dict(summary=summarize(list(current.values()),len(expected)),complete=set(current)==expected,
            diagnostic_only=True,job_id=p['job_id'],limitation='Selected startup tasks; not a benchmark estimate')
        write_json(root/'comparison.json',result)
        return result
    if require_complete and set(current)!=set(p['task_ids']): raise ValueError('Full 300 not complete')
    result=dict(summary=summarize(list(current.values()),300),complete=len(current)==300,paired={},
        job_id=p['job_id'],selector='gpt-5.6-sol',reasoning_effort='medium',
        limitation='Historical controls; live-site/date differences remain. All pilots included exactly once.')
    if require_complete:
        for mode in ['baseline','scalar','selection']:
            control=read_result_summaries(CONTROL/mode/'results')
            result['paired'][mode]=paired_report(control,current)
    write_json(root/'comparison.json',result)
    write_json(REPO/'openwebrl/docs/arm_results/sol-selection300.json',result)
    return result


def execute(p):
    from arm_launch_preflight import require_receipt
    readiness=require_receipt()
    job=p['job_id']
    if os.getenv('SLURM_JOB_ID')!=job or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Requires dedicated authorized Slurm allocation')
    resources=allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),job,requested_gpus=2)
    if resources['cpus']<16 or resources['allocated_memory_gib']<240: raise ValueError('Need 16 CPU/240 GiB')
    steps=subprocess.check_output(['squeue','--steps',f'--jobs={job}','--noheader','--format=%i'],text=True).split()
    if set(steps)-{f'{job}.{x}' for x in ('batch','extern',os.getenv('SLURM_STEP_ID'))}: raise ValueError('Another active worker')
    deadline=time.time()+min(resources['maximum_seconds'],p['requested_resources']['hours']*3600-180)
    from dotenv import dotenv_values
    env=clean_environment()
    for k,v in dotenv_values(REPO/'.env').items():
        if v and k.startswith(('WANDB_','JUDGE_','OPENAI_','AZURE_')): env.setdefault(k,v)
    if not env.get('OPENAI_API_KEY'): raise ValueError('OpenAI key unavailable')
    env.update(JUDGE_API_MODE='served',JUDGE_API_BASE='https://api.openai.com/v1',
        SLIME_BROWSER_LOCAL_PROCESS_PYTHON=str(PYTHON),OMP_NUM_THREADS='2',PYTHONPATH=str(REPO))
    devices=env.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(devices)!=2 or len(set(devices))!=2 or not all(devices): raise ValueError('Expected 2 assigned GPUs')
    root=Path(p['output']); root.mkdir(parents=True,exist_ok=False)
    write_json(root/'launch_manifest.json',p)
    write_json(root/'launch-readiness.json',readiness)
    write_json(root/'source-manifest.json',{name:hashlib.sha256((REPO/name).read_bytes()).hexdigest() for name in
        ['openwebrl/arm_eval.py','openwebrl/arm_inference.py','openwebrl/generate_browser.py','openwebrl/run_evaluate.py',
         'openwebrl/eval/reward_online_mind2web.py','scripts/serve_sol_selector.py']})
    children=[]; handles=[]; services=[]; stage='startup'
    def status(**kw): write_json(root/'status.json',dict(stage=stage,job_id=job,updated_at_epoch_seconds=time.time(),**kw))
    def spawn(command,name,extra=None):
        log=(root/name).open('w'); handles.append(log)
        child=subprocess.Popen(source_command(REPO,['env','OMP_NUM_THREADS=2',*command]),cwd=REPO,env=dict(env,**(extra or {})),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(child); return child
    def check():
        if time.time()>=deadline: raise TimeoutError('Allocation deadline; completed task files preserved')
        if any(c.poll() is not None for c in services): raise RuntimeError('Inference service exited')
        usage=root/'sol-api/usage.json'
        if (root/'sol-api/budget-exhausted.json').exists(): raise RuntimeError('Sol API usage cap reached')
        if usage.exists() and json.loads(usage.read_text()).get('consecutive_failures',0)>=5:
            raise RuntimeError('Five consecutive Sol API failures')
    def healthy(url,child):
        until=min(deadline,time.time()+600)
        while time.time()<until:
            check()
            if child.poll() is not None: raise RuntimeError('Service startup failed')
            try:
                with urllib.request.urlopen(url,timeout=3) as response:
                    if response.status==200: return
            except OSError: pass
            time.sleep(2)
        raise TimeoutError('Service startup timeout')
    def interrupted(*_): raise InterruptedError('Allocation shutdown requested')
    old=[signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT)]
    try:
        status()
        # Validate the CPU/API selector before loading either GPU actor.
        selector=spawn([str(PYTHON),str(REPO/'scripts/serve_sol_selector.py'),'--output',str(root/'sol-api'),
            '--port',str(p['selector_port']),'--max-cost-usd','200'], 'selector.log',dict(CUDA_VISIBLE_DEVICES=''))
        services.append(selector)
        endpoint=f"http://127.0.0.1:{p['selector_port']}"
        healthy(endpoint+'/health',selector)
        selector_preflight(endpoint,root,expected_api_model='gpt-5.6-sol')
        actors=[]
        for i in range(2):
            actors.append(spawn([str(PYTHON),'-m','sglang.launch_server','--model-path',str(INITIAL),
                '--host','127.0.0.1','--port',str(p['actor_port_base']+i),'--dtype','bfloat16','--tp','1',
                '--mem-fraction-static','0.4','--context-length','32768','--max-running-requests','24',
                '--chunked-prefill-size','4096','--disable-cuda-graph'],f'actor-{i}.log',dict(CUDA_VISIBLE_DEVICES=devices[i])))
        services.extend(actors)
        for i in range(2): healthy(f"http://127.0.0.1:{p['actor_port_base']+i}/health_generate",actors[i])
        phases=[('pilot',p['segments'][:2])]
        if p['segments'][2:]: phases.append(('full300',p['segments'][2:]))
        for phase,segments in phases:
            stage=phase; status()
            workers=[spawn(s['command'],s['name']+'.log',dict(CUDA_VISIBLE_DEVICES='')) for s in segments]
            while any(w.poll() is None for w in workers):
                check()
                if any(w.poll() not in (None,0) for w in workers): raise RuntimeError('Evaluator worker failed')
                status(completed_task_files=sum(len(list((root/s['name']/'results').glob('*.json'))) for s in p['segments']))
                time.sleep(10)
            if any(w.returncode for w in workers): raise RuntimeError('Evaluator worker failed')
            if phase=='pilot':
                traces=[json.loads(line) for s in segments for path in (root/s['name']/'selections').glob('*.jsonl') for line in path.read_text().splitlines()]
                if not traces or any(t.get('fallback') or len(t['candidates'])!=5 for t in traces):
                    raise RuntimeError('Pilot selector pipeline did not pass; do not scale')
                valid=0
                for s in segments:
                    summary=json.loads((root/s['name']/'summary.json').read_text())
                    valid+=summary['valid']
                if not valid: raise RuntimeError('Both pilot tasks invalid; inspect browser/API/judge before scaling')
                write_json(root/'pilot-gate.json',dict(passed=True,selection_turns=len(traces),valid_tasks=valid,
                    task_indices=[s['indices'][0] for s in segments]))
        result=report(p); stage='complete'; status(summary=result['summary'],complete=True)
    except BaseException as exc:
        try: partial=report(p,False)
        except Exception: partial={}
        status(failed=True,error_type=type(exc).__name__,error=str(exc)[:300],partial=partial.get('summary'))
        raise
    finally:
        for child in reversed(children):
            if child.poll() is None:
                try: os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError: pass
        for child in reversed(children):
            try: child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try: os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                child.wait()
        for handle in handles: handle.close()
        for s,h in zip((signal.SIGTERM,signal.SIGINT),old): signal.signal(s,h)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id',default='APPROVED_JOB'); parser.add_argument('--execute',action='store_true')
    parser.add_argument('--minutes',type=int,default=120)
    a=parser.parse_args(); p=plan(a.job_id,a.minutes)
    if a.execute: execute(p)
    else: print(json.dumps(p,indent=2))
