#!/usr/bin/env python3
"""Read-only ARM pilot monitoring; no submission, restart, signaling, or training.

Record full snapshots every 15 minutes and on stage changes. Check the small
controller status between snapshots so failures and handoffs surface promptly.
"""
import argparse
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

REPO=Path(__file__).resolve().parents[1]
RUNTIME=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL'}


def read(path):
    try: return json.loads(path.read_text())
    except (OSError,json.JSONDecodeError): return {}


def tail(path,limit=512*1024):
    if not path.exists(): return ''
    with path.open('rb') as handle:
        handle.seek(0,2)
        handle.seek(max(0,handle.tell()-limit))
        return handle.read().decode(errors='replace')


def scheduler(job):
    raw=subprocess.check_output(['sacct','-X','-j',job,'-n','-P',
        '--format=JobIDRaw,State,Elapsed,Timelimit,NodeList,ExitCode'],text=True,timeout=20)
    for line in raw.splitlines():
        fields=line.split('|')
        if fields[0]==job:
            return dict(zip(('job','state','elapsed','limit','node','exit_code'),fields))
    return dict(job=job,state='UNKNOWN')


def resource_sample(job,gpus=4):
    code="""import json,os,subprocess
from pathlib import Path
p=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split('::',1)[1].lstrip('/')
root=next((x for x in [p,*p.parents] if x.name.startswith('job_')),p)
assigned=['--id='+os.environ['CUDA_VISIBLE_DEVICES']] if os.environ.get('CUDA_VISIBLE_DEVICES') else []
print(json.dumps(dict(gpus=subprocess.check_output(['nvidia-smi',*assigned,'--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True,timeout=10).strip().splitlines(),memory={k:(root/k).read_text().strip() for k in ['memory.current','memory.peak','memory.max','memory.events'] if (root/k).exists()})))
"""
    command=['srun','--overlap',f'--jobid={job}','--nodes=1','--ntasks=1','--cpus-per-task=1',
        '--mem=1G',f'--gres=gpu:h200:{gpus}','--exact',str(RUNTIME/'venv/bin/python'),'-c',code]
    # A monitor owned by the training worker already has the allocation's
    # GPU/cgroup access. Nested srun can inherit incompatible step settings.
    if os.environ.get('SLURM_JOB_ID')==job and f'/job_{job}/' in Path('/proc/self/cgroup').read_text():
        command=[str(RUNTIME/'venv/bin/python'),'-c',code]
    try:
        proc=subprocess.run(command,text=True,capture_output=True,timeout=30)
        return json.loads(proc.stdout) if proc.returncode==0 else dict(error=f'srun return code {proc.returncode}')
    except (OSError,ValueError,subprocess.TimeoutExpired) as exc:
        return dict(error=type(exc).__name__)


def run_root(job):
    controllers=[RUNTIME/f'evaluations/arm-gate-b-recovery-{job}/controller-plan.json']
    controllers.extend((RUNTIME/'evaluations').glob(f'arm-ablation-*-{job}/controller-plan.json'))
    for controller in controllers:
        plan=read(controller);target=plan.get('current_target')
        expected=RUNTIME/f'evaluations/arm-failure-additive-{job}-tp2-iter{target}'
        if (plan.get('job_id')==job and target in (30,40,50,60,70,80,90)
                and plan.get('training_root')==str(expected)):
            return expected
    b90=RUNTIME/f'evaluations/arm-gate-b-to90-{job}/controller-plan.json'
    if b90.is_file():
        plan=read(b90);target=plan.get('current_target')
        expected=RUNTIME/f'evaluations/arm-failure-additive-{job}-tp2-iter{target}'
        if target in (70,80,90) and plan.get('training_root')==str(expected):return expected
    beta=RUNTIME/f'evaluations/arm-beta-to40-{job}/controller-plan.json'
    if beta.is_file():
        plan=read(beta);target=plan.get('current_target')
        expected=RUNTIME/f'evaluations/arm-failure-additive-{job}-tp2-iter{target}'
        if target in (30,40) and plan.get('training_root')==str(expected):return expected
    recovery=RUNTIME/f'evaluations/arm-coverage-recovery-{job}/controller-plan.json'
    if recovery.is_file():
        root=read(recovery).get('training_root')
        expected=RUNTIME/f'evaluations/arm-failure-additive-{job}-tp2-iter20'
        if root and Path(root)==expected:return expected
    for controller in (RUNTIME/'evaluations').glob(f'arm-failure-ablation-*-{job}-tp2/controller-plan.json'):
        root=read(controller).get('training_root')
        expected=RUNTIME/f'evaluations/arm-failure-additive-{job}-tp2-iter20'
        if root and Path(root)==expected:return expected
    for controller in (RUNTIME/'evaluations').glob(f'arm-gate-?-to60-{job}/controller-plan.json'):
        root=read(controller).get('training_root')
        if root and Path(root).parent==RUNTIME/'evaluations' and Path(root).name in (
                f'arm-failure-additive-{job}-iter40',f'arm-failure-additive-{job}-iter60',
                f'arm-failure-additive-{job}-tp2-iter40',f'arm-failure-additive-{job}-tp2-iter60'):
            return Path(root)
    coverage=RUNTIME/f'evaluations/arm-failure-coverage-pilot-{job}'
    if coverage.exists(): return coverage
    additive=RUNTIME/f'evaluations/arm-failure-additive-{job}'
    if additive.exists(): return additive
    failure=RUNTIME/f'evaluations/arm-turn-bonus-fresh-allfailure-{job}'
    if failure.exists(): return failure
    sol=RUNTIME/f'evaluations/sol-turn-bonus-fresh-{job}'
    if sol.exists(): return sol
    fresh=RUNTIME/f'evaluations/arm-turn-bonus-fresh-{job}'
    return fresh if fresh.exists() else RUNTIME/f'evaluations/arm-turn-bonus-calibration-{job}'


def wandb_sample(job):
    try:
        os.environ.setdefault('WANDB_CACHE_DIR',str(RUNTIME/'wandb-cache'))
        from dotenv import dotenv_values
        import wandb
        key=dotenv_values(REPO/'.env').get('WANDB_API_KEY')
        api=wandb.Api(api_key=key,timeout=20)
        root=run_root(job)
        manifest=read(root/'launch_manifest.json')
        run_id=manifest.get('wandb_run_id',f'arm-turn-bonus-beta0.5-after70-{job}')
        project=manifest.get('environment',{}).get('WANDB_PROJECT','openwebrl')
        run=api.run(f'zixianma/{project}/{run_id}')
        summary=dict(run.summary)
        return dict(url=run.url,state=run.state,last_history_step=run.lastHistoryStep,
            metrics={k:v for k,v in summary.items() if k.startswith(('arm_collection/','arm_calibration/','train/','rollout/task/'))})
    except Exception as exc:
        # Do not print exception details that could include credential context.
        return dict(error=type(exc).__name__)


def snapshot(job,slurm,remote=False,compare_baseline=True):
    root=run_root(job)
    current=read(root/'arm-config.json')
    iteration=Path(current.get('output',root))
    result=dict(checked_utc=datetime.now(timezone.utc).isoformat(),slurm=slurm,alerts=[])
    for name in ('status','live_metrics','selector_preflight','calibration','collection_complete',
                 'training_gate','trained_checkpoint_validation'):
        result[name]=read((root if name=='status' else iteration)/f'{name}.json')
    result['completed_checkpoints']=read(root/'completed-checkpoints.json')
    text=re.sub(r'\x1b\[[0-9;]*m','',tail(root/'collection.log')+'\n'+tail(root/'runtime/progress.log'))
    rows=[]
    for match in re.finditer(r"model.py:\d+ - step \d+: (\{[^\n]+\})",text):
        try: rows.append(ast.literal_eval(match[1]))
        except (ValueError,SyntaxError): pass
    if not rows:
        for line in text.splitlines():
            if '[TrainMetrics]' not in line: continue
            if not re.search(r'\bstep=\d+\b.*\bgrad_norm=',line): continue
            row={}
            for key,value in re.findall(r'([\w/]+)=([-+\d.eE]+)',line):
                try: row[key if key.startswith('train/') else 'train/'+key]=float(value)
                except ValueError: pass
            if row: rows.append(row)
    # Ray can forward the same event more than once; training step is explicit.
    rows=list({row.get('train/step',i):row for i,row in enumerate(rows)}.values())
    result['latest_train']=rows[-1] if rows else {}
    recent=rows[-5:]
    result['recent_training_steps_in_log_tail']=len(rows)
    result['recent_train_medians']={key:statistics.median(r[key] for r in recent if key in r)
        for key in ('train/loss','train/pg_loss','train/grad_norm','train/pg_clipfrac','train/ppo_kl')
        if any(key in r for r in recent)}
    groups=list(re.finditer(r'\[RolloutProgress\].*accepted_groups=(\d+)/(\d+) completed_groups=(\d+).*samples=(\d+) elapsed_secs=([\d.]+)',text))
    if groups:
        g=groups[-1]
        result['collection_progress']=dict(accepted_groups=int(g[1]),target_groups=int(g[2]),
            completed_groups=int(g[3]),accepted_trajectories=int(g[4]),elapsed_seconds=float(g[5]))
    if result['status'].get('failed') or slurm['state'].split()[0].rstrip('+') in TERMINAL-{'COMPLETED'}:
        result['alerts'].append('Worker or allocation failed; inspect status and logs immediately')
    if re.search(r"(?:grad_norm|train/(?:loss|pg_loss))['\"\s:=]+(?:nan|[+-]?inf)\b",text,re.I):
        result['alerts'].append('Nonfinite training loss/gradient; controller should stop worker')
    if compare_baseline and len(recent)>=3:
        if all(r.get('train/grad_norm',0)>3*1.079 for r in recent[-3:]):
            result['alerts'].append('Three gradients exceed 3x historical baseline median 1.079; review')
        if all(r.get('train/pg_clipfrac',0)>.10 for r in recent[-3:]):
            result['alerts'].append('Three clipping fractions exceed 10%; review')
    status_path=root/'status.json'
    worker_active=result['status'].get('stage')!='complete' and not result['status'].get('failed')
    if slurm['state']=='RUNNING' and worker_active and status_path.exists() and time.time()-status_path.stat().st_mtime>900:
        result['alerts'].append('Controller status has not updated for 15 minutes')
    if remote:
        # Do not create an overlapping telemetry step until the controller has
        # completed its exclusive-start check and written its first status.
        if slurm['state']=='RUNNING' and result['status'].get('stage'):
            gpus=read(root/'launch_manifest.json').get('requested_resources',{}).get('gpus',4)
            result['resources']=resource_sample(job,gpus=gpus)
        # Before worker startup there is no W&B run to query.
        if result['live_metrics'] or rows: result['wandb']=wandb_sample(job)
        online=result.get('wandb',{}).get('metrics',{})
        local=result['live_metrics'].get('arm_collection/elapsed_seconds',0)
        synced=online.get('arm_collection/elapsed_seconds')
        if synced is not None and local-synced>900:
            result['alerts'].append('W&B collection metrics lag local metrics by over 15 minutes')
    return result


def record(report,output):
    output.mkdir(parents=True,exist_ok=True)
    temporary=output/'latest.partial'
    temporary.write_text(json.dumps(report,indent=2)+'\n')
    temporary.replace(output/'latest.json')
    with (output/'history.jsonl').open('a') as handle: handle.write(json.dumps(report)+'\n')
    print(json.dumps({k:report[k] for k in ('checked_utc','slurm','alerts','collection_progress',
        'latest_train','recent_train_medians') if k in report}|dict(stage=report['status'].get('stage'),
        labels=report['live_metrics'].get('arm_collection/admitted_turns'),
        training_complete=report['status'].get('training_complete'))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job-id',required=True)
    p.add_argument('--watch',action='store_true')
    p.add_argument('--interval',type=int,default=900)
    p.add_argument('--hours',type=float,default=6)
    args=p.parse_args()
    if not re.fullmatch(r'\d+',args.job_id): raise ValueError('Numeric Slurm job id required')
    output=RUNTIME/f'arm-turn-bonus-preparation/monitor-{args.job_id}'
    last_key=None;next_full=0;next_scheduler=0;slurm={}
    end=time.monotonic()+args.hours*3600
    while time.monotonic()<end:
        now=time.monotonic()
        if now>=next_scheduler:
            try: slurm=scheduler(args.job_id)
            except (OSError,subprocess.TimeoutExpired,subprocess.CalledProcessError):
                slurm=dict(job=args.job_id,state='UNKNOWN')
            next_scheduler=now+(60 if slurm['state'] in ('PENDING','CONFIGURING','UNKNOWN') else 300)
        state=read(run_root(args.job_id)/'status.json')
        key=(slurm.get('state'),state.get('stage'),state.get('failed'),state.get('training_complete'),
            state.get('iteration'),state.get('completed_iterations'))
        finished=slurm.get('state','').split()[0].rstrip('+') in TERMINAL
        if key!=last_key or now>=next_full or not args.watch:
            report=snapshot(args.job_id,slurm,remote=True)
            record(report,output)
            last_key=key;next_full=time.monotonic()+args.interval
        if finished or not args.watch or state.get('stage')=='complete' or state.get('failed'): break
        # Cheap file-only watch for handoffs/failure; full telemetry uses 15 min.
        time.sleep(30)


if __name__=='__main__': main()
