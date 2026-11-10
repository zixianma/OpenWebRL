#!/usr/bin/env python3
"""Persistent approved candidate -> GPT5.5 Batch -> LF SFT/eval handoff.

Owns no GPUs while waiting for labels. Stops mutations on errors; no blind
request or job retries. Atomic receipts and a flock prevent duplicate charges.
"""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
from prepare_arm_forward_transfer import sha,write

REPO=Path(__file__).resolve().parents[1]
R=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime');CONTROL=R/'arm-turn-bonus-preparation/arm-refresh-20260924';C=CONTROL/'baseline40'
PY=R/'arm-reproduction/venv/bin/python';LFPY=R/'arm-reproduction/venv-refresh/bin/python'
ARM=R/'arm-reproduction/action-reward-models-refresh'

def command(cmd,phase):
    update(phase)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
        PYTHONPATH=str(REPO/'scripts')+':'+str(REPO))
    with (C/(phase+'.log')).open('a') as log:
        subprocess.run([str(x) for x in cmd],cwd=REPO,env=env,check=True,stdout=log,stderr=subprocess.STDOUT)

def update(stage,**extra):
    write(C/'pipeline-status.json',dict(stage=stage,updated_epoch=time.time(),watcher_pid=os.getpid(),**extra))

def queue_state(job):
    result=subprocess.run(['squeue','-h','-j',job,'-o','%T|%M|%R'],text=True,capture_output=True)
    if result.returncode and 'Invalid job id' not in result.stderr:
        raise RuntimeError('Slurm queue lookup failed: '+result.stderr[:250])
    return result.stdout.strip()

def training_progress():
    result={}
    log=C/'adapter/trainer_log.jsonl'
    if log.exists():
        for line in reversed(log.read_text().splitlines()):
            try:result['latest_metrics']=json.loads(line);break
            except json.JSONDecodeError:continue  # Worker may be appending.
    checkpoints=[]
    for path in (C/'adapter').glob('checkpoint-*'):
        needed=('adapter_model.safetensors','optimizer.pt','scheduler.pt','trainer_state.json','rng_state.pth')
        if not all((path/name).exists() and (path/name).stat().st_size>0 for name in needed):continue
        try:checkpoints.append(json.loads((path/'trainer_state.json').read_text())['global_step'])
        except json.JSONDecodeError:continue
    result['durable_checkpoint_steps']=sorted(checkpoints)
    result['evaluation_saved_records']={}
    for name in ('frozen','adapted'):
        path=C/'evaluation'/(name+'.jsonl');ids=set()
        if path.exists():
            for line in path.read_text().splitlines():
                try:ids.add(json.loads(line)['pass_id'])
                except json.JSONDecodeError:continue
        result['evaluation_saved_records'][name]=len(ids)
    return result

def tick():
    submission=json.loads((C/'candidate-submission.json').read_text());job=submission['job_id']
    root=R/f'evaluations/arm-refresh-candidates-{job}'
    result=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
    if not result.get('complete'):
        slurm=queue_state(job)
        progress=json.loads((root/'baseline40/progress.json').read_text()) if (root/'baseline40/progress.json').exists() else {}
        if result.get('stage')=='failed':raise RuntimeError('Candidate worker failed; inspect preserved logs')
        if not slurm:raise RuntimeError('Candidate allocation ended without verified completion')
        update('candidates',job=job,slurm=slurm,progress=progress);return False
    if not (C/'data/manifest.json').exists():
        command([PY,REPO/'scripts/assemble_arm_refresh_data.py','--job-id',job],'assemble')
    if not (C/'teacher/manifest.json').exists():
        if (C/'teacher').exists():raise RuntimeError('Partial teacher request build; reconcile before retry')
        command([PY,REPO/'scripts/prepare_arm_refresh_teacher.py','--data',C/'data','--output',C/'teacher'],'prepare-teacher')
    manifest=json.loads((C/'teacher/manifest.json').read_text())
    approved=json.loads((CONTROL/'baseline40-approval.json').read_text())
    if not approved['approved'] or manifest['cost_ceiling_usd']>approved['teacher_api_cap_usd']:
        raise RuntimeError('Teacher reservation exceeds approved cap; no API requests sent')
    api_approval=C/'teacher/approval.json'
    if not api_approval.exists():
        write(api_approval,dict(teacher_approved=True,model='gpt-5.5',api_budget_usd=approved['teacher_api_cap_usd'],
            requests_manifest_sha256=sha((C/'teacher/manifest.json').read_bytes()),
            authorization=str(CONTROL/'baseline40-approval.json'),reserved_usd=manifest['cost_ceiling_usd']))
    label=[PY,REPO/'scripts/label_arm_refresh.py']
    opts=['--root',C/'teacher','--approval',api_approval]
    if not (C/'teacher/pilot-passed.json').exists():command(label+['pilot']+opts,'teacher-pilot')
    chunks=list((C/'teacher').glob('batch-*.jsonl'))
    if any(not p.with_suffix('.submitted.json').exists() for p in chunks):command(label+['submit']+opts,'teacher-submit')
    status=json.loads((C/'teacher/label-status.json').read_text())
    if not status['complete']:
        command(label+['collect']+opts,'teacher-collect')
        status=json.loads((C/'teacher/label-status.json').read_text())
        if status['errors']:
            command(label+['recover']+opts,'teacher-recover-length')
            status=json.loads((C/'teacher/label-status.json').read_text())
            if status['errors']:raise RuntimeError('Unrecovered teacher errors require review; no substitute labels')
        if not status['complete']:
            batches=[json.loads(p.read_text()) for p in (C/'teacher').glob('batch-*.submitted.status.json')]
            states=[b['status'] for b in batches]
            if any(s in ('expired','failed','cancelled') for s in states):raise RuntimeError('Batch labeling ended incompletely')
            update('waiting-for-teacher',valid=status['valid'],expected=status['expected'],gpu_waiting=False,
                reserved_usd=status.get('total_reserved_usd',manifest['cost_ceiling_usd']),
                api_cap_usd=approved['teacher_api_cap_usd'],
                provider_completed_requests=sum((b.get('request_counts') or {}).get('completed',0) for b in batches),
                provider_failed_requests=sum((b.get('request_counts') or {}).get('failed',0) for b in batches));return False
    if not (C/'training-readiness.json').exists():
        command([LFPY,ARM/'training/llamafactory/prepare_refresh.py'],'prepare-training')
    receipt=C/'training-submission.json'
    if not receipt.exists():
        intent=C/'training-submission.intent.json'
        if intent.exists():raise RuntimeError('Unreconciled sbatch intent; prevent duplicate GPU submission')
        ready=json.loads((C/'training-readiness.json').read_text())
        if not ready['passed']:raise RuntimeError('Training readiness failed')
        write(intent,dict(approved_by=str(CONTROL/'baseline40-approval.json'),gpus=1,hours=4,cpus=8,memory_gib=120,created_epoch=time.time()))
        jobid=subprocess.check_output(['sbatch','--parsable',str(REPO/'scripts/run_arm_refresh_training_1gpu.sbatch')],text=True).strip().split(';')[0]
        if not jobid.isdigit():raise ValueError('Unexpected sbatch receipt')
        write(receipt,dict(job_id=jobid,submitted_epoch=time.time(),gpus=1,hours=4))
    training=json.loads(receipt.read_text());job=training['job_id']
    state=json.loads((C/'training-status.json').read_text()) if (C/'training-status.json').exists() else {}
    if state.get('job')!=job:state={}
    if state.get('complete'):
        report=json.loads((C/'evaluation/results.json').read_text())
        if not report.get('complete'):raise RuntimeError('Final report incomplete')
        update('complete',candidate_job=submission['job_id'],training_job=job,results=str(C/'evaluation/results.json'));return True
    if state.get('stage')=='failed':raise RuntimeError('Training/eval worker failed; inspect preserved stage logs')
    slurm=queue_state(job)
    if not slurm:
        accounting=subprocess.run(['sacct','-j',job,'-X','-n','-P','-o','JobID,State,ElapsedRaw,ExitCode'],text=True,capture_output=True)
        raise RuntimeError('Training/eval ended without complete results: '+accounting.stdout.strip()[:250])
    update('training-evaluation',job=job,slurm=slurm,worker=state,progress=training_progress());return False

def main():
    lock=(C/'watcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(C/'watcher.json',dict(pid=os.getpid(),started_epoch=time.time(),expires_epoch=time.time()+72*3600,
        observation=True,approved_stage_handoffs=True,automatic_bug_repair=False))
    end=time.monotonic()+72*3600;blocked=None;retry_seen=0
    while time.monotonic()<end:
        retry=C/'retry-request.json';stamp=retry.stat().st_mtime if retry.exists() else 0
        if stamp>retry_seen:blocked=None;retry_seen=stamp
        if blocked:update('needs-attention',error=blocked,mutations_stopped=True)
        else:
            try:
                if tick():return
            except Exception as exc:
                blocked=f'{type(exc).__name__}: {str(exc)[:500]}'
                update('needs-attention',error=blocked,mutations_stopped=True)
        # Bounded one-minute sleeps keep the persistent process interruptible.
        for _ in range(10):time.sleep(60)
    update('monitor-expired',complete=False)

if __name__=='__main__':main()
