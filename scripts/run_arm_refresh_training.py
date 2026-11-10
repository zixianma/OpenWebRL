#!/usr/bin/env python3
"""Await LF GPU smoke, standard SFT and paired evaluation in one allocation."""
import hashlib,json,os,re,subprocess,sys
from pathlib import Path
from run_arm_quality_audit import environment
from resume_baseline import write_json

R=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime');CONTROL=R/'arm-turn-bonus-preparation/arm-refresh-20260924';C=CONTROL/'baseline40'
ARM=R/'arm-reproduction/action-reward-models-refresh';LF=ARM/'training/llamafactory';PY=R/'arm-reproduction/venv-refresh/bin/python'

def completed_training_receipt(root):
    """Verify the predefined endpoint without loading tensors or retraining."""
    state=json.loads((root/'trainer_state.json').read_text())
    if state.get('global_step')!=90 or state.get('max_steps')!=90 or state.get('epoch')!=1.0:
        raise ValueError('Evaluation requires the completed predefined90-update endpoint')
    checkpoint=root/'checkpoint-90'
    if json.loads((checkpoint/'trainer_state.json').read_text()).get('global_step')!=90:
        raise ValueError('Final durable checkpoint is incomplete')
    for name in ('optimizer.pt','scheduler.pt','rng_state.pth'):
        if not (checkpoint/name).is_file() or (checkpoint/name).stat().st_size==0:
            raise ValueError('Final optimizer/scheduler/RNG state is missing')
    hashes={}
    for name in ('adapter_model.safetensors','adapter_config.json','trainer_state.json','train_results.json','eval_results.json'):
        path=root/name
        if not path.is_file() or path.stat().st_size==0:raise ValueError('Missing final artifact: '+name)
        with path.open('rb') as stream:hashes[name]=hashlib.file_digest(stream,'sha256').hexdigest()
    with (checkpoint/'adapter_model.safetensors').open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=hashes['adapter_model.safetensors']:
            raise ValueError('Exported adapter differs from final checkpoint')
    return hashes

def stage_names(mode):
    if mode=='evaluation':return ('environment','grammar-smoke','evaluation')
    if mode=='full':return ('environment','gpu-smoke','training','grammar-smoke','evaluation')
    raise ValueError('Unknown refresh recovery mode')

def verify_time_budget(fields,job,receipt):
    """Recovery spends only unused time from the originally approved four hours."""
    if receipt.get('job_id')!=job:raise ValueError('Allocation does not match current submission receipt')
    prior=receipt.get('prior_attempts',[])
    if len({p['job_id'] for p in prior})!=len(prior):raise ValueError('Duplicate budget ledger entry')
    spent=sum(int(p['elapsed_seconds']) for p in prior)
    if spent<0 or any(int(p['elapsed_seconds'])<0 for p in prior):raise ValueError('Invalid prior elapsed time')
    requested=int(receipt.get('time_limit_seconds',14400))
    value=fields.get('TimeLimit','')
    if not re.fullmatch(r'\d{2}:\d{2}:\d{2}',value):raise ValueError('Unexpected allocation time format')
    h,m,s=map(int,value.split(':'));actual=h*3600+m*60+s
    if actual!=requested or not 0<actual<=14400-spent:
        raise ValueError('Allocation exceeds remaining approved four-hour budget')

def worker(job):
    if os.getenv('SLURM_JOB_ID')!=job or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():raise ValueError('GPU cgroup required')
    ready=json.loads((C/'training-readiness.json').read_text())
    if not ready['passed'] or ready['expected_updates']!=90:raise ValueError('LF readiness gate missing')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if (sha(C/'data/manifest.json')!=ready['data_manifest_sha256']
        or sha(C/'teacher/labels.jsonl')!=ready['teacher_labels_sha256']):raise ValueError('Approved fit data changed')
    for root,entries in [(C/'llamafactory',ready['files']),(LF,ready['source_files'])]:
        if any(sha(root/name)!=expected for name,expected in entries.items()):raise ValueError('Prepared training inputs/source changed')
    env=environment();env.update(WANDB_PROJECT='openwebrl',WANDB_RUN_ID='arm-refresh-baseline40-gpt55',
        WANDB_RESUME='never',WANDB_DIR=str(C/'wandb'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',PYTHONPATH=str(LF)+':/gpfs/projects/krishna/zixianma/OpenWebRL/scripts:/gpfs/projects/krishna/zixianma/OpenWebRL')
    env['CPATH']=str(R/'src/python-headers/Include')+':'+str(R/'src/python-headers')
    env['TRITON_CACHE_DIR']=str(C/'triton-cache')
    receipt=json.loads((C/'training-submission.json').read_text())
    mode=receipt.get('mode','full')
    if mode=='evaluation':
        if completed_training_receipt(C/'adapter')!=receipt['trained_model_sha256']:
            raise ValueError('Trained endpoint changed since evaluation recovery was prepared')
        env['WANDB_PROJECT']='openwebrl-evals'
    env['ARM_REFRESH_EVAL_LINEAGE']=receipt.get('evaluation_wandb_lineage',job)
    (C/'wandb').mkdir(exist_ok=True)
    commands={'environment':[str(PY),str(Path(__file__).with_name('check_arm_refresh_environment.py'))],
        'gpu-smoke':[str(PY),str(LF/'gpu_smoke_refresh.py')],
        'training':[str(PY),str(LF/'train_refresh.py'),str(LF/'arm_refresh_baseline40.yaml')],
        'grammar-smoke':[str(PY),str(Path(__file__).with_name('check_arm_refresh_grammar.py'))],
        'evaluation':[str(PY),str(LF/'evaluate_refresh.py')]}
    for name in stage_names(mode):
        command=commands[name]
        write_json(C/'training-status.json',dict(stage=name,job=job,complete=False))
        with (C/(name+'.log')).open('w') as log:subprocess.run(command,env=env,cwd=ARM,stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads((C/'evaluation/results.json').read_text())
    if not result['complete']:raise ValueError('Evaluation incomplete')
    write_json(C/'training-status.json',dict(stage='complete',job=job,complete=True,results=str(C/'evaluation/results.json')))

def main():
    job=os.environ.get('SLURM_JOB_ID')
    if not job:raise ValueError('Explicitly approved Slurm job required')
    approval=json.loads((CONTROL/'baseline40-approval.json').read_text())
    if not approval['approved']:raise ValueError('No approval')
    fields=dict(re.findall(r'(\w+)=(\S+)',subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)))
    verify_time_budget(fields,job,json.loads((C/'training-submission.json').read_text()))
    tres=dict(x.split('=',1) for x in fields['AllocTRES'].split(','))
    if (fields.get('JobId')!=job or fields.get('JobState')!='RUNNING' or fields.get('NumNodes')!='1'
        or not fields.get('UserId','').endswith(f'({os.getuid()})') or int(fields.get('NumCPUs','0'))!=8
        or int(tres.get('gres/gpu:h200','0'))!=1 or tres.get('mem')!='120G'):
        raise ValueError('Allocation differs from approved one-H200 four-hour profile')
    try:
        if '--worker' in sys.argv:worker(job)
        else:subprocess.run(['srun',f'--jobid={job}','--nodes=1','--ntasks=1','--cpus-per-task=8','--gres=gpu:h200:1',
            '--exact','--cpu-bind=none',sys.executable,str(Path(__file__).resolve()),'--worker'],check=True)
    except BaseException as exc:
        write_json(C/'training-status.json',dict(stage='failed',job=job,complete=False,error_type=type(exc).__name__,error=str(exc)[:500]));raise

if __name__=='__main__':main()
