#!/usr/bin/env python3
"""Recover a verified startup/evaluation failure within unused approved GPU time.

Requires a diagnosed repair and refreshed CPU checks. This does not authorize
new budgets, retraining after optimizer updates, or retrying unknown job state.
"""
import argparse,fcntl,json,shutil,subprocess,time
from pathlib import Path
from run_arm_refresh_training import C,CONTROL,completed_training_receipt
from resume_baseline import write_json

REPO=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repair-note',required=True)
    parser.add_argument('--submit',action='store_true')
    parser.add_argument('--evaluation-only',action='store_true')
    args=parser.parse_args()
    lock=(C/'recovery.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    approval=json.loads((CONTROL/'baseline40-approval.json').read_text())
    assert approval['approved']
    original=[s for s in approval['proposal']['gpu_stages'] if s['name']=='LLaMA-Factory ARM refresh and offline evaluation']
    assert len(original)==1 and original[0]['gpus']==1 and original[0]['hours']==4
    assert json.loads((C/'environment-readiness.json').read_text())['passed']
    ready=json.loads((C/'training-readiness.json').read_text())
    assert ready['passed'] and ready['expected_updates']==90
    receipt=json.loads((C/'training-submission.json').read_text());old=receipt['job_id']
    worker=json.loads((C/'training-status.json').read_text())
    assert worker['job']==old and worker['stage']=='failed'
    trained=None
    if args.evaluation_only:trained=completed_training_receipt(C/'adapter')
    elif list((C/'adapter').glob('checkpoint-*')) or (C/'adapter/trainer_state.json').exists():
        raise ValueError('Training has checkpoints; prepare state-preserving resume instead')
    rows=subprocess.check_output(['sacct','-j',old,'-X','-n','-P','-o','JobID,State,ElapsedRaw,ExitCode'],text=True).strip().splitlines()
    assert len(rows)==1
    ident,state,elapsed,exitcode=rows[0].split('|')[:4]
    assert ident==old and state=='FAILED' and int(elapsed)>0
    prior=receipt.get('prior_attempts',[])+[dict(job_id=old,state=state,elapsed_seconds=int(elapsed),exit_code=exitcode)]
    assert len({r['job_id'] for r in prior})==len(prior)
    remaining=14400-sum(r['elapsed_seconds'] for r in prior)
    limit=(remaining//60)*60
    assert 0<limit<14400
    time_limit=f'{limit//3600:02d}:{limit%3600//60:02d}:00'
    plan=dict(previous_job=old,gpus=1,cpus=8,memory_gib=120,time_limit_seconds=limit,
        prior_attempts=prior,repair_note=args.repair_note,approved_total_seconds=14400,
        authorization='Original baseline40 approval plus AGENTS.md standing repair/relaunch within remaining resources')
    plan.update(mode='evaluation' if args.evaluation_only else 'full')
    if trained:plan.update(trained_model_sha256=trained,evaluation_wandb_lineage=receipt.get('evaluation_wandb_lineage',old))
    print(json.dumps(plan,indent=2),flush=True)
    if not args.submit:return
    intent=C/f'recovery-submission-{old}.intent.json'
    if intent.exists():raise ValueError('Reconcile existing recovery intent before any further submission')
    archive=C/'attempts'/old;archive.mkdir(parents=True,exist_ok=True)
    for name in ('training-status.json','training-submission.json','gpu-smoke.log','environment.log','evaluation.log','training-readiness.json'):
        p=C/name
        if p.exists() and not (archive/name).exists():shutil.copy2(p,archive/name)
    write_json(intent,dict(plan,created_epoch=time.time()))
    jobname='arm-refresh-offline-eval' if args.evaluation_only else 'arm-refresh-lf-sft'
    job=subprocess.check_output(['sbatch','--parsable','--hold',f'--time={time_limit}',f'--job-name={jobname}',
        str(REPO/'scripts/run_arm_refresh_training_1gpu.sbatch')],text=True).strip().split(';')[0]
    if not job.isdigit():raise ValueError('Unknown sbatch result; reconcile intent')
    write_json(C/'training-submission.json',dict(plan,job_id=job,submitted_epoch=time.time()))
    write_json(C/'training-status.json',dict(stage='queued',job=job,complete=False))
    subprocess.run(['scontrol','release',job],check=True)
    write_json(C/'retry-request.json',dict(job_id=job,reason=args.repair_note,created_epoch=time.time()))
    print('Released bounded recovery job',job,flush=True)

if __name__=='__main__':main()
