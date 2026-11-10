#!/usr/bin/env python3
"""One conditional submission explicitly approved on 2026-09-14; no retries."""
import json,os,re,subprocess,time
from pathlib import Path
R=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
REPO=Path('/gpfs/projects/krishna/zixianma/OpenWebRL')
ROOT=R/'evaluations/arm-failure-additive-295786'
CONTROL=R/'arm-turn-bonus-preparation/additive-to20-approved'
CONTROL.mkdir(parents=True,exist_ok=True)

def write(name,data):
    p=CONTROL/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(p)

def main():
    try:(CONTROL/'owner').mkdir()
    except FileExistsError:return
    write('status.json',dict(stage='waiting_for_test',test_job='295786',approved_gpus=4,approved_hours=[24,30],target_iteration=20))
    while True:
        state=json.loads((ROOT/'status.json').read_text()) if (ROOT/'status.json').exists() else {}
        if state.get('failed'):raise RuntimeError('Test failed; no long allocation submitted')
        if state.get('stage')=='complete':break
        slurm=subprocess.check_output(['sacct','-j','295786','-X','-n','--format=State'],text=True).strip()
        if any(x in slurm for x in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY')):raise RuntimeError('Test allocation ended unsuccessfully')
        time.sleep(60)
    if not state.get('has_saved_checkpoints'):raise RuntimeError('No durable test checkpoint')
    checkpoint=Path(state['last_valid_checkpoint']);rid=int(checkpoint.name.split('_')[1]);folder=ROOT/'iterations'/f'{rid:04d}'
    validation=json.loads((folder/'checkpoint-validation.json').read_text())
    calibration=json.loads((folder/'calibration.json').read_text())
    auxiliary=json.loads((folder/'failure_auxiliary.json').read_text())
    if not calibration['decision']['passed'] or auxiliary['failure_groups']<=0 or not auxiliary['records']:
        raise RuntimeError('No validated nonempty additive batch; long run withheld')
    if validation['completed_optimizer_updates']<=0:raise RuntimeError('No completed Adam updates')
    # A checkpoint beyond this batch requires the actual auxiliary actor path
    # (including its manifest validation and backward pass) to have returned.
    saved=json.loads((folder/'checkpoint-saved.json').read_text())['saved_at_epoch_seconds']
    cycle_seconds=calibration['elapsed_collection_seconds']+max(0,saved-(folder/'training_gate.json').stat().st_mtime)
    predicted_hours=(20-rid-1)*cycle_seconds/3600*1.2+1
    hours=24 if predicted_hours<=24 else 30
    import sys
    sys.path.insert(0,str(REPO/'scripts'))
    from resume_arm_failure_variants import plan,validate_plan
    p=plan('PREPARE',ROOT,hours);validate_plan(p)
    write('decision.json',dict(test_completed_iteration=rid+1,checkpoint=str(checkpoint),measured_cycle_seconds=cycle_seconds,
        projected_hours_with_20percent_margin=predicted_hours,selected_gpus=4,selected_hours=hours,
        target_iteration=20,approval='User approved both continuations, including additive 4 GPUs x24–30 hours, on 2026-09-14.'))
    env=os.environ.copy();env.pop('WANDB_SERVICE',None)
    command=['sbatch','--parsable','--job-name=arm-additive-to20',f'--time={hours}:00:00',
        f'--export=ALL,ARM_VARIANT_RESUME_FROM={ROOT},ARM_VARIANT_HOURS={hours}',
        str(REPO/'scripts/resume_arm_failure_variants_4gpu.sbatch')]
    result=subprocess.run(command,env=env,cwd=REPO,text=True,capture_output=True)
    write('submission.json',dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    if result.returncode:raise RuntimeError('Submission failed; no automatic resubmission')
    job=result.stdout.strip().splitlines()[-1].split(';')[0]
    if not re.fullmatch(r'\d+',job):raise RuntimeError('Cannot parse submitted job ID; inspect receipt before any action')
    write('status.json',dict(stage='submitted',job_id=job,gpus=4,hours=hours,target_iteration=20,checkpoint=str(checkpoint)))
    write('pointer.json',dict(experiment='additive failure',active_test='295786',next_job_id=job,next_job_status='submitted',checkpoint=str(checkpoint)))
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write('status.json',dict(stage='blocked',error=repr(exc),automatic_resubmission=False));raise
