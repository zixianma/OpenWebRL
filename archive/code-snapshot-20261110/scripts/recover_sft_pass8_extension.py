#!/usr/bin/env python3
"""Run a hash-pinned collector repair under the unchanged pass@8 campaign caps.

Never submits allocations and never changes the campaign plan or a live source.
Only the collector and this new entry point differ in the recovery source. Task,
checkpoint, judge, sampling, output, and persistent cap paths remain unchanged.
"""
from __future__ import annotations
import argparse
import copy
import difflib
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

REPO=Path('/gpfs/projects/krishna/zixianma/OpenWebRL')
RUNTIME=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
ROOT=RUNTIME/'evaluations/sft-training-pass8-20261006/extension3-misses-v1'
PARENT=RUNTIME/'reference-sft-pass8-misses-20261006-v1'
SOURCE=RUNTIME/'reference-sft-pass8-misses-20261006-v3'
SUPERSEDED=RUNTIME/'reference-sft-pass8-misses-20261006-v2'
SUPERSEDED_COLLECTOR_SHA='40151d27abe105226186c7ea4c5f23f0a9a38ef4e664b7738040f39674b7eb43'
RECOVERY_VERSION='native-invalid-stubs-v3'
COLLECTOR='openwebrl/arm_task_pass8.py'
ENTRY='scripts/recover_sft_pass8_extension.py'


def load_base():
    sys.path.insert(0,str(PARENT/'scripts'))
    spec=importlib.util.spec_from_file_location('pinned_pass8_campaign',PARENT/'scripts/run_sft_pass8_extension.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def stable_config(config):
    return {key:value for key,value in config.items() if key not in ('job_id','attempt_output')}


def validate_overlay(parent_manifest, overlay_manifest, source):
    before=parent_manifest['recipe_files_sha256'];after=overlay_manifest['recipe_files_sha256']
    if set(after)!=set(before)|{ENTRY}:
        raise ValueError('Recovery source has unexpected recipe additions or deletions')
    changed={name for name in before if before[name]!=after[name]}
    if changed!={COLLECTOR}:raise ValueError('Only the collector verification repair may change')
    for name,expected in after.items():
        path=(Path(source)/name).resolve()
        if not path.is_relative_to(Path(source).resolve()) or sha(path)!=expected:
            raise ValueError('Recovery source hash mismatch or path escape: '+name)
    return changed


def recommended_seconds(remaining):
    if type(remaining) is not int or remaining<600:raise ValueError('Insufficient approved time for recovery')
    return (remaining//60)*60


def verifier_chain():
    parent=read(PARENT/'reference_manifest.json')
    previous=read(SUPERSEDED/'reference_manifest.json')
    current=read(SOURCE/'reference_manifest.json')
    validate_overlay(parent,previous,SUPERSEDED)
    validate_overlay(parent,current,SOURCE)
    if sha(SUPERSEDED/COLLECTOR)!=SUPERSEDED_COLLECTOR_SHA:
        raise ValueError('Superseded verified collector identity changed')
    before=previous['recipe_files_sha256'];after=current['recipe_files_sha256']
    changed=sorted(name for name in before if before[name]!=after[name])
    if set(changed)!={COLLECTOR,ENTRY}:
        raise ValueError('V3 must change only the verifier and its versioned controller')
    return dict(original_source=str(PARENT),original_source_manifest_sha256=sha(PARENT/'reference_manifest.json'),
        previous_source=str(SUPERSEDED),previous_source_manifest_sha256=sha(SUPERSEDED/'reference_manifest.json'),
        previous_collector_sha256=sha(SUPERSEDED/COLLECTOR),
        current_source=str(SOURCE),current_source_manifest_sha256=sha(SOURCE/'reference_manifest.json'),
        current_collector_sha256=sha(SOURCE/COLLECTOR),changed_from_previous=changed,
        scientific_protocol_unchanged=True)


def overlay_launch(base,plan,recovery,job):
    launch=base.make_launch(plan,recovery['shard'],str(job))
    original=copy.deepcopy(launch['collector_config'])
    if stable_config(original)!=read(ROOT/f'shard-{recovery["shard"]}/collector-config.json'):
        raise ValueError('Recovery would change bound task, history, output, or budget paths')
    stage=launch['stage']
    if stage['command'][0:2]!=['bash',str(PARENT/'scripts/run_h200_browser.sh')]:
        raise ValueError('Unexpected native launch chain')
    stage['command'][1]=str(SOURCE/'scripts/run_h200_browser.sh')
    stage.update(source=str(SOURCE),campaign_source=str(PARENT),
        recovery_manifest=recovery['manifest_path'],recovery_manifest_sha256=digest(recovery),
        collector_sha256=recovery['collector_after_sha256'])
    launch.update(recovery_manifest=recovery['manifest_path'],recovery_manifest_sha256=digest(recovery),
                  execution_source=str(SOURCE),execution_source_manifest_sha256=recovery['source_manifest_sha256'])
    assert launch['collector_config']==original
    return launch


def preserve_inventory(shard,job):
    root=ROOT/f'shard-{shard}'
    canonical=sorted((root/'records').glob('*.json'))
    dispatches=sorted((root/'physical-attempts').glob('*/dispatch.json'))
    saved=sorted((root/'attempts'/job).glob('*/screen_records/*.json'))
    files={str(path):sha(path) for path in canonical+dispatches+saved}
    canonical_names={path.name for path in canonical}
    orphans=[dict(path=str(path),sha256=sha(path),task_id=read(path)['task_id'],attempt=read(path)['attempt'])
             for path in saved if path.name not in canonical_names]
    budget=root/'judge-budget/ledger.json'
    return dict(canonical_records_before=len(canonical),physical_dispatches_before=len(dispatches),
        saved_records_in_failed_job=len(saved),saved_orphans=orphans,immutable_files_sha256=files,
        judge_ledger_before=dict(path=str(budget),sha256=sha(budget),contents=read(budget)))


def prepare(shard,failed_job):
    base=load_base();plan=base.validate_prepared();approval=read(ROOT/'approval.json')
    attempts=[row for row in approval['attempts'] if row['shard']==shard]
    if not attempts or str(attempts[-1]['job_id'])!=failed_job:
        raise ValueError('Recovery must target the latest registered shard attempt')
    used,charges=base.accounted_seconds(approval,shard,'PREPARE')
    current=[row for row in charges if row['job_id']==failed_job]
    if len(current)!=1 or current[0]['state'].split()[0].rstrip('+')=='COMPLETED':
        raise ValueError('Requires a finalized failed/interrupted scheduler attempt')
    remaining=base.RESOURCES['total_seconds_per_shard']-used
    directory=ROOT/f'recoveries/{RECOVERY_VERSION}/shard-{shard}-after-{failed_job}'
    manifest_path=directory/'recovery.json'
    if manifest_path.exists():return validate_recovery(manifest_path)[2]
    patched=REPO/COLLECTOR
    if sha(patched)==sha(PARENT/COLLECTOR):raise ValueError('Refuse unchanged failure replay')
    if not SOURCE.exists():
        shutil.copytree(PARENT,SOURCE,symlinks=True,
            ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git','.browser_use_sessions'))
        shutil.copyfile(patched,SOURCE/COLLECTOR);shutil.copyfile(REPO/ENTRY,SOURCE/ENTRY)
        overlay=read(PARENT/'reference_manifest.json')
        overlay['recipe_files_sha256'].update({COLLECTOR:sha(SOURCE/COLLECTOR),ENTRY:sha(SOURCE/ENTRY)})
        overlay['pass8_collector_recovery']=dict(parent=str(PARENT),
            parent_manifest_sha256=sha(PARENT/'reference_manifest.json'),
            changed_existing_files=[COLLECTOR],added_files=[ENTRY],
            scientific_protocol_unchanged=True,global_plan_unchanged=True,
            supersedes_verifier_source=str(SUPERSEDED),
            supersedes_source_manifest_sha256=sha(SUPERSEDED/'reference_manifest.json'),
            supersedes_collector_sha256=sha(SUPERSEDED/COLLECTOR))
        base.write(SOURCE/'reference_manifest.json',overlay)
    overlay=read(SOURCE/'reference_manifest.json')
    validate_overlay(read(PARENT/'reference_manifest.json'),overlay,SOURCE)
    if sha(patched)!=sha(SOURCE/COLLECTOR) or sha(REPO/ENTRY)!=sha(SOURCE/ENTRY):
        raise ValueError('Recovery source already frozen; create a new version after edits')
    base.validate_launch_shells(SOURCE)
    chain=verifier_chain()
    recovery=dict(manifest_path=str(manifest_path),campaign_root=str(ROOT),shard=shard,
        failed_job=failed_job,campaign_plan_sha256=base.digest(plan),source=str(SOURCE),
        source_manifest_sha256=sha(SOURCE/'reference_manifest.json'),parent_source=str(PARENT),
        parent_source_manifest_sha256=sha(PARENT/'reference_manifest.json'),
        recovery_controller_sha256=sha(SOURCE/ENTRY),collector_before_sha256=sha(PARENT/COLLECTOR),
        collector_after_sha256=sha(SOURCE/COLLECTOR),prior_charged_seconds=used,prior_attempts=charges,
        supersedes_verifier_source=str(SUPERSEDED),
        supersedes_source_manifest_sha256=sha(SUPERSEDED/'reference_manifest.json'),
        supersedes_collector_sha256=sha(SUPERSEDED/COLLECTOR),verifier_chain=chain,
        changed_files=[COLLECTOR],added_files=[ENTRY],
        unchanged_scientific_files_sha256=read(base.PARENT/'reference_manifest.json')['recipe_files_sha256'],
        remaining_scheduler_seconds=remaining,recommended_time_limit_seconds=recommended_seconds(remaining),
        resources=plan['resources'],limits=plan['limits'],preserved=preserve_inventory(shard,failed_job),
        scientific_protocol_unchanged=True,optimizer_updates=0,added_budget=False,
        reason='Accept strict native timeout and local environment-server health-failure invalid stubs without relaxing successful/valid trajectory checks')
    directory.mkdir(parents=True,exist_ok=True)
    patch=''.join(difflib.unified_diff((PARENT/COLLECTOR).read_text().splitlines(True),
        (SOURCE/COLLECTOR).read_text().splitlines(True),fromfile='frozen-v1/'+COLLECTOR,tofile='recovery-v3/'+COLLECTOR))
    (directory/'collector.patch').write_text(patch)
    recovery['collector_patch_sha256']=sha(directory/'collector.patch')
    base.freeze(manifest_path,recovery)
    base.freeze(directory/'launch-preview.json',overlay_launch(base,plan,recovery,'PREPARE'))
    return recovery


def validate_recovery(path):
    base=load_base();plan=base.validate_prepared();path=Path(path).resolve();recovery=read(path)
    if (path!=Path(recovery['manifest_path']).resolve() or not path.is_relative_to(ROOT/'recoveries')
            or recovery['campaign_plan_sha256']!=base.digest(plan) or recovery['source']!=str(SOURCE)
            or recovery['resources']!=plan['resources'] or recovery['limits']!=plan['limits']
            or recovery['source_manifest_sha256']!=sha(SOURCE/'reference_manifest.json')
            or recovery['parent_source_manifest_sha256']!=sha(PARENT/'reference_manifest.json')
            or recovery['recovery_controller_sha256']!=sha(SOURCE/ENTRY)
            or recovery['collector_after_sha256']!=sha(SOURCE/COLLECTOR)
            or recovery['collector_before_sha256']!=sha(PARENT/COLLECTOR)
            or recovery['collector_patch_sha256']!=sha(path.parent/'collector.patch')):
        raise ValueError('Recovery receipt/source/protocol identity changed')
    validate_overlay(read(PARENT/'reference_manifest.json'),read(SOURCE/'reference_manifest.json'),SOURCE)
    if (recovery.get('supersedes_verifier_source')!=str(SUPERSEDED)
            or recovery.get('supersedes_source_manifest_sha256')!=sha(SUPERSEDED/'reference_manifest.json')
            or recovery.get('supersedes_collector_sha256')!=SUPERSEDED_COLLECTOR_SHA
            or recovery.get('verifier_chain')!=verifier_chain()):
        raise ValueError('Recovery verifier source chain changed')
    if recovery.get('changed_files')!=[COLLECTOR] or recovery.get('added_files')!=[ENTRY]:
        raise ValueError('Unexpected recovery change scope')
    expected_scientific=read(base.PARENT/'reference_manifest.json')['recipe_files_sha256']
    if recovery.get('unchanged_scientific_files_sha256')!=expected_scientific:
        raise ValueError('Historical scientific proof changed')
    for relative,expected in expected_scientific.items():
        if sha(SOURCE/relative)!=expected:raise ValueError('Historical scientific file changed')
    for original,expected in recovery['preserved']['immutable_files_sha256'].items():
        if sha(original)!=expected:raise ValueError('Preserved prior attempt evidence changed')
    return base,plan,recovery


def native_check(recovery):
    base,plan,recovery=validate_recovery(recovery['manifest_path'])
    from resume_baseline import clean_environment,source_command
    stage=overlay_launch(base,plan,recovery,'PREPARE')['stage'];directory=Path(recovery['manifest_path']).parent
    env=dict(clean_environment(),**stage['environment'],DRY_RUN='1',CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1')
    argv=shlex.split(subprocess.check_output(stage['command'],env=env,text=True,cwd=SOURCE));env.pop('DRY_RUN')
    code="""from unittest.mock import patch
from pathlib import Path
import hashlib,json
from slime.utils.arguments import parse_args
import openwebrl.arm_task_pass8 as collector
with patch('megatron.training.arguments.get_device_arch_version',return_value=9): a=parse_args()
expected={'num_rollout':0,'tensor_model_parallel_size':1,'actor_num_gpus_per_node':1,
'rollout_num_gpus_per_engine':1,'sequence_parallel':False,'browser_rollout_concurrency':8,
'judge_api_model':'gpt-4.1','judge_prompt_variant':'action_history','max_steps':15,
'context_num_screenshots':1,'turn_history_reasoning_mode':'full',
'eval_function_path':'openwebrl.arm_task_pass8.generate_rollout','wandb_project':'openwebrl-evals'}
assert all(getattr(a,k)==v for k,v in expected.items()),{k:getattr(a,k) for k in expected}
assert a.load=='/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT'
d=a.eval_datasets[0]
assert (d.n_samples_per_eval_prompt,d.temperature,d.top_p,d.top_k,d.max_response_len)==(3,.8,1.,-1,1024)
assert Path(collector.__file__).resolve()==Path(SOURCE)/'openwebrl/arm_task_pass8.py'
assert hashlib.sha256(Path(collector.__file__).read_bytes()).hexdigest()==COLLECTOR_SHA
assert str(d.path).startswith(PARENT+'/')
recovery=json.loads(Path(RECOVERY_MANIFEST).read_text())
for orphan in recovery['preserved']['saved_orphans']:
    path=Path(orphan['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==orphan['sha256']
    record=json.loads(path.read_text())
    collector.verify_record(record,record['task_id'],record['attempt'],deep=True)
print('SFT_PASS8_RECOVERY_SAVED_ORPHANS_DEEP_VERIFIED',len(recovery['preserved']['saved_orphans']))
print('SFT_PASS8_RECOVERY_NATIVE_PARSE_OK')
"""
    code='SOURCE='+repr(str(SOURCE))+'\nPARENT='+repr(str(PARENT))+'\nCOLLECTOR_SHA='+repr(recovery['collector_after_sha256'])+'\nRECOVERY_MANIFEST='+repr(recovery['manifest_path'])+'\n'+code
    with (directory/'native-parse.log').open('w') as log:
        subprocess.run(source_command(SOURCE,[str(RUNTIME/'venv/bin/python'),'-c',code,*argv[2:]]),
            cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    base.write(directory/'readiness.json',dict(recovery_manifest_sha256=digest(recovery),
        native_parse_passed=True,collector_import_verified=True,
        saved_orphan_deep_validation_passed=True,saved_orphans_deep_verified=len(recovery['preserved']['saved_orphans']),
        native_parse_log_sha256=sha(directory/'native-parse.log'),no_gpu_api_browser_calls=True))


def validate_readiness(recovery,ready):
    if (ready.get('recovery_manifest_sha256')!=digest(recovery)
            or ready.get('native_parse_passed') is not True
            or ready.get('collector_import_verified') is not True
            or ready.get('saved_orphan_deep_validation_passed') is not True
            or ready.get('saved_orphans_deep_verified')!=len(recovery['preserved']['saved_orphans'])):
        raise ValueError('Native preflight and saved-orphan deep verification required')


def check_registration(base,plan,recovery,approval,job,used):
    seconds=base.verify_approval(plan,approval,job,recovery['shard'],used)
    current=next(row for row in approval['attempts'] if str(row['job_id'])==job)
    if (current.get('recovery_manifest')!=recovery['manifest_path']
            or current.get('recovery_manifest_sha256')!=digest(recovery)
            or current.get('source')!=str(SOURCE)
            or used<recovery['prior_charged_seconds']
            or seconds>recovery['remaining_scheduler_seconds']):
        raise ValueError('Register the exact reviewed recovery source/receipt within remaining caps')
    return seconds


def worker(path,job):
    base,plan,recovery=validate_recovery(path)
    if Path(__file__).resolve()!=SOURCE/ENTRY or os.getenv('SLURM_JOB_ID')!=job:
        raise ValueError('Recovery worker must run the frozen entry point in its allocation')
    approval=read(ROOT/'approval.json');used,_=base.accounted_seconds(approval,recovery['shard'],job)
    check_registration(base,plan,recovery,approval,job,used)
    launch=overlay_launch(base,plan,recovery,job);root=Path(launch['stage']['output'])
    if launch!=read(ROOT/f'shard-{recovery["shard"]}/launch-{job}.json'):
        raise ValueError('Recovery launch changed')
    from resume_baseline import clean_environment,source_command
    from runtime_ports import PORT_ENV,lease_ports
    from dotenv import dotenv_values
    env=clean_environment()
    for key in list(env):
        if key.startswith(('OPENWEBRL_ARM_','OPENWEBRL_TASK_SCREEN_')):env.pop(key)
    for key,value in dotenv_values(REPO/'.env').items():
        if value and key.startswith(('OPENAI_','JUDGE_','WANDB_','AZURE_')):env.setdefault(key,value)
    if env.get('OPENAI_API_KEY') and not env.get('JUDGE_API_BASE'):
        env.update(JUDGE_API_BASE='https://api.openai.com/v1',JUDGE_API_MODE='served')
    env.update(launch['stage']['environment'],WANDB_PROJECT='openwebrl-evals')
    if len(env.get('CUDA_VISIBLE_DEVICES','').split(','))!=1 or not env.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Exactly one assigned GPU required')
    root.mkdir(parents=True,exist_ok=True);base.write(root/'collector-config.json',launch['collector_config'])
    base.write(root/'evaluation_manifest.json',launch['stage'])
    Path(env['OPENWEBRL_MULTIMODAL_STORAGE_DIR']).mkdir(parents=True,exist_ok=True)
    child=None;identity=None;lease,port=lease_ports(job);env[PORT_ENV]=str(port)
    try:
        with (root/'evaluation.log').open('a') as log:
            child=subprocess.Popen(source_command(SOURCE,launch['stage']['command']),env=env,cwd=SOURCE,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            identity=base.process_identity(child.pid)
            if child.wait():raise RuntimeError('Recovery collector failed; preserve and diagnose')
    finally:base.stop_owned(child,identity);lease.close()


def execute(path,job):
    base,plan,recovery=validate_recovery(path);shard=recovery['shard']
    if Path(__file__).resolve()!=SOURCE/ENTRY or os.getenv('SLURM_JOB_ID')!=job:
        raise ValueError('Execute the frozen recovery controller inside the approved allocation')
    approval=read(ROOT/'approval.json');used,charges=base.accounted_seconds(approval,shard,job)
    seconds=check_registration(base,plan,recovery,approval,job,used)
    remaining=base.validate_allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),job,seconds)
    ready=read(Path(path).parent/'readiness.json')
    validate_readiness(recovery,ready)
    from check_storage_quota import check_storage_quota
    check_storage_quota(1024**4)
    devices=os.getenv('CUDA_VISIBLE_DEVICES','').split(',')
    if len(devices)!=1 or not devices[0]:raise ValueError('Requires one assigned GPU')
    launch=overlay_launch(base,plan,recovery,job);root=Path(launch['output'])
    with (root/'owner.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        attempt=root/'scheduler-attempts'/job
        if (attempt/'started.json').exists():raise ValueError('Do not reuse scheduler attempt identities')
        base.write(attempt/'started.json',dict(job_id=job,shard=shard,epoch=time.time(),
            plan_sha256=base.digest(plan),prior_charged_seconds=used,prior_attempts=charges,
            time_limit_seconds=seconds,recovery_manifest=str(path),recovery_manifest_sha256=digest(recovery),
            source=str(SOURCE),source_manifest_sha256=recovery['source_manifest_sha256']))
        base.write(root/f'launch-{job}.json',launch)
        base.write(ROOT/f'current-shard-{shard}.json',dict(job_id=job,shard=shard,
            launch=str(root/f'launch-{job}.json'),status=str(root/'status.json'),plan_sha256=base.digest(plan),
            epoch=time.time(),verified_complete=False,source=str(SOURCE),recovery_manifest=str(path),
            recovery_manifest_sha256=digest(recovery)))
        child=None;identity=None;deadline=time.monotonic()+remaining
        try:
            with (attempt/'worker.log').open('a') as log:
                child=subprocess.Popen(['srun',f'--jobid={job}','--nodes=1','--ntasks=1','--cpus-per-task=8',
                    '--gres=gpu:h200:1','--mem=120G','--exact','--cpu-bind=none',
                    str(RUNTIME/'venv/bin/python'),'-B',str(SOURCE/ENTRY),'--recovery',str(path),
                    '--job-id',job,'--worker'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                identity=base.process_identity(child.pid)
                while child.poll() is None:
                    if time.monotonic()>deadline:raise TimeoutError('Recovery shutdown margin reached')
                    base.write(root/'status.json',dict(job_id=job,shard=shard,stage='running',complete=False,
                        epoch=time.time(),canonical_records=len(list((root/'records').glob('*.json'))),
                        prior_charged_seconds=used,source=str(SOURCE),recovery_manifest=str(path)))
                    with (attempt/'gpu.log').open('a') as gpu:
                        subprocess.run(['nvidia-smi','--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw',
                            '--format=csv,noheader,nounits',f'--id={devices[0]}'],stdout=gpu,stderr=subprocess.STDOUT,timeout=20,check=False)
                    time.sleep(30)
            if child.returncode:raise RuntimeError('Recovery worker failed; needs diagnosis')
            if read(root/'collection-complete.json').get('complete') is not True:
                raise ValueError('Missing audited collection receipt')
            base.write(root/'status.json',dict(job_id=job,shard=shard,stage='collection_complete',complete=True,
                verified_complete=False,epoch=time.time(),source=str(SOURCE),recovery_manifest=str(path),
                collection_receipt_sha256=sha(root/'collection-complete.json')))
        except BaseException as exc:
            base.write(root/'status.json',dict(job_id=job,shard=shard,stage='failed',complete=False,
                epoch=time.time(),error_type=type(exc).__name__,error=str(exc)[:800],
                source=str(SOURCE),recovery_manifest=str(path)));raise
        finally:
            base.stop_owned(child,identity)
            base.write(attempt/'controller-exited.json',dict(epoch=time.time(),scheduler_final_accounting_pending=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--native-check',action='store_true')
    parser.add_argument('--execute',action='store_true');parser.add_argument('--worker',action='store_true')
    parser.add_argument('--shard',type=int,choices=range(4),default=0)
    parser.add_argument('--failed-job');parser.add_argument('--job-id');parser.add_argument('--recovery',type=Path)
    args=parser.parse_args()
    if args.worker:worker(args.recovery,args.job_id);return
    if args.execute:execute(args.recovery,args.job_id);return
    recovery=prepare(args.shard,args.failed_job) if args.prepare else validate_recovery(args.recovery)[2]
    if args.native_check:native_check(recovery)
    print(json.dumps({key:recovery[key] for key in ('manifest_path','shard','failed_job','prior_charged_seconds',
        'remaining_scheduler_seconds','recommended_time_limit_seconds','source','collector_after_sha256')},indent=2))


if __name__=='__main__':main()
