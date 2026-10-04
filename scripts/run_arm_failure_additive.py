#!/usr/bin/env python3
"""Prepare additive all-failure ARM RL; never submits an allocation."""
import argparse,hashlib,json,os,shlex,shutil,subprocess,sys
from pathlib import Path
from prepare_arm_turn_bonus import copy_plain
from resume_baseline import validate_source,write_json,source_command
import run_arm_turn_bonus_cycles as training
REPO,RUNTIME=training.REPO,training.RUNTIME
PARENT=RUNTIME/'reference-arm-turn-bonus-cycles-20260913-v3'
SOURCE=RUNTIME/'reference-arm-failure-additive-20260914-v3'
MODULES=['openwebrl/arm_failure_additive.py','openwebrl/arm_failure_aux.py','openwebrl/arm_failure_bonus.py']

def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError('Patch target missing or ambiguous: '+old[:90])
    return text.replace(old,new,1)

def prepare():
    validate_source(PARENT)
    if SOURCE.exists():
        validate_source(SOURCE)
        assert all((SOURCE/f).read_bytes()==(REPO/f).read_bytes() for f in MODULES)
        return
    shutil.copytree(PARENT,SOURCE,symlinks=True,copy_function=copy_plain,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git'))
    for f in MODULES:copy_plain(REPO/f,SOURCE/f)
    changed=list(MODULES)
    f='openwebrl/arm_turn_bonus.py';p=SOURCE/f;s=p.read_text()
    anchor='    # Native outcomes remain unchanged. Beta=0/shadow preserves object identity.'
    s=replace_once(s,anchor,"    if config.get('additive_failure_groups'):\n        from openwebrl.arm_failure_additive import write_auxiliary\n        write_auxiliary(args,samples,current,report)\n"+anchor);p.write_text(s);changed.append(f)
    f='slime/backends/megatron_utils/actor.py';p=SOURCE/f;s=p.read_text()
    anchor='    def train_actor(self, rollout_id: int, rollout_data: RolloutBatch) -> None:\n'
    s=replace_once(s,anchor,anchor+'        from openwebrl.arm_failure_aux import prepare_auxiliary, attach_auxiliary\n        arm_auxiliary = prepare_auxiliary(self, rollout_id)\n')
    for indent,epoch in [('                    ','ppo_epoch_id'),('            ','0')]:
        old=indent+'with timer("actor_train"):\n'+indent+'    train(\n'
        new=indent+'with timer("actor_train"):\n'+indent+'    train_iterators, train_counts = attach_auxiliary(self.args, data_iterator, num_microbatches, arm_auxiliary)\n'+indent+'    train(\n'
        s=replace_once(s,old,new)
        old=indent+'        data_iterator,\n'+indent+'        num_microbatches,\n'+indent+f'        ppo_epoch_id={epoch},'
        new=indent+'        train_iterators,\n'+indent+'        train_counts,\n'+indent+f'        ppo_epoch_id={epoch},'
        s=replace_once(s,old,new)
    p.write_text(s);changed.append(f)
    f='slime/backends/megatron_utils/model.py';p=SOURCE/f;s=p.read_text()
    s=replace_once(s,'                "teacher_log_probs",\n','                "teacher_log_probs",\n                "arm_source", "arm_scale", "arm_advantage", "arm_old_log_probs",\n')
    s=replace_once(s,'            logging_utils.log(args, log_dict, step_key="train/step")',
        '            from openwebrl.arm_failure_aux import normalize_metrics\n            normalize_metrics(log_dict)\n            logging_utils.log(args, log_dict, step_key="train/step")')
    p.write_text(s);changed.append(f)
    f='slime/backends/megatron_utils/loss.py';p=SOURCE/f;s=p.read_text()
    anchor='    if args.recompute_loss_function:\n'
    s=replace_once(s,anchor,'    if batch.get("arm_source") is not None:\n        from openwebrl.arm_failure_aux import loss as func\n\n'+anchor);p.write_text(s);changed.append(f)
    manifest=json.loads((SOURCE/'reference_manifest.json').read_text())
    manifest['recipe_files_sha256'].update({f:hashlib.sha256((SOURCE/f).read_bytes()).hexdigest() for f in changed})
    manifest['arm_failure_additive']=dict(parent=str(PARENT),mixed_groups=48,failure_group_cap=8,coefficient=1/6,gpu_validated=False)
    write_json(SOURCE/'reference_manifest.json',manifest);validate_source(SOURCE)

def plan(job='UNAPPROVED',minutes=240):
    validate_source(SOURCE)
    p=training.plan(job,'arm',minutes);old=p['output'];out=RUNTIME/f'evaluations/arm-failure-additive-{job}';run=f'arm-failure-additive-{job}'
    p['command']=[x.replace(old,str(out)).replace(str(PARENT),str(SOURCE)) for x in p['command']]
    p['environment']={k:v.replace(old,str(out)).replace(str(PARENT),str(SOURCE)) for k,v in p['environment'].items()}
    p['command'][p['command'].index('--wandb-group')+1]='executed-turn-bonus-additive-failure'
    p['command']+=['--dynamic-sampling-filter-path','openwebrl.arm_failure_additive.filter_groups']
    p.update(source=str(SOURCE),output=str(out),wandb_run_id=run,experiment='additive-all-failure',compute_approved=False)
    p['environment'].update(WANDB_RUN_ID=run,WANDB_RESUME='never',BROWSER_CONCURRENCY='32',SGLANG_CONCURRENCY='32',
        OPENWEBRL_ARM_FAILURE_AUX_MANIFEST=str(out/'iterations/{rollout_id:04d}/failure_auxiliary.json'))
    p['browser_config']['browser_rollout_concurrency']=32;p['requested_resources']['browsers']=32
    p['arm_config'].update(run_id=run,policy_id=f'{run}:uninitialized',run_output=str(out),output=str(out),
        admit_all_failure_groups=True,additive_failure_groups=True,failure_group_cap=8,failure_loss_coefficient=1/6,
        minimum_cycle_seconds=5400,seconds_per_optimizer_update=180)
    p['comparison']=dict(initialization='original SFT, fresh optimizer; compare original ARM from update zero',
        outcome_groups=48,auxiliary_groups='0..8 encountered without extra collection',
        auxiliary_objective='N/48 (N<=8) times all-failure turn-population mean clipped executed-turn PPO; beta .5',
        outcome_updates='unchanged; auxiliary backward passes appended inside original optimizer windows',
        old_auxiliary_logprobs='recomputed once on frozen actor at collection start, temperature .8',
        max_auxiliary_turns_per_update=32,gpu_validation_pending=True)
    return p

def native_check(p,out):
    env=dict(os.environ,**p['environment']);env.update(DRY_RUN='1',BROWSER_TRAIN_CONFIG=str(out/'browser.json'),
        FLASHINFER_WORKSPACE_BASE=str(RUNTIME),CUDA_VISIBLE_DEVICES='')
    write_json(out/'browser.json',p['browser_config'])
    argv=shlex.split(subprocess.check_output(p['command'],env=env,text=True));env.pop('DRY_RUN')
    code="""from unittest.mock import patch
from slime.utils.arguments import parse_args
with patch('megatron.training.arguments.get_device_arch_version',return_value=9): a=parse_args()
expected={'actor_num_gpus_per_node':4,'tensor_model_parallel_size':4,'global_batch_size':256,'micro_batch_size':1,'ppo_epochs':2,'rollout_batch_size':48,'browser_rollout_concurrency':32,'judge_api_model':'gpt-4.1','judge_prompt_variant':'action_history','dynamic_sampling_filter_path':'openwebrl.arm_failure_additive.filter_groups','use_rollout_logprobs':True}
assert all(getattr(a,k)==v for k,v in expected.items()),{k:getattr(a,k) for k in expected}
from openwebrl.arm_failure_aux import check_topology
check_topology(a)
print('ADDITIVE_NATIVE_PARSE_OK')
"""
    with (out/'native-parse.log').open('w') as log:
        subprocess.run(source_command(SOURCE,[sys.executable,'-c',code,*argv[2:]]),env=env,cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)

def fingerprint():
    validate_source(SOURCE)
    files=MODULES+['scripts/run_arm_failure_additive.py','scripts/run_arm_failure_additive_4gpu.sbatch',
        'scripts/run_arm_turn_bonus_cycles.py','scripts/monitor_arm_turn_bonus.py','tests/test_arm_failure_additive.py']
    return {str(REPO/f):hashlib.sha256((REPO/f).read_bytes()).hexdigest() for f in files} | {str(SOURCE/'reference_manifest.json'):hashlib.sha256((SOURCE/'reference_manifest.json').read_bytes()).hexdigest()}


def require_receipt(p):
    out=RUNTIME/'arm-turn-bonus-preparation/additive-failure'
    receipt=json.loads((out/'readiness.json').read_text())
    if not receipt.get('cpu_tests_passed') or not receipt.get('native_parse_passed') or receipt['fingerprint']!=fingerprint():
        raise ValueError('Additive readiness missing or stale')
    if p!=plan(p['job_id'],round(p['requested_resources']['hours']*60)):
        raise ValueError('Additive launch plan changed')
    return receipt

def check():
    prepare()
    subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_arm_failure_additive.py','-v'],cwd=REPO,check=True)
    subprocess.run(['bash','-n',str(REPO/'scripts/run_arm_failure_additive_4gpu.sbatch')],check=True)
    p=plan();out=RUNTIME/'arm-turn-bonus-preparation/additive-failure';out.mkdir(parents=True,exist_ok=True)
    write_json(out/'plan.json',p)
    native_check(p,out)
    receipt=dict(cpu_tests_passed=True,native_parse_passed=True,gpu_validated=False,compute_approved=False,
        source=str(SOURCE),plan=str(out/'plan.json'),fingerprint=fingerprint())
    write_json(out/'readiness.json',receipt);return receipt

def main():
    a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--check',action='store_true');a.add_argument('--execute',action='store_true');a.add_argument('--job-id',default='UNAPPROVED');a.add_argument('--minutes',type=int,default=240);args=a.parse_args()
    if args.prepare:prepare();print(SOURCE);return
    if args.check:print(json.dumps(check(),indent=2));return
    if not args.execute:prepare();print(json.dumps(plan(args.job_id,args.minutes),indent=2));return
    if os.environ.get('SLURM_JOB_ID')!=args.job_id:raise ValueError('Existing authorized allocation required')
    check();p=plan(args.job_id,args.minutes)
    # Own the actual training worker and monitor for the entire allocation.
    with (RUNTIME/f'logs/arm-additive-{args.job_id}-monitor.log').open('w') as log:
        monitor=subprocess.Popen([sys.executable,str(REPO/'scripts/monitor_arm_turn_bonus.py'),'--job-id',args.job_id,'--watch','--interval','900','--hours',str(args.minutes/60)],stdout=log,stderr=subprocess.STDOUT)
        try:training.execute(p)
        finally:
            monitor.terminate()
            try:monitor.wait(timeout=30)
            except subprocess.TimeoutExpired:monitor.kill();monitor.wait()
if __name__=='__main__':main()
