#!/usr/bin/env python3
"""Three isolated native TP1 evaluations in one approved 3-H200 allocation."""
import argparse, hashlib, json, os, re, runpy, shlex, shutil, subprocess, sys, time
from pathlib import Path
from evaluate_baseline_checkpoint import build_plan, configure_evaluation_tracking, finalize_evaluation, record_restore_evidence
from resume_baseline import clean_environment, source_command, validate_source, write_json
from prepare_arm_turn_bonus import copy_plain
REPO=Path(__file__).resolve().parents[1]
RUNTIME=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
PARENT=RUNTIME/'reference-stage1-tp4-eval-cache-20260909'
SOURCE=Path(os.environ.get('ARM_EVAL_SOURCE',str(RUNTIME/'reference-arm-threeway-eval-20260914-v2')))
SAMPLE=REPO/'openwebrl/docs/arm_c2_scaling_100.json'
CASES={
 'baseline46':(RUNTIME/'runs/openwebrl-4b-reference-282346-20260908T024410/iter_0000002',46,1),
 'arm46':(RUNTIME/'evaluations/arm-turn-bonus-fresh-295067/runtime/iter_0000002',46,0),
 'allfailure42':(RUNTIME/'evaluations/arm-turn-bonus-fresh-allfailure-295353/runtime/iter_0000002',42,0)}

def prepare():
    validate_source(PARENT)
    sample=json.loads(SAMPLE.read_text())
    assert sample['count']==100 and len(set(sample['task_ids']))==100
    assert hashlib.sha256(Path(sample['task_file']).read_bytes()).hexdigest()==sample['task_file_sha256']
    rows=[json.loads(x) for x in (PARENT/'online_mind2web_monitor.jsonl').read_text().splitlines()]
    chosen=[rows[i] for i in sample['indices']]
    assert [x['metadata']['task_id'] for x in chosen]==sample['task_ids']
    if SOURCE.exists():validate_source(SOURCE);return
    shutil.copytree(PARENT,SOURCE,symlinks=True,copy_function=copy_plain,
        ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git','.browser_use_sessions'))
    # Preserve the existing deterministic monitor; add durable per-task records.
    p=SOURCE/'openwebrl/eval_monitor.py'
    p.write_text(p.read_text()+'''\n_original_generate = generate\nasync def generate(args, sample, sampling_params, evaluation=False):\n    turns = await _original_generate(args, sample, sampling_params, evaluation)\n    from openwebrl.eval_benchmark import save_completed_task\n    save_completed_task(args, turns)\n    return turns\n''')
    copy_plain(REPO/'openwebrl/eval_benchmark.py',SOURCE/'openwebrl/eval_benchmark.py')
    p=SOURCE/'online_mind2web_monitor.jsonl';p.write_text(''.join(json.dumps(x)+'\n' for x in chosen))
    m=json.loads((SOURCE/'reference_manifest.json').read_text())
    for f in ['openwebrl/eval_monitor.py','openwebrl/eval_benchmark.py','online_mind2web_monitor.jsonl']:
        m['recipe_files_sha256'][f]=hashlib.sha256((SOURCE/f).read_bytes()).hexdigest()
    m['threeway_eval']=dict(parent=str(PARENT),sample=str(SAMPLE),sample_sha256=hashlib.sha256(SAMPLE.read_bytes()).hexdigest(),zero_optimizer_updates=True)
    write_json(SOURCE/'reference_manifest.json',m);validate_source(SOURCE)

def make_plan(job,label):
    ck,updates,offset=CASES[label];idx=list(CASES).index(label)
    suffix=os.environ.get('ARM_EVAL_ATTEMPT_SUFFIX','')
    out=RUNTIME/f'evaluations/arm-threeway-{job}/{label}{suffix}'
    p=build_plan(SOURCE,ck,out,job,gpus=2)
    p.update(gpus=1,expected_task_count=100,expected_task_ids=json.loads(SAMPLE.read_text())['task_ids'],
        expected_optimizer_updates=updates,scheduler_offset=offset,wandb_run_id=f'arm-threeway-{job}-{label}{suffix}',
        protocol='Fixed100 native GPT-4.1/action_history monitor: T0,p1,k1,response4096,horizon30; actor-only',parent_training_run=label)
    p['command'][p['command'].index('--wandb-group')+1]='arm-threeway-early-100'
    p['environment'].update(NUM_GPUS='1',TP_SIZE='1',BROWSER_CONCURRENCY='8',SGLANG_CONCURRENCY='8',
        SLIME_BROWSER_ENV_MODE='local_process',SLIME_BROWSER_LOCAL_PROCESS_PYTHON=str(RUNTIME/'venv/bin/python'),
        SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES='8',SLIME_BROWSER_ROLLOUT_CONCURRENCY='8',
        OPENWEBRL_CUDA_CACHE_LIMIT_GIB='100',MODEL_ARGS_ROTARY_BASE='5000000',
        ARM_EVAL_METRICS_PORT=str(24000+idx),ARM_EVAL_SOURCE=str(SOURCE),
        WANDB_RUN_ID=p['wandb_run_id'],WANDB_RESUME='never',WANDB_CACHE_DIR=str(out/'wandb-cache'),
        OPENWEBRL_BENCHMARK_RESULTS_DIR=str(out/'completed_tasks'),
        OPENWEBRL_MULTIMODAL_STORAGE_DIR=f'/tmp/a3-{job}-{label}-images',
        BROWSER_TRAIN_CONFIG=str(out/'browser.json'),RAY_ADDRESS='local')
    # Each worker has an independent port range; no unrelated browser processes.
    p['browser_ports']=[20000+idx*200,20199+idx*200]
    return configure_evaluation_tracking(p)

def expanded(p,env):
    argv=shlex.split(subprocess.check_output(p['command'],env=dict(env,DRY_RUN='1'),text=True))
    argv.remove('--sequence-parallel')
    assert argv[argv.index('--num-rollout')+1]=='0'
    assert argv[argv.index('--tensor-model-parallel-size')+1]=='1'
    assert '--eval-interval' in argv
    return argv

def worker(job,label,task_ids=None):
    import yaml
    from dotenv import dotenv_values
    assert os.environ.get('SLURM_JOB_ID')==job and f'/job_{job}/' in Path('/proc/self/cgroup').read_text()
    devices=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',');assert len(devices)==1 and devices[0]
    prepare();p=make_plan(job,label);out=Path(p['output']);out.mkdir(parents=True,exist_ok=False)
    ck=Path(p['checkpoint']);view=out/'checkpoint-view';view.mkdir();(view/ck.name).symlink_to(ck,target_is_directory=True)
    (view/'rollout').symlink_to(ck.parent/'rollout',target_is_directory=True)
    (view/'latest_checkpointed_iteration.txt').write_text(str(p['checkpoint_index'])+'\n')
    cfg=yaml.safe_load((SOURCE/'openwebrl/browser_training_config.yaml').read_text());cfg['browser_rollout_concurrency']=8
    write_json(out/'browser.json',cfg)
    if task_ids is not None:
        rows=[json.loads(x) for x in (SOURCE/'online_mind2web_monitor.jsonl').read_text().splitlines()]
        chosen=[x for x in rows if x['metadata']['task_id'] in set(task_ids)]
        assert len(chosen)==len(set(task_ids))==len(task_ids)
        tasks=out/'retry_tasks.jsonl';tasks.write_text(''.join(json.dumps(x)+'\n' for x in chosen))
        eval_cfg=yaml.safe_load((SOURCE/'openwebrl/online_mind2web_monitor.yaml').read_text())
        eval_cfg['eval']['datasets'][0]['path']=str(tasks)
        eval_path=out/'retry_eval.yaml';eval_path.write_text(yaml.safe_dump(eval_cfg))
        p['command'][p['command'].index('--eval-config')+1]=str(eval_path)
        p.update(expected_task_count=len(task_ids),expected_task_ids=task_ids,retry_of_job='295690',attempt=1)
    env=clean_environment()
    for k,v in dotenv_values(REPO/'.env').items():
        if v and k.startswith(('WANDB_','JUDGE_','OPENAI_','AZURE_')):env.setdefault(k,v)
    env.update(p['environment']);env.update(SLIME_BROWSER_LOCAL_PROCESS_PORT_START=str(p['browser_ports'][0]),SLIME_BROWSER_LOCAL_PROCESS_PORT_END=str(p['browser_ports'][1]))
    if env.get('OPENAI_API_KEY') and not env.get('JUDGE_API_BASE'):env.update(JUDGE_API_MODE='served',JUDGE_API_BASE='https://api.openai.com/v1')
    argv=expanded(p,env);p['native_argv']=argv;write_json(out/'evaluation_manifest.json',p)
    inspect=[sys.executable,str(REPO/'scripts/inspect_training_checkpoint.py'),str(view),'--expected-updates',str(p['expected_optimizer_updates']),'--expected-scheduler-offset-updates',str(p['scheduler_offset']),'--report',str(out/'checkpoint_validation.json')]
    subprocess.run(source_command(SOURCE,inspect),env=env,check=True,stdout=subprocess.DEVNULL)
    record=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
    from datetime import datetime
    deadline=datetime.fromisoformat(re.search(r'EndTime=(\S+)',record)[1]).timestamp()-180
    seconds=int(deadline-time.time());assert seconds>1800
    # Explicit Ray initialization prevents automatic attachment to another worker.
    command=[sys.executable,str(Path(__file__).resolve()),'--native',label,'--job-id',job,'--',*argv[2:]]
    with (out/'evaluation.log').open('w') as log:
        proc=subprocess.Popen(source_command(SOURCE,['timeout','--signal=INT','--kill-after=60',str(seconds),*command]),env=env,cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT)
        while proc.poll() is None:record_restore_evidence(p);time.sleep(5)
        record_restore_evidence(p);code=proc.returncode
    write_json(out/'launcher_exit.json',dict(returncode=code));finalize_evaluation(p,code)

def controller(job):
    prepare();root=RUNTIME/f'evaluations/arm-threeway-{job}';root.mkdir(parents=True,exist_ok=False)
    # A single step allocates all three tasks together. Separate srun steps in
    # 295690 serialized despite --exact, leaving the third checkpoint unevaluated.
    command=['srun','--jobid='+job,'--nodes=1','--ntasks=3','--cpus-per-task=8',
        '--mem=720G','--gpus-per-task=h200:1','--gpu-bind=single:1','--exact',
        '--cpu-bind=none','--output='+str(root/'rank-%t-launcher.log'),
        sys.executable,str(Path(__file__).resolve()),'--worker-rank','--job-id',job]
    code=subprocess.run(command).returncode
    write_json(root/'controller-result.json',{'step_returncode':code})
    if code:raise RuntimeError('Some evaluation workers failed; preserved partial task records')

def retry_manifest():
    import math
    result={}
    for label,count in [('baseline46',35),('allfailure42',30)]:
        root=RUNTIME/'evaluations/arm-threeway-295690'/label
        assert json.loads((root/'status.json').read_text())['complete']
        records=[json.loads(x.read_text()) for x in (root/'completed_tasks').glob('*.json')]
        assert len(records)==100 and len({x['task_id'] for x in records})==100
        ids=[]
        for x in records:
            reward=x['transport_reward']
            if isinstance(reward,dict):reward=reward.get('combined')
            valid=not x['remove_sample'] and x['status']!='aborted' and reward is not None and math.isfinite(float(reward))
            if not valid:ids.append(x['task_id'])
        assert len(ids)==count
        result[label]=sorted(ids)
    return result

def retry_controller(job):
    prepare();root=RUNTIME/f'evaluations/arm-threeway-{job}';root.mkdir(parents=True,exist_ok=False)
    write_json(root/'retry_manifest.json',retry_manifest())
    command=['srun','--jobid='+job,'--nodes=1','--ntasks=2','--cpus-per-task=8',
        '--mem=480G','--gpus-per-task=h200:1','--gpu-bind=single:1','--exact',
        '--cpu-bind=none','--output='+str(root/'rank-%t-launcher.log'),
        sys.executable,str(Path(__file__).resolve()),'--retry-rank','--job-id',job]
    proc=subprocess.Popen(command);started=time.monotonic()
    while proc.poll() is None:
        receipts=list(root.glob('rank-*-started.json'))
        if len(receipts)<2 and time.monotonic()-started>180:
            proc.terminate();proc.wait()
            raise RuntimeError('Both GPU workers did not start within 180 seconds')
        write_json(root/'controller-progress.json',{'started_workers':len(receipts),'updated':time.time(),
            'completed_tasks':{k:len(list((root/k/'completed_tasks').glob('*.json'))) for k in CASES}})
        time.sleep(15)
    write_json(root/'controller-result.json',{'step_returncode':proc.returncode})
    if proc.returncode:raise RuntimeError('Evaluation worker failed; partial records preserved')

def retry_rank(job):
    rank=int(os.environ['SLURM_PROCID']);assert rank in (0,1)
    root=RUNTIME/f'evaluations/arm-threeway-{job}'
    write_json(root/f'rank-{rank}-started.json',{'rank':rank,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'time':time.time()})
    if rank==0:worker(job,'arm46')
    else:
        manifest=json.loads((root/'retry_manifest.json').read_text())
        for label in ['baseline46','allfailure42']:worker(job,label,manifest[label])

def main():
    if '--native' in sys.argv:
        a=sys.argv;label=a[a.index('--native')+1];job=a[a.index('--job-id')+1];args=a[a.index('--')+1:]
        import ray
        ray.init(address='local',num_gpus=1,num_cpus=8,include_dashboard=False,_temp_dir=f'/tmp/a3-{job}-{label}',object_store_memory=8*1024**3)
        sys.path.insert(0,str(SOURCE));sys.argv=[str(SOURCE/'train.py'),*args]
        try:runpy.run_path(str(SOURCE/'train.py'),run_name='__main__')
        finally:ray.shutdown()
        return
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--controller',action='store_true');p.add_argument('--worker',choices=list(CASES));p.add_argument('--worker-rank',action='store_true');p.add_argument('--retry-controller',action='store_true');p.add_argument('--retry-rank',action='store_true');p.add_argument('--job-id',default='PREPARE');a=p.parse_args()
    if a.prepare:prepare();print(SOURCE)
    elif a.controller:controller(a.job_id)
    elif a.worker_rank:worker(a.job_id,list(CASES)[int(os.environ['SLURM_PROCID'])])
    elif a.worker:worker(a.job_id,a.worker)
    elif a.retry_controller:retry_controller(a.job_id)
    elif a.retry_rank:retry_rank(a.job_id)
if __name__=='__main__':main()
