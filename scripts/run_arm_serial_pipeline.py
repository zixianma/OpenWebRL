#!/usr/bin/env python3
"""Own each approved serial SFT -> checkpoint export -> fixed-100 evaluation."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from train_arm_joint_sft import REPO,RUNTIME,ACTOR,digest,read_rows,write_json
from resume_arm_c2_training import allocation_deadline,parse_job_record

CONFIG=REPO/'openwebrl/docs/arm_results/serial_sft'
PYTHON=RUNTIME/'venv/bin/python'


def prepare():
    data=RUNTIME/'serial-data-v1';report=json.loads((data/'verification.json').read_text())
    if not report['passed']:raise ValueError('Data did not pass verification')
    CONFIG.mkdir(parents=True,exist_ok=True)
    cohorts=json.loads((REPO/'openwebrl/docs/arm_c2_full300_cohorts.json').read_text())
    cohort=cohorts['cohorts']['checkpoint_selection_100']
    ordered=sorted(zip(cohort['indices'],cohort['task_ids']))
    shardmap=dict(task_file_sha256=cohorts['task_file_sha256'],cohorts={
        f'shard-{rank}':dict(count=50,indices=[v[0] for v in ordered[rank::2]],task_ids=[v[1] for v in ordered[rank::2]]) for rank in range(2)})
    assert len({x[1] for x in ordered})==100
    write_json(CONFIG/'om2w-shards.json',shardmap)
    validation=read_rows(data/'all5.validation.jsonl')
    import hashlib
    panel=sorted(validation,key=lambda r:hashlib.sha256(('serial-panel:42:'+r['id']).encode()).hexdigest())[:128]
    code=['scripts/train_arm_serial_sft.py','scripts/train_arm_joint_sft.py','scripts/train_arm_c2.py',
          'scripts/run_arm_serial_pipeline.py','scripts/run_arm_full300_checkpoint_eval.py',
          'scripts/merge_arm_c2_student.py','openwebrl/serial_actions.py','openwebrl/arm_eval.py',
          'openwebrl/generate_browser.py','openwebrl/reward_browser.py','openwebrl/run_evaluate.py']
    for variant in ['all5','diverse3']:
        path=CONFIG/f'{variant}-config.json'
        if path.exists():raise ValueError('Config already frozen')
        config=dict(variant=variant,actor=str(ACTOR),data=str(data),output=str(RUNTIME/f'runs/serial-{variant}-v1'),
            train_examples=5437,validation_examples=644,batch_size=32,world_size=2,gradient_accumulation_per_rank=16,
            epochs=1,total_updates=170,final_batch_size=29,learning_rate=1e-5,warmup_examples=512,
            endpoint_lr_factor=.5,lora_rank=16,lora_alpha=32,lora_dropout=.05,max_tokens=32768,
            segment_weights=[.5,.1,.4],checkpoint_updates=[43,85,128,170],seed=42,
            validation_panel_ids=[r['id'] for r in panel],wandb_id=uuid.uuid4().hex[:8],
            wandb_project='openwebrl-arm',wandb_group='serial-alternatives-v1',
            data_hashes={name:checksum for name,checksum in report['data_hashes'].items() if name.startswith(variant+'.')},
            code_hashes={name:digest(REPO/name) for name in code},
            eval_shards_sha256=digest(CONFIG/'om2w-shards.json'),
            evaluation=dict(tasks=100,checkpoint=170,parallel_per_gpu=12,max_new_tokens=6144,judge='o4-mini',protocol='online_mind2web/AgentTrek'),
            budget=dict(gpus=2,hours=4,cpus=16,memory_gib=240))
        write_json(path,config);print(str(path))


def aggregate(config):
    from summarize_arm_c2_full300 import rate_report,paired_report
    from openwebrl.arm_eval import summarize
    root=Path(config['output']);shards=json.loads((CONFIG/'om2w-shards.json').read_text());records={}
    for name,cohort in shards['cohorts'].items():
        out=root/'evaluation'/name/'online-mind2web'
        manifest=json.loads((out/'manifest.json').read_text())
        assert manifest['task_ids']==cohort['task_ids'] and manifest['serial_variant']==config['variant']
        assert manifest['judge']=='o4-mini' and manifest['judge_protocol']=='online_mind2web/AgentTrek'
        rows=[json.loads(p.read_text()) for p in (out/'results').glob('*.json')]
        assert {r['task_id'] for r in rows}==set(cohort['task_ids'])
        for row in rows:
            if row['task_id'] in records:raise ValueError('Duplicate task')
            records[row['task_id']]=row
    assert len(records)==100
    report=dict(variant=config['variant'],updates=170,summary=summarize(list(records.values()),100),
        completed_utc=datetime.now(timezone.utc).isoformat(),shards=shards,
        note='Same fixed development 100 tasks; initial outcomes only, no retry replacement.')
    write_json(root/'evaluation/fixed100-records.json',records)
    write_json(CONFIG/f'{config["variant"]}-results.json',report)
    other='diverse3' if config['variant']=='all5' else 'all5'
    opath=RUNTIME/f'runs/serial-{other}-v1/evaluation/fixed100-records.json'
    if opath.exists():
        both={config['variant']:records,other:json.loads(opath.read_text())};ids=sorted(records)
        write_json(CONFIG/'comparison.json',dict(rates={v:rate_report(rs,ids) for v,rs in both.items()},
            paired_common_valid=paired_report(both['all5'],both['diverse3'],ids),
            note='One training seed; fixed 100 development tasks; no evaluation retries merged.'))
    return report


def run(path):
    config=json.loads(path.read_text());root=Path(config['output']);root.mkdir(parents=True,exist_ok=True)
    job=os.environ.get('SLURM_JOB_ID')
    if not job or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():raise ValueError('Requires approved allocation')
    rec=parse_job_record(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True,timeout=20))
    if int(rec['NumNodes'])!=1 or int(rec['NumCPUs'])<16:raise ValueError('Requires one node and 16 CPUs')
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],text=True,timeout=20).strip().splitlines()
    if len(devices)!=2 or any('H200' not in d for d in devices):raise ValueError('Requires two H200s')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=20).strip():raise ValueError('Allocated GPUs occupied')
    for name,checksum in config['code_hashes'].items():
        if digest(REPO/name)!=checksum:raise ValueError(f'Frozen source changed: {name}')
    if digest(CONFIG/'om2w-shards.json')!=config['eval_shards_sha256']:raise ValueError('Cohort changed')
    lock=(root/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline=allocation_deadline(rec,margin_minutes=5).timestamp();training_stop=deadline-75*60
    children=[];logs=[];stopping=False
    def status(phase,**extra):write_json(root/'status.json',dict(phase=phase,job=job,variant=config['variant'],
        checked_utc=datetime.now(timezone.utc).isoformat(),deadline_unix=deadline,**extra))
    def launch(cmd,logname,env=None):
        log=(root/logname).open('a');logs.append(log)
        child=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(child);return child
    def stop(*_):
        nonlocal stopping
        stopping=True
        for child in children:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def wait(child):
        while child.poll() is None:
            if stopping or time.time()>=deadline:stop()
            time.sleep(5)
        return child.wait()
    try:
        while not (root/'student/complete.json').exists():
            if stopping or time.time()>=training_stop:raise RuntimeError('Training budget exhausted before endpoint')
            status('training',training_stop_unix=training_stop)
            cmd=[str(PYTHON),'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',
                 str(REPO/'scripts/train_arm_serial_sft.py'),'--config',str(path),'--stop-unix',str(training_stop)]
            if (root/'student/latest-checkpoint.json').exists():cmd.append('--resume')
            code=wait(launch(cmd,'training.log'))
            if code==0 and (root/'student/complete.json').exists():break
            if stopping or time.time()>=training_stop:raise RuntimeError('Training stopped before endpoint')
            # Keep ownership briefly for a concrete repair without a new paid job.
            repair_until=min(time.time()+600,training_stop)
            status('awaiting_repair',returncode=code,repair_until_unix=repair_until)
            request=root/'retry-controller.request'
            while time.time()<repair_until and not stopping:
                if request.exists():request.rename(root/f'retry-consumed-{int(time.time())}');break
                time.sleep(10)
            else:raise RuntimeError('Training failed; repair window expired')
        endpoint=root/'student/update-000170';merged=root/'evaluation/merged-endpoint'
        if stopping:return
        status('merging')
        if wait(launch([str(PYTHON),str(REPO/'scripts/merge_arm_c2_student.py'),'--run-root',str(root),
            '--checkpoint',str(endpoint),'--output',str(merged)],'merge.log')):raise RuntimeError('Merge failed')
        if stopping:return
        status('evaluating_fixed100');workers=[]
        base=28110 if config['variant']=='all5' else 29110
        for rank in range(2):
            port=base+rank*300;env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=devices[rank].split(',',1)[0]
            cmd=[str(PYTHON),str(REPO/'scripts/run_arm_full300_checkpoint_eval.py'),'--job-id',job,
                '--run-root',str(root),'--label',f'shard-{rank}','--model',str(merged),'--expected-checkpoint',str(endpoint),
                '--dataset-audit',str(root/'dataset-audit.json'),'--output-group',str(root/'evaluation'),
                '--task-cohort',str(CONFIG/'om2w-shards.json'),'--cohort-name',f'shard-{rank}',
                '--actor-port',str(port),'--browser-port-start',str(port+90),'--browser-port-end',str(port+289),
                '--parallel','12','--mem-fraction-static','0.45','--serial-variant',config['variant']]
            workers.append(launch(cmd,f'eval-shard-{rank}.log',env))
        while any(w.poll() is None for w in workers):
            if any(w.poll() not in (None,0) for w in workers):raise RuntimeError('Evaluation worker failed')
            if stopping or time.time()>=deadline:stop();raise RuntimeError('Evaluation deadline')
            time.sleep(10)
        for rank in range(2):
            if json.loads((root/f'evaluation/shard-{rank}/status.json').read_text())['phase']!='complete':raise RuntimeError('Incomplete evaluation')
        result=aggregate(config);write_json(root/'complete.json',result);status('complete',summary=result['summary'])
    except Exception as exc:
        status('failed',error=repr(exc));raise
    finally:
        stop()
        for child in children:
            if child.poll() is None:
                try:child.wait(timeout=60)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        for log in logs:log.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--config',type=Path)
    args=p.parse_args()
    if args.prepare:prepare()
    elif args.config:run(args.config)
    else:p.error('--prepare or --config required')
