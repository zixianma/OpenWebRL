#!/usr/bin/env python3
"""Own joint-data training -> merge -> two-shard OM2W evaluation in one job."""
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

from train_arm_joint_sft import REPO, RUNTIME, digest, read_rows, write_json
from resume_arm_c2_training import allocation_deadline, parse_job_record
try:
    from project_docs import write_document_section
except ModuleNotFoundError:  # Imported as scripts.run_arm_joint_pipeline.
    from scripts.project_docs import write_document_section

PYTHON=RUNTIME/'venv/bin/python'
CONFIG_DIR=REPO/'openwebrl/docs/arm_results/joint_data_v2'
TASK_FILE=REPO/'openwebrl/data/eval/online-mind2web.jsonl'


def cohorts():
    source=json.loads((REPO/'openwebrl/docs/arm_c2_full300_cohorts.json').read_text())
    if digest(TASK_FILE)!=source['task_file_sha256']:raise ValueError('OM2W task file changed')
    full=source['cohorts']['all_300'];shards={}
    for rank in range(2):
        shards[f'shard-{rank}']=dict(count=150,indices=full['indices'][rank::2],task_ids=full['task_ids'][rank::2])
    if len(set(shards['shard-0']['task_ids'])|set(shards['shard-1']['task_ids']))!=300:
        raise ValueError('Evaluation shards must partition 300 tasks')
    return dict(task_file_sha256=source['task_file_sha256'],cohorts=shards)


def prepare():
    from train_arm_joint_ddp import frozen_config
    CONFIG_DIR.mkdir(parents=True,exist_ok=True)
    write_json(CONFIG_DIR/'om2w-shards.json',cohorts())
    for objective in ['sft','dpo']:
        config=frozen_config(objective)
        config['eval_shards_sha256']=digest(CONFIG_DIR/'om2w-shards.json')
        config['handoff_hashes']={name:digest(REPO/'scripts'/name) for name in
                                ['merge_arm_c2_student.py','run_arm_full300_checkpoint_eval.py']}
        path=CONFIG_DIR/f'{objective}-2gpu-config.json';write_json(path,config)
        print(json.dumps(dict(config=str(path),objective=objective,gpus=2,
                 hours=3 if objective=='sft' else 5,updates=174,evaluation_tasks=300,submitted=False)))


def aggregate(root,objective,shards):
    from summarize_arm_c2_full300 import load_records,rate_report,paired_report,percent
    from summarize_arm_c2_vs_1a_full300 import behavior_from_records
    records={}
    for label,cohort in shards['cohorts'].items():
        output=root/'evaluation'/label/'online-mind2web'
        manifest=json.loads((output/'manifest.json').read_text())
        if manifest['task_ids']!=cohort['task_ids'] or manifest.get('judge')!='o4-mini' or manifest.get('judge_protocol')!='online_mind2web/AgentTrek':
            raise ValueError('Evaluation cohort/protocol mismatch')
        rows=load_records(output,cohort['task_ids'])
        if set(rows)!=set(cohort['task_ids']):raise ValueError('Missing evaluation tasks')
        if set(rows)&set(records):raise ValueError('Duplicate cross-shard evaluation tasks')
        records.update(rows)
    ids=[r['task_id'] for r in read_rows(TASK_FILE)]
    if len(records)!=300 or set(records)!=set(ids):raise ValueError('Incomplete OM2W result')
    report=dict(objective=objective,checkpoint_update=174,rates=rate_report(records,ids),
                behavior=behavior_from_records(records,ids),shards=shards,
                evaluation_completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(root/'evaluation/all300-summary.json',report)
    write_json(root/'evaluation/all300-records.json',records)
    rates=report['rates']
    doc=REPO/f'openwebrl/docs/ARM_JOINT_{objective.upper()}_RESULTS.md'
    write_document_section(doc,f'# Joint-data {objective.upper()}: endpoint OM2W evaluation\n\n'
      f"Update 174, 5,540 training states, fresh evaluation of all 300 tasks.\n\n"
      f"- Overall: {rates['successes']}/300 = {percent(rates['overall'])}.\n"
      f"- Valid-only: {rates['successes']}/{rates['valid']} = {percent(rates['valid_only'])}.\n"
      f"- Unavailable: {rates['unavailable']}; missing: {rates['missing']}.\n\n"
      f"Protocol: one candidate, no inference ARM, o4-mini/AgentTrek judge. "
      f"Unavailable results were not silently replaced.\n\n"
      f"[Full report]({root}/evaluation/all300-summary.json) · "
      f"[Rollouts]({root}/evaluation)\n")
    # The later finisher creates the paired report without launching more compute.
    other='dpo' if objective=='sft' else 'sft'
    other_file=RUNTIME/f'runs/joint-v2-{other}-2gpu/evaluation/all300-records.json'
    if other_file.exists():
        all_records={objective:records,other:json.loads(other_file.read_text())}
        if set(all_records[other])!=set(ids):raise ValueError('Other endpoint has incomplete coverage')
        comparison=dict(rates={name:rate_report(rows,ids) for name,rows in all_records.items()},
               paired_common_valid=paired_report(all_records['sft'],all_records['dpo'],ids),
               note='Independent training from original SFT actor; evaluation times differ; one training seed.')
        write_json(CONFIG_DIR/'joint-sft-vs-dpo-om2w.json',comparison)


def run(args):
    job=os.environ.get('SLURM_JOB_ID')
    if not job or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Pipeline must run in its approved allocation')
    record=parse_job_record(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True,timeout=20))
    if record.get('JobState')!='RUNNING' or int(record['NumNodes'])!=1 or int(record['NumCPUs'])<16:
        raise ValueError('Expected one running node and 16 allocated CPUs')
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],text=True,timeout=20).strip().splitlines()
    if len(devices)!=2 or any('H200' not in d for d in devices):raise ValueError('Expected two allocated H200s')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=20).strip():
        raise ValueError('Assigned GPUs are occupied')
    config_path=args.config or CONFIG_DIR/f'{args.objective}-2gpu-config.json';config=json.loads(config_path.read_text())
    for name,checksum in {**config['code_hashes'],**config['handoff_hashes']}.items():
        if digest(REPO/'scripts'/name)!=checksum:raise ValueError(f'Code changed: {name}')
    shard_path=CONFIG_DIR/'om2w-shards.json'
    if digest(shard_path)!=config['eval_shards_sha256']:raise ValueError('Evaluation shard manifest changed')
    root=Path(config['output']);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    shards=json.loads(shard_path.read_text());deadline=allocation_deadline(record,margin_minutes=5).timestamp()
    # Reserve 65 minutes for merge, two 150-task workers, report, and shutdown.
    training_stop=deadline-65*60
    if (root/'complete.json').exists():raise ValueError('Pipeline already complete')
    processes=[];logs=[];stopping=False
    def status(phase,**extra):
        write_json(root/'status.json',dict(phase=phase,allocation=job,objective=args.objective,
          updated_utc=datetime.now(timezone.utc).isoformat(),allocation_end_utc=datetime.fromtimestamp(deadline+300,timezone.utc).isoformat(),**extra))
    def launch(command,logname,env=None):
        log=(root/logname).open('a');logs.append(log)
        process=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(process);return process
    def terminate():
        for process in processes:
            if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
    def stop(*_):
        nonlocal stopping
        stopping=True;terminate()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        if not (root/'student/complete.json').exists():
            status('training_startup',training_stop_unix=training_stop)
            command=[str(PYTHON),'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',
                     str(REPO/'scripts/train_arm_joint_ddp.py'),'--config',str(config_path),'--stop-unix',str(training_stop)]
            if (root/'student/latest-checkpoint.json').exists():command.append('--resume')
            train=launch(command,'training.log');code=train.wait()
            if code or not (root/'student/complete.json').exists():
                status('training_paused_or_failed',returncode=code)
                raise RuntimeError(f'Training failed or paused before the endpoint: returncode={code}')
        if stopping:return
        endpoint=root/'student/update-000174';merged=root/'evaluation/merged-endpoint'
        status('merging_endpoint')
        merge=launch([str(PYTHON),str(REPO/'scripts/merge_arm_c2_student.py'),'--run-root',str(root),
                     '--checkpoint',str(endpoint),'--output',str(merged)],'merge.log')
        if merge.wait():raise RuntimeError('Endpoint merge failed')
        if stopping:return
        status('evaluating_all300')
        workers=[]
        base_port=23110 if args.objective=='sft' else 24110
        for rank in range(2):
            port=base_port+rank*300;env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=devices[rank].split(',',1)[0]
            command=[str(PYTHON),str(REPO/'scripts/run_arm_full300_checkpoint_eval.py'),
                     '--job-id',job,'--run-root',str(root),'--label',f'shard-{rank}',
                     '--model',str(merged),'--expected-checkpoint',str(endpoint),
                     '--dataset-audit',str(root/'dataset-audit.json'),'--output-group',str(root/'evaluation'),
                     '--task-cohort',str(shard_path),'--cohort-name',f'shard-{rank}',
                     '--actor-port',str(port),'--browser-port-start',str(port+90),'--browser-port-end',str(port+289),
                     '--parallel','12','--mem-fraction-static','0.45']
            workers.append(launch(command,f'eval-shard-{rank}.log',env))
        while any(p.poll() is None for p in workers):
            if any(p.poll() not in (None,0) for p in workers):
                terminate();raise RuntimeError('An evaluation worker failed')
            if stopping:return
            time.sleep(30)
        if any(p.returncode for p in workers):raise RuntimeError('Evaluation worker failed')
        for rank in range(2):
            state=json.loads((root/f'evaluation/shard-{rank}/status.json').read_text())
            if state['phase']!='complete':status('evaluation_paused',shard=rank);return
        aggregate(root,args.objective,shards)
        write_json(root/'complete.json',dict(objective=args.objective,updates=174,evaluated_tasks=300,allocation=job))
        status('complete',updates=174,evaluated_tasks=300)
    except Exception as exc:
        status('failed',error=repr(exc));raise
    finally:
        terminate()
        for process in processes:
            if process.poll() is None:
                try:process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
        for log in logs:log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--objective',choices=['sft','dpo'])
    parser.add_argument('--config',type=Path)
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.objective:run(args)
    else:parser.error('--objective is required to run')
