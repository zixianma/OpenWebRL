#!/usr/bin/env python3
"""One isolated live collection/update mechanics test; dry-run by default.

This does not submit compute or update the baseline's persistent pointer.
Unit auxiliary weight is a mechanics fixture, not the scientific training weight.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import urllib.request

from evaluate_baseline_checkpoint import REPO, RUNTIME, build_plan, record_restore_evidence, restore_pattern
from resume_baseline import allocation, clean_environment, source_command, write_json

SOURCE = RUNTIME / 'reference-arm-rl-mechanics-20260912'
CHECKPOINT = RUNTIME / 'runs/openwebrl-4b-reference-288861-20260912T040027/iter_0000061'
SELECTOR = Path('/gpfs/scrubbed/zixianma/checkpoints/web/arm/selection/81b452d800d9f859687074f82680dd5257e02d89')


def plan(track, job):
    output = RUNTIME / f'evaluations/arm-rl-mechanics-{track}-{job}'
    p = build_plan(SOURCE, CHECKPOINT, output, job, gpus=2)
    run_id = f'arm-rl-{track}-mechanics-{job}'
    p.update(track=track, mechanics_only=True, scientific_training=False,
             protocol='ARM live collection/backward mechanics; unit auxiliary coefficient; not a benchmark',
             optimizer_updates_requested='one small collection, two PPO epochs; count depends on valid turns',
             wandb_run_id=run_id, parent_training_run='qcq7i4ug')
    p['environment'].update(NUM_ROLLOUT='63', ROLLOUT_BATCH_SIZE='4', GLOBAL_BATCH_SIZE='32',
        BROWSER_CONCURRENCY='8', SGLANG_CONCURRENCY='16', WANDB_RUN_ID=run_id,
        OPENWEBRL_ARM_CONFIG=str(output/'arm-config.json'), OPENWEBRL_ARM_POLICY_ID=run_id,
        OPENWEBRL_ARM_AUX_MANIFEST=str(output/'aux'/'round-{rollout_id}'/'manifest.json'))
    command = p['command']
    command[command.index('--wandb-project')+1] = 'openwebrl-arm'
    command[command.index('--wandb-group')+1] = 'arm-rl-mechanics-v1'
    command += ['--custom-generate-function-path', 'openwebrl.arm_rl_runtime.generate',
                '--sglang-mem-fraction-static', '.35']
    p['arm_config'] = dict(track=track, tokenizer='/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT',
        evaluation_tasks=str(REPO/'openwebrl/data/eval/online-mind2web.jsonl'),
        selector_endpoint='http://127.0.0.1:27501', scored_fraction=.2,
        max_local_states=16, max_pending_per_trajectory=2, rescue_task_cap=2,
        eta=1. if track=='rescue' else 0., **{'lambda':1. if track=='local_preference' else 0.},
        coefficient_purpose='unit-weight mechanics only; calibrate before scientific training')
    p['requested_resources'] = dict(gpus=2, gpu_type='H200', hours=2, cpus=16, memory_gib=480,
        estimated_gpu_cost_usd=3.60, judge_usage_additional=True, approval_required=True)
    return p


def execute(p):
    job = p['job_id']
    if os.getenv('SLURM_JOB_ID') != job or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Execute only in the dedicated, authorized GPU step')
    record = subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
    resources = allocation(record,job,requested_gpus=2)
    steps = subprocess.check_output(['squeue','--steps',f'--jobs={job}','--noheader','--format=%i'],text=True).split()
    allowed = {f'{job}.{x}' for x in ('batch','extern',os.getenv('SLURM_STEP_ID'))}
    if set(steps)-allowed:
        raise ValueError('Another step is active in this allocation')
    from dotenv import dotenv_values
    env = clean_environment()
    for k,v in dotenv_values(REPO/'.env').items():
        if v and k.startswith(('WANDB_','JUDGE_','OPENAI_','AZURE_')): env.setdefault(k,v)
    if env.get('OPENAI_API_KEY') and not env.get('JUDGE_API_BASE'):
        env.update(JUDGE_API_MODE='served',JUDGE_API_BASE='https://api.openai.com/v1')
    env.update(p['environment'])
    output = Path(p['output']);output.mkdir(parents=True,exist_ok=False)
    view=output/'checkpoint-view';view.mkdir()
    (view/CHECKPOINT.name).symlink_to(CHECKPOINT,target_is_directory=True)
    (view/'rollout').symlink_to(CHECKPOINT.parent/'rollout',target_is_directory=True)
    (view/'latest_checkpointed_iteration.txt').write_text('61\n')
    write_json(output/'pilot-manifest.json',p);write_json(output/'arm-config.json',p['arm_config'])
    children=[];handles=[]
    def stop(*unused):
        for child in reversed(children):
            if child.poll() is None: os.killpg(child.pid,signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        devices=env.get('CUDA_VISIBLE_DEVICES','').split(',')
        if len(devices)!=2 or not all(devices): raise ValueError('Expected exactly two assigned CUDA devices')
        selector_env=dict(env,CUDA_VISIBLE_DEVICES=devices[-1])
        handles.append((output/'selector.log').open('w'))
        selector=subprocess.Popen([str(RUNTIME/'arm-reproduction/venv/bin/python'),str(SOURCE/'scripts/serve_arm.py'),
            '--mode','selection','--model',str(SELECTOR),'--port','27501'],cwd=SOURCE,env=selector_env,
            stdout=handles[-1],stderr=subprocess.STDOUT,start_new_session=True)
        children.append(selector)
        deadline=time.monotonic()+600
        while True:
            if selector.poll() is not None:raise RuntimeError('Selector failed to load')
            try:
                with urllib.request.urlopen(p['arm_config']['selector_endpoint']+'/health',timeout=2) as r:
                    if r.status==200:break
            except OSError:pass
            if time.monotonic()>deadline:raise TimeoutError('Selector startup timeout')
            time.sleep(5)
        command=source_command(SOURCE,['timeout','--signal=INT','--kill-after=120',str(resources['maximum_seconds']-600),*p['command']])
        handles.append((output/'training.log').open('w'))
        worker=subprocess.Popen(command,cwd=SOURCE,env=env,stdout=handles[-1],stderr=subprocess.STDOUT,start_new_session=True)
        children.append(worker)
        while worker.poll() is None:
            if selector.poll() is not None:raise RuntimeError('Selector exited during the mechanics test')
            record_restore_evidence(p, log_filename='training.log')
            time.sleep(10)
        text=(output/'training.log').read_text(errors='replace')
        checkpoint=output/'runtime/iter_0000062/common.pt'
        auxpath=output/'aux/round-62/manifest.json'
        aux=json.loads(auxpath.read_text()) if auxpath.exists() else {}
        record_restore_evidence(p, log_filename='training.log')
        receipt=output/'checkpoint_restore_evidence.json'
        restore_text=json.loads(receipt.read_text()).get('restore_line','') if receipt.exists() else text
        restored=bool(re.search(restore_pattern(p),restore_text))
        trained_aux=any(float(x)>0 for x in re.findall(r"arm_aux_rows[\"']?\s*[:=]\s*([0-9.]+)",text))
        complete=worker.returncode==0 and checkpoint.is_file() and bool(aux.get('records')) and trained_aux and restored
        write_json(output/'status.json',dict(mechanics_completed=complete,exit_code=worker.returncode,
            aux_rows=len(aux.get('records',[])),checkpoint=str(checkpoint),scientific_result=False,
            selected_checkpoint_restored=restored,
            auxiliary_rows_reached_backward=trained_aux,
            remaining_gates=['component gradient calibration','TP2 checkpoint save/reload verification','full-size collection parity']))
        if not complete:raise RuntimeError('Mechanics gate incomplete; inspect logs before scaling')
    finally:
        stop()
        for child in children:
            try:child.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()
        for handle in handles:handle.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track',choices=['rescue','local_preference'],required=True)
    parser.add_argument('--job-id',default='APPROVED_JOB')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();p=plan(args.track,args.job_id)
    if args.execute:execute(p)
    else:print(json.dumps(p,indent=2))
