#!/usr/bin/env python3
"""Fresh, repeated ARM RL collections in a dedicated user-authorized allocation."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import urllib.request

from run_arm_turn_bonus_calibration import SELECTOR, selector_preflight, health_issue
from evaluate_baseline_checkpoint import REPO, RUNTIME
from resume_baseline import allocation, clean_environment, source_command, validate_source, write_json

SOURCE = RUNTIME/'reference-arm-turn-bonus-cycles-20260913-v3'
INITIAL = Path('/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT')


def plan(job, provider='arm', minutes=240, interactive_2gpu=False):
    if provider not in ('arm','sol'): raise ValueError('Unknown selector provider')
    if not 1<=minutes<=1440: raise ValueError('Training allocation cap is 1440 minutes')
    validate_source(SOURCE)
    if not json.loads((SOURCE/'reference_manifest.json').read_text())['arm_turn_bonus_calibration'].get('supports_multiple_collections'):
        raise ValueError('Repeated-collection source required')
    import yaml
    output = RUNTIME/f'evaluations/{provider}-turn-bonus-fresh-{job}'
    run_id = f'{provider}-turn-bonus-fresh-{job}'
    selector_port = 31000 + int(job)%20000 if str(job).isdigit() else 27511
    browser = yaml.safe_load((SOURCE/'openwebrl/browser_training_config.yaml').read_text())
    browser['browser_rollout_concurrency'] = 48
    env = dict(NUM_GPUS='4',TP_SIZE='4',NUM_ROLLOUT='4',HF_CHECKPOINT=str(INITIAL),
        SLIME_LOAD_CHECKPOINT=str(INITIAL),BROWSER_MAX_STEPS='15',ROLLOUT_BATCH_SIZE='48',N_SAMPLES='5',
        GLOBAL_BATCH_SIZE='256',CONTEXT_LEN='32768',RESPONSE_LEN='1024',BROWSER_CONCURRENCY='48',
        SGLANG_CONCURRENCY='48',LEARNING_RATE='1e-6',RECOMPUTE_ACTIVATIONS='1',SAVE_INTERVAL='1',
        SAVE_DIR=str(output/'runtime'),WANDB_MODE='online',WANDB_RUN_ID=run_id,JUDGE_MODEL='gpt-4.1',
        OMP_NUM_THREADS='2',RAY_ADDRESS='local',RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=str(8*1024**3),
        OPENWEBRL_MULTIMODAL_STORAGE_DIR=f'/tmp/arm-turn-bonus-{job}-multimodal',
        SLIME_ADAPTIVE_QUERY_BLACKLIST_PATH=str(SOURCE/'reference_empty_blacklist.txt'),
        SLIME_BROWSER_QUERY_BLACKLIST_PATH=str(SOURCE/'reference_empty_blacklist.txt'),
        BROWSER_TRAIN_CONFIG=str(output/'browser-training-config.json'),FLASHINFER_WORKSPACE_BASE=str(RUNTIME),
        OPENWEBRL_ARM_TURN_BONUS_CONFIG=str(output/'arm-config.json'),WANDB_CACHE_DIR=str(output/'wandb-cache'))
    command = ['bash',str(SOURCE/'scripts/run_h200_browser.sh'),'--use-wandb','--wandb-mode','online',
        '--wandb-project','openwebrl','--wandb-team','zixianma','--wandb-group','executed-turn-bonus-from-zero',
        '--disable-wandb-random-suffix','--wandb-dir',str(output/'wandb'),'--sglang-disable-cuda-graph',
        '--rollout-health-check-first-wait','180','--use-fault-tolerance',
        '--router-balance-abs-threshold','2','--skip-eval-before-train','--lr-decay-iters','1',
        '--save-debug-rollout-data',str(output/'runtime/rollout_recovery/{rollout_id}.pt'),
        '--custom-generate-function-path','openwebrl.arm_turn_bonus.generate','--sglang-mem-fraction-static','.35']
    arm = dict(shadow_only=False,train_after_calibration=True,beta=.5,scored_fraction=.2,k=5,seed=42,
        seconds_per_optimizer_update=150,max_pending_per_trajectory=2,label_timeout_seconds=120,
        selector_endpoint=f'http://127.0.0.1:{selector_port}',selector_checkpoint=str(SELECTOR),
        checkpoint=str(INITIAL),initial_checkpoint=str(INITIAL),policy_id=f'{run_id}:uninitialized',
        run_id=run_id,output=str(output),run_output=str(output),minimum_cycle_seconds=3300,
        evaluation_tasks=str(REPO/'openwebrl/data/eval/online-mind2web.jsonl'))
    if provider=='sol':
        arm['selector_checkpoint']='gpt-5.6-sol'
        arm['selector_provider']='openai-responses'
        arm['reasoning_effort']='medium'
    prepared = dict(job_id=str(job),output=str(output),source=str(SOURCE),checkpoint=str(INITIAL),
        requested_resources=dict(gpus=4,gpu_type='H200',hours=minutes/60,gpu_hours=4*minutes/60,cpus=32,memory_gib=480,browsers=48),
        command=command,environment=env,browser_config=browser,arm_config=arm,wandb_run_id=run_id,
        requested_iterations=4,fresh_optimizer=True,evaluation_in_allocation=False,selector_provider=provider,
        selector_port=selector_port)
    if interactive_2gpu:
        # Preserve the real data, batch, loss and calibration criteria. Only
        # topology/concurrency change to fit an existing small allocation.
        prepared['requested_resources'].update(gpus=2,cpus=8,memory_gib=240,
            browsers=8,gpu_hours=2*minutes/60)
        prepared['environment'].update(NUM_GPUS='2',TP_SIZE='2',BROWSER_CONCURRENCY='8',
            SGLANG_CONCURRENCY='8')
        prepared['browser_config']['browser_rollout_concurrency']=8
        prepared['interactive_2gpu']=True
    return prepared


def tail(path, limit=262144):
    if not path.exists(): return ''
    with path.open('rb') as f:
        f.seek(0,2); f.seek(max(0,f.tell()-limit)); return f.read().decode(errors='replace')


def verify_live_shadow(p,root,no_optimizer_updates):
    """Separate live-hook diagnostics from statistical calibration/training success."""
    if not p.get('diagnostic_only') or not p['arm_config'].get('shadow_only') or not no_optimizer_updates:
        raise ValueError('Live shadow diagnostic must execute zero optimizer updates')
    current=Path(root)/'iterations/0000'
    report=json.loads((current/'calibration.json').read_text())
    complete=json.loads((current/'collection_complete.json').read_text())
    if (report.get('applied_beta')!=0 or report.get('collection_records',0)<1
            or report.get('admitted',0)<1 or complete.get('optimizer_updates')!=0):
        raise ValueError('Live shadow diagnostic lacks real admitted turn labels or changed actor rewards')
    return report


def execute(p):
    if p.get('variant_continuation'):
        from resume_arm_failure_variants import validate_plan
        readiness=validate_plan(p)
    elif p.get('experiment') == 'additive-all-failure':
        from run_arm_failure_additive import require_receipt
        readiness=require_receipt(p)
    elif p.get('experiment') == 'arm-all-failure-bonus':
        from run_arm_failure_bonus import require_receipt
        readiness=require_receipt(p)
    else:
        from arm_launch_preflight import require_receipt
        readiness=require_receipt()
    SOURCE = Path(p['source'])
    validate_source(SOURCE)
    job=p['job_id']
    if os.getenv('SLURM_JOB_ID') != job or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Execute inside the dedicated authorized allocation')
    requested=p['requested_resources']
    resources=allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),job,requested_gpus=requested['gpus'],maximum_hours=requested['hours'])
    if resources['cpus']<requested['cpus'] or resources['allocated_memory_gib']<requested['memory_gib']:
        raise ValueError('Allocation does not meet the prepared CPU/memory profile')
    steps=subprocess.check_output(['squeue','--steps',f'--jobs={job}','--noheader','--format=%i'],text=True).split()
    if set(steps)-{f'{job}.{x}' for x in ('batch','extern','interactive',os.getenv('SLURM_STEP_ID'))}:
        raise ValueError('Another worker is active in this allocation')
    deadline=time.time()+min(resources['maximum_seconds'],p['requested_resources']['hours']*3600-180)
    p['arm_config']['deadline_epoch_seconds']=deadline
    from dotenv import dotenv_values
    env={k:v for k,v in clean_environment().items() if not k.startswith('OPENWEBRL_ARM_')}
    for k,v in dotenv_values(REPO/'.env').items():
        if v and k.startswith(('WANDB_','JUDGE_','OPENAI_','AZURE_')): env.setdefault(k,v)
    if env.get('OPENAI_API_KEY') and not env.get('JUDGE_API_BASE'):
        env.update(JUDGE_API_MODE='served',JUDGE_API_BASE='https://api.openai.com/v1')
    if not env.get('WANDB_API_KEY') or not any(env.get(k) for k in ('OPENAI_API_KEY','JUDGE_API_KEY','AZURE_OPENAI_API_KEY')):
        raise ValueError('Missing W&B or terminal-judge credentials')
    env.update(p['environment'])
    devices=env.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(devices)!=requested['gpus'] or len(set(devices))!=requested['gpus'] or not all(devices):
        raise ValueError('GPU assignment does not match the prepared profile')
    if p.get('resume_from'):
        from resume_arm_turn_bonus import validate_resume_plan
        validate_resume_plan(p,require_gpu_restore=True)
    root=Path(p['output']); root.mkdir(parents=True,exist_ok=False)
    if p.get('resume_from'):
        from resume_arm_turn_bonus import stage_checkpoint_reference
        stage_checkpoint_reference(p)
    for name,value in [('launch_manifest',p),('arm-config',p['arm_config']),('browser-training-config',p['browser_config'])]:
        write_json(root/f'{name}.json',value)
    write_json(root/'launch-readiness.json',readiness)
    children=[]; handles=[]; selector=None; worker=None; validated={}; stage='actor-startup'; current=None
    result=dict(training_complete=False,no_optimizer_updates=True,completed_iterations=0)
    def status(**values):
        result.update(values,updated_at_epoch_seconds=time.time(),stage=stage,seconds_remaining=int(deadline-time.time()))
        write_json(root/'status.json',result)
    def interrupted(*unused): raise InterruptedError('Allocation shutdown requested')
    old=[signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT)]
    def stop_child(child):
        if child is not None and child.poll() is None:
            try: os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try: child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                try: os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                child.wait()
    try:
        status()
        handles.append((root/'collection.log').open('w'))
        worker=subprocess.Popen(source_command(SOURCE,['timeout','--signal=INT','--kill-after=60',str(int(deadline-time.time())),*p['command']]),
            cwd=SOURCE,env=env,stdout=handles[-1],stderr=subprocess.STDOUT,start_new_session=True)
        children.append(worker)
        last_request=None
        while True:
            if time.time()>=deadline: raise TimeoutError('Allocation deadline reached')
            request_path=root/'teacher-request.json'
            request=json.loads(request_path.read_text()) if request_path.exists() else None
            if request is not None and request!=last_request and worker.poll() is None:
                current=root/'iterations'/f"{request['rollout_id']:04d}"
                if request['phase']=='collection':
                    if selector is not None: raise ValueError('Teacher from previous collection was not released')
                    stage='selector-startup'; status(iteration=request['rollout_id']+1)
                    handles.append((current/'selector.log').open('w'))
                    selector_env=dict(env,CUDA_VISIBLE_DEVICES=devices[-1],PYTHONPATH=str(SOURCE),
                        CPATH=str(RUNTIME/'src/python-headers/Include')+':'+str(RUNTIME/'src/python-headers'))
                    selector_command=[str(RUNTIME/'arm-reproduction/venv/bin/python'),str(SOURCE/'scripts/serve_arm.py'),
                        '--mode','selection','--model',str(SELECTOR),'--port',str(p['selector_port'])]
                    if p['selector_provider']=='sol':
                        selector_command=[str(RUNTIME/'venv/bin/python'),str(REPO/'scripts/serve_sol_selector.py'),
                            '--output',str(root/'sol-api'),'--port',str(p['selector_port']),'--max-cost-usd','200']
                        selector_env['CUDA_VISIBLE_DEVICES']=''
                    selector=subprocess.Popen(selector_command,cwd=SOURCE,env=selector_env,
                        stdout=handles[-1],stderr=subprocess.STDOUT,start_new_session=True)
                    children.append(selector)
                    startup=min(deadline-120,time.time()+540)
                    while True:
                        if selector.poll() is not None: raise RuntimeError('Selector failed to start')
                        try:
                            with urllib.request.urlopen(p['arm_config']['selector_endpoint']+'/health',timeout=2) as response:
                                if response.status==200: break
                        except OSError: pass
                        if time.time()>startup: raise TimeoutError('Selector startup timeout')
                        time.sleep(3)
                    selector_preflight(p['arm_config']['selector_endpoint'],current,
                        expected_api_model='gpt-5.6-sol' if p['selector_provider']=='sol' else None)
                    stage='collection'
                elif request['phase']=='training':
                    stop_child(selector); selector=None; stage='training'
                else: raise ValueError('Unknown teacher lifecycle request')
                write_json(root/'teacher-ready.json',request); last_request=request
            if selector is not None and selector.poll() is not None: raise RuntimeError('Selector exited during collection')
            api_usage=root/'sol-api/usage.json'
            if (root/'sol-api/budget-exhausted.json').exists():
                raise RuntimeError('Sol API usage cap reached; preserve collection for inspection')
            if api_usage.exists() and json.loads(api_usage.read_text()).get('consecutive_failures',0)>=5:
                raise RuntimeError('Five consecutive Sol selector failures; preserve collection for inspection')
            recent=tail(root/'runtime/progress.log')+'\n'+tail(root/'collection.log',65536)
            live_path=current/'live_metrics.json' if current else root/'unused'
            live=json.loads(live_path.read_text()) if live_path.exists() else {}
            if issue:=health_issue(live,recent): raise RuntimeError(issue)
            if re.search(r'\[TrainMetrics\][^\n]*\bstep=\d+\b[^\n]*\bgrad_norm=',recent):
                result['no_optimizer_updates']=False
            # Validate each saved checkpoint against accumulated native PPO steps.
            for saved in sorted((root/'iterations').glob('*/checkpoint-saved.json')):
                row=json.loads(saved.read_text()); rid=row['rollout_id']
                if rid in validated: continue
                marker=root/'runtime/latest_checkpointed_iteration.txt'
                if not marker.exists() or int(marker.read_text())!=rid:
                    raise ValueError('Checkpoint marker disagrees with completed native save')
                expected=p.get('initial_optimizer_updates',0)+sum(json.loads(x.read_text())['expected_optimizer_updates']
                    for x in (root/'iterations').glob('*/training_gate.json')
                    if int(x.parent.name)<=rid)
                report=saved.parent/'checkpoint-validation.json'
                subprocess.run(source_command(SOURCE,['python',str(REPO/'scripts/inspect_training_checkpoint.py'),
                    str(root/'runtime'),'--expected-updates',str(expected),'--report',str(report)]),
                    env=env,check=True,stdout=subprocess.DEVNULL,timeout=120)
                validated[rid]=json.loads(report.read_text())
                write_json(root/'completed-checkpoints.json',list(validated.values()))
                status(completed_iterations=len(validated),completed_optimizer_updates=expected,
                    last_valid_checkpoint=row['checkpoint'])
            status(live_metrics=live)
            if worker.poll() is not None: break
            time.sleep(10)
        if worker.returncode: raise RuntimeError(f'Native training exited {worker.returncode}; inspect preserved artifacts')
        stage='complete'
        if p.get('diagnostic_live_shadow'):
            report=verify_live_shadow(p,root,result['no_optimizer_updates'])
            status(shadow_complete=True,training_complete=False,has_saved_checkpoints=False,exit_code=0,
                completed_optimizer_updates=0,live_admitted_labels=report['admitted'],
                stop_reason='Live executed-turn hook verified; no training or statistical calibration claim')
            return
        status(training_complete=len(validated)==p['requested_iterations'],has_saved_checkpoints=bool(validated),exit_code=0,
            stop_reason='requested iterations complete' if len(validated)==p['requested_iterations'] else 'budget or calibration gate; inspect iteration artifacts')
        if not validated: raise RuntimeError('No completed ARM training iteration')
    except BaseException as exc:
        status(failed=True,error_type=type(exc).__name__,error=str(exc)[:500])
        raise
    finally:
        for child in reversed(children): stop_child(child)
        for handle in handles: handle.close()
        for sig,handler in zip((signal.SIGTERM,signal.SIGINT),old): signal.signal(sig,handler)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id',default='APPROVED_JOB')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--provider',choices=['arm','sol'],default='arm')
    parser.add_argument('--minutes',type=int,default=240)
    parser.add_argument('--interactive-2gpu',action='store_true',help='Real training in an existing 2 H200 / 8 CPU / 240 GiB allocation')
    args=parser.parse_args(); p=plan(args.job_id,args.provider,args.minutes,args.interactive_2gpu)
    if args.execute: execute(p)
    else: print(json.dumps(p,indent=2))
