#!/usr/bin/env python3
"""Verify ARM/Sol launch readiness before any explicitly approved submission.

--check runs CPU tests/native parsers and one live Sol HTTP preflight, no GPUs.
--submit requires separate user approval of the printed resources and budget.
Both job controllers also reject missing or stale receipts before model loading.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from evaluate_baseline_checkpoint import REPO,RUNTIME
from resume_baseline import validate_source,write_json,source_command

DEFAULT_RECEIPT=RUNTIME/'arm-turn-bonus-preparation/launch-readiness/receipt.json'
FILES=(
    'scripts/run_arm_sol_gpu_diagnostic.py','scripts/arm_gpu_smoke_fixture.py',
    'scripts/run_arm_sol_gpu_diagnostic.sbatch',
    'scripts/run_arm_live_shadow.py',
    'scripts/arm_launch_preflight.py','scripts/check_arm_native_launch.py',
    'scripts/run_arm_turn_bonus_cycles.py','scripts/run_arm_turn_bonus_calibration.py',
    'scripts/resume_arm_turn_bonus.py','scripts/resume_arm_turn_bonus_4gpu.sbatch',
    'scripts/resume_arm_turn_bonus_8gpu.sbatch','scripts/resume_arm_turn_bonus_4gpu_16h.sbatch','scripts/start_arm_resume_8gpu.py','scripts/resume_baseline.py',
    'scripts/check_arm_gpu_checkpoint_restore.py','scripts/monitor_arm_turn_bonus.py',
    'scripts/run_sol_inference.py','scripts/serve_sol_selector.py','scripts/inspect_training_checkpoint.py',
    'scripts/run_arm_turn_bonus_fresh_4gpu.sbatch','scripts/run_sol_turn_bonus_fresh_4gpu.sbatch',
    'scripts/run_sol_inference_2gpu.sbatch','scripts/h200_env.sh',
    'openwebrl/arm_eval.py','openwebrl/arm_inference.py','openwebrl/generate_browser.py',
    'openwebrl/run_evaluate.py','openwebrl/eval/reward_online_mind2web.py',
    'openwebrl/env/local_process_env.py','openwebrl/arm_turn_bonus.py',
    'openwebrl/arm_turn_bonus_cycles.py','openwebrl/arm_turn_bonus_runtime.py',
    'openwebrl/data/eval/online-mind2web.jsonl')


def fingerprint():
    from run_arm_turn_bonus_cycles import SOURCE,INITIAL
    validate_source(SOURCE)
    paths=[REPO/name for name in FILES]
    paths += sorted((REPO/'tests').glob('test_arm_turn_bonus*.py'))
    paths += sorted((REPO/'tests').glob('test_sol*.py'))
    paths += [SOURCE/'reference_manifest.json',INITIAL/'config.json',INITIAL/'tokenizer_config.json',
        RUNTIME/'arm-reproduction/source/inference/selection_prompt.py']
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def require_receipt(path=None):
    path=Path(path or os.environ.get('ARM_READINESS_FILE',DEFAULT_RECEIPT)).resolve()
    if not path.is_file(): raise ValueError('Missing readiness receipt; run scripts/arm_launch_preflight.py --check before submission')
    receipt=json.loads(path.read_text())
    if not receipt.get('passed') or receipt.get('fingerprint')!=fingerprint():
        raise ValueError('Readiness receipt is failed or stale; rerun scripts/arm_launch_preflight.py --check')
    for key in ('arm_native','sol_native','sol_http'):
        artifact=Path(receipt['artifacts'][key])
        if not artifact.is_file() or not json.loads(artifact.read_text()).get('passed'):
            raise ValueError('Missing passing readiness artifact: '+key)
        if hashlib.sha256(artifact.read_bytes()).hexdigest()!=receipt['artifact_sha256'][key]:
            raise ValueError('Readiness artifact changed: '+key)
    return receipt


def live_http_check(output):
    """Run the production server CLI, then the production /select preflight."""
    from run_arm_turn_bonus_calibration import selector_preflight
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    command=[str(RUNTIME/'venv/bin/python'),str(REPO/'scripts/serve_sol_selector.py'),
        '--port','0','--output',str(output/'api'),'--concurrency','2','--max-requests','2','--max-cost-usd','1']
    with (output/'server.log').open('w') as log:
        child=subprocess.Popen(source_command(REPO,command),cwd=REPO,env=env,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True)
        try:
            limit=time.monotonic()+45
            while True:
                if child.poll() is not None: raise RuntimeError('Sol HTTP server failed to start; inspect server.log')
                text=(output/'server.log').read_text(errors='replace')
                match=re.search(r'Uvicorn running on (http://127\.0\.0\.1:\d+)',text)
                if match: break
                if time.monotonic()>limit: raise TimeoutError('Sol HTTP server startup timed out')
                time.sleep(.2)
            selector_preflight(match[1],output,expected_api_model='gpt-5.6-sol')
        finally:
            if child.poll() is None:
                try: os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError: pass
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try: os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                child.wait()
    return output/'selector_preflight.json'


def check(receipt_path):
    receipt_path=Path(receipt_path).resolve()
    started=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=receipt_path.parent/started; output.mkdir(parents=True,exist_ok=False)
    before=fingerprint()
    write_json(receipt_path,dict(passed=False,stage='checking',output=str(output)))
    result=dict(passed=False,checked_utc=started,output=str(output),fingerprint=before,artifacts={})
    try:
        # Keep isolated CPU test workers below the login-node cgroup limit.
        # In particular, importing the Sol selector stack can otherwise be
        # SIGKILLed before the readiness receipt is produced.
        test_env=dict(os.environ, MALLOC_ARENA_MAX='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        # Isolate test modules so Torch/native imports do not accumulate across
        # suites under the login user's small host-memory cgroup limit.
        for pattern in ('test_arm_turn_bonus*.py','test_sol*.py'):
            for test_file in sorted((REPO/'tests').glob(pattern)):
                with (output/(test_file.stem+'.log')).open('w') as log:
                    subprocess.run([str(RUNTIME/'venv/bin/python'),'-m','unittest','discover','-s','tests','-p',test_file.name,'-v'],
                        cwd=REPO,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=120,env=test_env)
        for name in FILES:
            if name.endswith('.sbatch'): subprocess.run(['bash','-n',str(REPO/name)],check=True)
        for provider in ('arm','sol'):
            with (output/f'{provider}-native-command.log').open('w') as log:
                subprocess.run([str(RUNTIME/'venv/bin/python'),str(REPO/'scripts/check_arm_native_launch.py'),
                    '--provider',provider,'--output',str(output/'native')],cwd=REPO,
                    check=True,stdout=log,stderr=subprocess.STDOUT,timeout=150)
            result['artifacts'][provider+'_native']=str(output/'native'/f'{provider}-report.json')
        with (output/'interactive-native-command.log').open('w') as log:
            subprocess.run([str(RUNTIME/'venv/bin/python'),str(REPO/'scripts/check_arm_native_launch.py'),
                '--provider','arm','--interactive-2gpu','--output',str(output/'interactive-native')],
                cwd=REPO,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=150)
        result['artifacts']['interactive_native']=str(output/'interactive-native/arm-report.json')
        with (output/'gpu-diagnostic-native.log').open('w') as log:
            subprocess.run([str(RUNTIME/'venv/bin/python'),str(REPO/'scripts/run_arm_sol_gpu_diagnostic.py'),
                '--native-check',str(output/'gpu-diagnostic-native')],cwd=REPO,
                check=True,stdout=log,stderr=subprocess.STDOUT,timeout=180)
        result['artifacts']['gpu_diagnostic_native']=str(output/'gpu-diagnostic-native/parse.json')
        result['artifacts']['sol_http']=str(live_http_check(output/'sol-http'))
        if before!=fingerprint(): raise ValueError('Launch sources changed during preflight; rerun after edits finish')
        result.update(passed=True,artifact_sha256={k:hashlib.sha256(Path(v).read_bytes()).hexdigest()
            for k,v in result['artifacts'].items()},gpu_validated=False,
            limitation='CPU fixtures exercise native control flow; real GPU loading and optimizer execution remain unverified.')
        write_json(output/'receipt.json',result); write_json(receipt_path,result)
        require_receipt(receipt_path)
        return result
    except BaseException as exc:
        result.update(error_type=type(exc).__name__,error=str(exc)[:300])
        write_json(output/'receipt.json',result); write_json(receipt_path,result)
        raise


def submission_commands(receipt,workflow,train_minutes=239,inference_minutes=119):
    if not 1<=train_minutes<=240 or not 1<=inference_minutes<=120: raise ValueError('Minutes exceed prepared budget caps')
    path=str(Path(receipt).resolve())
    if ',' in path: raise ValueError('Receipt path cannot contain a Slurm export separator')
    specs=[('arm','run_arm_turn_bonus_fresh_4gpu.sbatch',train_minutes,'ARM_TRAIN_MINUTES',4,32,480),
        ('sol','run_sol_turn_bonus_fresh_4gpu.sbatch',train_minutes,'ARM_TRAIN_MINUTES',4,32,480),
        ('inference','run_sol_inference_2gpu.sbatch',inference_minutes,'SOL_INFERENCE_MINUTES',2,16,240)]
    return [dict(workflow=name,gpus=gpus,minutes=minutes,cpus=cpus,memory_gib=memory,
        command=['sbatch','--parsable',f'--time={minutes//60:02d}:{minutes%60:02d}:00',
                 f'--export=ALL,ARM_READINESS_FILE={path},{variable}={minutes}',str(REPO/'scripts'/script)])
        for name,script,minutes,variable,gpus,cpus,memory in specs if workflow in ('all',name)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--verify-only',action='store_true')
    mode.add_argument('--submit',choices=['all','arm','sol','inference'],help='Only after exact user allocation/budget approval')
    p.add_argument('--receipt',type=Path,default=DEFAULT_RECEIPT)
    p.add_argument('--train-minutes',type=int,default=239); p.add_argument('--inference-minutes',type=int,default=119)
    args=p.parse_args()
    if args.check:
        r=check(args.receipt); print(json.dumps(dict(passed=True,receipt=str(args.receipt),artifacts=r['artifacts']))); return
    require_receipt(args.receipt)
    if args.verify_only: print(json.dumps(dict(passed=True,receipt=str(args.receipt)))); return
    records=[]
    for spec in submission_commands(args.receipt,args.submit,args.train_minutes,args.inference_minutes):
        completed=subprocess.run(spec['command'],cwd=REPO,check=True,text=True,capture_output=True)
        spec['job_id']=completed.stdout.strip().split(';')[0]
        records.append(spec)
        write_json(args.receipt.parent/'latest-submissions.json',records)
        print(json.dumps(spec),flush=True)


if __name__=='__main__': main()
