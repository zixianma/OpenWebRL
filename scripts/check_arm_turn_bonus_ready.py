#!/usr/bin/env python3
"""Small CPU readiness audit. No sbatch, model loading, browser or paid API calls."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_arm_turn_bonus_calibration import SOURCE,CHECKPOINT,SELECTOR,REPO,RUNTIME,plan
from resume_baseline import validate_source


def check(output,run_tests=True):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    validate_source(SOURCE)
    manifest=json.loads((SOURCE/'reference_manifest.json').read_text())['arm_turn_bonus_calibration']
    if not manifest.get('supports_gated_first_batch_training'):
        raise ValueError('Prepared source lacks the training gate')
    for rel in manifest['changed_files_sha256']:
        if rel.endswith('.py'):
            compile((SOURCE/rel).read_text(),str(SOURCE/rel),'exec')
    for rel in ('scripts/run_arm_turn_bonus_calibration.py','openwebrl/arm_turn_bonus.py',
                'openwebrl/arm_turn_bonus_runtime.py'):
        compile((REPO/rel).read_text(),str(REPO/rel),'exec')
    for rel in ('scripts/run_arm_turn_bonus_calibration_2gpu.sbatch','scripts/run_arm_turn_bonus_pilot_2gpu.sbatch',
                'scripts/run_arm_turn_bonus_pilot_4gpu.sbatch'):
        subprocess.run(['bash','-n',str(REPO/rel)],check=True)
    for rel in ('config.json','tokenizer.json','chat_template.jinja'):
        if not (SELECTOR/rel).is_file():
            raise ValueError(f'Missing selector file: {rel}')
    index=SELECTOR/'model.safetensors.index.json'
    shards=set(json.loads(index.read_text())['weight_map'].values())
    if not shards or any(not (SELECTOR/name).is_file() or (SELECTOR/name).stat().st_size==0 for name in shards):
        raise ValueError('Missing selector tensor shard')
    headers=[RUNTIME/'src/python-headers/Include',RUNTIME/'src/python-headers']
    compiler=subprocess.run(['cc','-fsyntax-only','-x','c',*[arg for p in headers for arg in ('-I',str(p))],'-'],
        input='#include <Python.h>\nint main(void) { return 0; }\n',text=True,capture_output=True)
    if compiler.returncode:
        raise RuntimeError('Python header compiler check failed: '+compiler.stderr[-2000:])
    selector_python=RUNTIME/'arm-reproduction/venv/bin/python'
    modules=('torch','transformers','fastapi','uvicorn','xgrammar','triton','PIL')
    probe=subprocess.run([str(selector_python),'-c',
        'import importlib.util,json; names='+repr(modules)+'; print(json.dumps({n:importlib.util.find_spec(n) is not None for n in names}))'],
        check=True,capture_output=True,text=True)
    deps=json.loads(probe.stdout)
    if not all(deps.values()): raise ValueError('Missing selector dependency: '+str(deps))
    for phase in ('calibrate','calibrate-train'):
        prepared=plan('NEXT_APPROVED_JOB',phase=phase)
        (output/f'{phase}-plan.json').write_text(json.dumps(prepared,indent=2)+'\n')
        prepared=plan('NEXT_APPROVED_JOB',phase=phase,gpus=4)
        (output/f'{phase}-4gpu-plan.json').write_text(json.dumps(prepared,indent=2)+'\n')
    if run_tests:
        with (output/'tests.log').open('w') as log:
            subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_arm_turn_bonus*.py','-v'],
                cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    report=dict(checked_at_epoch_seconds=time.time(),cpu_ready=True,source=str(SOURCE),checkpoint=str(CHECKPOINT),
        selector=str(SELECTOR),selector_dependencies=deps,python_headers_compile=True,
        tests_run=run_tests,tests_log=str(output/'tests.log'),prepared_source=manifest,
        gpu_validated=False,new_compute_submitted=False,
        remaining='Real selector preflight, complete browser collection, calibration gates and GPU optimizer/checkpoint validation run only in the next explicitly approved allocation.')
    (output/'readiness.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=RUNTIME/'arm-turn-bonus-preparation/readiness-v8')
    parser.add_argument('--skip-tests',action='store_true')
    args=parser.parse_args()
    print(json.dumps(check(args.output,not args.skip_tests),indent=2))
