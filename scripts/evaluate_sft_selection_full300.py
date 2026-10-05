#!/usr/bin/env python3
"""Prepare or execute full300 SFT+Jev/Kev; submission requires exact approval."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from evaluate_sft_decision_selection import run, ACTOR, WORKER_DEPS
from openwebrl.decision_selection_eval import FULL300_PROTOCOL as PROTOCOL
from openwebrl.decision_selection import write_json
from openwebrl.kev_eval import file_hash, model_spec
from prepare_arm_turn_bonus import copy_plain
from resume_baseline import validate_source, source_command

RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
PILOT = RUNTIME / 'evaluations/sft-decision-selection-pilot-20261004'
SOURCE = RUNTIME / 'reference-sft-selection-full300-20261004-v2'
CHANGED = ('openwebrl/decision_selection.py', 'openwebrl/decision_selection_eval.py',
           'openwebrl/selection_budget.py')
CODE = ('scripts/evaluate_sft_selection_full300.py',
        'scripts/evaluate_sft_decision_selection.py', 'scripts/audit_sft_decision_selection.py')


def read(path):
    return json.loads(Path(path).read_text())


def control(mode):
    return RUNTIME / 'evaluations' / ('sft-' + mode.replace('-', '') + '-full300-20261004')


def limits(mode):
    return dict(browser_sessions=330, concurrent_browsers=2, browser_expiry_minutes=12,
        jev_requests=9900 if mode == 'jev' else 0,
        local_kev_requests=9910 if mode == 'kev-27b' else 0,
        judge_http_attempts=1320, judge_completion_tokens=4096,
        actor_proposals=50000, text_helper_calls=0)


def prepare(mode):
    prior = read(PILOT / 'plan.json')
    validate_source(Path(prior['source']))
    if not SOURCE.exists():
        shutil.copytree(prior['source'], SOURCE, symlinks=True, copy_function=copy_plain,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.env*', '.browser_use_sessions'))
        for name in CHANGED:
            copy_plain(REPO / name, SOURCE / name)
        manifest = read(SOURCE / 'reference_manifest.json')
        manifest['recipe_files_sha256'].update({n: file_hash(SOURCE / n) for n in CHANGED})
        manifest['selection_full300'] = dict(parent=prior['source'], changed=list(CHANGED),
            protocol=PROTOCOL, terminal_judge='Automated equivalent of pilot step-limit recovery')
        write_json(SOURCE / 'reference_manifest.json', manifest)
    validate_source(SOURCE)
    if any(file_hash(REPO/n) != file_hash(SOURCE/n) for n in CHANGED):
        raise ValueError('Frozen full300 worker differs; create a new source revision')
    rows = [json.loads(line) for line in (REPO/'openwebrl/data/eval/online-mind2web.jsonl').read_text().splitlines()]
    ids = [row['metadata']['task_id'] for row in rows]
    if len(ids) != 300 or len(set(ids)) != 300:
        raise ValueError('Full300 must contain all300 unique benchmark tasks')
    root = control(mode); root.mkdir(parents=True, exist_ok=True)
    tasks = root/'tasks.jsonl'
    contents = ''.join(json.dumps(row) + '\n' for row in rows)
    if tasks.exists() and tasks.read_text() != contents:
        raise ValueError('Frozen task file differs')
    tasks.write_text(contents)
    actor_weights = {str(p.relative_to(ACTOR)): dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                     for p in ACTOR.glob('*.safetensors')}
    if actor_weights != prior['actor_weights'] or file_hash(ACTOR/'config.json') != prior['actor_config_sha256']:
        raise ValueError('SFT checkpoint differs from pilot')
    models = read(RUNTIME/'kev-preparation-20261004/verified-models.json')
    if mode == 'kev-27b':
        if not models['27b']['verified'] or models['27b']['spec'] != model_spec('27b'):
            raise ValueError('Kev pin changed')
        for filename, receipt in models['27b']['files'].items():
            stat=Path(filename).stat()
            if (stat.st_size, stat.st_mtime_ns) != (receipt['size'], receipt['mtime_ns']):
                raise ValueError('Kev weights differ from the independent hash audit')
    resources=dict(gpus=2 if mode=='kev-27b' else 1, gpu_type='H200',
        cpus=16 if mode=='kev-27b' else 8, memory_gib=240 if mode=='kev-27b' else 120,
        total_seconds=36000)
    caps=limits(mode)
    budget=dict(root=str(root/'budget'), limits={k:caps[k] for k in
        ('browser_sessions','jev_requests','local_kev_requests','judge_http_attempts','actor_proposals')})
    baseline_root=RUNTIME/'arm-turn-bonus-preparation/controlled-inference-20261004'
    baseline=read(baseline_root/'experiment.json')
    complete=read(baseline_root/'completion.json')
    assert complete['complete'] and complete['tasks']==300
    baseline_ids={json.loads(line)['metadata']['task_id'] for line in (baseline_root/'tasks.jsonl').read_text().splitlines()}
    assert baseline_ids==set(ids)
    plan=copy.deepcopy(prior)
    plan.update(status='prepared_awaiting_exact_resource_approval', protocol=PROTOCOL,
        modes=[mode], resources=resources, limits=caps, source=str(SOURCE),
        source_manifest_sha256=file_hash(SOURCE/'reference_manifest.json'),
        tasks=str(tasks), task_ids=ids, tasks_sha256=file_hash(tasks), budget=budget,
        worker_options=dict(expected_task_count=300, selector_request_cap=9900,
            judge_step_limit=True, budget=budget, wandb_group='SFT-selection-full300-20261004',
            wandb_id='sft-selection-full300-20261004-'+mode),
        code_sha256={name:file_hash(REPO/name) for name in CODE},
        baseline_reuse=dict(root=str(baseline_root), protocol=baseline['protocol'],
            fixed_actor0=dict(successes=106,tasks=300), pooled_pass1=dict(successes=528,episodes=1500),
            matched_protocol=False,
            step_limit_scoring='Baseline: automatic failure without judge; selector pilot/full300: saved terminal evidence receives canonical judge',
            interpretation='Historical reference; browser, sampling, context-response allowance, seeds, timeouts and step-limit judge dispatch differ'),
        budget_policy='Separate new10h total per mode including every retry; no unused pilot budget transferred.660 browser reservations across both runs; no automatic reroll of valid failures.',
        comparison='Fresh full300 cohort for this selector, including the original pilot10; pilot reported separately',
        completion_requires='Audit all300 records, final screenshots and saved judge verdicts or explicit invalid diagnoses; scheduler/browser shutdown, W&B, docs and private review')
    path=root/'plan.json'
    if path.exists() and read(path)!=plan:
        raise ValueError('Prepared plan changed; preserve the previous plan before revising')
    write_json(path, plan)
    approval=root/'approval.json'
    if not approval.exists():
        write_json(approval,dict(approved=False,resources=resources,limits=caps,
            plan_sha256=file_hash(path),attempts=[],authorization='Exact resource/API approval pending'))
    batch=f'''#!/usr/bin/env bash
# Prepared only: no allocation is authorized by this file.
#SBATCH --job-name=sft-{mode}-300
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:{resources['gpus']}
#SBATCH --cpus-per-task={resources['cpus']}
#SBATCH --mem={resources['memory_gib']}G
#SBATCH --time=10:00:00
#SBATCH --signal=B:TERM@90
#SBATCH --output={RUNTIME}/logs/sft-{mode}-300-%j.out
set -euo pipefail
cd {REPO}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export WANDB_PROJECT=openwebrl-evals
exec {RUNTIME}/venv/bin/python scripts/evaluate_sft_selection_full300.py --mode {mode} --execute
'''
    (root/'launch.sbatch').write_text(batch)
    return plan


def verify_prepared(mode):
    root=control(mode);plan=read(root/'plan.json')
    validate_source(Path(plan['source']))
    if plan['modes'] != [mode] or plan['protocol'] != PROTOCOL:
        raise ValueError('Mode/protocol changed')
    for name, sha in plan['code_sha256'].items():
        if file_hash(REPO/name)!=sha:raise ValueError('Prepared controller changed: '+name)
    if file_hash(Path(plan['tasks'])) != plan['tasks_sha256']:
        raise ValueError('Frozen task file changed')
    weights={str(p.relative_to(ACTOR)):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
             for p in ACTOR.glob('*.safetensors')}
    if weights!=plan['actor_weights'] or file_hash(ACTOR/'config.json')!=plan['actor_config_sha256']:
        raise ValueError('Prepared actor checkpoint changed')
    if file_hash(WORKER_DEPS/'verified.json')!=plan['worker_dependencies_sha256']:
        raise ValueError('Prepared browser dependencies changed')
    return plan


def submit(mode):
    root=control(mode);plan=verify_prepared(mode);approval=read(root/'approval.json')
    if (not approval['approved'] or approval['resources']!=plan['resources'] or
            approval['limits']!=plan['limits'] or approval['plan_sha256']!=file_hash(root/'plan.json')):
        raise ValueError('Explicit exact new compute/API approval required before sbatch')
    used=0
    for attempt in approval['attempts']:
        lines=subprocess.check_output(['sacct','-X','-n','-P','-j',attempt['job_id'],
            '--format=JobIDRaw,State,ElapsedRaw'],text=True).splitlines()
        record=next(line.split('|') for line in lines if line.split('|')[0]==attempt['job_id'])
        if record[1].split()[0] not in ('COMPLETED','FAILED','CANCELLED','TIMEOUT'):
            raise ValueError('An existing attempt is still active')
        attempt.update(state=record[1],scheduler_seconds=int(record[2]));used+=int(record[2])
    minutes=(plan['resources']['total_seconds']-used)//60
    if minutes<10:raise ValueError('Insufficient remaining approved compute')
    job=subprocess.check_output(['sbatch','--hold','--parsable',f'--time={minutes}',str(root/'launch.sbatch')],text=True).strip().split(';')[0]
    assert job.isdecimal()
    approval['attempts'].append(dict(job_id=job,time_limit_seconds=minutes*60,submitted_unix=time.time()))
    write_json(root/'approval.json',approval)
    pointer=dict(status='submitted',job_id=job,run_directory=str(root),verified_complete=False,
        approved_resources=plan['resources'],consumed_scheduler_seconds=used,
        remaining_approved_seconds=plan['resources']['total_seconds']-used,updated_unix=time.time(),
        continuation_active=False,supervision_required=True)
    write_json(RUNTIME/f'current_sft_{mode.replace("-", "")}_full300.json',pointer)
    subprocess.run(['scontrol','release',job],check=True)
    print(json.dumps(pointer,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['jev','kev-27b'],required=True)
    action=parser.add_mutually_exclusive_group()
    action.add_argument('--execute',action='store_true');action.add_argument('--submit',action='store_true')
    args=parser.parse_args()
    if args.submit:
        with (control(args.mode)/'submit.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);submit(args.mode)
    elif args.execute:
        plan=verify_prepared(args.mode)
        with (control(args.mode)/'owner.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(plan,control=control(args.mode))
    else:
        plan=prepare(args.mode)
        print(json.dumps({k:plan[k] for k in ('status','modes','resources','limits','baseline_reuse')},indent=2))
