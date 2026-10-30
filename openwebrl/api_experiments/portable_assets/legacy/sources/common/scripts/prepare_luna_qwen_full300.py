#!/usr/bin/env python3
"""Freeze a full300 comparison and exact budget request; never submit jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from openwebrl.luna_qwen_metrics import PRICING, digest, write_json
from openwebrl.luna_qwen_full_eval import protocol_for_family
from prepare_arm_turn_bonus import copy_plain
from resume_baseline import validate_source

RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
ROOT = RUNTIME / 'luna-qwen-full300-20261004'
PILOT = RUNTIME / 'luna-qwen-inference-20261004'
SOURCE = RUNTIME / 'reference-luna-qwen-full300-20261004-v1'
CODE = ('openwebrl/luna_qwen_full_eval.py', 'scripts/run_luna_qwen_full300.py',
        'scripts/prepare_luna_qwen_full300.py', 'scripts/report_luna_qwen_full300.py',
        'scripts/report_luna_qwen_inference.py')
HELPERS = ('scripts/runtime_ports.py', 'scripts/prepare_arm_turn_bonus.py',
           'scripts/evaluate_baseline_checkpoint.py', 'scripts/run_arm_quality_audit.py',
           'scripts/run_arm_turn_bonus_calibration.py')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    if (ROOT / 'approval.json').exists() and read(ROOT / 'approval.json').get('approved'):
        raise ValueError('Never rewrite an approved experiment')
    base, sft = read(PILOT / 'plan.json'), read(PILOT / 'sft-extension/plan.json')
    parent = Path(base['source'])
    validate_source(parent)
    if not SOURCE.exists():
        shutil.copytree(parent, SOURCE, symlinks=True, copy_function=copy_plain,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.env*', '.browser_use_sessions'))
    for name in CODE:
        copy_plain(REPO / name, SOURCE / name)
    helper_source = Path(sft['source'])
    validate_source(helper_source)
    for name in HELPERS:
        copy_plain(helper_source / name, SOURCE / name)
    manifest = read(SOURCE / 'reference_manifest.json')
    manifest['recipe_files_sha256'].update({n: sha(SOURCE/n) for n in (*CODE, *HELPERS)})
    manifest['luna_full300'] = dict(parent=str(parent), changed=list(CODE),
        inherited_browser_measurement_hooks=True, pilot_records_reused=False)
    write_json(SOURCE / 'reference_manifest.json', manifest)
    validate_source(SOURCE)
    rows = [json.loads(x) for x in (REPO/'openwebrl/data/eval/online-mind2web.jsonl').read_text().splitlines()]
    ids = [row['metadata']['task_id'] for row in rows]
    if len(ids) != 300 or len(set(ids)) != 300:
        raise ValueError('Need all300 unique OM2W tasks')
    ROOT.mkdir(parents=True, exist_ok=True)
    tasks = ROOT / 'tasks.jsonl'
    contents = ''.join(json.dumps(row) + '\n' for row in rows)
    if tasks.exists() and tasks.read_text() != contents:
        raise ValueError('Task cohort changed')
    tasks.write_text(contents)
    modes = dict(qwen=['qwen', 'qwen_luna5'],
                 sft=['sft_luna5', 'sft_luna10'], luna=['luna'])
    n = sum(map(len, modes.values())) * 300
    limits = dict(luna_usd=100., luna_calls=39600,
                  judge_usd=25., judge_calls=int(n*4*1.1))
    pools = dict(gpu=dict(resources=dict(gpus=8, gpu_type='H200', cpus=64, memory_gib=960,
                                         total_seconds=16*3600),
                         families=['sft','qwen'], workers=8),
                 cpu=dict(resources=dict(gpus=0, cpus=16, memory_gib=32, total_seconds=4*3600),
                         families=['luna'], workers=4))
    family_plans = {}
    for family, enabled in modes.items():
        prior = sft if family == 'sft' else base
        actor = Path(prior['actor_path'])
        for filename, expected in prior['actor_files'].items():
            stat = (actor / filename).stat()
            if dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns) != expected:
                raise ValueError('Actor changed since independent weight verification')
        shuffled = list(ids)
        random.Random(20261004).shuffle(shuffled)
        schedule = []
        for task_id in shuffled:
            order = list(enabled)
            random.Random(digest([task_id, family, 20261004])).shuffle(order)
            schedule.append(dict(task_id=task_id, modes=order))
        prompt = Path(prior['selector_prompt_source'])
        prompt_file = prompt / 'inference/selection_prompt.py'
        plan = dict(family=family, protocol=protocol_for_family(family), task_ids=ids,
            task_file=str(tasks), task_file_sha256=sha(tasks), schedule=schedule,
            actor_path=str(actor), actor_files=prior['actor_files'],
            actor_revision=prior['actor_revision'],
            actor_config_sha256=sha(actor/'config.json'),
            actor_preprocessor_sha256=sha(actor/'preprocessor_config.json'),
            selector_prompt_source=str(prompt), selector_prompt_sha256=sha(prompt_file),
            source=str(SOURCE), source_manifest_sha256=sha(SOURCE/'reference_manifest.json'),
            output=str(ROOT/family/'results'), budget_root=str(ROOT/'budget'), limits=limits,
            pricing=PRICING, experiment_id=ROOT.name, wandb_group=ROOT.name,
            wandb_prefix=ROOT.name+'-'+family)
        path = ROOT/family/'plan.json'
        write_json(path, plan)
        family_plans[family] = dict(plan=str(path), plan_sha256=digest(plan))
    baseline_root = RUNTIME/'arm-turn-bonus-preparation/controlled-inference-20261004'
    complete = read(baseline_root/'completion.json')
    if not complete['complete'] or complete['tasks'] != 300:
        raise ValueError('Historical baseline not verified')
    plan = dict(status='prepared_awaiting_exact_resource_approval', experiment_id=ROOT.name,
        task_ids=ids, task_file=str(tasks), task_file_sha256=sha(tasks), modes=modes,
        tasks=300, new_episodes=n, pools=pools, limits=limits, families=family_plans,
        source=str(SOURCE), source_manifest_sha256=sha(SOURCE/'reference_manifest.json'),
        controller_sha256=sha(SOURCE/'scripts/run_luna_qwen_full300.py'), pricing=PRICING,
        metrics=['overall_success','valid_only_success','paired_success_difference','cost',
                 'episode_latency','input_output_tokens','browser_steps','local_compute'],
        plots=['success_vs_cost','success_vs_latency','success_vs_input_output_tokens'],
        baseline_reuse=dict(actor='OpenWebRL/OpenWebRL-4B-SFT', root=str(baseline_root),
            fixed_actor0=dict(successes=106,tasks=300,valid=272), pooled_pass1=dict(successes=528,episodes=1500),
            protocol=read(baseline_root/'experiment.json')['protocol'], matched_protocol=False,
            comparable_exclusive_gpu_cost=False, comparable_episode_latency=False,
            interpretation='Historical SFT reference only; T0.7/p0.9/1024 output, thoughts-inclusive judge and shared serving differ. No new SFT-alone episodes.'),
        exclusions=['Luna actor + Luna selector', 'new official SFT actor-only run',
                    'Qwen + Luna N=10; user keeps N=10 only for official SFT'],
        pilot_separation='New T1/p0.95 study: no T1/p0.9 pilot episodes or remaining budget transferred.',
        comparison='Same300 tasks; fixed seeds; within-family arm order shuffled. Family workers run concurrently.',
        completion_requires='Exact new task/mode pairs, frozen identities/receipts/verdicts, final accounting, W&B and three reviewed plots.')
    write_json(ROOT/'plan.json', plan)
    write_json(ROOT/'approval.json', dict(approved=False, plan_sha256=digest(plan),
        pools={p:d['resources'] for p,d in pools.items()}, limits=limits, attempts=[],
        authorization='Full-set scope requested; exact new resource and API budget approval pending'))
    for pool, spec in pools.items():
        r = spec['resources']
        gpu_flag = '#SBATCH --gpus=h200:8' if r['gpus'] else '#SBATCH --gres=none'
        batch = f'''#!/usr/bin/env bash
# Prepared only; no new allocation is authorized until exact approval.
#SBATCH --job-name=luna-full300-{pool}
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
{gpu_flag}
#SBATCH --cpus-per-task={r['cpus']}
#SBATCH --mem={r['memory_gib']}G
#SBATCH --time={r['total_seconds']//3600:02d}:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output={RUNTIME}/logs/luna-full300-{pool}-%j.out
set -euo pipefail
cd {SOURCE}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 WANDB_PROJECT=openwebrl-evals
exec {RUNTIME}/venv/bin/python scripts/run_luna_qwen_full300.py --root {ROOT} --execute {pool} "$SLURM_JOB_ID"
'''
        (ROOT/f'launch-{pool}.sbatch').write_text(batch)
    write_json(ROOT/'current.json', dict(stage='prepared_awaiting_exact_resource_approval',
        new_episodes=n, jobs=[], pilot_job_separate='344655'))
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    p = prepare()
    print(json.dumps({k:p[k] for k in ('status','new_episodes','pools','limits')},indent=2))
