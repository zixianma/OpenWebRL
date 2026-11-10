#!/usr/bin/env python3
"""Freeze Jev-only local full300; no job, browser, or API call is launched."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
sys.path.insert(0, str(REPO))
from openwebrl.controlled_sft_state import digest
from openwebrl.jev_eval import UPSTREAM_COMMIT, UPSTREAM_FILES, load_tasks


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path, value):
    with Path(path).open('x') as handle: json.dump(value, handle, indent=2, sort_keys=True); handle.write('\n')


def schedule(tasks):
    ids = [t['task_id'] for t in tasks]
    random.Random(20261006).shuffle(ids)
    return [dict(item_id=digest([category, task, 'L06']), category=category, condition='L06', task_id=task)
        for category, chosen in [('smoke', ids[:3]), ('primary', ids)] for task in chosen]


def prepare(output, reference_plan):
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError('Preserve prepared experiments; select a fresh output')
    reference = read(reference_plan)
    tasks_path = REPO/'openwebrl/data/eval/online-mind2web.jsonl'
    tasks = load_tasks(tasks_path)
    if len(tasks) != 300: raise ValueError('Expected all300 released tasks')
    upstream = RUNTIME/'jev-ultrafast-upstream'
    installed = RUNTIME/'jev-eval-venv/lib64/python3.12/site-packages'
    head = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if head != UPSTREAM_COMMIT: raise ValueError('Native Jev revision changed')
    for name in UPSTREAM_FILES:
        raw = subprocess.check_output(['git', '-C', str(upstream), 'show', 'HEAD:jev_ultrafast/'+name])
        if raw != (installed/'jev_ultrafast'/name).read_bytes() or raw != (upstream/'jev_ultrafast'/name).read_bytes():
            raise ValueError('Pinned native Jev source differs: '+name)
    output.mkdir(parents=True)
    source, vendor = output/'source', output/'source/vendor'
    shutil.copytree(reference['source'], source, symlinks=True,
        ignore=shutil.ignore_patterns('.git', '.env*', '__pycache__', '*.pyc'))
    for name in ('jev_ultrafast', 'browser_harness'):
        shutil.copytree(installed/name, vendor/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    relative = [p.relative_to(REPO) for p in (REPO/'openwebrl').rglob('*.py')]
    relative += [Path('scripts')/n for n in ('prepare_controlled_jev_actor.py', 'run_controlled_jev_actor.py',
        'run_controlled_sft_suite.py', 'resume_baseline.py', 'runtime_ports.py', 'h200_env.sh')]
    for name in relative:
        (source/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO/name, source/name)
    files = {str(n): sha(source/n) for n in relative}
    files.update({str(p.relative_to(source)): sha(p) for p in vendor.rglob('*') if p.is_file()})
    manifest = read(source/'reference_manifest.json')
    manifest['recipe_files_sha256'].update(files)
    (source/'reference_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    shutil.copyfile(tasks_path, output/'tasks.jsonl')
    browser = dict(reference['worker_config']['browser_manifest'], artifact_directory=str(output/'browser-template'))
    if sha(browser['binary_path']) != browser['binary_sha256']: raise ValueError('Chromium binary changed')
    resources = dict(gpus=0, cpus=8, memory_gib=32, total_seconds=14400, collectors=4)
    limits = dict(browser_sessions=330, concurrent_browsers=4, jev_requests=19800,
        text_http_attempts=19800, text_usd=15., om2w_judge_http_attempts=1320, judge_usd=8.)
    config = dict(browser_manifest=browser, canonical_reward_sha256=reference['worker_config']['canonical_reward_sha256'],
        parser_checkpoint=reference['worker_config']['actor_checkpoint'], upstream_commit=head,
        actor_model='jev-1.13.0', text_model='gpt-4.1-mini-2025-04-14', judge_model='o4-mini-2025-04-16',
        harness_revision='actionable-v2', actor_input='Native DOM text; no screenshots',
        actor_sampling='Native operation/target probability argmax; no temperature or top-p',
        typing_sampling=dict(temperature=.6, top_p=.95, max_tokens=1024),
        native_thoughts='No generated thoughts; exact actual operations map to canonical empty-thought turns',
        terminal_rule='Only native DONE becomes COMPLETED; other terminals get canonical0 without API; infrastructure validity separate',
        max_browser_operations=30, max_decisions=60, task_timeout_seconds=1800,
        transport_retries=dict(jev=0, text=0, judge='unchanged canonical four-attempt loop'),
        public_prompt_hashes={n:sha(vendor/'jev_ultrafast'/n) for n in UPSTREAM_FILES})
    if sha(source/'openwebrl/eval/reward_online_mind2web.py') != config['canonical_reward_sha256']:
        raise ValueError('Canonical source differs from live local SFT reference')
    pricing = dict(verified_date_utc='2026-10-06',
        text=dict(source='https://developers.openai.com/api/docs/models/gpt-4.1-mini', input_per_million=.4, output_per_million=1.6,
                  cached_input_discount_ignored=True, snapshot='gpt-4.1-mini-2025-04-14'),
        judge=dict(source='https://developers.openai.com/api/docs/models/o4-mini', input_per_million=1.1, output_per_million=4.4))
    write(output/'pricing-verification.json', pricing)
    plan = dict(protocol_name='local-openwebrl-om2w-v2', actor_protocol='native-jev-actionable-v2',
        experiment_id='jev-actor-local-full300-20261006', root=str(output), source=str(source), vendor_path=str(vendor),
        source_manifest_sha256=sha(source/'reference_manifest.json'), controller_sha256=sha(source/'scripts/run_controlled_jev_actor.py'),
        task_file=str(output/'tasks.jsonl'), task_file_sha256=sha(output/'tasks.jsonl'), tasks=tasks,
        conditions={'L06': dict(actor='Jev Ultrafast alone', selector=None)}, schedule=schedule(tasks),
        primary_episodes=300, smoke_episodes=3, recovery_episodes_max=27,
        resources=resources, limits=limits, worker_config=config,
        pricing_sha256=sha(output/'pricing-verification.json'), reference_plan_sha256=sha(reference_plan),
        reference_plan=str(reference_plan), budget_inheritance='none; separate approval and cumulative attempts',
        wandb_project='openwebrl-evals', final_independent_audit_required=True)
    write(output/'plan.json', plan)
    write(output/'approval.json', dict(approved=False, plan_sha256=digest(plan), resources=resources, limits=limits, attempts=[]))
    batch = f'''#!/usr/bin/env bash
# PREPARED ONLY: exact separate budget approval and registered job required.
#SBATCH --job-name=jev-local300
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output={output}/slurm-%j.log
set -euo pipefail
source {source}/scripts/h200_env.sh
export CUDA_VISIBLE_DEVICES="" WANDB_PROJECT=openwebrl-evals OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
exec {RUNTIME}/venv/bin/python {source}/scripts/run_controlled_jev_actor.py --root {output} --execute "$SLURM_JOB_ID"
'''
    (output/'launch.sbatch').write_text(batch)
    write(output/'preparation.json', dict(prepared=True, launch_authorized=False, model_api_calls=0, browser_episodes=0,
        gpu_allocations=0, plan_sha256=digest(plan), live_smoke_gate_required=True))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-plan', type=Path, default=RUNTIME/'evaluations/local-sft-selector-controlled-20261006/prepared-parallel-v3/shard-0/plan.json')
    args = parser.parse_args()
    plan = prepare(args.output, args.reference_plan)
    print(json.dumps({k: plan[k] for k in ('root', 'resources', 'limits', 'primary_episodes', 'smoke_episodes')}, indent=2))


if __name__ == '__main__': main()
