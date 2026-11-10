#!/usr/bin/env python3
"""Prepare the uniform G8 60->90 continuation control (CPU only; never submits or deletes).

Approved by the user on 2026-10-24: 8 H200 / 64 CPU / 960 GiB for 42 h total across all
attempts (24 h + 18 h afterok), GPT-4.1 training/eval judge cap $300 / 30,000 requests,
full-300 local GPT-4.1/action_history/T0 evaluations at 70/80/90. The earlier to60
budget's unused balance is not added. Scientific recipe, optimizer/scheduler/cursor and
W&B run are those of the verified iteration-60 checkpoint.
"""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
REPO = Path(__file__).resolve().parents[1]
LINEAGE = RUNTIME/'training-controls/uniform8-original2102-20261006'
TO60 = LINEAGE/'to60-20261008'
CONTROL = LINEAGE/'to90-20261024'
SOURCE = CONTROL/'source'
RUN = RUNTIME/'runs/outcome-uniform8-original2102-20261006'
RESOURCES = dict(gpus=8, gpu_type='H200', cpus=64, memory_gib=960, total_hours=42, total_gpu_hours=336,
                 allocations=[24, 18], max_hours_per_allocation=24)
CAPS = dict(judge_cap_usd=300, judge_call_cap=30000, browser_dispatch_cap=40000, browser_concurrency=64)
CONTROLLER = REPO/'scripts/run_uniform8_to90.py'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, d):
    Path(p).write_text(json.dumps(d, indent=2)+'\n')


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Expected exactly one occurrence of: '+old[:80])
    return text.replace(old, new)


def prepare():
    CONTROL.mkdir(exist_ok=True)
    if not SOURCE.exists():
        shutil.copytree(TO60/'source', SOURCE, symlinks=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
        f = 'openwebrl/rollout_sampling_budget.py'
        text = replace_once((SOURCE/f).read_text(),
            'EXPECTED_CAPS = dict(judge_cap_usd=400, judge_call_cap=40000,\n                     browser_dispatch_cap=65000, browser_concurrency=64)',
            'EXPECTED_CAPS = dict(judge_cap_usd=300, judge_call_cap=30000,\n                     browser_dispatch_cap=40000, browser_concurrency=64)')
        (SOURCE/f).write_text(text)
        manifest = json.loads((SOURCE/'reference_manifest.json').read_text())
        manifest['recipe_files_sha256'][f] = sha(SOURCE/f)
        manifest['uniform8_to90'] = dict(parent_source=str(TO60/'source'), changed_files=[f], scientific_change=None,
                                         scope='New 60->90 judge/browser caps only')
        save(SOURCE/'reference_manifest.json', manifest)
    recipe = (TO60/'campaign_recipe.py').read_text()
    for old, new in [(f'CONTROL=Path({str(TO60)!r})', f'CONTROL=Path({str(CONTROL)!r})'),
                     (f'SOURCE=Path({str(TO60/"source")!r})', f'SOURCE=Path({str(SOURCE)!r})'),
                     ('TARGET_ITERATION=60', 'TARGET_ITERATION=90'),
                     ('EVAL_ITERATIONS=(20,30,40,50,60)', 'EVAL_ITERATIONS=(70,80,90)'),
                     ('TRAINING_STAGES=tuple(range(11,61))', 'TRAINING_STAGES=tuple(range(61,91))'),
                     ("NUM_ROLLOUT='60'", "NUM_ROLLOUT='90'")]:
        recipe = replace_once(recipe, old, new)
    (CONTROL/'campaign_recipe.py').write_text(recipe)
    shutil.copyfile(TO60/'legacy_support.py', CONTROL/'legacy_support.py')
    link = CONTROL/'combined-tasks.parquet'
    if not link.exists():
        link.symlink_to((TO60/'combined-tasks.parquet').resolve())
    if not (CONTROL/'approval-request.json').exists():
        save(CONTROL/'approval-request.json', dict(
            approved=True, approval_id='uniform8-to90-20261024', approved_unix=time.time(),
            user_message='yes I approve',
            approved_in_response_to='8 x H200 / 64 CPU / 960 GiB for 42 hours total (24 h + 18 h), plus the $300 / 30,000-request judge cap',
            resources=RESOURCES, **CAPS, target_iteration=90, resume_completed_iteration=60,
            full300_evaluations=[70, 80, 90], includes_all_startup_retries_and_evaluations=True,
            unused_prior_budgets_added=False))
    if not (CONTROL/'attempts.json').exists():
        save(CONTROL/'attempts.json', dict(attempts=[], approved_total_seconds=42*3600, total_charged_seconds=0,
                                           prior_budget_seconds_added=0))
    cp = RUN/'iter_0000059'
    proofs = [cp/'common.pt', cp/'.metadata', RUN/'rollout/global_dataset_state_dict_59.pt',
              TO60/'verified-complete.json' if (TO60/'verified-complete.json').exists() else TO60/'status.json']
    save(CONTROL/'resume-anchor.json', dict(
        completed_iteration=60, native_iteration=59, checkpoint=str(cp),
        audit=str(REPO/'openwebrl/docs/arm_results/rl_integration/uniform8-iteration60-audit.json'),
        adam_updates=1318, scheduler_num_steps=337408, training_wandb_run_id='outcome-uniform8-original2102-20261006',
        proof_sha256={str(p): sha(p) for p in proofs},
        shard_bytes={p.name: p.stat().st_size for p in sorted(cp.iterdir()) if p.is_file()}))
    sys.path[:0] = [str(CONTROL), str(REPO/'scripts')]
    from resume_baseline import validate_source
    validate_source(SOURCE)
    execution = [CONTROLLER, CONTROL/'campaign_recipe.py', CONTROL/'legacy_support.py',
                 SOURCE/'openwebrl/rollout_sampling_budget.py', SOURCE/'reference_manifest.json']
    save(CONTROL/'readiness.json', dict(cpu_checks_passed=True, prepared_epoch=time.time(),
                                        execution_file_sha256={str(p): sha(p) for p in execution}))
    return CONTROL


if __name__ == '__main__':
    print(prepare())
