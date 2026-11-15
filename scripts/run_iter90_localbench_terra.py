#!/usr/bin/env python3
"""Iteration-90 WebVoyager (595) + DeepShop (150) evaluation, judge deferred to offline shared Terra.

Approved 2026-11-15: 12 jobs, each 1 H200 / 8 CPU / 240 GiB / 5 h, vlm / normal. Each job runs one model x one
repeat: WebVoyager, then DeepShop. Models at iteration 90: Gate B 40M+16F and Gate B (best two by shared Terra on
OM2W), outcome-only 48 and outcome-only 56; three repeats each, all submitted together so they share a window.
Harness: local browser, 30 turns, actor T0.6 / top-p 0.95 / top-k 20, 4,096 tokens, 8 browsers per job
(reference-localbench-deferjudge-20261115 = the October 8 local-bench source plus an OPENWEBRL_DEFER_JUDGE switch).
The in-harness GPT-4o judge is skipped and the plan's OpenAI key is a non-working placeholder, so any judge call
fails instead of spending. Episodes are graded afterwards by scripts/rejudge_localbench_shared_terra.py.
Every stage clones the verified October 8 outcome-only 48 plan and swaps only the checkpoint, harness, scheduler
offset, seed, output and W&B identity.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path('/gpfs/projects/krishna/zixianma/OpenWebRL')
sys.path.insert(0, str(REPO/'scripts'))
from resume_baseline import write_json  # noqa: E402

RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
CONTROL = RUNTIME/'training-controls/iter90-localbench-terra-20261115'
OLDCTL = RUNTIME/'arm-turn-bonus-preparation/original90-localbench-20261007'
TEMPLATE = OLDCTL/'controller-local-baseline-349503'
OLD = OLDCTL/'source-local-v2-prompt-assets'
HARNESS = RUNTIME/'reference-localbench-deferjudge-20261115'
PYTHON = RUNTIME/'venv/bin/python'
CK = RUNTIME/'evaluations'
RUNS = {  # label: (checkpoint, expected scheduler offset, parent run)
    'g40m16f': (CK/'arm-failure-additive-40m16f-413757-iter90/runtime/iter_0000089', '0', 'arm-gate-b-40m16f-392093'),
    'gateb': (CK/'arm-failure-additive-335729-b100-iter90/runtime/iter_0000089', '0', 'arm-gate-b'),
    'outcome48': (RUNTIME/'runs/openwebrl-4b-reference-294421-20260913T211532/iter_0000089', '1', 'qcq7i4ug'),
    'outcome56': (RUNTIME/'runs/outcome-only-56groups-20261006/iter_0000089', '0', 'outcome-only-56groups-20261006'),
}
REPEATS, BENCHMARKS = 3, ('webvoyager', 'deepshop')
HOURS = 5


def read(p):
    return json.loads(Path(p).read_text())


def name(label, repeat, benchmark):
    return f'lb90-terra-{label}-{benchmark}-r{repeat}'


def plan(job, label, repeat, benchmark):
    template = read(TEMPLATE/f'{benchmark}-plan.json')
    checkpoint, offset, parent = RUNS[label]
    old_out = Path(template['output'])
    output = RUNTIME/f'evaluations/{name(label, repeat, benchmark)}-{job}'
    text = (json.dumps(template).replace(str(old_out), str(output)).replace(template['checkpoint'], str(checkpoint))
            .replace(str(OLD), str(HARNESS)).replace('/tmp/openwebrl-eval-349503-', f'/tmp/openwebrl-eval-{job}-'))
    p = json.loads(text)
    env = p['environment']
    env.update(WANDB_RUN_ID=output.name, OPENWEBRL_EXPECTED_SCHEDULER_OFFSET=offset, OPENWEBRL_DEFER_JUDGE='1',
               OPENAI_API_KEY='deferred-judge-no-key', RAY_TMPDIR=f'/tmp/lb-{job}-{benchmark[0]}')
    cmd = p['command']
    cmd[cmd.index('--wandb-group')+1] = 'iter90-localbench-terra-20261115'
    cmd[cmd.index('--seed')+1] = str(1233 + repeat)
    p.update(job_id=job, attempt=0, output=str(output), checkpoint=str(checkpoint), source=str(HARNESS),
             wandb_run_id=output.name, wandb_url=f'https://wandb.ai/zixianma/openwebrl-evals/runs/{output.name}',
             parent_training_run=parent, method=label, repeat=repeat, allocation_hours_cap=HOURS,
             judge_model='deferred: offline shared-Terra (gpt-5.6-terra)',
             protocol=f'local_process/{benchmark}; judge deferred to offline shared Terra; T0.6/p0.95/k20; 4096 tokens; 30 turns',
             note='Iteration-90 local WebVoyager/DeepShop, same window for all models; judge deferred.')
    for k in ('completed_optimizer_updates', 'source_manifest_sha256', 'checkpoint_validation_sha256'):
        p.pop(k, None)
    leaked = [x for x in (str(old_out), str(OLD), '349503') if x in json.dumps(p)]
    if leaked:
        raise ValueError(f'Template identity leaked into plan: {leaked}')
    return p


def complete(p):
    d = Path(p['output'])
    s = d/'status.json'
    return s.is_file() and read(s).get('complete') and len(list((d/'rollouts').glob('*.json'))) == p['expected_task_count']


def worker(job, label, repeat, benchmark):
    if os.getenv('SLURM_JOB_ID') != job:
        raise ValueError('Run inside the approved allocation')
    p = plan(job, label, repeat, benchmark)
    os.environ['OPENWEBRL_EXPECTED_SCHEDULER_OFFSET'] = p['environment']['OPENWEBRL_EXPECTED_SCHEDULER_OFFSET']
    os.environ['OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT'] = str(p['expected_task_count'])
    from evaluate_baseline_checkpoint import run
    run(p, REPO/'.env')


def run_job(job, label, repeat):
    approval = read(CONTROL/'approval.json')
    if approval.get('approved') is not True or approval.get('hours') != HOURS:
        raise PermissionError('Exact approval required')
    if not RUNS[label][0].is_dir():
        raise FileNotFoundError(f'{label}: checkpoint missing')
    status = dict(job_id=job, label=label, repeat=repeat, stages=[])
    for benchmark in BENCHMARKS:
        p = plan(job, label, repeat, benchmark)
        if complete(p):
            status['stages'].append(dict(benchmark=benchmark, output=p['output'], reused=True))
            continue
        write_json(CONTROL/f'status-{label}-r{repeat}.json', dict(status, stage=benchmark, current=p['output']))
        step = ['srun', f'--jobid={job}', '--nodes=1', '--ntasks=1', '--cpus-per-task=8', '--gres=gpu:h200:1',
                '--exact', '--cpu-bind=none', str(PYTHON), '-B', str(Path(__file__).resolve()), '--job-id', job,
                '--worker', label, str(repeat), benchmark]
        code = subprocess.run(step).returncode
        ok = code == 0 and complete(p)
        status['stages'].append(dict(benchmark=benchmark, output=p['output'], exit=code, complete=bool(ok)))
        write_json(CONTROL/f'status-{label}-r{repeat}.json', dict(status, stage='running'))
        if not ok:
            write_json(CONTROL/f'status-{label}-r{repeat}.json', dict(status, stage='failed'))
            return 1
    write_json(CONTROL/f'status-{label}-r{repeat}.json', dict(status, stage='complete'))
    return 0


if __name__ == '__main__':
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--job-id', required=True)
    a.add_argument('--run', nargs=2, metavar=('LABEL', 'REPEAT'))
    a.add_argument('--worker', nargs=3, metavar=('LABEL', 'REPEAT', 'BENCHMARK'))
    a = a.parse_args()
    if a.worker:
        worker(a.job_id, a.worker[0], int(a.worker[1]), a.worker[2])
    elif a.run:
        sys.exit(run_job(a.job_id, a.run[0], int(a.run[1])))
    else:
        print(json.dumps({f'{l}-r{r}-{b}': {k: v for k, v in plan(a.job_id, l, r, b).items()
                                            if k not in ('command', 'environment', 'expected_rollout_task_ids')}
                          for r in range(1, REPEATS+1) for l in RUNS for b in BENCHMARKS}, indent=1)[:3000])
