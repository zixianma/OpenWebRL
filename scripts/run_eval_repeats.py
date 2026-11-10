#!/usr/bin/env python3
"""Repeat full-300 local GPT-4.1/action_history/T0 evaluations of iteration-90 checkpoints.

Approved 2026-10-24: one 8 H200 / 64 CPU / 960 GiB job, at most 8 h including retries,
GPT-4.1 judge cap $60. Three new repeats each for the 48-group outcome-only baseline,
the 56-group outcome-only run and Gate B at iteration 90.

Every evaluation clones the verified 56-group iteration-90 evaluation plan (same harness
source, config, 300 tasks, judge, decoding, TP2/64 browsers) and swaps only the
checkpoint, scheduler-offset check, output and W&B identity, so all repeats share one
harness. Evaluations run in rounds (each run once per round). The batch process holds
the allocation after a failure for in-allocation repair (1 h per failure, 2 h total).
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'scripts'))
from resume_baseline import write_json  # noqa: E402

RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
CONTROL = RUNTIME/'training-controls/eval-repeats-iter90-20261024'
TEMPLATE = RUNTIME/'evaluations/outcome-only-56groups-20261006-iter90'
PYTHON = RUNTIME/'venv/bin/python'
RUNS = {  # label: (iteration-90 checkpoint, expected scheduler offset, parent training run)
    'outcome48': (RUNTIME/'runs/openwebrl-4b-reference-294421-20260913T211532/iter_0000089', '1', 'qcq7i4ug'),
    'outcome56': (RUNTIME/'runs/outcome-only-56groups-20261006/iter_0000089', '0', 'outcome-only-56groups-20261006'),
    'gate-b': (RUNTIME/'evaluations/arm-failure-additive-335729-b100-iter90/runtime/iter_0000089', '0', 'arm-gate-b'),
}
REPEATS = 3
MAX_LAUNCHES = 10  # 9 planned + 1 retry: at most ~3,000 judge calls, inside the $60 cap
MIN_SECONDS_PER_EVAL = 2700
HOLD_IDLE_SECONDS, HOLD_TOTAL_IDLE_SECONDS = 3600, 7200


def read(p):
    return json.loads(Path(p).read_text())


def items():
    return [(label, r) for r in range(1, REPEATS+1) for label in RUNS]


def name(label, repeat):
    return f'repeat-iter90-{label}-r{repeat}'


def complete(label, repeat):
    for d in sorted(RUNTIME.glob(f'evaluations/{name(label, repeat)}-*')):
        s = d/'status.json'
        if s.is_file() and read(s).get('complete') and len(list((d/'rollouts').glob('*.json'))) == 300:
            return d
    return None


def plan(job, label, repeat, attempt):
    template = read(TEMPLATE/'evaluation_manifest.json')
    checkpoint, _, parent = RUNS[label]
    output = RUNTIME/f'evaluations/{name(label, repeat)}-{job}-a{attempt}'
    run_id = output.name
    text = json.dumps(template).replace(str(TEMPLATE), str(output)).replace(template['checkpoint'], str(checkpoint))
    p = json.loads(text)
    p['environment']['WANDB_RUN_ID'] = run_id
    p['command'][p['command'].index('--wandb-group')+1] = 'eval-repeats-iter90-20261024'
    p.update(job_id=job, attempt=0, output=str(output), checkpoint=str(checkpoint), checkpoint_index=89,
             wandb_run_id=run_id, wandb_url=f'https://wandb.ai/zixianma/openwebrl-evals/runs/{run_id}',
             parent_training_run=parent, allocation_hours_cap=8, repeat=repeat, run_label=label,
             note='Repeat evaluation; harness cloned from outcome56 iteration-90 evaluation, checkpoint swapped.')
    return p


def ledger():
    path = CONTROL/'launches.json'
    return read(path) if path.is_file() else dict(launches=[])


def worker(job, label, repeat, attempt):
    if os.getenv('SLURM_JOB_ID') != job:
        raise ValueError('Run inside the approved allocation')
    os.environ['OPENWEBRL_EXPECTED_SCHEDULER_OFFSET'] = RUNS[label][1]
    os.environ['OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT'] = '300'
    from evaluate_baseline_checkpoint import run
    run(plan(job, label, repeat, attempt), REPO/'.env')


def remaining_seconds(job):
    import re
    from datetime import datetime
    fields = dict(re.findall(r'(\w+)=(\S+)', subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)))
    return datetime.strptime(fields['EndTime'], '%Y-%m-%dT%H:%M:%S').timestamp()-time.time()


def run_all(job):
    """Run pending evaluations; return 0 when all are complete or time/launch caps stop it cleanly."""
    approval = read(CONTROL/'approval.json')
    if approval.get('approved') is not True or approval.get('hours') != 8 or approval.get('judge_cap_usd') != 60:
        raise PermissionError('Exact approval required')
    for label, repeat in items():
        if complete(label, repeat):
            continue
        if remaining_seconds(job) < MIN_SECONDS_PER_EVAL:
            write_json(CONTROL/'status.json', dict(stage='time-stop', pending=[name(*x) for x in items() if not complete(*x)]))
            return 0
        book = ledger()
        if len(book['launches']) >= MAX_LAUNCHES:
            raise RuntimeError('Launch cap reached; judge budget protection')
        attempt = sum(1 for x in book['launches'] if x['name'] == name(label, repeat)) + 1
        book['launches'].append(dict(name=name(label, repeat), job_id=job, attempt=attempt, started=time.time()))
        write_json(CONTROL/'launches.json', book)
        write_json(CONTROL/'status.json', dict(stage='evaluation', current=name(label, repeat), attempt=attempt, job_id=job,
                                               completed=[name(*x) for x in items() if complete(*x)], updated=time.time()))
        step = ['srun', f'--jobid={job}', '--nodes=1', '--ntasks=1', '--cpus-per-task=64', '--gres=gpu:h200:8', '--exact',
                '--cpu-bind=none', str(PYTHON), '-B', str(Path(__file__).resolve()), '--job-id', job,
                '--worker', label, str(repeat), str(attempt)]
        code = subprocess.run(step).returncode
        if code or not complete(label, repeat):
            raise RuntimeError(f'{name(label, repeat)} attempt {attempt} failed (exit {code})')
    write_json(CONTROL/'status.json', dict(stage='complete', completed=[str(complete(*x)) for x in items()], updated=time.time()))
    return 0


def hold(job, reason):
    request = CONTROL/f'{job}-hold-request.json'
    idle_used = 0.
    while True:
        limit = min(HOLD_IDLE_SECONDS, HOLD_TOTAL_IDLE_SECONDS-idle_used)
        write_json(CONTROL/f'hold-{job}.json', dict(holding=True, reason=str(reason)[:600], idle_limit_seconds=int(limit),
                                                    request_path=str(request), epoch=time.time()))
        start, r = time.time(), None
        while time.time()-start < limit:
            if request.is_file():
                r = read(request)
                request.rename(request.with_name(f'{request.name}.{int(time.time())}.consumed'))
                break
            time.sleep(15)
        idle_used += time.time()-start
        if r is None or r.get('action') == 'release':
            write_json(CONTROL/f'hold-{job}.json', dict(holding=False, released=True, idle_cap=r is None, epoch=time.time()))
            return 1
        argv = ([str(PYTHON), '-B', str(Path(__file__).resolve()), '--job-id', job, '--run-all']
                if r.get('action') == 'retry' else list(map(str, r.get('argv', []))))
        code = subprocess.run(argv).returncode if argv else 2
        with (CONTROL/f'hold-{job}.jsonl').open('a') as log:
            log.write(json.dumps(dict(epoch=time.time(), argv=argv, exit_code=code))+'\n')
        if code == 0 and r.get('action') == 'retry':
            return 0
        reason = f'{r.get("action")} exited {code}'


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job-id', required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--run-all', action='store_true')
    p.add_argument('--worker', nargs=3)
    p.add_argument('--preview', action='store_true')
    a = p.parse_args()
    if a.worker:
        worker(a.job_id, a.worker[0], int(a.worker[1]), int(a.worker[2]))
    elif a.run_all:
        sys.exit(run_all(a.job_id))
    elif a.execute:
        code = subprocess.run([str(PYTHON), '-B', str(Path(__file__).resolve()), '--job-id', a.job_id, '--run-all']).returncode
        sys.exit(0 if code == 0 else hold(a.job_id, f'run-all exited {code}; see {CONTROL}/status.json'))
    else:
        print(json.dumps({f'{l}-r{r}': {k: v for k, v in plan(a.job_id, l, r, 1).items() if k not in ('command', 'environment', 'expected_rollout_task_ids')}
                          for l, r in items()[:3]}, indent=1))
