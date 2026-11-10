#!/usr/bin/env python3
"""Harness step-error fix validation and checkpoint-average ("soup") evaluations (#5).

Approved 2026-11-03: one 8 H200 / 64 CPU / 960 GiB job, 4 h, account zixianma, QoS urgent, GPT-4.1
judge cap $30. Five full-300 local GPT-4.1/action_history/T0 evaluations, run in this order:
  1. gateb90-old     Gate B iteration 90, old harness (the iteration-90 repeats harness, job 385593)
  2. gateb90-fix     Gate B iteration 90, same harness plus only the step-error fix (commit c9caa02)
  3. gatebsoup-fix   uniform weight average of Gate B 70/80/90, fixed harness
  4. outcome56-90-fix   outcome-only 56 groups iteration 90, fixed harness
  5. outcome56soup-fix  uniform weight average of outcome-only 56 groups 70/80/90, fixed harness
Every evaluation clones the verified outcome56 iteration-90 plan (as the repeats did) and swaps only the
checkpoint, harness source, output and W&B identity. The batch process holds the allocation after a
failure for in-allocation repair (retry via <job>-hold-request.json).
"""
import argparse
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
CONTROL = RUNTIME/'training-controls/harness-soup-evals-20261103'
TEMPLATE = RUNTIME/'evaluations/outcome-only-56groups-20261006-iter90'
OLD = RUNTIME/'arm-turn-bonus-preparation/outcome-only-56groups-20261006/to90-20261013/source'
FIX = RUNTIME/'reference-om2w-eval-stepfix-20261103'
PYTHON = RUNTIME/'venv/bin/python'
GB90 = RUNTIME/'evaluations/arm-failure-additive-335729-b100-iter90/runtime/iter_0000089'
O90 = RUNTIME/'runs/outcome-only-56groups-20261006/iter_0000089'
RUNS = {  # label: (checkpoint, harness source, expected scheduler offset, parent run)
    'gateb90-old': (GB90, OLD, '0', 'arm-gate-b'),
    'gateb90-fix': (GB90, FIX, '0', 'arm-gate-b'),
}
MAX_LAUNCHES = 3  # 2 planned + 1 retry: at most ~900 judge calls, inside the $12 cap
MIN_SECONDS_PER_EVAL = 2400
HOLD_IDLE_SECONDS, HOLD_TOTAL_IDLE_SECONDS = 1800, 3600
HOURS, JUDGE_CAP = 2, 12


def read(p):
    return json.loads(Path(p).read_text())


def items():
    return list(RUNS)


def name(label):
    return f'harness-soup-{label}'


def complete(label):
    for d in sorted(RUNTIME.glob(f'evaluations/{name(label)}-*')):
        s = d/'status.json'
        if s.is_file() and read(s).get('complete') and len(list((d/'rollouts').glob('*.json'))) == 300:
            return d
    return None


def plan(job, label, attempt):
    template = read(TEMPLATE/'evaluation_manifest.json')
    checkpoint, source, _, parent = RUNS[label]
    if template['source'] != str(OLD):
        raise ValueError('Template harness is not the repeats harness')
    output = RUNTIME/f'evaluations/{name(label)}-{job}-a{attempt}'
    run_id = output.name
    text = (json.dumps(template).replace(str(TEMPLATE), str(output)).replace(template['checkpoint'], str(checkpoint))
            .replace(str(OLD), str(source)))
    p = json.loads(text)
    p['environment']['WANDB_RUN_ID'] = run_id
    p['command'][p['command'].index('--wandb-group')+1] = 'harness-soup-evals-20261103'
    p.update(job_id=job, attempt=0, output=str(output), checkpoint=str(checkpoint), checkpoint_index=89,
             wandb_run_id=run_id, wandb_url=f'https://wandb.ai/zixianma/openwebrl-evals/runs/{run_id}',
             parent_training_run=parent, allocation_hours_cap=HOURS, run_label=label, source=str(source),
             note='Harness-fix / checkpoint-average evaluation; plan cloned from outcome56 iteration-90 evaluation.')
    leftovers = [x for x in (str(TEMPLATE),) if x in json.dumps(p)]
    if leftovers or (source == FIX and str(OLD) in json.dumps(p)):
        raise ValueError('Template identity leaked into plan')
    return p


def ledger():
    path = CONTROL/'launches.json'
    return read(path) if path.is_file() else dict(launches=[])


def worker(job, label, attempt):
    if os.getenv('SLURM_JOB_ID') != job:
        raise ValueError('Run inside the approved allocation')
    os.environ['OPENWEBRL_EXPECTED_SCHEDULER_OFFSET'] = RUNS[label][2]
    os.environ['OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT'] = '300'
    from evaluate_baseline_checkpoint import run
    run(plan(job, label, attempt), REPO/'.env')


def remaining_seconds(job):
    import re
    from datetime import datetime
    fields = dict(re.findall(r'(\w+)=(\S+)', subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)))
    return datetime.strptime(fields['EndTime'], '%Y-%m-%dT%H:%M:%S').timestamp()-time.time()


def run_all(job):
    """Run pending evaluations; return 0 when all are complete or time/launch caps stop it cleanly."""
    approval = read(CONTROL/'approval.json')
    if approval.get('approved') is not True or approval.get('hours') != HOURS or approval.get('judge_cap_usd') != JUDGE_CAP:
        raise PermissionError('Exact approval required')
    for label in items():
        if not RUNS[label][0].is_dir():
            raise FileNotFoundError(f'{label}: checkpoint missing')
        if complete(label):
            continue
        if remaining_seconds(job) < MIN_SECONDS_PER_EVAL:
            write_json(CONTROL/'status.json', dict(stage='time-stop', pending=[name(x) for x in items() if not complete(x)]))
            return 0
        book = ledger()
        if len(book['launches']) >= MAX_LAUNCHES:
            raise RuntimeError('Launch cap reached; judge budget protection')
        attempt = sum(1 for x in book['launches'] if x['name'] == name(label)) + 1
        book['launches'].append(dict(name=name(label), job_id=job, attempt=attempt, started=time.time()))
        write_json(CONTROL/'launches.json', book)
        write_json(CONTROL/'status.json', dict(stage='evaluation', current=name(label), attempt=attempt, job_id=job,
                                               completed=[name(x) for x in items() if complete(x)], updated=time.time()))
        step = ['srun', f'--jobid={job}', '--nodes=1', '--ntasks=1', '--cpus-per-task=64', '--gres=gpu:h200:8', '--exact',
                '--cpu-bind=none', str(PYTHON), '-B', str(Path(__file__).resolve()), '--job-id', job,
                '--worker', label, str(attempt)]
        code = subprocess.run(step).returncode
        if code or not complete(label):
            raise RuntimeError(f'{name(label)} attempt {attempt} failed (exit {code})')
    write_json(CONTROL/'status.json', dict(stage='complete', completed=[str(complete(x)) for x in items()], updated=time.time()))
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
    p.add_argument('--worker', nargs=2)
    p.add_argument('--preview', action='store_true')
    a = p.parse_args()
    if a.worker:
        worker(a.job_id, a.worker[0], int(a.worker[1]))
    elif a.run_all:
        sys.exit(run_all(a.job_id))
    elif a.execute:
        code = subprocess.run([str(PYTHON), '-B', str(Path(__file__).resolve()), '--job-id', a.job_id, '--run-all']).returncode
        sys.exit(0 if code == 0 else hold(a.job_id, f'run-all exited {code}; see {CONTROL}/status.json'))
    else:
        print(json.dumps({l: {k: v for k, v in plan(a.job_id, l, 1).items() if k not in ('command', 'environment', 'expected_rollout_task_ids')}
                          for l in items()}, indent=1))
