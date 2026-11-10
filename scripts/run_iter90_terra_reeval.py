#!/usr/bin/env python3
"""Same-day iteration-90 re-evaluation: Gate B variants vs outcome-only, OpenWebRL paper actor protocol, Terra judge.

Approved 2026-11-07: one 8 H200 / 64 CPU / 960 GiB job, vlm / normal, 8 h incl. retries, judge cap $100. Five
iteration-90 checkpoints (Gate B, 40M+16F, 48M+16F, outcome-only 48 and 56 groups) x three full-300 repeats,
interleaved by repeat so every checkpoint sees the same live-site window. Harness: local browser, 30 turns,
actor T0.6 / top-p 0.95 / top-k 20, 4,096 response tokens (reference-om2w-eval-deferjudge-20261107). No GPT-4.1
or other in-harness judge: OPENWEBRL_DEFER_JUDGE=1 stubs it (tripwire ledger $0.01 / 1 call). All completed
episodes are graded afterwards on CPU by scripts/rejudge_rl_evals_shared_terra.py (gpt-5.6-terra, frozen
terminal-success prompt, medium effort, 4,096 tokens, seed 42, final screenshot). In-harness success counts are
not results. Every evaluation clones the verified outcome56 iteration-90 plan and swaps only the checkpoint,
harness, judge settings, output and W&B identity.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path('/gpfs/projects/krishna/zixianma/OpenWebRL')
sys.path.insert(0, str(REPO/'scripts'))
from resume_baseline import write_json  # noqa: E402

RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
CONTROL = RUNTIME/'training-controls/iter90-terra-reeval-20261107'
TEMPLATE = RUNTIME/'evaluations/outcome-only-56groups-20261006-iter90'
OLD = RUNTIME/'arm-turn-bonus-preparation/outcome-only-56groups-20261006/to90-20261013/source'
HARNESS = RUNTIME/'reference-om2w-eval-deferjudge-20261107'
LEDGER = CONTROL/'judge-ledger'
PYTHON = RUNTIME/'venv/bin/python'
CK = RUNTIME/'evaluations'


def g48m16f_checkpoint():
    """Latest 48M+16F to-90 root holding a validated iteration-90 checkpoint."""
    for root in sorted(CK.glob('arm-failure-additive-48m16f-*'), key=lambda p: p.stat().st_mtime, reverse=True):
        if (root/'iterations/0089/checkpoint-validation.json').is_file() and (root/'runtime/iter_0000089').is_dir():
            return root/'runtime/iter_0000089'
    return CK/'arm-failure-additive-48m16f-PENDING/runtime/iter_0000089'


RUNS = {  # label: (checkpoint, expected scheduler offset, parent run)
    'gateb': (CK/'arm-failure-additive-335729-b100-iter90/runtime/iter_0000089', '0', 'arm-gate-b'),
    'g40m16f': (CK/'arm-failure-additive-40m16f-413757-iter90/runtime/iter_0000089', '0', 'arm-gate-b-40m16f-392093'),
    'g48m16f': (g48m16f_checkpoint(), '0', 'arm-gate-b-48m16f'),
    'outcome48': (RUNTIME/'runs/openwebrl-4b-reference-294421-20260913T211532/iter_0000089', '1', 'qcq7i4ug'),
    'outcome56': (RUNTIME/'runs/outcome-only-56groups-20261006/iter_0000089', '0', 'outcome-only-56groups-20261006'),
}
REPEATS = 3
JUDGE_MODEL, JUDGE_TIMEOUT = 'gpt-5.6-terra', '300'
MAX_LAUNCHES = 20  # 15 planned + 5 retries
MIN_SECONDS_PER_EVAL = 1800
HOLD_IDLE_SECONDS, HOLD_TOTAL_IDLE_SECONDS = 1800, 3600
HOURS, JUDGE_CAP = 8, 100



def read(p):
    return json.loads(Path(p).read_text())


def items():
    return [f'{label}-r{r}' for r in range(1, REPEATS+1) for label in RUNS]


def base(item):
    return item.rsplit('-r', 1)[0]


def name(label):
    return f'iter90-terra-{label}'


def complete(label):
    for d in sorted(RUNTIME.glob(f'evaluations/{name(label)}-*')):
        s = d/'status.json'
        m = d/'evaluation_manifest.json'
        if not m.is_file() or read(m).get('source') != str(HARNESS):
            continue
        if s.is_file() and read(s).get('complete') and len(list((d/'rollouts').glob('*.json'))) == 300:
            return d
    return None


def plan(job, label, attempt):
    template = read(TEMPLATE/'evaluation_manifest.json')
    checkpoint, _, parent = RUNS[base(label)]
    source = HARNESS
    if template['source'] != str(OLD):
        raise ValueError('Template harness is not the repeats harness')
    output = RUNTIME/f'evaluations/{name(label)}-{job}-a{attempt}'
    run_id = output.name
    text = (json.dumps(template).replace(str(TEMPLATE), str(output)).replace(template['checkpoint'], str(checkpoint))
            .replace(str(OLD), str(source)))
    p = json.loads(text)
    p['environment']['WANDB_RUN_ID'] = run_id
    p['environment']['JUDGE_MODEL'] = JUDGE_MODEL
    p['environment']['OPENWEBRL_EXPANDED_BASELINE_CONTROL'] = str(LEDGER)
    p['environment']['OPENWEBRL_JUDGE_TIMEOUT_SECS'] = JUDGE_TIMEOUT
    p['environment']['OPENWEBRL_DEFER_JUDGE'] = '1'
    index = int(checkpoint.name.split('_')[-1])
    p['environment']['SLIME_CKPT_STEP'] = str(index)
    p['completed_training_iterations'] = index+1
    p['command'] += ['--judge-timeout-secs', JUDGE_TIMEOUT]
    p['command'][p['command'].index('--wandb-group')+1] = 'iter90-terra-reeval-20261107'
    p.update(job_id=job, attempt=0, output=str(output), checkpoint=str(checkpoint), checkpoint_index=index,
             wandb_run_id=run_id, wandb_url=f'https://wandb.ai/zixianma/openwebrl-evals/runs/{run_id}',
             parent_training_run=parent, allocation_hours_cap=HOURS, run_label=label, source=str(source),
             note='Same-day iteration-90 re-evaluation, paper actor sampling, judge deferred to offline shared Terra.',
             judge_model='deferred: offline shared-Terra (gpt-5.6-terra)',
             protocol='OpenWebRL paper actor (T0.6/p0.95/k20, 4,096 tokens, 30 turns), local browser, 300 tasks; offline shared-Terra judge')
    leftovers = [x for x in (str(TEMPLATE),) if x in json.dumps(p)]
    if leftovers:
        raise ValueError('Template identity leaked into plan')
    return p


def ledger():
    path = CONTROL/'launches.json'
    return read(path) if path.is_file() else dict(launches=[])


def worker(job, label, attempt):
    if os.getenv('SLURM_JOB_ID') != job:
        raise ValueError('Run inside the approved allocation')
    os.environ['OPENWEBRL_EXPECTED_SCHEDULER_OFFSET'] = RUNS[base(label)][1]
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
        if not RUNS[base(label)][0].is_dir():
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
