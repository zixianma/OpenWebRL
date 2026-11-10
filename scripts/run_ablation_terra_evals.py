#!/usr/bin/env python3
"""Same-week re-evaluation of Gate B, 40M+16F and 48M+16F at iterations 50 and 60 with a GPT-5.6 Terra judge.

Approved 2026-11-04: one 8 H200 / 64 CPU / 960 GiB job, zixianma / urgent. Six full-300 evaluations,
interleaved by iteration (Gate B, 40M+16F, 48M+16F at 50, then at 60), back to back on one harness: the
local-browser, actor-T0, 30-turn harness of the iteration-90 repeats (job 385593) and of the 2026-11-03
Gate B harness test. The only protocol change is the judge model (gpt-5.6-terra, same action_history prompt,
seed 42) with a 300 s judge timeout for the reasoning model. Every evaluation clones the verified outcome56
iteration-90 plan and swaps only the checkpoint, judge model/timeout, output and W&B identity.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path('/gpfs/projects/krishna/zixianma/OpenWebRL')  # copy runs from the control dir (project quota full)
sys.path.insert(0, str(REPO/'scripts'))
from resume_baseline import write_json  # noqa: E402

RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
CONTROL = RUNTIME/'training-controls/ablation-terra-evals-20261104'
TEMPLATE = RUNTIME/'evaluations/outcome-only-56groups-20261006-iter90'
OLD = RUNTIME/'arm-turn-bonus-preparation/outcome-only-56groups-20261006/to90-20261013/source'
# OLD plus Terra pricing in the judge ledger and an env-set judge timeout (TERRA.txt); 418387 a2 showed the
# OLD guard refuses every non-GPT-4.1 call ('Unpriced judge model') and eval_monitor pins the timeout to 120 s.
TERRA_HARNESS = RUNTIME/'reference-om2w-eval-terra-20261104'
LEDGER = CONTROL/'judge-ledger'
PYTHON = RUNTIME/'venv/bin/python'
CK = RUNTIME/'evaluations'
RUNS = {  # label: (checkpoint, harness source, expected scheduler offset, parent run)
    'gateb50': (CK/'arm-failure-additive-329908-iter60/runtime/iter_0000049', TERRA_HARNESS, '0', 'arm-gate-b'),
    'g40m16f50': (CK/'arm-failure-additive-40m16f-405274-iter50/runtime/iter_0000049', TERRA_HARNESS, '0', 'arm-gate-b-40m16f-392093'),
    'g48m16f50': (CK/'arm-failure-additive-48m16f-413756-iter50/runtime/iter_0000049', TERRA_HARNESS, '0', 'arm-gate-b-48m16f'),
    'gateb60': (CK/'arm-failure-additive-329908-iter60/runtime/iter_0000059', TERRA_HARNESS, '0', 'arm-gate-b'),
    'g40m16f60': (CK/'arm-failure-additive-40m16f-405274-iter60/runtime/iter_0000059', TERRA_HARNESS, '0', 'arm-gate-b-40m16f-392093'),
    'g48m16f60': (CK/'arm-failure-additive-48m16f-413756-iter60/runtime/iter_0000059', TERRA_HARNESS, '0', 'arm-gate-b-48m16f'),
}
JUDGE_MODEL, JUDGE_TIMEOUT = 'gpt-5.6-terra', '300'
MAX_LAUNCHES = 9  # 3 used by 418387 (gateb50 a1/a2, g40m16f50 a1; $0 judge spend: refused before any API call) + 6 planned; $100 ledger cap unchanged
MIN_SECONDS_PER_EVAL = 2400
HOLD_IDLE_SECONDS, HOLD_TOTAL_IDLE_SECONDS = 1800, 3600
HOURS, JUDGE_CAP = 4, 100


def read(p):
    return json.loads(Path(p).read_text())


def items():
    return list(RUNS)


def name(label):
    return f'terra-reeval-{label}'


def complete(label):
    for d in sorted(RUNTIME.glob(f'evaluations/{name(label)}-*')):
        s = d/'status.json'
        m = d/'evaluation_manifest.json'
        if not m.is_file() or read(m).get('source') != str(TERRA_HARNESS):
            continue  # 418387 a1/a2 ran the unpatched harness; every judge call was refused
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
    p['environment']['JUDGE_MODEL'] = JUDGE_MODEL
    p['environment']['OPENWEBRL_EXPANDED_BASELINE_CONTROL'] = str(LEDGER)
    p['environment']['OPENWEBRL_JUDGE_TIMEOUT_SECS'] = JUDGE_TIMEOUT
    index = int(checkpoint.name.split('_')[-1])
    p['environment']['SLIME_CKPT_STEP'] = str(index)
    p['completed_training_iterations'] = index+1
    p['command'] += ['--judge-timeout-secs', JUDGE_TIMEOUT]
    p['command'][p['command'].index('--wandb-group')+1] = 'ablation-terra-evals-20261104'
    p.update(job_id=job, attempt=0, output=str(output), checkpoint=str(checkpoint), checkpoint_index=index,
             wandb_run_id=run_id, wandb_url=f'https://wandb.ai/zixianma/openwebrl-evals/runs/{run_id}',
             parent_training_run=parent, allocation_hours_cap=HOURS, run_label=label, source=str(source),
             note='Same-week Terra-judged re-evaluation; plan cloned from outcome56 iteration-90 evaluation.', judge_model=JUDGE_MODEL)
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
