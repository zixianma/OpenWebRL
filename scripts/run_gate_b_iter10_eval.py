#!/usr/bin/env python3
"""Full-300 local GPT-4.1/action_history/T0 evaluation of Gate B at iteration 10.

Approved 2026-10-31: one 8 H200 / 64 CPU / 960 GiB job, 1.5 h, account vlm, GPT-4.1 judge cap $10.
Gate B was never evaluated at iteration 10. The plan clones the verified 40M+16F iteration-50
evaluation manifest (same harness source, config, 300 tasks, judge, decoding, TP2/64 browsers)
and swaps only the checkpoint, its step, output and W&B identity, so the result is directly
comparable with the 40M+16F evaluations. After a failure the batch process holds the
allocation for in-allocation repair (retry via <job>-hold-request.json, 30 min idle cap).
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
CONTROL = RUNTIME/'training-controls/gate-b-iter10-eval-20261031'
TEMPLATE = RUNTIME/'evaluations/arm-gate-b-40m16f-iter50-405274'
CHECKPOINT = RUNTIME/'evaluations/arm-failure-additive-313208/runtime/iter_0000009'
STEP = 9
PARENT = 'arm-gate-b-309053'
PYTHON = RUNTIME/'venv/bin/python'
HOLD_IDLE_SECONDS = 1800


def read(p):
    return json.loads(Path(p).read_text())


def plan(job, attempt):
    template = read(TEMPLATE/'evaluation_manifest.json')
    output = RUNTIME/f'evaluations/gate-b-iter10-full300-{job}-a{attempt}'
    run_id = output.name
    text = (json.dumps(template).replace(template['output'], str(output)).replace(template['checkpoint'], str(CHECKPOINT))
            .replace(f"/tmp/openwebrl-eval-{template['job_id']}-{template['checkpoint_index']}-multimodal",
                     f'/tmp/openwebrl-eval-{job}-{STEP}-multimodal'))
    p = json.loads(text)
    p['environment'].update(WANDB_RUN_ID=run_id, SLIME_CKPT_STEP=str(STEP), OPENWEBRL_EXPECTED_SCHEDULER_OFFSET='0')
    p['command'][p['command'].index('--wandb-group')+1] = 'ARM-B | iter10 | full300'
    p.update(job_id=job, attempt=0, output=str(output), checkpoint=str(CHECKPOINT), checkpoint_index=STEP,
             completed_training_iterations=STEP+1, wandb_run_id=run_id, arm_variant='gate-b',
             wandb_url=f'https://wandb.ai/zixianma/openwebrl-evals/runs/{run_id}', parent_training_run=PARENT,
             note='Gate B iteration-10 evaluation; harness cloned from the 40M+16F iteration-50 evaluation, checkpoint swapped.')
    leftovers = [s for s in ('40m16f', 'iter_0000049', template['job_id']) if s in json.dumps(p)]
    if leftovers:
        raise ValueError(f'Template identity leaked into plan: {leftovers}')
    return p


def complete(job, attempt):
    d = Path(plan(job, attempt)['output'])
    s = d/'status.json'
    return s.is_file() and read(s).get('complete') and len(list((d/'rollouts').glob('*.json'))) == 300


def worker(job, attempt):
    if os.getenv('SLURM_JOB_ID') != job:
        raise ValueError('Run inside the approved allocation')
    os.environ['OPENWEBRL_EXPECTED_SCHEDULER_OFFSET'] = '0'
    os.environ['OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT'] = '300'
    from evaluate_baseline_checkpoint import run
    run(plan(job, attempt), REPO/'.env')


def run_once(job):
    a = read(CONTROL/'approval.json')
    if a.get('approved') is not True or a.get('hours') != 1.5 or a.get('judge_cap_usd') != 10:
        raise PermissionError('Exact approval required')
    launches = read(CONTROL/'launches.json')['launches'] if (CONTROL/'launches.json').is_file() else []
    if len(launches) >= 2:
        raise RuntimeError('Launch cap reached; judge budget protection')
    attempt = len(launches)+1
    launches.append(dict(job_id=job, attempt=attempt, started=time.time()))
    write_json(CONTROL/'launches.json', dict(launches=launches))
    write_json(CONTROL/'status.json', dict(stage='evaluation', attempt=attempt, output=plan(job, attempt)['output']))
    code = subprocess.run(['srun', f'--jobid={job}', '--nodes=1', '--ntasks=1', '--cpus-per-task=64', '--gres=gpu:h200:8',
                           '--exact', '--cpu-bind=none', str(PYTHON), '-B', str(Path(__file__).resolve()),
                           '--job-id', job, '--worker', str(attempt)]).returncode
    if code or not complete(job, attempt):
        raise RuntimeError(f'attempt {attempt} failed (exit {code})')
    write_json(CONTROL/'status.json', dict(stage='complete', output=plan(job, attempt)['output'], updated=time.time()))
    return 0


def hold(job, reason):
    request = CONTROL/f'{job}-hold-request.json'
    write_json(CONTROL/f'hold-{job}.json', dict(holding=True, reason=str(reason)[:600], request_path=str(request),
                                                idle_limit_seconds=HOLD_IDLE_SECONDS, epoch=time.time()))
    start = time.time()
    while time.time()-start < HOLD_IDLE_SECONDS:
        if request.is_file():
            r = read(request)
            request.rename(request.with_name(f'{request.name}.{int(time.time())}.consumed'))
            write_json(CONTROL/f'hold-{job}.json', dict(holding=False, running=r.get('action'), epoch=time.time()))
            if r.get('action') != 'retry':
                return 1
            return subprocess.run([str(PYTHON), '-B', str(Path(__file__).resolve()), '--job-id', job, '--run']).returncode
        time.sleep(15)
    write_json(CONTROL/f'hold-{job}.json', dict(holding=False, released=True, idle_cap=True, epoch=time.time()))
    return 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--job-id', required=True)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--worker', type=int)
    ap.add_argument('--preview', action='store_true')
    a = ap.parse_args()
    if a.worker:
        worker(a.job_id, a.worker)
    elif a.run:
        sys.exit(run_once(a.job_id))
    elif a.execute:
        code = subprocess.run([str(PYTHON), '-B', str(Path(__file__).resolve()), '--job-id', a.job_id, '--run']).returncode
        sys.exit(0 if code == 0 else hold(a.job_id, f'run exited {code}; see {CONTROL}/status.json'))
    else:
        p = plan(a.job_id, 1)
        print(json.dumps({k: v for k, v in p.items() if k not in ('command', 'expected_rollout_task_ids')}, indent=1))
