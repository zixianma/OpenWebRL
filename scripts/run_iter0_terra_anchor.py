#!/usr/bin/env python3
"""Iteration-0 (OpenWebRL-4B-SFT) same-day anchor for the iteration-90 Terra re-evaluation.

Is the RL gain real against the starting model? Three arms x three full-300 repeats, interleaved by repeat:
the SFT start model (iteration 0, loaded from its HF weights through slime's bridge path, the same way every
fresh RL run starts), and outcome-only 48 and Gate B at iteration 90 as same-day anchors back to the
2026-11-09/10 nine-arm cohort. Harness, actor sampling (T0.6 / top-p 0.95 / top-k 20, 4,096 tokens, 30 turns),
deferred in-harness judge (tripwire ledger) and offline shared-Terra grading are identical to
scripts/run_iter90_terra_reeval.py; only the iteration-0 arm differs, by skipping the Megatron checkpoint view and
validation and requiring the HF-load log line as restore evidence.
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
CONTROL = RUNTIME/'training-controls/iter0-terra-anchor-20261111'
TEMPLATE = RUNTIME/'evaluations/outcome-only-56groups-20261006-iter90'
OLD = RUNTIME/'arm-turn-bonus-preparation/outcome-only-56groups-20261006/to90-20261013/source'
HARNESS = RUNTIME/'reference-om2w-eval-deferjudge-20261107'
LEDGER = CONTROL/'judge-ledger'
PYTHON = RUNTIME/'venv/bin/python'
CK = RUNTIME/'evaluations'


SFT = Path('/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT')  # 15e777db2ddba2e0e82080ebccd3ad8d215b7f0a
RUNS = {  # label: (checkpoint, expected scheduler offset, parent run)
    'sft0': (SFT, '0', 'OpenWebRL/OpenWebRL-4B-SFT@15e777db'),
    'outcome48': (RUNTIME/'runs/openwebrl-4b-reference-294421-20260913T211532/iter_0000089', '1', 'qcq7i4ug'),
    'gateb': (CK/'arm-failure-additive-335729-b100-iter90/runtime/iter_0000089', '0', 'arm-gate-b'),
}
REPEATS = 3
JUDGE_MODEL, JUDGE_TIMEOUT = 'gpt-5.6-terra', '300'
MAX_LAUNCHES = 12  # 9 planned + 3 retries
MIN_SECONDS_PER_EVAL = 1800
HOLD_IDLE_SECONDS, HOLD_TOTAL_IDLE_SECONDS = 1800, 3600
HOURS, JUDGE_CAP = 4, 100



def read(p):
    return json.loads(Path(p).read_text())


def items():
    return [f'{label}-r{r}' for r in range(1, REPEATS+1) for label in RUNS]


def base(item):
    return item.rsplit('-r', 1)[0]


def name(label):
    return f'anchor1111-terra-{label}'


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
    if hf(label):
        index = 0
        p['environment'].pop('SLIME_CKPT_STEP', None)
        p['environment']['SLIME_LOAD_CHECKPOINT'] = str(checkpoint)
        p['environment']['HF_CHECKPOINT'] = str(checkpoint)
        p['completed_training_iterations'] = 0
    else:
        index = int(checkpoint.name.split('_')[-1])
        p['environment']['SLIME_CKPT_STEP'] = str(index)
        p['completed_training_iterations'] = index+1
    p['command'] += ['--judge-timeout-secs', JUDGE_TIMEOUT]
    p['command'][p['command'].index('--wandb-group')+1] = 'iter0-terra-anchor-20261111'
    p.update(job_id=job, attempt=0, output=str(output), checkpoint=str(checkpoint), checkpoint_index=index,
             wandb_run_id=run_id, wandb_url=f'https://wandb.ai/zixianma/openwebrl-evals/runs/{run_id}',
             parent_training_run=parent, allocation_hours_cap=HOURS, run_label=label, source=str(source),
             note='Iteration-0 SFT anchor + same-day iteration-90 anchors, paper actor sampling, judge deferred to offline shared Terra.',
             judge_model='deferred: offline shared-Terra (gpt-5.6-terra)',
             protocol='OpenWebRL paper actor (T0.6/p0.95/k20, 4,096 tokens, 30 turns), local browser, 300 tasks; offline shared-Terra judge')
    leftovers = [x for x in (str(TEMPLATE),) if x in json.dumps(p)]
    if leftovers:
        raise ValueError('Template identity leaked into plan')
    return p


def hf(label):
    return not RUNS[base(label)][0].name.startswith('iter_')


def run_hf(plan, env_file):
    """evaluate_baseline_checkpoint.run for an HF start model: no Megatron view/inspection; HF-load restore evidence."""
    import re
    import evaluate_baseline_checkpoint as ebc
    ebc.restore_pattern = lambda p: (r'Load checkpoint from HuggingFace model into Megatron \(path='
                                     + re.escape(p['checkpoint']) + r'\)')
    real_inspect = ebc.source_command

    def skip_inspection(source, argv):
        if any(str(a).endswith('inspect_training_checkpoint.py') for a in argv):
            ebc.write_json(Path(plan['output'])/'checkpoint_validation.json',
                           dict(skipped='HF start model; no Megatron checkpoint to inspect', hf_checkpoint=plan['checkpoint']))
            return ['true']
        return real_inspect(source, argv)
    ebc.source_command = skip_inspection
    # run() links <checkpoint>/rollout and writes latest_checkpointed_iteration.txt into a view that this arm never loads.
    real_symlink = Path.symlink_to

    def guarded_symlink(self, target, target_is_directory=False):
        if self.parent.name == 'checkpoint-view':
            return None
        return real_symlink(self, target, target_is_directory)
    Path.symlink_to = guarded_symlink
    ebc.run(plan, env_file)


def ledger():
    path = CONTROL/'launches.json'
    return read(path) if path.is_file() else dict(launches=[])


def worker(job, label, attempt):
    if os.getenv('SLURM_JOB_ID') != job:
        raise ValueError('Run inside the approved allocation')
    os.environ['OPENWEBRL_EXPECTED_SCHEDULER_OFFSET'] = RUNS[base(label)][1]
    os.environ['OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT'] = '300'
    if hf(label):
        run_hf(plan(job, label, attempt), REPO/'.env')
    else:
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
