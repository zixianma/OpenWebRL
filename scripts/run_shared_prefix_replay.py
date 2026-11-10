#!/usr/bin/env python3
"""Own a bounded paired replay allocation; never submit or extend one."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


SOURCE = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-turn-bonus-preparation/'
              'branch-selector-scaleup-20261008/pilot32-v1/source')
RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
CONTROLLER = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / 'scripts'))
sys.path.insert(0, str(CONTROLLER))
from openwebrl.arm_branch_state import BranchState
from openwebrl.arm_branch_worker import sha, write
from openwebrl.controlled_sft_state import digest
from owned_cleanup import remember_child, stop_owned, cleanup_browser_groups
from resume_baseline import source_command, validate_source
from run_arm_branches import allocation_deadline
from run_controlled_sft_suite import check_approval, scheduler_accounting
from runtime_ports import lease_ports


def read(path):
    return json.loads(Path(path).read_text())


def validate(root):
    root = Path(root).resolve()
    plan = read(root / 'plan.json')
    if (Path(plan['root']) != root or Path(plan['source']) != SOURCE
            or sha(__file__) != plan['controller_sha256']):
        raise ValueError('Replay controller, source, or ledger root changed')
    validate_source(SOURCE)
    if (len(plan['states']) != 16 or len(plan['schedule']) != 640
            or plan['candidate_count'] != 5 or plan['continuation_repetitions'] != 8
            or plan['resources']['total_seconds'] != 28800
            or plan['limits']['browser_sessions'] != 900):
        raise ValueError('Frozen paired design or caps changed')
    cohort = read(plan['cohort_manifest'])
    if (cohort['cohort_states'] != 8 or cohort['task_groups'] != 5
            or cohort['branch_records_planned'] != 640):
        raise ValueError('Frozen external cohort changed')
    for path, expected in plan['artifacts'].items():
        if sha(path) != expected:
            raise ValueError('Frozen input changed: ' + path)
    for name, expected in plan['actor_files'].items():
        info = (Path(plan['worker_config']['actor_checkpoint']) / name).stat()
        if dict(size=info.st_size, mtime_ns=info.st_mtime_ns) != expected:
            raise ValueError('Pinned actor file changed: ' + name)
    by_state = {}
    for state in plan['states']:
        by_state.setdefault(state['source_state'], []).append(state)
        reference = read(root / 'states' / state['id'] / 'anchor-reference.json')
        if (reference['sha256'] != state['anchor_sha256']
                or sha(reference['path']) != reference['sha256']):
            raise ValueError('Branch anchor identity changed')
    if any({s['condition'] for s in states} != {'original', 'shared'}
           or len(states) != 2 for states in by_state.values()):
        raise ValueError('Every source state needs both paired conditions')
    return plan


def result(state, item):
    row, attempts = state.item(item['item_id'])
    finished = [attempt for attempt in attempts if attempt['status'] == 'finished']
    if (row['status'] != 'finished' or len(finished) != 1
            or attempts[-1]['status'] != 'finished'
            or any(attempt['status'] not in ('interrupted', 'finished') for attempt in attempts)):
        raise ValueError('Branch attempt is not durably finished; diagnose before retry')
    value = json.loads(finished[0]['result_json'])
    if sha(value['result_path']) != value['result_sha256']:
        raise ValueError('Finished result payload changed')
    return value


def execute(root, job):
    root = Path(root).resolve()
    plan = validate(root)
    if str(job) != os.getenv('SLURM_JOB_ID') or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Controller must own its registered Slurm allocation')
    approval = read(root / 'approval.json')
    used, previous = scheduler_accounting(approval, job)
    seconds = check_approval(plan, approval, job, used)
    info = subprocess.check_output(['scontrol', 'show', 'job', '-o', str(job)], text=True)
    deadline = allocation_deadline(info, job, seconds, plan['resources'])
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(devices) != 1 or not devices[0]:
        raise ValueError('Exactly one approved H200 must be visible')
    from dotenv import dotenv_values
    env = dict(os.environ)
    for key, value in dotenv_values('/gpfs/projects/krishna/zixianma/OpenWebRL/.env').items():
        if value and key.startswith(('OPENAI_', 'JUDGE_', 'WANDB_')):
            env.setdefault(key, value)
    if not (env.get('JUDGE_API_KEY') or env.get('OPENAI_API_KEY')):
        raise ValueError('Canonical outcome judge credential is unavailable')
    env.update(WANDB_PROJECT='openwebrl-evals', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    journal = root / 'scheduler-attempts' / str(job)
    journal.mkdir(parents=True, exist_ok=True)
    write(journal / 'prior-accounting.json', dict(prior_seconds=used, attempts=previous))
    write(journal / 'allocation-validated.json', dict(job_id=str(job), plan_sha256=digest(plan),
          resources=plan['resources'], limits=plan['limits'], prior_actual_seconds=used,
          current_attempt_limit=seconds, work_deadline_unix=deadline))
    children, handles, servers = [], [], []
    state = lease = None
    completed = False

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def spawn(argv, name, extra=None):
        handle = (journal / (name + '.log')).open('a')
        handles.append(handle)
        child = subprocess.Popen(source_command(SOURCE, argv), cwd=SOURCE,
                                 env=dict(env, **(extra or {})), stdout=handle,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        children.append(remember_child(child, job))
        return child

    def health(stage):
        if time.time() >= deadline:
            raise TimeoutError('All-attempt allocation boundary reached')
        if any(child.poll() is not None for child in servers):
            raise RuntimeError('Actor server stopped during ' + stage)
        snapshot = state.snapshot()
        if snapshot['halt']:
            raise RuntimeError('Durable budget or provider halt needs diagnosis')
        (root / 'current.json').write_text(json.dumps(dict(stage=stage, job_id=str(job),
            updated_utc=datetime.now(timezone.utc).isoformat(), **snapshot), indent=2) + '\n')

    def await_workers(workers, stage):
        while any(child.poll() is None for child in workers):
            health(stage)
            if any(child.poll() not in (None, 0) for child in workers):
                raise RuntimeError('Worker failed in ' + stage)
            time.sleep(2)
        if any(child.returncode != 0 for child in workers):
            raise RuntimeError('Worker failed in ' + stage)

    def worker(item, slot, actor_url):
        return spawn([str(RUNTIME / 'venv/bin/python'), '-m', 'openwebrl.arm_branch_worker',
                      '--root', str(root), '--item', item['item_id'], '--actor-url', actor_url],
                     item['item_id'], dict(CUDA_VISIBLE_DEVICES='',
                     OPENWEBRL_ROLLOUT_PORT_BASE=str(base + 64 + slot * 12)))

    def panel(state_info, actor_url):
        panel_id = state_info['id']
        panel_dir = root / 'states' / panel_id
        items = sorted((item for item in plan['schedule'] if item['state'] == panel_id),
                       key=lambda item: (item['candidate'], item['repeat']))
        if (panel_dir / 'release.json').exists():
            for item in items:
                value = result(state, item)
                if not value['released']:
                    raise ValueError('Previously released branch lacks an outcome')
            return True
        if (panel_dir / 'reject.json').exists():
            for item in items:
                row, attempts = state.item(item['item_id'])
                if row['status'] == 'active' or any(a['status'] == 'interrupted' for a in attempts):
                    raise ValueError('Rejected panel contains unresolved attempts')
            return False
        if any(state.item(item['item_id'])[0]['status'] != 'pending' for item in items):
            raise ValueError('Unreleased panel has nonpending work; diagnose before retry')
        if deadline - time.time() < 2700:
            raise TimeoutError('Insufficient approved time to start a whole 40-branch panel')
        workers = [worker(items[0], 0, actor_url)]
        first_attempt = None
        until = min(deadline, time.time() + 600)
        while time.time() < until:
            health('first_reconstruction')
            attempts = [attempt for attempt in state.item(items[0]['item_id'])[1]
                        if attempt['job_id'] == str(job) and attempt['status'] == 'active']
            if len(attempts) == 1:
                first_attempt = attempts[0]
                if (Path(first_attempt['artifact_directory']) / 'ready.json').exists():
                    break
            if workers[0].poll() is not None:
                if workers[0].returncode != 0:
                    raise RuntimeError('First strict replay worker failed')
                write(panel_dir / 'reject.json', dict(reason='first_reconstruction_rejected'))
                return False
            time.sleep(1)
        else:
            raise TimeoutError('First strict reconstruction did not reach readiness')
        for slot, item in enumerate(items[1:], 1):
            workers.append(worker(item, slot, actor_url))
            time.sleep(1.0)
        until = min(deadline, time.time() + 900)
        while time.time() < until:
            health('reconstructing_panel')
            attempts = [[attempt for attempt in state.item(item['item_id'])[1]
                         if attempt['job_id'] == str(job) and attempt['status'] == 'active']
                        for item in items]
            if all(len(entry) == 1 for entry in attempts):
                dirs = [entry[0]['artifact_directory'] for entry in attempts]
                if all((Path(directory) / 'ready.json').exists() for directory in dirs):
                    state.release(panel_id, state_info['anchor_sha256'], dirs)
                    break
            if any(child.poll() is not None for child in workers):
                if any(child.poll() not in (None, 0) for child in workers):
                    raise RuntimeError('Strict replay worker failed')
                write(panel_dir / 'reject.json', dict(reason='strict_reconstruction_rejected'))
                await_workers(workers, 'rejected_panel_closeout')
                return False
            time.sleep(1)
        else:
            write(panel_dir / 'reject.json', dict(reason='reconstruction_timeout'))
            await_workers(workers, 'timed_out_reconstruction_closeout')
            return False
        await_workers(workers, 'continuations')
        values = [result(state, item) for item in items]
        if not all(value['released'] for value in values):
            raise ValueError('Released panel lacks a continuation record')
        write(panel_dir / 'complete.json', dict(panel_id=panel_id,
              condition=state_info['condition'], states=1, records=40,
              valid=sum(value['valid'] for value in values),
              invalid=sum(not value['valid'] for value in values)))
        return True

    with (root / 'controller.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            state = BranchState(root)
            snapshot = state.snapshot()
            if snapshot['halt'] or any(snapshot['queue'].get(key) for key in ('active', 'interrupted')):
                raise ValueError('Unresolved prior work requires diagnosis before retry')
            feature_dir = root / 'results' / 'features'
            feature_dir.mkdir(parents=True, exist_ok=True)
            if not (feature_dir / 'features-0000.pt').exists():
                extract = spawn([str(RUNTIME / 'venv/bin/python'),
                    str(CONTROLLER / 'extract_shared_prefix_features.py'),
                    '--panels', plan['feature_panels'], '--output', str(feature_dir)], 'features',
                    {'CUDA_VISIBLE_DEVICES': devices[0]})
                await_workers([extract], 'feature_replay')
            score = spawn([str(RUNTIME / 'venv/bin/python'),
                str(CONTROLLER / 'score_shared_prefix_features.py'),
                '--root', str(root), '--score'], 'frozen_picks',
                {'CUDA_VISIBLE_DEVICES': ''})
            await_workers([score], 'outcome_blind_scoring')
            if not (root / 'model/frozen-picks.json').exists():
                raise ValueError('Actor and value choices were not frozen before replay')
            lease, base = lease_ports(job, bases=range(30000, 54000, 1024), size=640)
            actor_url = f'http://127.0.0.1:{base}'
            servers.append(spawn([str(RUNTIME / 'venv/bin/python'),
                str(CONTROLLER / 'actor_entrypoint.py'), '--skip-server-warmup',
                '--model-path', plan['worker_config']['actor_checkpoint'],
                '--host', '127.0.0.1', '--port', str(base), '--dtype', 'bfloat16',
                '--tp', '1', '--mem-fraction-static', '.65', '--context-length', '32768',
                '--max-running-requests', '16', '--chunked-prefill-size', '4096',
                '--disable-cuda-graph'], 'actor',
                dict(CUDA_VISIBLE_DEVICES=devices[0], ARM_ACTOR_REPLICA='0')))
            until = min(deadline, time.time() + 900)
            while time.time() < until:
                if servers[0].poll() is not None:
                    raise RuntimeError('Actor failed startup')
                try:
                    with urllib.request.urlopen(actor_url + '/get_model_info', timeout=2) as response:
                        card = json.load(response)
                    if card.get('model_path') != plan['worker_config']['actor_checkpoint']:
                        raise ValueError('Unexpected actor model')
                    write(journal / 'actor-identity.json', card)
                    break
                except OSError:
                    time.sleep(2)
            else:
                raise TimeoutError('Actor startup timed out')
            finished = []
            for info in plan['states']:
                if panel(info, actor_url):
                    finished.append(info['id'])
                write(journal / ('panel-' + info['id'] + '.json'),
                      dict(panel_id=info['id'], complete=info['id'] in finished,
                           finished_panels=len(finished), recorded_panels=len(plan['states'])))
                if len(finished) < 2 and len([x for x in plan['states']
                        if (root / 'states' / x['id'] / 'reject.json').exists()]) >= 4:
                    write(journal / 'replay-yield-stop.json',
                          dict(reason='Four prespecified panels rejected before two completed'))
                    break
            complete = len(finished) == len(plan['states'])
            write(journal / 'collection-closeout.json', dict(
                status='complete_pending_independent_analysis' if complete else 'partial',
                job_id=str(job), completed_panels=len(finished), target_panels=len(plan['states']),
                completed_records=40 * len(finished), target_records=640,
                rejected_panels=len(list((root / 'states').glob('*/reject.json'))),
                source_states=8, task_groups=5))
            completed = complete
        finally:
            direct = stop_owned(children, job)
            browsers = cleanup_browser_groups(root, job)
            clean = all(row['process_gone'] for row in direct + browsers)
            write(journal / 'owned-process-teardown.json', dict(verified=clean,
                  direct=direct, browsers=browsers))
            if state is not None:
                state.mark_interrupted_after_teardown(teardown_verified=clean,
                    reason='shared-prefix controller closeout')
            if lease is not None:
                lease.close()
            for handle in handles:
                handle.close()
    if not completed:
        raise RuntimeError('Paired replay is partial; inspect saved causes and budget')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dry-run', action='store_true')
    group.add_argument('--execute')
    args = parser.parse_args()
    if args.dry_run:
        plan = validate(args.root)
        print(json.dumps(dict(prepared=True,
              approved=read(args.root / 'approval.json')['approved'],
              source_states=8, condition_panels=16, branch_records=640,
              resources=plan['resources'], limits=plan['limits']), indent=2))
    else:
        execute(args.root, args.execute)


if __name__ == '__main__':
    main()
