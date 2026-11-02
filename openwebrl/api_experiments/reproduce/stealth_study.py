"""Prepare and own a fresh, paired 300-task US-residential BrowserCode study.

Preparation is offline. Execution requires a plan-bound financial approval and
the exact CPU allocation. Historical twenty-pair pilot sources remain intact.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
from decimal import Decimal
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from . import common, stealth_pilot as pilot, stealth_plan

ARMS = pilot.ARMS
need, read, write, digest, money = pilot.need, pilot.read, pilot.write, pilot.digest, pilot.money
SHARDS, PAIRS, SECONDS = 2, 150, 57600
MODEL_USD = {ARMS[0]: '4', ARMS[1]: '18'}
MODEL_CALLS = {ARMS[0]: 9000, ARMS[1]: 54000}


def plan_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def exclusive(path, value):
    """Publish complete JSON atomically, refusing to replace existing evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, default=str)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)


def prepare(runtime, output, shard):
    runtime, output = Path(runtime).resolve(), Path(output).resolve()
    need(type(shard) is int and 0 <= shard < SHARDS, 'Exactly two fixed shards')
    need(not output.is_relative_to(Path('/gpfs/home/zixianma')), 'Home is read-only')
    need(not output.is_relative_to(runtime), 'Output must be outside source runtime')
    need(not (output / ('shard-%02d' % shard)).exists(), 'Existing output needs finite recovery review')
    raw = (runtime / 'tasks.json').read_bytes()
    tasks = json.loads(raw)
    reference = Path(__file__).resolve().parents[1] / 'portable_assets/campaign_specs/browsercode_gpt_high/task-roster.json'
    expected = {x['task_id'] for x in read(reference)}
    actual = [x['task_id'] for x in tasks if x.get('benchmark', 'online-mind2web').lower() == 'online-mind2web']
    need(len(actual) == len(expected) == 300 and set(actual) == expected, 'Exact full OM2W300 roster required')
    # Archived single-model task files use the display spelling Online-Mind2Web.
    # Normalize only planner copies; retain all original task/runtime bytes.
    ordered = stealth_plan.build([dict(x, benchmark='online-mind2web') for x in tasks
                                  if x['task_id'] in expected], count=300)['pairs']
    ids = [x['task_id'] for x in ordered][shard * PAIRS:(shard + 1) * PAIRS]
    root = output / ('shard-%02d' % shard)
    arms = {}
    for arm in ARMS:
        args = common.parser('gpt').parse_args([
            '--runtime', str(runtime), '--arm', arm, '--output', str(root / arm),
            '--limit', str(PAIRS), '--model-usd', MODEL_USD[arm], '--judge-usd', '4',
            '--model-calls', str(MODEL_CALLS[arm]), '--judge-calls', str(PAIRS),
            '--browser-starts', '152', '--wandb-mode', 'disabled'])
        plan, _ = common.prepare(args, 'gpt')
        need(plan.get('required_cloud', {}).get('proxy_country') == 'us', 'Bound US cloud runtime required')
        need((plan['model'], plan['candidate_effort'], plan['max_output_tokens'], plan['max_decisions'])
             == ('gpt-6-luna', 'high', 12288, 30), 'Exact high/1x/30 actor protocol required')
        need(plan['candidate_count'] == (1 if arm == ARMS[0] else 5), 'Single versus N5 required')
        if arm == ARMS[1]:
            need((plan['selector_effort'], plan['selector_max_output_tokens']) == ('high', 12288), 'Exact selector required')
        plan.update(selected_task_ids=ids, task_selection_count=PAIRS, task_count=PAIRS)
        arms[arm] = dict(args=vars(args), plan=plan)
    judge = read(runtime / 'runtime-config.json')['judge_protocol']
    need({k: judge.get(k) for k in ('model', 'reasoning_effort', 'max_completion_tokens', 'seed')} ==
         dict(model='gpt-5.6-terra', reasoning_effort='medium', max_completion_tokens=4096, seed=42), 'Exact Terra protocol required')
    sources = {str(Path(__file__).with_name(n).resolve()): digest(Path(__file__).with_name(n)) for n in
               ('stealth_study.py', 'stealth_plan.py', 'stealth_pilot.py', 'common.py',
                'browser_lifecycle.py', 'cloud_browser.py', 'cloud_browsercode.py')}
    return dict(schema='browsercode-us-stealth-full300-shard-v1', approved=False,
                runtime=str(runtime), output=str(root), campaign_output=str(output), shard=shard,
                task_ids=ids, all_task_ids=[x['task_id'] for x in ordered], arms=arms,
                task_source_sha256=hashlib.sha256(raw).hexdigest(), reference_roster_sha256=digest(reference),
                runtime_manifest_sha256=digest(runtime / 'source-manifest.json'), source_sha256=sources,
                global_limits=dict(api_usd='60', browser_target_usd='90', browser_admission_stop_usd='80',
                    browser_create_cap=608, actor_counter_calls=126000, terra_calls=600,
                    concurrent_browsers=4, concurrent_actor_http=12, collection_calendar_seconds=604800),
                local_limits=dict(task_pairs=PAIRS, browser_creates=304, initial_mechanical_probes=2,
                    recovery_mechanical_probe_reserve=2, cpus=2, memory_gib=16, gpus=0,
                    seconds_all_attempts=SECONDS, actor_usd=MODEL_USD, terra_usd_per_arm='4'),
                provider_browser_target_is_not_a_hard_cap=True, fresh_cohort=True,
                no_failed_actor_retries=True, all_attempts_counted=True)


def approval_check(plan, approval):
    need(approval.get('approved') is True and approval.get('proxy_overshoot_authorized') is True,
         'Exact financial and inflight-proxy overshoot approval required')
    need(approval.get('plan_sha256') == plan_hash(plan), 'Approval must bind this exact shard plan')
    need(approval.get('global_limits') == plan['global_limits'] and approval.get('local_limits') == plan['local_limits'],
         'Approved caps must exactly match the prepared caps')


class GlobalControl:
    """Cross-shard provider accounting; observed fees never claim a hard cap."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        need(not self.root.is_relative_to(Path('/gpfs/home/zixianma')), 'Home is read-only')

    @contextlib.contextmanager
    def locked(self, *, readonly=False):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / 'financial.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = read(self.root / 'financial.json')
            yield state
            if not readonly:
                write(self.root / 'financial.json', state)

    def initialize(self, first_start, limits):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / 'financial.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.root / 'financial.json'
            if path.exists():
                state = read(path)
                need(state['limits'] == limits, 'Shared financial limits changed')
                # Arrival order can differ from actual Slurm start order.
                state['first_start'] = min(first_start, state['first_start'])
                state['deadline'] = state['first_start'] + limits['collection_calendar_seconds']
                write(path, state)
            else:
                state = dict(limits=limits, first_start=first_start,
                             deadline=first_start + limits['collection_calendar_seconds'],
                             creates={}, sessions={}, abort=None)
                write(path, state)
            return state['deadline']

    @staticmethod
    def total(state):
        return sum((money(x['browserCost']) + money(x['proxyCost'])
                    for x in state['sessions'].values()), Decimal(0))

    def check(self, *, new_pair=False):
        with self.locked(readonly=True) as state:
            need(state['abort'] is None, 'Shared provider pool halted')
            need(time.time() < state['deadline'] - 660, 'Seven-day collection deadline reached')
            limit = state['limits']['browser_admission_stop_usd'] if new_pair else state['limits']['browser_target_usd']
            need(self.total(state) < money(limit), 'Shared provider operational threshold reached')

    def halt(self, reason):
        with self.locked() as state:
            if state['abort'] is None:
                state['abort'] = dict(reason=reason, epoch=time.time())


class Control(pilot.Control):
    """One paired shard barrier plus the shared provider monetary gate."""
    def __init__(self, root, **kwargs):
        super().__init__(root, **kwargs)
        self.config = read(self.root.parent / 'study-control-config.json')
        self.global_control = GlobalControl(self.config['global_control'])

    def halt(self, reason):
        super().halt(reason)
        if str(reason).startswith(('cloud_', 'provider_', 'usage_')):
            self.global_control.halt(str(reason))

    def check(self, state, *, new_pair=False):
        need(state['abort'] is None, 'Paired shard halted')
        need(self.clock() < state['deadline'] - (1860 if new_pair else 660), 'Original time guard')
        self.global_control.check(new_pair=new_pair)

    def pair(self, arm, index, *, after=False, sleep=time.sleep):
        need(arm in ARMS and type(index) is int and 0 <= index < PAIRS, 'Exact paired task identity required')
        field, key = ('finished' if after else 'arrivals'), str(index)
        with self.locked() as state:
            self.check(state)
            if not after:
                need(self.clock() < state['deadline'] - 1860, 'Drain before next task claim')
                if index:
                    need(state['released_pair_phases'].get('finished:' + str(index - 1)) is True, 'Previous pair not released')
            values = state[field].setdefault(key, [])
            need(arm not in values, 'Duplicate paired transition')
            values.append(arm)
        while True:
            with self.locked() as state:
                self.check(state)
                phase = field + ':' + key
                if state['released_pair_phases'].get(phase) is True:
                    return
                if set(state[field][key]) == set(ARMS):
                    self.require_settled(state)
                    if not after:
                        self.check(state, new_pair=True)
                    state['released_pair_phases'][phase] = True
                    return
            sleep(0.1)

    def admission(self, directory, payload):
        key = str(Path(directory).resolve())
        with self.locked() as local:
            self.check(local, new_pair=True)
            need(key not in local['creates'] and len(local['creates']) < 304, 'No extra local create or retry')
            with self.global_control.locked() as state:
                need(state['abort'] is None and GlobalControl.total(state) < money('80'), 'Global admission stopped')
                need(key not in state['creates'] and len(state['creates']) < 608, 'Global browser create cap')
                entry = dict(epoch=self.clock(), payload_sha256=plan_hash(payload), shard=self.config['shard'])
                state['creates'][key] = entry
                local['creates'][key] = entry
        return dict(approved=True, proxy_overshoot_authorized=True, creation_retries=0, traffic_guard='best_effort_not_hard')

    def usage(self, directory, view):
        try:
            key = str(Path(directory).resolve())
            known = all(view.get(k) is not None for k in ('browserCost', 'proxyCost', 'proxyUsedMb'))
            with self.locked() as local:
                need(key in local['creates'], 'Usage lacks local create identity')
                with self.global_control.locked() as global_state:
                    need(key in global_state['creates'], 'Usage lacks global create identity')
                    if not known:
                        global_state['abort'] = global_state['abort'] or dict(reason='provider_usage_unknown', epoch=self.clock())
                        local['abort'] = local['abort'] or dict(reason='provider_usage_unknown', epoch=self.clock())
                        return False
                    previous = local['sessions'].get(key, {})
                    maximum = previous.get('admission_maximum_observed', {})
                    maximum = {k: str(max(money(view[k]), money(maximum.get(k, 0))))
                               for k in ('browserCost', 'proxyCost', 'proxyUsedMb')}
                    local['sessions'][key] = dict(previous, id=view['id'], known=True,
                        status=view.get('status'), epoch=self.clock(), admission_maximum_observed=maximum,
                        latest_provider_usage={k:view[k] for k in maximum})
                    global_state['sessions'][key] = dict(maximum, id=view['id'], status=view.get('status'))
                    if GlobalControl.total(global_state) >= money('90'):
                        global_state['abort'] = global_state['abort'] or dict(reason='observed_browser_target_reached', epoch=self.clock())
                    return global_state['abort'] is None and local['abort'] is None
        except BaseException:
            self.halt('usage_callback_evidence_error')
            return False


def allocation(raw, job, now):
    rows = [dict(x.split('=', 1) for x in line.split() if '=' in x) for line in raw.splitlines()]
    rows = [x for x in rows if x.get('JobId') == job]
    need(len(rows) == 1, 'Exactly one actual Slurm job record required')
    f = rows[0]
    need(f.get('JobState') == 'RUNNING' and f.get('Partition') == 'gpu-h200', 'Actual main-H200 allocation required')
    need(f.get('NumCPUs') == '2' and f.get('NumNodes') == '1' and f.get('Requeue') == f.get('Restarts') == '0', 'Exact non-requeued resource profile required')
    for key in ('ReqTRES', 'AllocTRES'):
        tres = dict(x.split('=', 1) for x in f[key].split(',') if '=' in x)
        need(tres.get('cpu') == '2' and tres.get('mem') in ('16G', '16384M') and tres.get('node') == '1', 'Exact 2CPU/16GiB/node required')
        need(not any('gpu' in k and money(v) != 0 for k, v in tres.items()), 'Zero GPUs required')
    start, end = [datetime.fromisoformat(f[x]).replace(tzinfo=timezone.utc).timestamp()
                  for x in ('StartTime', 'EndTime')]
    need(start <= now < end and 0 < end - start <= SECONDS, 'At most original sixteen-hour allocation')
    return start, end, f


def _child_entry(arm, plan, env_file, control_root):
    # This changes the owner class in this fresh spawned process only. The
    # historical pilot file and its ordinary entry point retain original caps.
    pilot.Control = Control
    raise SystemExit(pilot._worker(arm, plan, env_file, control_root))


def _halt_safely(control, reason, errors):
    """Durable evidence must never prevent the in-memory owner shutdown."""
    try:
        control.halt(reason)
    except BaseException as error:
        errors.append(dict(component='halt_record', error_type=type(error).__name__))


def _signal_started(workers, signalled, errors):
    for worker in workers:
        if worker.pid is None:
            continue
        try:
            if worker.is_alive() and worker.pid not in signalled:
                os.kill(worker.pid, signal.SIGTERM)
                signalled.add(worker.pid)
        except ProcessLookupError:
            pass
        except BaseException as error:
            errors.append(dict(component='child_signal', error_type=type(error).__name__))


def _drain(workers, control, signalled, errors):
    if any(w.pid is not None and w.is_alive() for w in workers):
        _halt_safely(control, 'owner_draining', errors)
        _signal_started(workers, signalled, errors)
    for worker in workers:
        if worker.pid is not None:
            try:
                worker.join()
            except BaseException as error:
                # Attempt every other owned child even if one join itself fails.
                errors.append(dict(component='child_await', error_type=type(error).__name__))


def run(plan, approval, env_file, *, shutdown_event=None):
    approval_check(plan, approval)
    for path, expected in plan['source_sha256'].items():
        need(digest(path) == expected, 'Executed source changed')
    fresh = prepare(plan['runtime'], plan['campaign_output'], plan['shard'])
    need(plan_hash(fresh) == plan_hash(plan), 'Fresh plan differs')
    job = os.environ.get('SLURM_JOB_ID', '')
    need(job.isdigit(), 'Exact approved allocation required')
    result = subprocess.run(['scontrol', 'show', 'job', '-o', job], check=True, text=True,
                            capture_output=True, timeout=30, env=dict(os.environ, TZ='UTC'))
    start, end, fields = allocation(result.stdout, job, time.time())
    global_root = Path(plan['campaign_output']) / 'global-provider-control'
    calendar = GlobalControl(global_root).initialize(start, plan['global_limits'])
    root = Path(plan['output'])
    root.mkdir(parents=True, exist_ok=False)
    exclusive(root / 'execution-plan.json', plan)
    exclusive(root / 'execution-approval.json', approval)
    exclusive(root / 'allocation.json', dict(job_id=job, actual=fields, raw=result.stdout))
    exclusive(root / 'study-control-config.json', dict(global_control=str(global_root), shard=plan['shard']))
    control = Control(root / 'shared-control')
    control.initialize(min(end, calendar))
    shutdown = shutdown_event if shutdown_event is not None else threading.Event()
    need(not shutdown.is_set(), 'External admission stop before any worker starts')
    ctx = multiprocessing.get_context('spawn')
    workers = [ctx.Process(target=_child_entry, args=(arm, plan, env_file, str(control.root))) for arm in ARMS]
    signalled, errors, halt_attempted = set(), [], False
    signal.signal(signal.SIGTERM, lambda *_: shutdown.set())
    try:
        for worker in workers:
            if shutdown.is_set():
                break
            worker.start()
        while any(x.is_alive() for x in workers):
            if any(x.exitcode not in (None, 0) for x in workers):
                shutdown.set()
            if shutdown.is_set() or time.time() >= min(end, calendar) - 660:
                shutdown.set()
                if not halt_attempted:
                    _halt_safely(control, 'owner_shutdown', errors)
                    halt_attempted = True
                _signal_started(workers, signalled, errors)
            for worker in workers:
                if worker.pid is not None:
                    worker.join(timeout=0.25)
    finally:
        # Never put a fallible filesystem write before the final signal/await.
        _drain(workers, control, signalled, errors)
    write(root / 'owner-summary.json', dict(worker_exitcodes={a:w.exitcode for a,w in zip(ARMS,workers)},
          control=control.snapshot(), all_children_awaited=all(w.pid is None or not w.is_alive() for w in workers),
          owner_errors=errors, final_invoice_asserted=False))
    return 0 if all(w.exitcode == 0 for w in workers) and not errors else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--shard', type=int, choices=(0, 1), required=True)
    p.add_argument('--approval', type=Path)
    p.add_argument('--env-file', type=Path)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    plan = prepare(a.runtime, a.output, a.shard)
    if not a.execute:
        print(json.dumps(plan, indent=2, default=str))
        return 0
    need(a.approval is not None, 'Exact separate financial approval required')
    return run(plan, read(a.approval), str(a.env_file) if a.env_file else None)


if __name__ == '__main__':
    raise SystemExit(main())
