#!/usr/bin/env python3
"""Owner of one approved Online-Mind2Web Table 5-style stage; preview by default.

One allocation hosts the three pinned local critics (Shepherd-8B, Arbiter-7B, Piotr
SelectionARM) on one H200 and one shared budget broker, then drains the stage's fixed
cells through the resumable claim store with bounded browser concurrency. Every
claimed cell is a primary attempt that is never rerun. The controller owns and
awaits every server, broker and worker before it exits. No resources are requested
here; execution needs an exact approval naming the API cap inside its own job.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import time
from urllib.request import urlopen

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from openwebrl.webprm_campaign_queue import ClaimStore, checked, run_queue, write_once
from openwebrl.webprm_lifecycle import _process, _processes, _subreaper, OwnedEpisodeTree
from openwebrl.webprm_om2w_table5 import (CAMPAIGN_PROTOCOL, JUDGE, STAGES, cell_outcome, cohort as om2w_cohort,
    protocol_contract, routes as om2w_routes)
from openwebrl.webprm_webarena import digest

CRITICS = ('shepherd8b', 'arbiter7b', 'piotr_selectionarm')
PREFIX = dict(zip(CRITICS, ('shepherd', 'arbiter', 'piotr')))
BROWSER = '/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/browser-env/bin/python'
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE'}
REQUIRED_CODE = ('openwebrl/webprm_om2w_table5.py', 'scripts/run_webprm_om2w_table5_cell.py',
    'scripts/launch_webprm_om2w_table5.py', 'openwebrl/webprm_webarena_worker.py', 'openwebrl/webprm_webarena_episode.py',
    'openwebrl/webprm_webarena.py', 'openwebrl/webprm_worker.py', 'openwebrl/webprm_lifecycle.py',
    'openwebrl/webprm_arbiter_harness.py', 'openwebrl/webprm_source.py', 'openwebrl/webprm_runtime.py',
    'openwebrl/webprm_broker.py', 'openwebrl/webprm_concurrent_broker.py', 'openwebrl/webprm_budget_journal.py',
    'openwebrl/webprm_campaign_queue.py', 'openwebrl/jev_eval.py', 'openwebrl/webprm_webarena_piotr.py',
    'openwebrl/webprm_reddit_pilot_preflight.py', 'openwebrl/eval/reward_online_mind2web.py',
    'scripts/check_webprm_reddit_reset.py', 'scripts/launch_webprm_reddit_pilot.py',
    'scripts/prepare_webprm_piotr_processor.py', 'scripts/runtime_ports.py')


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def put(path, value): write_once(checked(path), value)
def cell_key(cell): return hashlib.sha256(cell['cell_id'].encode()).hexdigest()
def episode_dir(folder, cell): return Path(folder)/'episodes'/('om2w_'+cell['task_id'])/(cell['actor']+'__'+cell['selector'])


def stage_cells(proposal, cohort):
    cells = [c for c in cohort['cells'] if c['stage'] == proposal['stage']]
    if 'cell_ids' in proposal:
        # Tail campaign: only the named, hash-bound cells (e.g. cells a capped predecessor never claimed).
        wanted = proposal['cell_ids']
        if (not isinstance(wanted, list) or not wanted or len(set(wanted)) != len(wanted)
                or not set(wanted) <= {c['cell_id'] for c in cells}):
            raise ValueError('Tail cell list must be a nonempty unique subset of the stage')
        cells = [c for c in cells if c['cell_id'] in set(wanted)]
    return cells


def repair_parent(proposal):
    """Bind a paced operational successor to its unchanged original approval.

    Old attempts keep their original source/plan identities. Only request spacing
    and the controller/collector code needed to preserve that lineage may change.
    This grants no new cells, attempts, dispatches, money or allocation time.
    """
    repair = proposal.get('repair_of')
    if repair is None:
        return None
    if (set(repair) != {'proposal', 'approval', 'reason', 'paid_dispatch_interval_seconds'}
            or repair['reason'] != 'provider429_paid_dispatch_pacing'
            or type(repair['paid_dispatch_interval_seconds']) not in (int, float)
            or repair['paid_dispatch_interval_seconds'] != 3.0):
        raise ValueError('Exact bounded paid dispatch pacing repair required')
    values = {}
    for name in ('proposal', 'approval'):
        item = repair[name]
        if not isinstance(item, dict) or set(item) != {'path', 'sha256'}:
            raise ValueError('Repair parent must be an exact immutable reference')
        path = checked(Path(item['path']))
        if sha(path) != item['sha256'] or proposal['source_sha256'].get(str(path)) != item['sha256']:
            raise ValueError('Repair parent source binding changed')
        values[name] = read(path)
    parent, approval = values['proposal'], values['approval']
    if parent.get('repair_of') is not None:
        raise ValueError('Only the original approved campaign may parent this repair')
    if (approval.get('status') != 'approved' or approval.get('proposal_sha256') != digest(parent)
            or approval.get('resources') != parent['resources'] or approval.get('limits') != parent['limits']
            or approval.get('api_usd_cap') != parent['limits']['api_usd']
            or approval.get('account') != 'zixianma' or not approval.get('user_approval_quote')
            or sha(checked(Path(approval['user_approval_receipt']))) != approval['user_approval_receipt_sha256']):
        raise ValueError('Original exact approval does not authorize this campaign')
    for key in ('protocol', 'stage', 'stage_tasks', 'stage_cells', 'campaign_root', 'cohort_sha256',
                'protocol_sha256', 'broker_routes', 'cell_routes', 'env_file', 'models', 'resources',
                'limits', 'cell_ids', 'candidate_controls', 'scientific_scope'):
        if proposal.get(key) != parent.get(key):
            raise ValueError('Repair changed frozen science or resource authority: '+key)
    for path, expected in parent['source_sha256'].items():
        if sha(checked(Path(path))) != expected:
            raise ValueError('Original frozen source changed: '+path)
    for key in ('task_file', 'cohorts_file', 'cohort_path', 'reference_proposal'):
        if sha(proposal[key]) != sha(parent[key]):
            raise ValueError('Repair changed frozen task/cohort/model inputs: '+key)
    old_template, template = read(parent['worker_template']), read(proposal['worker_template'])
    if ({k: v for k, v in old_template.items() if k != 'code_sha256'}
            != {k: v for k, v in template.items() if k != 'code_sha256'}):
        raise ValueError('Repair changed the worker scientific template')
    allowed = {'openwebrl/webprm_concurrent_broker.py', 'scripts/launch_webprm_om2w_table5.py',
               'scripts/collect_webprm_om2w_table5.py', 'scripts/audit_webprm_om2w_saved_calls.py'}
    old_root = Path(parent['worker_template']).parent/'source'
    new_root = Path(proposal['worker_template']).parent/'source'
    old_code = {str(Path(p).relative_to(old_root)): value for p, value in old_template['code_sha256'].items()}
    new_code = {str(Path(p).relative_to(new_root)): value for p, value in template['code_sha256'].items()}
    if old_code.keys() != new_code.keys() or any(old_code[p] != new_code[p] for p in old_code if p not in allowed):
        raise ValueError('Repair changed scientific worker implementation')
    for p in old_root.rglob('*'):
        if p.is_file():
            rel = p.relative_to(old_root)
            if str(rel) not in allowed and not str(rel).startswith('tests/') and sha(new_root/rel) != sha(p):
                raise ValueError('Repair changed an unrelated frozen implementation: '+str(rel))
    return parent, old_template


def validate_proposal(proposal):
    """Bind cohort, protocol, routes, critics, source and caps; returns (cohort, template)."""
    if proposal.get('protocol') != CAMPAIGN_PROTOCOL or proposal.get('stage') not in STAGES:
        raise ValueError('Explicit OM2W campaign protocol and prespecified stage required')
    source = proposal['source_sha256']
    if any(str(REPO/name) not in source for name in REQUIRED_CODE):
        raise ValueError('Complete controller/worker/broker source freeze required')
    for path, value in source.items():
        if sha(path) != value:
            raise ValueError('Frozen source changed: ' + path)
    for key in ('task_file', 'cohorts_file', 'cohort_path', 'worker_template', 'reference_proposal'):
        if source.get(proposal[key]) is None:
            raise ValueError('Package input must be hash-bound: ' + key)
    template = read(proposal['worker_template'])
    from openwebrl.webprm_om2w_table5 import candidate_control_spec, candidate_control_ids, candidate_control_envelope, CONTROL_SELECTORS
    from openwebrl.webprm_webarena_worker import PRIMARY_SELECTORS
    controls = 'candidate_controls' in proposal
    if controls:
        task_count = proposal['stage_tasks']
        focused = proposal['candidate_controls'].get('identity') == 'om2w_random_candidate_control_100x4_v1'
        spec = candidate_control_spec(task_count, focused=focused)
        if (proposal['candidate_controls'] != spec or proposal['stage'] != 'rest200'
                or template['actors'] != ['gpt-4o-2024-05-13'] or template['selectors'] != spec['selectors']
                or (focused and proposal['limits']['api_usd'] != 500)
                or proposal.get('predecessor') is not None or proposal.get('preproduction_ledger') is not None):
            raise ValueError('Fixed fresh candidate-control study differs')
    elif template['selectors'] != list(PRIMARY_SELECTORS):
        raise ValueError('Primary selectors differ; controls require their explicit development contract')
    expected = om2w_cohort(proposal['task_file'], proposal['cohorts_file'], template['actors'], template['selectors'])
    if controls and (proposal.get('cell_ids') != candidate_control_ids(expected, task_count, focused=focused)
            or proposal['stage_cells'] != task_count*len(spec['selectors'])
            or proposal['resources'] != candidate_control_envelope(task_count, focused=focused)['resources']
            or any(proposal['limits'].get(k) != v for k,v in candidate_control_envelope(task_count, focused=focused)['limits'].items())):
        raise ValueError('Candidate-control membership, resource or dispatch envelope differs')
    if read(proposal['cohort_path']) != expected or proposal['cohort_sha256'] != digest(expected):
        raise ValueError('Fixed 3,000-cell OM2W membership/order differs')
    if any(source.get(path) != value for path, value in template['code_sha256'].items()):
        raise ValueError('Worker/controller implementation freezes differ')
    if (template['protocol_contract'] != protocol_contract(template['arbiter_source'])
            or digest(template['protocol_contract']) != proposal['protocol_sha256']):
        raise ValueError('Common scientific protocol changed')
    reference = read(proposal['reference_proposal'])
    broker_routes, cell_routes = om2w_routes(reference['broker_routes'], reference['cell_routes'], template['actors'],
                                           candidate_controls=controls)
    if (proposal['broker_routes'] != broker_routes or proposal['cell_routes'] != cell_routes
            or proposal['env_file'] != reference['env_file'] or template['env_file'] != reference['env_file']):
        raise ValueError('Exact actor/critic/judge routes and preserved provider environment required')
    if proposal['models'] != reference['models'] or set(proposal['models']) != set(CRITICS):
        raise ValueError('Critic identities must equal the validated Table 5 servers')
    for model in proposal['models'].values():
        if (sha(Path(model['path'])/'config.json') != model['config_sha256']
                or any((Path(model['path'])/f).stat().st_size != size for f, size in model['weight_sizes'].items())):
            raise ValueError('Pinned critic weights/config changed')
    resources, limits = proposal['resources'], proposal['limits']
    if (resources != dict(gpus=1, gpu_type='H200', cpus=8, memory_gib=resources['memory_gib'],
                          total_seconds=resources['total_seconds'])
            or not 0 < resources['memory_gib'] <= 240 or type(resources['total_seconds']) is not int
            or resources['total_seconds'] <= 0):
        raise ValueError('Exact one-H200/8-CPU/<=240 GiB envelope required')
    integers = ('paid_requests', 'local_requests', 'total_requests', 'max_workers', 'per_cell_memory_bytes',
                'fixed_memory_bytes', 'min_start_seconds', 'cleanup_seconds', 'retained_bytes',
                'max_consecutive_invalid', 'maximum_attempts')
    if any(type(limits.get(k)) is not int or limits[k] <= 0 for k in integers):
        raise ValueError('Exact positive dispatch/concurrency/memory/time caps required')
    if (type(limits.get('api_usd')) not in (int, float) or not math.isfinite(limits['api_usd']) or limits['api_usd'] <= 0
            or type(limits.get('min_cell_budget_usd')) not in (int, float) or not 0 < limits['min_cell_budget_usd'] < limits['api_usd']
            or limits['max_workers'] > 8 or limits['total_requests'] != limits['paid_requests'] + limits['local_requests']
            or limits['cleanup_seconds'] < 180
            or limits['fixed_memory_bytes'] + limits['max_workers']*limits['per_cell_memory_bytes'] > resources['memory_gib']*2**30):
        raise ValueError('Inconsistent explicit cumulative caps')
    root = checked(proposal['campaign_root'])
    if 'plan_overrides' in proposal and (proposal.get('fixture_only') is not True or Path('/tmp') not in root.parents
            or set(proposal['plan_overrides']) - {'api_base', 'local_endpoint', 'env_file'}):
        raise ValueError('Provider/critic endpoint overrides are allowed only in /tmp fixture packages')
    repair_parent(proposal)
    return expected, template


def preview(proposal):
    cohort, template = validate_proposal(proposal)
    cells = stage_cells(proposal, cohort)
    return dict(status='prepared_not_executed', stage=proposal['stage'], stage_cells=len(cells),
        stage_tasks=len({c['task_id'] for c in cells}), full_cohort_cells=len(cohort['cells']),
        resources=proposal['resources'], limits=proposal['limits'], launches_models=list(CRITICS),
        judge=JUDGE, fixed_primary_attempts_per_cell=1, allocations=0, provider_calls=0, browser_calls=0)


def cell_plan(template, proposal, context, cell, *, socket_path, ports):
    plan = deepcopy(template)
    plan.update(authorization_status='approved', job_id=context['job_id'], deadline=context['deadline'],
        actor_model=cell['actor'], selector=cell['selector'], task_id=cell['task_id'],
        task_config_sha256=cell['task_config_sha256'], task_file=proposal['task_file'],
        api_usd_cap=proposal['limits']['api_usd'], budget_path=str(Path(proposal['campaign_root'])/'shared-budget.json'),
        broker_socket=str(socket_path))
    for name, model in proposal['models'].items():
        plan[PREFIX[name]+'_model'] = model['served_model']
        plan[PREFIX[name]+'_endpoint'] = f'http://127.0.0.1:{ports[name]}/v1'
    for key in ('api_base', 'env_file'):  # fixture-only overrides, enforced by validate_proposal
        if key in proposal.get('plan_overrides', {}):
            plan[key] = proposal['plan_overrides'][key]
    return plan


def previous_closed_epoch(path):
    """Hash of the last closed financial epoch; any open/incomplete tail blocks a new allocation."""
    path = Path(path)
    if not path.exists():
        return None
    previous, last = '0'*64, None
    with path.open('rb') as stream:
        for sequence, raw in enumerate(stream):
            if not raw.endswith(b'\n'):
                raise ValueError('Incomplete financial tail requires recovery audit')
            event = json.loads(raw); core = {k: v for k, v in event.items() if k != 'event_sha256'}
            if event.get('sequence') != sequence or event.get('previous_sha256') != previous or event.get('event_sha256') != digest(core):
                raise ValueError('Financial journal chain differs')
            previous, last = event['event_sha256'], event
    if last is None or last['event'] not in ('header', 'epoch_closed'):
        raise ValueError('Financial epoch is not closed; independent recovery required before a new allocation')
    return last['event_sha256'] if last['event'] == 'epoch_closed' else None


def allocation_memory_bytes(job_id, *, proc=Path('/proc/self/cgroup'), cgroups=Path('/sys/fs/cgroup')):
    for line in proc.read_text().splitlines():
        _, controllers, relative = line.split(':', 2)
        if controllers and 'memory' not in controllers.split(','):
            continue
        path = cgroups/('memory' if controllers else '')/relative.lstrip('/')
        for candidate in (path, *path.parents):
            if candidate.name == 'job_'+str(job_id):
                text = (candidate/('memory.usage_in_bytes' if controllers else 'memory.current')).read_text().strip()
                if not text.isdigit():
                    raise ValueError('Allocation cgroup memory usage is not finite')
                return int(text)
    raise ValueError('Owned allocation memory cgroup is unavailable')


def tree_bytes(root):
    total = 0
    for directory, _, files in os.walk(root):
        for name in files:
            try:
                total += os.lstat(os.path.join(directory, name)).st_size
            except FileNotFoundError:
                pass
    return total


async def wait_process(process, deadline, check, *, allowed_returncodes=(0,)):
    while process.poll() is None:
        if time.time() >= deadline:
            raise TimeoutError('Shared allocation deadline reached')
        check(); await asyncio.sleep(.5)
    if process.returncode not in allowed_returncodes:
        raise RuntimeError('Owned command failed; attempt retained without automatic retry')


class Runner:
    def __init__(self, proposal, template, context, output, scratch):
        from scripts.launch_webprm_reddit_pilot import environment
        self.proposal, self.template, self.context = proposal, template, context
        self.output, self.scratch = Path(output), Path(scratch)
        # Chromium's singleton socket lives at TMPDIR/.org.chromium.Chromium.XXXXXX/SingletonSocket (46 chars)
        # under the browser runtime root <scratch>/browser/wprm-XXXXXXXX; AF_UNIX paths stop at 107 bytes.
        # Fixture 358267: a 72-char root made every Chromium launch exit immediately.
        if len(os.fsencode(self.scratch/'browser'))+14+46 > 107:
            raise ValueError('Campaign scratch path too long for Chromium singleton socket')
        self.env = environment(self.scratch)
        # Keep local readiness and model services off inherited cluster proxies.
        bypass = {item.strip() for name in ('NO_PROXY', 'no_proxy')
                  for item in self.env.get(name, '').split(',') if item.strip()}
        bypass.update(('localhost', '127.0.0.1', '::1'))
        self.env['NO_PROXY'] = self.env['no_proxy'] = ','.join(sorted(bypass))
        self.env['PYTHONPATH'] = str(REPO)
        os.environ.update(self.env)
        self.servers, self.children, self.plans, self.folders = [], [], {}, {}
        self.broker = self.port_lease = None
        self.halt_reason = None
        self.consecutive_invalid = 0
        self.last_storage, self.retained = 0., 0
        self.sequence = 0

    def check(self):
        if any(p.poll() is not None for p in self.servers):
            raise RuntimeError('An owned critic server exited')
        if self.broker is not None and not self.broker.process.is_alive():
            raise RuntimeError('Shared budget broker exited')
        if time.time() >= self.context['deadline']:
            raise TimeoutError('Allocation deadline reached')

    def command(self, argv, log, env=None):
        from scripts.launch_webprm_reddit_pilot import _command
        process = _command(argv, log, self.env if env is None else env)
        self.children.append(process)
        return process

    async def start_servers(self):
        from scripts.launch_webprm_reddit_pilot import model_command, piotr_compatible_config
        from openwebrl.webprm_worker import loopback_urlopen
        from scripts.runtime_ports import lease_ports
        ports_dir = self.scratch/'ports'; ports_dir.mkdir(mode=0o700)
        self.port_lease, base = lease_ports(self.context['job_id'], size=3, directory=ports_dir)
        ports = {name: base+i for i, name in enumerate(CRITICS)}
        for name, model in self.proposal['models'].items():
            compatible = None
            if name == 'piotr_selectionarm':
                compatible = self.output/'piotr-compatible-config.json'; put(compatible, piotr_compatible_config(model))
            env = dict(self.env, CUDA_VISIBLE_DEVICES=str(model['gpu_index']))
            argv = model_command(name, model, ports[name], compatible_config=compatible)
            put(self.output/(name+'-command.json'), dict(argv=argv, gpu_index=model['gpu_index']))
            self.servers.append(self.command(argv, self.output/(name+'-server.log'), env))
        until = min(self.context['deadline'], time.time()+1200); ready = set()
        while len(ready) < len(CRITICS):
            self.check()
            if time.time() >= until:
                raise TimeoutError('Critic startup deadline reached')
            for name in CRITICS:
                try:
                    with loopback_urlopen(f'http://127.0.0.1:{ports[name]}/v1/models', timeout=2) as response:
                        if self.proposal['models'][name]['served_model'] not in [v['id'] for v in json.load(response)['data']]:
                            raise ValueError('Served critic model differs')
                    ready.add(name)
                except OSError:
                    pass
            await asyncio.sleep(1)
        return ports

    async def start(self, cells, ports):
        from openwebrl.webprm_concurrent_broker import ConcurrentBrokerProcess
        from openwebrl.webprm_budget_journal import VERSION as BUDGET_VERSION
        from openwebrl.webprm_broker import make_broker_client
        from openwebrl.webprm_reddit_pilot_preflight import validate_local_critics
        self.socket = self.scratch/'broker/socket'; self.socket.parent.mkdir(mode=0o700)
        if len(os.fsencode(self.socket)) >= 108:
            raise ValueError('Broker socket path too long')
        route_table = {}
        for route in self.proposal['broker_routes']:
            endpoint = (self.proposal.get('plan_overrides', {}).get('local_endpoint') or
                        f"http://127.0.0.1:{ports[route['critic']]}/v1") if route['local'] else None
            route_table[(route['model'], route['purpose'])] = dict(local=route['local'], endpoint=endpoint,
                max_input_tokens=route['max_input_tokens'], max_output_tokens=route['max_output_tokens'])
        registrations = {}
        for cell in cells:
            folder = self.output/'cells'/cell_key(cell); folder.mkdir(parents=True, mode=0o700)
            plan = cell_plan(self.template, self.proposal, self.context, cell, socket_path=self.socket, ports=ports)
            put(folder/'worker-plan.json', plan); self.plans[cell['cell_id']] = plan; self.folders[cell['cell_id']] = folder
            journal = episode_dir(folder, cell)/'api.jsonl'; journal.parent.mkdir(parents=True, mode=0o700)
            registrations[cell['cell_id']] = dict(config=dict(plan, identity_sha256=digest(plan)),
                journal=str(journal), allowed_routes=self.proposal['cell_routes'][cell['actor']][cell['selector']])
        if not registrations:
            raise ValueError('No pending cells in this stage')
        diagnostic = next(iter(registrations.values()))['config']
        registrations['local_preflight'] = dict(config=diagnostic, journal=str(self.output/'local-preflight-api.jsonl'),
            allowed_routes=[list(k) for k, v in route_table.items() if v['local']])
        limits = self.proposal['limits']
        financial = dict(schema_version=1, budget_backend=BUDGET_VERSION, cohort_sha256=self.proposal['cohort_sha256'],
            protocol_sha256=self.proposal['protocol_sha256'], prices_usd_per_token=diagnostic['prices_usd_per_token'],
            api_usd_cap=limits['api_usd'], max_dispatches=limits['total_requests'],
            max_paid_dispatches=limits['paid_requests'], max_local_dispatches=limits['local_requests'])
        put(self.output/'financial-contract.json', financial)
        previous = previous_closed_epoch(Path(self.proposal['campaign_root'])/'shared-budget.json.events.jsonl')
        put(self.output/'financial-handoff.json', dict(previous_closed_epoch_sha256=previous))
        self.broker = ConcurrentBrokerProcess(socket_path=str(self.socket), cells=registrations, routes=route_table,
            deadline=self.context['deadline'], max_dispatches=limits['total_requests'],
            max_paid_dispatches=limits['paid_requests'], max_local_dispatches=limits['local_requests'],
            max_concurrency=limits['max_workers'], financial_contract=financial, epoch_id=self.context.get('epoch_id', 'job-'+self.context['job_id']),
            paid_dispatch_interval_seconds=self.proposal.get('repair_of', {}).get('paid_dispatch_interval_seconds', 0.0),
            previous_closed_epoch_sha256=previous, audit_path=str(self.output/'broker.jsonl')).start()
        client = make_broker_client(str(self.socket), cell_id='local_preflight', identity_sha256=diagnostic['identity_sha256'],
            deadline=min(self.context['deadline'], time.time()+600))
        proof = await asyncio.to_thread(validate_local_critics, diagnostic, client, self.output)
        put(self.output/'local_preflight-receipt.json', proof)

    def domain_ready(self, domain):
        return domain == 'om2w-'+self.proposal['stage']

    def decide(self, cell, active, allowed, reason, **details):
        folder = self.output/'admission-decisions'; folder.mkdir(mode=0o700, exist_ok=True)
        put(folder/f'{self.sequence:05d}.json', dict(protocol='om2w_admission_decision_v1', cell_id=cell['cell_id'],
            active_workers=active, epoch=time.time(), allowed=allowed, reason=reason, **details))
        self.sequence += 1
        return allowed

    def can_start(self, cell, active):
        self.check(); limits = self.proposal['limits']
        if self.halt_reason is not None:
            return self.decide(cell, active, False, 'halted', halt_reason=self.halt_reason)
        if time.time()+limits['min_start_seconds'] >= self.context['deadline']:
            return self.decide(cell, active, False, 'insufficient_start_time')
        memory = allocation_memory_bytes(self.context['job_id'])
        projected = memory+(active+1)*limits['per_cell_memory_bytes']
        if projected > self.proposal['resources']['memory_gib']*2**30:
            return self.decide(cell, active, False, 'allocation_memory_capacity', projected_memory_bytes=projected)
        if time.monotonic()-self.last_storage > 600:
            self.retained = tree_bytes(self.proposal['campaign_root']); self.last_storage = time.monotonic()
        if self.retained > limits['retained_bytes']:
            return self.decide(cell, active, False, 'retained_artifact_cap', retained_bytes=self.retained)
        status = self.broker.status(); budget = status['budget']
        if (status.get('admission_open') is not True or budget.get('halt_reason') is not None
                or float(budget['remaining']) < limits['min_cell_budget_usd']
                or budget['paid_dispatches'] >= limits['paid_requests'] or budget['local_dispatches'] >= limits['local_requests']):
            return self.decide(cell, active, False, 'broker_admission_closed', budget_remaining_usd=budget['remaining'])
        return self.decide(cell, active, True, 'all_gates_passed', budget_remaining_usd=budget['remaining'],
                           current_memory_bytes=memory)

    async def run_one(self, cell):
        from scripts.launch_webprm_reddit_pilot import _stop
        folder, plan = self.folders[cell['cell_id']], self.plans[cell['cell_id']]
        worker = None
        try:
            argv = [BROWSER, '-B', str(REPO/'scripts/run_webprm_om2w_table5_cell.py'), '--plan', str(folder/'worker-plan.json'),
                    '--output', str(folder/'episodes'), '--task-id', cell['task_id'], '--actor', cell['actor'],
                    '--selector', cell['selector']]
            put(folder/'worker-launch.json', dict(argv=argv, cell_id=cell['cell_id'], resampling_forbidden=True))
            worker = self.command(argv, folder/'worker.log')
            await wait_process(worker, self.context['deadline'], self.check)
            outcome = cell_outcome(episode_dir(folder, cell)/'result.json', cell, digest(plan))
            status = self.broker.status()
            if cell['cell_id'] in status['active_cell_ids']:
                raise RuntimeError('Cell still owns an in-flight model request after worker closure')
            put(folder/'broker-status.json', status)
            put(folder/'cell-outcome.json', outcome)
            self.consecutive_invalid = 0 if outcome['valid'] else self.consecutive_invalid+1
            return outcome
        except BaseException:
            self.consecutive_invalid += 1
            raise
        finally:
            _stop(worker)
            if self.consecutive_invalid >= self.proposal['limits']['max_consecutive_invalid'] and self.halt_reason is None:
                self.halt_reason = f'{self.consecutive_invalid} consecutive invalid cells; admission closed for review'
                put(self.output/'halt.json', dict(reason=self.halt_reason, epoch=time.time()))

    async def run(self, cells, store):
        from scripts.launch_webprm_reddit_pilot import _stop
        try:
            pending = [c for c in cells if store.state(c) == 'pending']
            if not pending:
                # Claimed cells (recorded or attempted without outcome) are never rerun.
                return dict(reason='no_pending_cells', admitted=[], closed=[], pending=[],
                    recorded=[c['cell_id'] for c in cells if store.state(c) == 'recorded'],
                    attempted_without_outcome=[c['cell_id'] for c in cells if store.state(c) == 'attempted_without_outcome'])
            ports = await self.start_servers()
            await self.start(pending, ports)
            return await run_queue(cells, store, domain_ready=self.domain_ready, can_start=self.can_start,
                run_one=self.run_one, max_workers=self.proposal['limits']['max_workers'],
                stop_admission=lambda: self.halt_reason is not None or time.time()+self.proposal['limits']['min_start_seconds'] >= self.context['deadline'])
        finally:
            for process in reversed(self.children):
                _stop(process)
            if self.broker is not None:
                put(self.output/'broker-closure.json', self.broker.close())
            if self.port_lease is not None:
                self.port_lease.close()


def _runner(proposal, template, context, output, scratch, cells):
    os.setsid()
    with ClaimStore(Path(proposal['campaign_root'])/'queue', cells) as store:
        runner = Runner(proposal, template, context, output, scratch)
        # A fork inherits the outer owner's handlers, whose private stop flag is
        # never read here. Close admission while retaining ownership of workers.
        def stop_admission(*_):
            if runner.halt_reason is None:
                runner.halt_reason = 'supervisor_requested_stop'
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, stop_admission)
        summary = asyncio.run(runner.run(cells, store))
        put(Path(output)/'queue-summary.json', dict(summary, halt_reason=runner.halt_reason))


def scheduler_context(proposal, job, prior_seconds, *, now=None):
    from scripts.launch_webprm_reddit_pilot import cpu_scheduler_epochs
    text = subprocess.check_output(['scontrol', 'show', 'job', '-o', job], text=True, timeout=15)
    raw = subprocess.check_output(['scontrol', '--json', 'show', 'job', job], text=True, timeout=15)
    fields = dict(re.findall(r'(\w+)=(\S+)', text))
    tres = dict(item.split('=', 1) for item in fields.get('AllocTRES', '').split(',') if '=' in item)
    resources = proposal['resources']
    if (fields.get('JobId') != job or fields.get('JobState') != 'RUNNING' or fields.get('NumNodes') != '1'
            or fields.get('NumCPUs') != str(resources['cpus']) or tres.get('gres/gpu:h200') != str(resources['gpus'])
            or tres.get('mem') not in (str(resources['memory_gib'])+'G', str(resources['memory_gib']*1024)+'M')):
        raise ValueError('Actual scheduler allocation differs from approval')
    epochs = cpu_scheduler_epochs(raw, job)
    if prior_seconds+epochs['limit_seconds'] > resources['total_seconds']:
        raise ValueError('Every prior allocation attempt counts against the cumulative approved time')
    deadline = min(epochs['end_epoch'], epochs['start_epoch']+epochs['limit_seconds'])-proposal['limits']['cleanup_seconds']
    now = time.time() if now is None else now
    if not epochs['start_epoch'] <= now < deadline:
        raise ValueError('No approved runtime remains before mandatory cleanup')
    return dict(job_id=job, deadline=deadline, scheduler_epochs=epochs, prior_seconds=prior_seconds)


def validate_approval(proposal, approval):
    if (approval.get('status') != 'approved' or approval.get('proposal_sha256') != digest(proposal)
            or approval.get('api_usd_cap') != proposal['limits']['api_usd'] or approval.get('resources') != proposal['resources']
            or approval.get('limits') != proposal['limits'] or approval.get('stage') != proposal['stage']
            or not isinstance(approval.get('user_approval_quote'), str) or not approval['user_approval_quote'].strip()
            or ('user_approval_receipt' in approval and
                hashlib.sha256(Path(approval['user_approval_receipt']).read_bytes()).hexdigest()
                != approval.get('user_approval_receipt_sha256'))):
        raise ValueError('Exact user-approved OM2W stage request (API cap, resources, quote) required')


def preproduction_seconds(proposal, *, history=None):
    """Charge explicitly bound earlier fixture allocations without importing scripted API usage."""
    ledger_path = proposal.get('preproduction_ledger')
    if ledger_path is None:
        return 0
    if proposal['source_sha256'].get(ledger_path) != sha(ledger_path):
        raise ValueError('Preproduction accounting ledger changed')
    rows = read(ledger_path)['allocations']
    jobs = [row['job_id'] for row in rows]
    if (not rows or len(set(jobs)) != len(jobs) or
            any(not isinstance(j, str) or not j.isdigit() for j in jobs) or
            any(type(row.get('elapsed_seconds')) is not int or row['elapsed_seconds'] < 0 for row in rows)):
        raise ValueError('Unique explicit preproduction allocation accounting required')
    if history is None:
        history = subprocess.check_output(['sacct', '-j', ','.join(jobs), '-X', '--noheader', '--parsable2',
            '--format=JobID,State,ElapsedRaw'], text=True, timeout=15)
    actual = [line.split('|') for line in history.strip().splitlines()]
    if (len(actual) != len(rows) or {r[0] for r in actual} != set(jobs) or
            any(len(r) < 3 or r[1].split()[0] not in TERMINAL or not r[2].isdigit() for r in actual)):
        raise ValueError('Preproduction allocations are not terminal/accounted')
    expected = {row['job_id']: row['elapsed_seconds'] for row in rows}
    if any(int(r[2]) != expected[r[0]] for r in actual):
        raise ValueError('Preproduction scheduler time changed')
    return sum(expected.values())


def register_attempt(proposal, approval, job):
    """Durable attempt registry; prior attempts must be terminal and their time is charged."""
    root = checked(proposal['campaign_root']); root.mkdir(mode=0o700, exist_ok=True)
    if root.stat().st_mode & 0o077:
        raise ValueError('Campaign root must be private 0700')
    folder = root/'allocations'; folder.mkdir(mode=0o700, exist_ok=True)
    rows = [read(p) for p in sorted(folder.glob('*.json'))]
    if [r['sequence'] for r in rows] != list(range(1, len(rows)+1)):
        raise ValueError('Allocation registry is not contiguous')
    jobs = [r['job_id'] for r in rows]
    prior = preproduction_seconds(proposal)
    preproduction_jobs = ([r['job_id'] for r in read(proposal['preproduction_ledger'])['allocations']]
                          if proposal.get('preproduction_ledger') else [])
    if job in preproduction_jobs or set(jobs) & set(preproduction_jobs):
        raise ValueError('Preproduction allocation cannot be reused')
    if job in jobs:
        raise ValueError('An allocation attempt is never reused or requeued')
    if len(jobs) >= proposal['limits']['maximum_attempts']:
        raise ValueError('Approved allocation-attempt cap reached')
    if jobs:
        for old in jobs:
            closure = root/('attempt-'+old)/'controller-closure.json'
            if closure.exists() and read(closure).get('remaining_owned_processes') != {}:
                raise ValueError('Previous allocation process closure is incomplete')
        history = subprocess.check_output(['sacct', '-j', ','.join(jobs), '-X', '--noheader', '--parsable2',
            '--format=JobID,State,ElapsedRaw'], text=True, timeout=15)
        rows_ = [line.split('|') for line in history.strip().splitlines()]
        if {r[0] for r in rows_} != set(jobs) or any(r[1].split()[0] not in TERMINAL or not r[2].isdigit() for r in rows_):
            raise ValueError('Prior allocation attempts are not terminal/accounted')
        prior += sum(int(r[2]) for r in rows_)
    put(folder/f'{len(jobs)+1:04d}-{job}.json', dict(sequence=len(jobs)+1, job_id=job, previous_job_ids=jobs,
        proposal_sha256=digest(proposal), approval_sha256=digest(approval), epoch=time.time()))
    return prior


def execute(proposal, approval):
    from scripts.launch_webprm_reddit_pilot import SCRATCH
    validate_approval(proposal, approval)
    cohort, template = validate_proposal(proposal)
    job = os.environ.get('SLURM_JOB_ID', '')
    if not job.isdigit() or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Controller must run inside its own Slurm allocation')
    lock = checked(Path(proposal['campaign_root']).parent/'controller.lock')
    with lock.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = register_attempt(proposal, approval, job)
        context = scheduler_context(proposal, job, prior)
        root = Path(proposal['campaign_root'])
        output = root/('attempt-'+job); output.mkdir(mode=0o700)
        scratch = SCRATCH/('o'+secrets.token_hex(3)); scratch.mkdir(mode=0o700, parents=True)
        context.update(proposal_sha256=digest(proposal), approval_sha256=digest(approval), scratch=str(scratch),
            output=str(output), controller_sha256=sha(__file__), owner_pid=os.getpid(),
            owner_start_ticks=_process(os.getpid())['start_ticks'])
        put(output/'execution-context.json', context)
        cells = stage_cells(proposal, cohort)
        previous = _subreaper(True); baseline = set(_processes()); tree = None; stopped = [False]
        handlers = {sig: signal.signal(sig, lambda *_: stopped.__setitem__(0, True)) for sig in (signal.SIGTERM, signal.SIGINT)}
        process = multiprocessing.get_context('fork').Process(target=_runner,
            args=(proposal, template, context, str(output), str(scratch), cells))
        try:
            process.start(); tree = OwnedEpisodeTree(process, baseline)
            while process.is_alive() and time.time() < context['deadline']+60 and not stopped[0]:
                tree.refresh(); time.sleep(.5)
            if process.is_alive():
                os.kill(process.pid, signal.SIGTERM)
                # Preserve the allocation cleanup bound while allowing an owned
                # browser phase and its cleanup to finish after admission closes.
                grace = min(240, max(0, context['deadline']+proposal['limits']['cleanup_seconds']-time.time()))
                process.join(timeout=grace)
        finally:
            closure = tree.stop(10) if tree else dict(remaining_owned_processes={}, child_exitcode=None)
            put(output/'controller-closure.json', closure)
            _subreaper(previous)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        if closure['remaining_owned_processes'] or closure['child_exitcode'] != 0:
            raise RuntimeError('Campaign attempt failed; durable claims and budget remain charged')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal', required=True, type=Path)
    parser.add_argument('--approval', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    proposal = read(args.proposal)
    if not args.execute:
        print(json.dumps(preview(proposal), indent=2)); return
    if args.approval is None:
        parser.error('--execute requires --approval')
    execute(proposal, read(args.approval))


if __name__ == '__main__':
    main()
